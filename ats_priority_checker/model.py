"""Text -> priority model, and the mismatch-flagging logic built on top of it.

Given only a few hundred reports, this deliberately does NOT train a neural
net from scratch. It uses a pretrained sentence-embedding model (frozen,
not fine-tuned) to turn report text into vectors, then trains a simple
classifier on top to predict "what priority does this text imply". A
report is flagged when the stated priority disagrees with what the text
implies.

This design is meant to scale with more data later: same pipeline, same
saved artifacts format, just re-run train_priority_classifier on a bigger
dataset.csv as more labeled/parsed reports come in.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def report_text(row: pd.Series) -> str:
    """Combine recommendations + comments into the text the model sees."""
    recs = row.get("recommendations") or ""
    comments = row.get("comments") or ""
    return f"Recommendations: {recs}\nComments: {comments}".strip()


def embed_texts(texts: list[str], model_name: str = EMBEDDING_MODEL_NAME) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model.encode(list(texts), show_progress_bar=True, normalize_embeddings=True)


def cross_validated_predictions(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
    """Out-of-fold predictions: for every row, the predicted priority comes
    from a model that never saw that row during training. This is what you
    want when flagging mismatches on your existing labeled reports - using
    a model's in-sample predictions on its own training data would make
    the stated priority look "confirmed" more often than it should.
    """
    class_counts = pd.Series(y).value_counts()
    n_splits = min(n_splits, int(class_counts.min())) if len(class_counts) > 0 else n_splits
    n_splits = max(n_splits, 2)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")

    pred = cross_val_predict(clf, X, y, cv=skf, method="predict")
    proba = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")
    classes = np.unique(y)

    return {"pred": pred, "proba": proba, "classes": classes, "n_splits": n_splits}


def fit_final_model(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """Fit on ALL available data - this is the model you save and reuse for
    scoring brand-new reports (not for evaluating the training set itself,
    use cross_validated_predictions for that)."""
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X, y)
    return clf


def flag_mismatches(
    df: pd.DataFrame,
    pred: np.ndarray,
    proba: np.ndarray,
    classes: np.ndarray,
    low_confidence_threshold: float = 0.15,
) -> pd.DataFrame:
    """Compare stated priority to text-implied priority, and to the
    Spectrum peak-amplitude and cross-report escalation graph signals
    where available.

    Flags a row when:
      - the predicted (argmax) priority differs from the stated priority, OR
      - the model's confidence in the STATED priority is very low, even if
        it isn't the top prediction for another class (catches "text reads
        as ambiguous / doesn't clearly support this priority" cases, not
        just clean disagreements), OR
      - the Spectrum peak-amplitude hint (see graph_signals.py) suggests a
        MORE urgent priority than stated, OR
      - the escalation hint (see dataset.add_escalation_signals) - this
        equipment's reading jumped to a more severe priority bucket, or
        dropped drastically, versus its own most recent earlier test -
        suggests a MORE urgent priority than stated.

    Both graph hints are allowed to flag on their own, even when the text
    agrees with the stated priority - the whole reason they exist is to
    catch cases the text can't: report writers sometimes under-state
    urgency for equipment that's been high-priority before, in which case
    the text itself may honestly match a priority that's actually too low.

    Graph hints only flag in ONE direction: hint < stated (lower number =
    more urgent). A hint suggesting LESS urgency than stated is not a
    mismatch signal - it just means the graph alone doesn't capture
    whatever else supports the higher stated priority (comments, repair
    history, other context), which is expected and not what these signals
    exist to catch. Flagging both directions was tried and produced
    disagreements on ~9-14% of otherwise-correct reports with zero gain in
    real mismatch detection (see the conversation this fix is from for the
    concrete evidence) - real prioritization is holistic, so a raw
    threshold rule landing one level more cautious than the stated
    priority is normal and not evidence of an error.
    """
    out = df.copy()
    out["predicted_priority"] = pred

    class_to_idx = {c: i for i, c in enumerate(classes)}
    stated_conf = []
    for stated, row_proba in zip(out["priority_num"], proba):
        idx = class_to_idx.get(stated)
        stated_conf.append(row_proba[idx] if idx is not None else np.nan)
    out["confidence_in_stated_priority"] = stated_conf

    text_disagrees = out["predicted_priority"] != out["priority_num"]
    low_conf = out["confidence_in_stated_priority"] < low_confidence_threshold

    def _disagrees(col: str) -> pd.Series:
        if col not in out.columns:
            return pd.Series(False, index=out.index)
        hint = out[col]
        return hint.notna() & (hint < out["priority_num"])

    spectrum_disagrees = _disagrees("spectrum_priority_hint")
    escalation_disagrees = _disagrees("escalation_priority_hint")

    out["flag_mismatch"] = text_disagrees | low_conf.fillna(False) | spectrum_disagrees | escalation_disagrees

    def reason(r):
        parts = []
        if r["predicted_priority"] != r["priority_num"]:
            parts.append(f"text implies priority {r['predicted_priority']:g}, report states {r['priority_num']:g}")
        elif r["confidence_in_stated_priority"] < low_confidence_threshold:
            parts.append("text is a weak/ambiguous match for the stated priority")
        spectrum_hint = r.get("spectrum_priority_hint")
        if pd.notna(spectrum_hint) and spectrum_hint < r["priority_num"]:
            parts.append(f"Spectrum peak reading suggests priority {spectrum_hint:g}, report states {r['priority_num']:g}")
        escalation_hint = r.get("escalation_priority_hint")
        if pd.notna(escalation_hint) and escalation_hint < r["priority_num"]:
            escalation_why = r.get("escalation_reason", "escalating vs. this equipment's prior test")
            parts.append(f"escalation ({escalation_why}) suggests priority {escalation_hint:g}, report states {r['priority_num']:g}")
        return "; ".join(parts)

    out["flag_reason"] = out.apply(reason, axis=1)
    return out


def priority_signal_reports(df: pd.DataFrame, true_col: str = "true_priority") -> dict:
    """Score each priority SIGNAL independently against true_col, each with
    its own per-class precision/recall/f1 - not folded into one flag/no-flag
    number the way flag_mismatch is.

    `predicted_priority` (text-only, from the Recommendations/Comments
    embedding model) is not the only signal worth a report card of its own:
    a report can be written in the same mild, boilerplate language every
    other Priority 4 report uses while its own Spectrum chart reads far
    more severe than that language suggests - the text model has no way to
    catch that, since it never sees the chart. Scoring
    spectrum_priority_hint and escalation_priority_hint the same way
    predicted_priority already gets scored surfaces exactly that gap.

    Reports on, when present in df:
      - predicted_priority: the text-only model. Broadest coverage (every
        row that went into training/cross-validation).
      - spectrum_priority_hint: THIS report's own Spectrum peak reading -
        no report history needed, so it's available on nearly every report
        with a readable chart. This is the one that catches "the writeup
        reads mild but the chart doesn't" on a single report, standalone.
      - escalation_priority_hint: cross-report signal - THIS report's
        Spectrum peak compared against the SAME equipment's prior dated
        test. Only populated when escalation_flag is True, so its n will
        be much smaller than the other two - that's expected, not a bug:
        most reports have nothing escalating to flag, and the first-ever
        report for a piece of equipment never has a prior test to compare
        against at all.

    Each signal is scored ONLY on the rows where it isn't null - you can't
    score a hint that was never produced, and a signal's n here is itself
    informative about how often it actually has something to say, not just
    how accurate it is when it does.

    Returns {signal_name: {"n": int, "report_text": str, "report_dict": dict}}
    for whichever of the three columns are present in df; a report_dict
    entry is None if all class-metric arrays end up empty (currently only
    ever thrown by sklearn on genuinely undefined input, e.g. n=0).
    """
    from sklearn.metrics import classification_report

    signal_cols = ["predicted_priority", "spectrum_priority_hint", "escalation_priority_hint"]
    results = {}
    for col in signal_cols:
        if col not in df.columns:
            continue
        sub = df.dropna(subset=[col, true_col])
        if len(sub) == 0:
            results[col] = {"n": 0, "report_text": "(no rows with both a value and a ground-truth priority)", "report_dict": None}
            continue
        results[col] = {
            "n": len(sub),
            "report_text": classification_report(sub[true_col], sub[col], zero_division=0),
            "report_dict": classification_report(sub[true_col], sub[col], zero_division=0, output_dict=True),
        }
    return results


def priority_signal_table(df: pd.DataFrame, true_col: str = "true_priority") -> pd.DataFrame:
    """Row-level companion to priority_signal_reports: one row per report
    with predicted_priority, spectrum_priority_hint, and
    escalation_priority_hint side by side, plus a `*_correct` bool for each
    against true_col - for eyeballing exactly which signal got which report
    right, not just the aggregate precision/recall.
    """
    signal_cols = [c for c in ("predicted_priority", "spectrum_priority_hint", "escalation_priority_hint") if c in df.columns]
    id_cols = [c for c in ("report_id", "equipment_id", "priority_num", true_col) if c in df.columns]
    out = df[id_cols + signal_cols].copy()
    for col in signal_cols:
        # NaN (signal didn't fire) is "not applicable", not "wrong" - keep
        # it as a null rather than letting `NaN == true_col` silently read
        # as False, which would make an escalation-didn't-fire row look
        # identical to an escalation-fired-and-was-wrong row. Uses pandas'
        # nullable "boolean" dtype specifically so this prints as
        # True/False/<NA> - plain np.where here upcasts the whole column to
        # float the moment a NaN is mixed in, so "correct" silently renders
        # as 1.0/0.0 sitting right next to the priority-number columns,
        # which reads at a glance like a 4th priority decision instead of a
        # true/false flag.
        correct = pd.Series(np.where(out[col].isna(), pd.NA, out[col] == out[true_col]), index=out.index)
        out[f"{col}_correct"] = correct.astype("boolean")
    return out


def save_bundle(clf: LogisticRegression, path: str | Path, embedding_model_name: str = EMBEDDING_MODEL_NAME) -> None:
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, path)
    with open(path.with_suffix(".meta.json"), "w") as f:
        json.dump({"embedding_model_name": embedding_model_name}, f)


def load_bundle(path: str | Path) -> tuple[LogisticRegression, str]:
    import joblib

    path = Path(path)
    clf = joblib.load(path)
    with open(path.with_suffix(".meta.json")) as f:
        meta = json.load(f)
    return clf, meta["embedding_model_name"]
