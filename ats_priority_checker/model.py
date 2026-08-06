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
    Spectrum Fund Amp and Trend graph signals where available.

    Flags a row when:
      - the predicted (argmax) priority differs from the stated priority, OR
      - the model's confidence in the STATED priority is very low, even if
        it isn't the top prediction for another class (catches "text reads
        as ambiguous / doesn't clearly support this priority" cases, not
        just clean disagreements), OR
      - the Spectrum Fund Amp hint (velocity/in-s reports only - see
        graph_signals.py) suggests a MORE urgent priority than stated, OR
      - the Trend hint (velocity/in-s reports only, escalating cases only -
        see graph_signals.py) suggests a MORE urgent priority than stated.

    Both graph hints are allowed to flag on their own, even when the text
    agrees with the stated priority - the whole reason they exist is to
    catch cases the text can't: report writers sometimes under-state
    urgency for equipment that's been high-priority before, in which case
    the text itself may honestly match a priority that's actually too low.

    Graph hints only flag in ONE direction: hint < stated (lower number =
    more urgent). A hint suggesting LESS urgency than stated is not a
    mismatch signal - it just means the graph alone doesn't capture
    whatever else supports the higher stated priority (spectrum/waterfall/
    other context), which is expected and not what these signals exist to
    catch. Flagging both directions was tried and produced disagreements
    on ~9-14% of otherwise-correct reports with zero gain in real mismatch
    detection (see the conversation this fix is from for the concrete
    evidence) - real prioritization is holistic, so a raw threshold rule
    landing one level more cautious than the stated priority is normal
    and not evidence of an error.

    Waterfall is not yet incorporated - see graph_signals.py for status.
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
    trend_disagrees = _disagrees("trend_priority_hint")

    out["flag_mismatch"] = text_disagrees | low_conf.fillna(False) | spectrum_disagrees | trend_disagrees

    def reason(r):
        parts = []
        if r["predicted_priority"] != r["priority_num"]:
            parts.append(f"text implies priority {r['predicted_priority']:g}, report states {r['priority_num']:g}")
        elif r["confidence_in_stated_priority"] < low_confidence_threshold:
            parts.append("text is a weak/ambiguous match for the stated priority")
        spectrum_hint = r.get("spectrum_priority_hint")
        if pd.notna(spectrum_hint) and spectrum_hint < r["priority_num"]:
            parts.append(f"Spectrum Fund Amp (in/s) suggests priority {spectrum_hint:g}, report states {r['priority_num']:g}")
        trend_hint = r.get("trend_priority_hint")
        if pd.notna(trend_hint) and trend_hint < r["priority_num"]:
            escalation = r.get("trend_escalation", "escalating")
            parts.append(f"Trend ({escalation}) suggests priority {trend_hint:g}, report states {r['priority_num']:g}")
        return "; ".join(parts)

    out["flag_reason"] = out.apply(reason, axis=1)
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
