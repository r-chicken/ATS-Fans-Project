"""Standalone scoring app (Streamlit version) - upload a report PDF, get
back whether the text and/or the Spectrum chart agree with the priority
the report states.

This is the Streamlit Community Cloud counterpart to webapp/app.py (a
Flask app meant for Render/Azure App Service). Same underlying scoring
pipeline and the same "never retrains, never looks at report history,
just this upload against its own stated priority" behavior described in
webapp/README.md - only the web framework differs, because Streamlit
Community Cloud runs a Streamlit-format app directly rather than an
arbitrary Dockerfile. See streamlit_app/README.md for deployment steps.
"""
from __future__ import annotations

import base64
import dataclasses
import io
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

# ats_priority_checker lives one directory up from this file (repo root).
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ats_priority_checker.extract import process_pdf  # noqa: E402
from ats_priority_checker.model import priority_recommendation_table, report_text  # noqa: E402

st.set_page_config(page_title="ATS Priority Checker", layout="wide")

# Tighter padding/spacing on each report card so more fit on screen at
# once - targets Streamlit's current internal class names for a
# bordered container, so this is the one part of the page that could
# go back to normal spacing (harmlessly, not break anything) if a
# future Streamlit version renames them.
st.markdown(
    """
    <style>
    [data-testid="stVerticalBlockBorderWrapper"] { margin-bottom: 0.4rem; }
    [data-testid="stVerticalBlockBorderWrapper"] > div > div { gap: 0.35rem; padding: 0.6rem 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model (first run only)...")
def _load_model_state(secrets_key: str) -> dict:
    """Loaded once per running process per secrets_key (st.cache_resource
    caches per distinct argument, so "model_fans" and "model_pumps" each
    get their own cached model - this holds live model objects, not
    serializable data, hence cache_resource rather than cache_data).

    The trained model still never goes into git (same reasoning as
    webapp/.gitignore) - here it comes from this app's Secrets instead
    of a file on disk, since Streamlit Community Cloud has no separate
    "copy files in before building" step the way Docker did. See
    streamlit_app/README.md for how those Secrets get set.
    """
    import joblib
    from sentence_transformers import SentenceTransformer

    try:
        section = st.secrets[secrets_key]
        model_b64 = section["joblib_b64"]
        meta_json = section["meta_json"]
    except (KeyError, FileNotFoundError) as exc:
        raise RuntimeError(
            f"No trained model found in this app's Secrets under [{secrets_key}]. "
            f"Add joblib_b64 and meta_json there - see streamlit_app/README.md."
        ) from exc

    clf = joblib.load(io.BytesIO(base64.b64decode(model_b64)))
    meta = json.loads(meta_json)
    embedder = SentenceTransformer(meta["embedding_model_name"])
    return {"clf": clf, "embedder": embedder}


def _score_pdfs(
    paths: list[Path], model_states: dict[str, dict | None], pdf_bytes_by_stem: dict[str, bytes] | None = None
) -> pd.DataFrame:
    """Same per-report extraction path as process_pdf, for every page of
    every uploaded PDF - no batch dataset, no escalation history, just
    each report's own text and its own Spectrum chart against its own
    stated priority (see model.priority_recommendation_table).

    One shared upload, auto-sorted by equipment_kind (a ReportRecord
    field now - see graph_signals.classify_equipment_kind - computed
    inside process_pdf itself, since it's also needed there to pick the
    right spectrum-reading thresholds, not just here to pick the right
    text classifier): fans are scored against model_states["fans"],
    pumps against model_states["pumps"] - each gets the model actually
    trained for it rather than one model guessing across both. A report
    that can't be told apart, or whose kind's model isn't configured
    (model_states[...] is None - see _load_model_state), still gets a
    spectrum reading (process_pdf already picked the right threshold set
    for it), just no text-based prediction, with a note explaining why.

    pdf_bytes_by_stem (source filename without extension -> original
    uploaded bytes) rides along per row so each result card can offer
    the original PDF back - the temp files themselves are gone by the
    time results render (cleaned up when the caller's
    TemporaryDirectory exits), so this is the only copy left."""
    rows = []
    for path in paths:
        try:
            records = process_pdf(path, max_pages=1)
        except Exception as exc:  # noqa: BLE001 - one bad upload shouldn't kill the batch
            rows.append({"report_id": path.name, "priority_raw": None, "parse_notes": f"failed to process: {exc}"})
            continue
        for page_number, rec in enumerate(records, start=1):
            d = dataclasses.asdict(rec)
            d["report_id"] = f"{path.stem}_p{page_number}"
            d["source_filename"] = path.name
            d["pdf_bytes"] = (pdf_bytes_by_stem or {}).get(path.stem)
            rows.append(d)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    usable = df.dropna(subset=["priority_num"]).copy()
    if usable.empty:
        df["predicted_priority"] = None
        return df

    if "parse_notes" not in usable.columns:
        usable["parse_notes"] = ""
    usable["parse_notes"] = usable["parse_notes"].fillna("")
    if "equipment_kind" not in usable.columns:
        usable["equipment_kind"] = None
    usable["predicted_priority"] = None

    unclassified = usable["equipment_kind"].isna()
    usable.loc[unclassified, "parse_notes"] = (
        usable.loc[unclassified, "parse_notes"]
        + "; couldn't tell fan vs. pump from the equipment description - text still scored with the "
        "fan model as a general default, but no spectrum priority (neither threshold set is fitted "
        "for this equipment type)"
    ).str.strip("; ")

    for kind, state in model_states.items():
        subset = usable[usable["equipment_kind"] == kind]
        if subset.empty:
            continue
        if state is None:
            usable.loc[subset.index, "parse_notes"] = (
                subset["parse_notes"] + f"; no {kind} model configured yet"
            ).str.strip("; ")
            continue
        texts = subset.apply(report_text, axis=1).tolist()
        embeddings = state["embedder"].encode(texts, normalize_embeddings=True)
        usable.loc[subset.index, "predicted_priority"] = state["clf"].predict(embeddings)

    # Unclassified equipment (blowers, or anything else that isn't
    # recognizably a fan or pump) still gets a text prediction, using
    # the fans model as a general-purpose default - the text classifier
    # reads language patterns in the Recommendations/Comments, not
    # equipment-specific vibration thresholds, so there's no reason to
    # withhold it just because the equipment type is unclear. The
    # spectrum reading is intentionally left blank for these instead
    # (see graph_signals.spectrum_priority_hint) - unlike text, the
    # amplitude thresholds genuinely are equipment-specific, and
    # neither fitted set applies to a machine type this app doesn't
    # know about.
    fans_state = model_states.get("fans")
    unclassified_subset = usable[unclassified]
    if fans_state is not None and not unclassified_subset.empty:
        texts = unclassified_subset.apply(report_text, axis=1).tolist()
        embeddings = fans_state["embedder"].encode(texts, normalize_embeddings=True)
        usable.loc[unclassified_subset.index, "predicted_priority"] = fans_state["clf"].predict(embeddings)

    table = priority_recommendation_table(usable)
    table["equipment_kind"] = usable["equipment_kind"]
    # priority_recommendation_table only copies over a curated set of
    # columns (see its docstring - it's a verdict table, not a debug
    # dump), so anything else worth showing on a card has to be attached
    # explicitly here, same as parse_notes above.
    table["parse_notes"] = usable["parse_notes"]
    for col in ("site", "source_filename", "pdf_bytes"):
        if col in usable.columns:
            table[col] = usable[col]
    unusable = df[df["priority_num"].isna()][["report_id"]].copy()
    if not unusable.empty:
        for col in ("parse_notes", "site", "source_filename", "pdf_bytes"):
            if col in df.columns:
                unusable[col] = df.loc[unusable.index, col]
        if "parse_notes" not in unusable.columns:
            unusable["parse_notes"] = "no priority found on this page"
        table = pd.concat([table, unusable], ignore_index=True)
    return table


def _format_table_for_display(table: pd.DataFrame) -> pd.DataFrame:
    """Blank for missing, Yes/No for booleans (including pandas' nullable
    "boolean" dtype, which otherwise prints as the confusing literal
    string "<NA>") - same idea as webapp/app.py's version of this, just
    returning a DataFrame here since st.dataframe wants one. Only used
    for the optional raw-table expander below the cards."""
    display = table.copy()
    for col in display.columns:
        if str(display[col].dtype) == "boolean":
            display[col] = display[col].map({True: "Yes", False: "No"}).fillna("")
        elif display[col].dtype == bool:
            display[col] = display[col].map({True: "Yes", False: "No"})
        else:
            display[col] = display[col].fillna("")
    return display


def _priority_str(value) -> str:
    return "-" if pd.isna(value) else f"P{int(value)}"


# Matched to the reference colors sent in chat (P1 red -> P4 green).
# P3's yellow needs dark text instead of white to stay readable - the
# other three keep white.
_PRIORITY_COLORS = {
    1: ("#e03131", "white"),
    2: ("#f76707", "white"),
    3: ("#ffd43b", "#1a1a1a"),
    4: ("#2f9e44", "white"),
}


def _priority_badge(value, flagged: bool = False) -> str:
    """A colored pill for a priority value, with a red star appended
    when this particular source disagreed with the stated priority."""
    if pd.isna(value):
        return '<span style="color:#888;">-</span>'
    bg, fg = _PRIORITY_COLORS.get(int(value), ("#888", "white"))
    star = ' <span style="color:#d32f2f;" title="Disagrees with stated priority">&#9733;</span>' if flagged else ""
    return (
        f'<span style="background-color:{bg}; color:{fg}; padding:2px 10px; '
        f'border-radius:6px; font-size:1.1rem; font-weight:600; display:inline-block;">'
        f"P{int(value)}</span>{star}"
    )


def _disagrees(value) -> bool:
    """True only when this source actually gave an answer AND it didn't
    match - a missing reading (spectrum couldn't be read, etc.) is a
    different situation from a disagreement and shouldn't be flagged as
    one."""
    return pd.notna(value) and not bool(value)


_KIND_LABELS = {"fans": "Fan model", "pumps": "Pump model"}


def _render_report_card(row: pd.Series) -> None:
    with st.container(border=True):
        equipment = row.get("equipment_id")
        header = str(equipment) if pd.notna(equipment) else "Unknown equipment"
        point = row.get("measurement_point")
        if pd.notna(point):
            header += f"  ·  {point}"

        title_col, link_col = st.columns([6, 1])
        with title_col:
            site = row.get("site")
            if pd.notna(site):
                st.caption(str(site))
            st.markdown(f"**{header}**")
            # Visible confirmation of which model actually scored this
            # report's text - fans and pumps are routed to separate joblib
            # bundles (see _score_pdfs), this is here so that's checkable
            # at a glance instead of just trusted. Unclassified equipment
            # (blowers, etc.) still gets a label - it was scored too, with
            # the fan model as a default (see _score_pdfs), not skipped.
            kind = row.get("equipment_kind")
            if pd.notna(kind):
                st.caption(f"Scored with: {_KIND_LABELS.get(kind, kind)}")
            else:
                st.caption("Scored with: Fan model (default - equipment type not recognized)")
        with link_col:
            pdf_bytes = row.get("pdf_bytes")
            if isinstance(pdf_bytes, (bytes, bytearray)):
                st.download_button(
                    "PDF",
                    data=bytes(pdf_bytes),
                    file_name=str(row.get("source_filename") or "report.pdf"),
                    mime="application/pdf",
                    key=f"pdf_{row.get('report_id', id(row))}",
                )

        stated = row.get("priority_raw")
        text_pred = row.get("text_recommended_priority")
        spectrum_pred = row.get("graph_recommended_priority")

        text_disagrees = _disagrees(row.get("text_agrees_with_stated"))
        graph_disagrees = _disagrees(row.get("graph_agrees_with_stated"))

        cols = st.columns(3, gap="small")
        with cols[0]:
            st.caption("Stated Priority")
            st.markdown(_priority_badge(stated), unsafe_allow_html=True)
        with cols[1]:
            st.caption("Text Recommends")
            st.markdown(_priority_badge(text_pred, flagged=text_disagrees), unsafe_allow_html=True)
        with cols[2]:
            st.caption("Spectrum Recommends")
            st.markdown(_priority_badge(spectrum_pred, flagged=graph_disagrees), unsafe_allow_html=True)

        notes = []
        if text_disagrees:
            notes.append(f"Check text; AI flagged it as {_priority_str(text_pred)} instead of {_priority_str(stated)}.")
        if graph_disagrees:
            notes.append(f"Check spectrum; AI flagged it as {_priority_str(spectrum_pred)} instead of {_priority_str(stated)}.")

        if notes:
            st.warning("  \n".join(notes))
        else:
            st.success("Text and spectrum both agree with the stated priority.")

        # A missing reading isn't a disagreement (nothing to compare),
        # but it's still worth surfacing why - could be no chart image,
        # OCR failing, an unclear fan-vs-pump call, or that kind's model
        # not being configured yet.
        if row.get("parse_notes"):
            st.caption(f"Note: {row['parse_notes']}")


def _try_load_model_state(secrets_key: str) -> dict | None:
    """None (not an exception) when that kind's Secrets aren't
    configured yet, so one missing model doesn't block scoring the
    other kind - see _score_pdfs's per-kind handling of a None state."""
    try:
        return _load_model_state(secrets_key)
    except RuntimeError:
        return None


st.title("ATS Vibration Priority Checker")
st.caption(
    "Upload report PDF(s) - fans and pumps together is fine, each report is "
    "automatically sorted and scored against the right model based on what "
    "equipment it's for."
)

uploaded = st.file_uploader("Report PDF(s)", type=["pdf"], accept_multiple_files=True)
go = st.button("Analyze", type="primary", disabled=not uploaded)

if go:
    with st.spinner("Reading and scoring report(s)..."):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = []
            pdf_bytes_by_stem = {}
            for f in uploaded:
                data = f.getvalue()
                path = Path(tmp_dir) / f.name
                path.write_bytes(data)
                paths.append(path)
                pdf_bytes_by_stem[path.stem] = data
            model_states = {
                "fans": _try_load_model_state("model_fans"),
                "pumps": _try_load_model_state("model_pumps"),
            }
            table = _score_pdfs(paths, model_states, pdf_bytes_by_stem)

    if table.empty:
        st.warning("No readable report pages found in the uploaded PDF(s).")
    else:
        flagged = table["any_disagreement"].eq(True).fillna(False) if "any_disagreement" in table.columns else pd.Series(False, index=table.index)
        st.caption(
            f"{len(table)} report(s) analyzed - {int(flagged.sum())} flagged for review."
            if flagged.any()
            else f"{len(table)} report(s) analyzed - text and spectrum agree with the stated priority on all of them."
        )

        has_kind_column = "equipment_kind" in table.columns
        for kind, heading in (("fans", "Fans"), ("pumps", "Pumps")):
            subset = table[table["equipment_kind"] == kind] if has_kind_column else pd.DataFrame()
            if subset.empty:
                continue
            st.header(heading)
            for _, row in subset.iterrows():
                _render_report_card(row)

        other = table[table["equipment_kind"].isna()] if has_kind_column else table
        if not other.empty:
            st.header("Needs Review")
            st.caption(
                "Couldn't automatically sort these into Fans or Pumps, or the page "
                "itself couldn't be read - see the note on each card."
            )
            for _, row in other.iterrows():
                _render_report_card(row)

        with st.expander("Show full data table"):
            st.dataframe(_format_table_for_display(table), use_container_width=True)

st.divider()
st.markdown(
    "**Does:** OCR + parse each uploaded PDF, read the Spectrum peak off the chart pixels, "
    "embed the Recommendations/Comments text with the same frozen sentence-embedding model "
    "training used, and run the loaded classifier - then show the same text/spectrum-vs-stated "
    "comparison as `priority_recommendation_table`.\n\n"
    "**Doesn't:** retrain, look at any report history (no escalation signal), or write "
    "anything back to Drive/dataset.csv. Every upload is scored independently, fresh."
)
