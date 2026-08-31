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


@st.cache_resource(show_spinner="Loading model (first run only)...")
def _load_state() -> dict:
    """Loaded once per running process (st.cache_resource, not
    cache_data - this holds live model objects, not serializable data).

    The trained model still never goes into git (same reasoning as
    webapp/.gitignore) - here it comes from this app's Secrets instead
    of a file on disk, since Streamlit Community Cloud has no separate
    "copy files in before building" step the way Docker did. See
    streamlit_app/README.md for how those Secrets get set.
    """
    import joblib
    from sentence_transformers import SentenceTransformer

    try:
        model_b64 = st.secrets["model"]["joblib_b64"]
        meta_json = st.secrets["model"]["meta_json"]
    except (KeyError, FileNotFoundError) as exc:
        raise RuntimeError(
            "No trained model found in this app's Secrets. Add a [model] section "
            "with joblib_b64 and meta_json - see streamlit_app/README.md."
        ) from exc

    clf = joblib.load(io.BytesIO(base64.b64decode(model_b64)))
    meta = json.loads(meta_json)
    embedder = SentenceTransformer(meta["embedding_model_name"])
    return {"clf": clf, "embedder": embedder}


def _score_pdfs(paths: list[Path]) -> pd.DataFrame:
    """Same per-report extraction path as process_pdf, for every page of
    every uploaded PDF, scored against the loaded model - no batch
    dataset, no escalation history, just this report's own text and its
    own Spectrum chart against its own stated priority (see
    model.priority_recommendation_table)."""
    state = _load_state()
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
            rows.append(d)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    usable = df.dropna(subset=["priority_num"]).copy()
    if usable.empty:
        df["predicted_priority"] = None
        return df

    texts = usable.apply(report_text, axis=1).tolist()
    embeddings = state["embedder"].encode(texts, normalize_embeddings=True)
    usable["predicted_priority"] = state["clf"].predict(embeddings)

    table = priority_recommendation_table(usable)
    # priority_recommendation_table doesn't carry this through on its own
    # (see its docstring - it's a verdict table, not a debug dump), but
    # it's the only place that explains a blank spectrum reading (no
    # chart image found on the page vs. OCR failing vs. genuinely nothing
    # to report), so it's worth showing here.
    if "parse_notes" in usable.columns:
        table["parse_notes"] = usable["parse_notes"]
    unusable = df[df["priority_num"].isna()][["report_id"]].copy()
    if not unusable.empty:
        unusable["parse_notes"] = df.loc[unusable.index, "parse_notes"] if "parse_notes" in df.columns else "no priority found on this page"
        table = pd.concat([table, unusable], ignore_index=True)
    return table


def _format_table_for_display(table: pd.DataFrame) -> pd.DataFrame:
    """Blank for missing, Yes/No for booleans (including pandas' nullable
    "boolean" dtype, which otherwise prints as the confusing literal
    string "<NA>") - same idea as webapp/app.py's version of this, just
    returning a DataFrame here since st.dataframe wants one."""
    display = table.copy()
    for col in display.columns:
        if str(display[col].dtype) == "boolean":
            display[col] = display[col].map({True: "Yes", False: "No"}).fillna("")
        elif display[col].dtype == bool:
            display[col] = display[col].map({True: "Yes", False: "No"})
        else:
            display[col] = display[col].fillna("")
    return display


st.title("ATS Vibration Priority Checker")
st.caption(
    "Upload a report PDF to see whether the text and the Spectrum chart "
    "agree with the priority the report states."
)

uploaded = st.file_uploader("Report PDF(s)", type=["pdf"], accept_multiple_files=True)
go = st.button("Analyze", type="primary", disabled=not uploaded)

if go:
    with st.spinner("Reading and scoring report(s)..."):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = []
            for f in uploaded:
                path = Path(tmp_dir) / f.name
                path.write_bytes(f.getvalue())
                paths.append(path)
            table = None
            try:
                table = _score_pdfs(paths)
            except RuntimeError as exc:
                st.error(str(exc))

    if table is not None:
        if table.empty:
            st.warning("No readable report pages found in the uploaded PDF(s).")
        else:
            flagged = table["any_disagreement"].eq(True).fillna(False) if "any_disagreement" in table.columns else pd.Series(False, index=table.index)
            display = _format_table_for_display(table)

            def _highlight(row: pd.Series) -> list[str]:
                color = "background-color: #fff3cd" if flagged.loc[row.name] else ""
                return [color] * len(row)

            st.dataframe(display.style.apply(_highlight, axis=1), use_container_width=True)
            if flagged.any():
                st.caption(f"{int(flagged.sum())} row(s) highlighted - text and/or spectrum disagree with the stated priority.")

st.divider()
st.markdown(
    "**Does:** OCR + parse each uploaded PDF, read the Spectrum peak off the chart pixels, "
    "embed the Recommendations/Comments text with the same frozen sentence-embedding model "
    "training used, and run the loaded classifier - then show the same text/spectrum-vs-stated "
    "comparison as `priority_recommendation_table`.\n\n"
    "**Doesn't:** retrain, look at any report history (no escalation signal), or write "
    "anything back to Drive/dataset.csv. Every upload is scored independently, fresh."
)
