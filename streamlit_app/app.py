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


def _score_pdfs(paths: list[Path], state: dict) -> pd.DataFrame:
    """Same per-report extraction path as process_pdf, for every page of
    every uploaded PDF, scored against the given model state (see
    _load_model_state - which model this is, fans vs. pumps, is the
    caller's choice) - no batch dataset, no escalation history, just
    this report's own text and its own Spectrum chart against its own
    stated priority (see model.priority_recommendation_table)."""
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


def _render_report_card(row: pd.Series) -> None:
    with st.container(border=True):
        equipment = row.get("equipment_id")
        header = str(equipment) if pd.notna(equipment) else "Unknown equipment"
        point = row.get("measurement_point")
        if pd.notna(point):
            header += f"  ·  {point}"
        st.markdown(f"**{header}**")

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

        # A missing spectrum reading isn't a disagreement (nothing to
        # compare), but it's still worth surfacing why - same reasoning
        # as adding parse_notes to the table in the first place.
        if pd.isna(spectrum_pred) and row.get("parse_notes"):
            st.caption(f"Spectrum not read: {row['parse_notes']}")


def _render_scoring_section(label: str, secrets_key: str, key_prefix: str) -> None:
    """One independent "upload -> Analyze -> results" section, backed by
    its own model (own Secrets key, own cache_resource entry). Widget
    keys are prefixed so the Fans and Pumps sections don't collide with
    each other - Streamlit requires unique keys per widget on a page."""
    st.header(label)
    uploaded = st.file_uploader(
        "Report PDF(s)", type=["pdf"], accept_multiple_files=True, key=f"{key_prefix}_upload"
    )
    go = st.button("Analyze", type="primary", disabled=not uploaded, key=f"{key_prefix}_analyze")

    if not go:
        return

    with st.spinner("Reading and scoring report(s)..."):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = []
            for f in uploaded:
                path = Path(tmp_dir) / f.name
                path.write_bytes(f.getvalue())
                paths.append(path)
            table = None
            try:
                state = _load_model_state(secrets_key)
                table = _score_pdfs(paths, state)
            except RuntimeError as exc:
                st.error(str(exc))

    if table is None:
        return

    if table.empty:
        st.warning("No readable report pages found in the uploaded PDF(s).")
        return

    flagged = table["any_disagreement"].eq(True).fillna(False) if "any_disagreement" in table.columns else pd.Series(False, index=table.index)
    st.caption(
        f"{len(table)} report(s) analyzed - {int(flagged.sum())} flagged for review."
        if flagged.any()
        else f"{len(table)} report(s) analyzed - text and spectrum agree with the stated priority on all of them."
    )
    for _, row in table.iterrows():
        _render_report_card(row)

    with st.expander("Show full data table"):
        st.dataframe(_format_table_for_display(table), use_container_width=True)


st.title("ATS Vibration Priority Checker")
st.caption(
    "Upload a report PDF to see whether the text and the Spectrum chart "
    "agree with the priority the report states."
)

_render_scoring_section("Score Fans", "model_fans", "fans")
st.divider()
_render_scoring_section("Score Pumps", "model_pumps", "pumps")

st.divider()
st.markdown(
    "**Does:** OCR + parse each uploaded PDF, read the Spectrum peak off the chart pixels, "
    "embed the Recommendations/Comments text with the same frozen sentence-embedding model "
    "training used, and run the loaded classifier - then show the same text/spectrum-vs-stated "
    "comparison as `priority_recommendation_table`.\n\n"
    "**Doesn't:** retrain, look at any report history (no escalation signal), or write "
    "anything back to Drive/dataset.csv. Every upload is scored independently, fresh."
)
