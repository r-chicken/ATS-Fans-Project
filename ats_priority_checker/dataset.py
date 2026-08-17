"""Build a flat dataset CSV from a folder of ATS vibration report PDFs."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd

from .extract import DEFAULT_STYLE_THRESHOLD, classify_style_by_text, process_pdf

REC_SEP = " | "

# A current-vs-previous-test amplitude ratio at or below this counts as
# "much much lower" - a drop this steep is more often a broken/loosened
# sensor, a since-fixed fault, or a missed read than routine improvement,
# so it's worth a flagged second look either way. Not independently
# derived from the labeled set - retune once you have enough labeled
# examples of real sensor/mount issues vs. genuine improvement to check it
# against.
ESCALATION_DROP_RATIO = 0.25


def _parse_report_date(date_str) -> pd.Timestamp | None:
    if date_str is None or (isinstance(date_str, float) and pd.isna(date_str)) or str(date_str).strip() == "":
        return None
    try:
        return pd.to_datetime(date_str)
    except (ValueError, TypeError):
        return None


def add_escalation_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Compare each report's Spectrum peak reading against the SAME
    equipment's most recent earlier dated report, and flag escalation.

    Replaces the old Waterfall/Trend pixel-based escalation check - this
    project's report set now spans a real timeline per machine (one report
    per date, not a single snapshot), so "is this getting worse?" is
    answered by comparing dated reports for the same equipment_id directly,
    the same thing a human reviewer would do by eye.

    Groups rows by (site, equipment_id), sorts by date_tested, and for each
    report walks backward to the nearest EARLIER report for the same
    equipment that has a comparable reading (same spectrum_unit, both
    amplitudes present) - not just the literal previous row, since a
    between-visit report that failed to parse or had no usable chart image
    shouldn't break the comparison. Flags a row when, versus that prior
    reading:
      - its spectrum_priority_hint bucket is MORE severe (a lower number -
        see graph_signals.py) than the prior reading's, or
      - its spectrum_peak_amplitude has dropped to ESCALATION_DROP_RATIO or
        less of the prior reading's - a sudden, drastic drop is flagged for
        review same as a jump, not treated as improvement, since it can
        mean a sensor came loose or a fault is intermittent rather than
        gone (see ESCALATION_DROP_RATIO above).

    Adds these columns (present but blank/NaN when there's no comparable
    prior reading, e.g. the equipment's first report in the set):
      prior_date_tested, prior_spectrum_peak_amplitude,
      prior_spectrum_priority_hint, escalation_flag, escalation_reason,
      escalation_priority_hint (the more-severe of the two hints, when
      flagged for a threshold jump; the current reading's own hint when
      flagged for a sharp drop, since a drop doesn't imply a specific
      "should be" number the way a jump does)

    Future idea, not implemented here: also compare each peak's FREQUENCY
    across dated reports for the same equipment, to catch a resonance
    shifting frequency even when its amplitude doesn't move much. Would
    need x-axis (frequency) calibration added to graph_signals.py, not
    just the y-axis calibration read_spectrum_peak already does.
    """
    df = df.copy()
    df["prior_date_tested"] = None
    df["prior_spectrum_peak_amplitude"] = float("nan")
    df["prior_spectrum_priority_hint"] = float("nan")
    df["escalation_flag"] = False
    df["escalation_reason"] = ""
    df["escalation_priority_hint"] = float("nan")

    parsed_dates = df["date_tested"].apply(_parse_report_date)

    group_cols = [c for c in ("site", "equipment_id") if c in df.columns]
    if not group_cols or parsed_dates.isna().all():
        return df

    for _, idxs in df.groupby(group_cols, dropna=False).groups.items():
        ordered = parsed_dates.loc[list(idxs)].dropna().sort_values()
        history = []  # (idx, amplitude, unit, priority_hint) for rows seen so far, most recent last
        for idx in ordered.index:
            row = df.loc[idx]
            cur_amp = row.get("spectrum_peak_amplitude")
            cur_unit = row.get("spectrum_unit")
            cur_hint = row.get("spectrum_priority_hint")

            # Walk backward for the nearest earlier reading with a
            # comparable (same-unit, both-present) amplitude.
            prior = None
            for prev_idx, prev_amp, prev_unit, prev_hint in reversed(history):
                if pd.notna(prev_amp) and prev_unit == cur_unit:
                    prior = (prev_idx, prev_amp, prev_unit, prev_hint)
                    break

            if prior is not None:
                prev_idx, prev_amp, _prev_unit, prev_hint = prior
                df.at[idx, "prior_date_tested"] = df.at[prev_idx, "date_tested"]
                df.at[idx, "prior_spectrum_peak_amplitude"] = prev_amp
                df.at[idx, "prior_spectrum_priority_hint"] = prev_hint

                reasons = []
                escalation_hint = None
                if pd.notna(cur_hint) and pd.notna(prev_hint) and cur_hint < prev_hint:
                    reasons.append(
                        f"priority threshold jumped from {prev_hint:g} to {cur_hint:g} "
                        f"vs. the {df.at[prev_idx, 'date_tested']} test"
                    )
                    escalation_hint = cur_hint
                if pd.notna(cur_amp) and prev_amp > 0 and cur_amp <= prev_amp * ESCALATION_DROP_RATIO:
                    reasons.append(
                        f"amplitude dropped sharply vs. the {df.at[prev_idx, 'date_tested']} test "
                        f"({prev_amp:g} -> {cur_amp:g}) - verify sensor/mount before reading this as improvement"
                    )
                    if escalation_hint is None:
                        escalation_hint = cur_hint
                if reasons:
                    df.at[idx, "escalation_flag"] = True
                    df.at[idx, "escalation_reason"] = "; ".join(reasons)
                    if escalation_hint is not None:
                        df.at[idx, "escalation_priority_hint"] = escalation_hint

            history.append((idx, cur_amp, cur_unit, cur_hint))

    return df


def _split_and_write(df: pd.DataFrame, out_dir: Path, filter_style: bool) -> dict:
    """Split the full extracted set into the three output CSVs and write
    them + summary.txt. Shared by build_dataset and recompute_dataset so
    both always split the same way.
    """
    # Defense in depth: an object-dtype parse_ok column (e.g. from
    # concatenating with an empty dataframe upstream) makes `~` do Python
    # bitwise invert instead of logical negation, silently matching every
    # row below. Always force a real bool dtype before masking with it.
    df = df.copy()
    df["parse_ok"] = df["parse_ok"].astype(bool)

    if filter_style:
        usable = df[(df["style"] == "waterfall") & (df["parse_ok"])].copy()
        excluded_style = df[df["style"] == "colored_spectrum"].copy()
        parse_errors = df[~df["parse_ok"] | (df["style"] == "unknown")].copy()
    else:
        usable = df[df["parse_ok"]].copy()
        excluded_style = df.iloc[0:0].copy()
        parse_errors = df[~df["parse_ok"]].copy()

    usable = add_escalation_signals(usable)

    usable.to_csv(out_dir / "dataset.csv", index=False)
    excluded_style.to_csv(out_dir / "excluded_style.csv", index=False)
    parse_errors.to_csv(out_dir / "parse_errors.csv", index=False)

    summary = {
        "pages_total": len(df),
        "usable_rows": len(usable),
        "excluded_colored_spectrum_rows": len(excluded_style),
        "parse_error_rows": len(parse_errors),
        "escalation_flagged_rows": int(usable["escalation_flag"].sum()) if "escalation_flag" in usable.columns else 0,
    }
    with open(out_dir / "summary.txt", "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
    return summary


def build_dataset(
    pdf_dir: str | Path,
    out_dir: str | Path,
    style_threshold: float = DEFAULT_STYLE_THRESHOLD,
    max_pages: int | None = None,
    filter_style: bool = True,
) -> dict:
    """Walk every PDF in pdf_dir, extract fields, and write three CSVs to out_dir:

    - dataset.csv         usable rows
    - excluded_style.csv  colored-spectrum-style rows, kept aside for reference
                          (always written, empty when filter_style=False)
    - parse_errors.csv    rows that failed to parse cleanly, for regex fixes /
                          manual review

    This is the slow step (OCR + pixel analysis run on every PDF's chart
    image). Every row also carries chart_ocr_text - the raw OCR output
    cached alongside it - specifically so that a later change to style
    detection *logic* doesn't require re-running this against every PDF
    again: use recompute_dataset() instead. See recompute_dataset's
    docstring for exactly what it can and can't refresh from the cache -
    the Spectrum peak reading itself needs the actual chart image (pixel
    analysis), unlike the old Fund Amp text field, so it's not one of the
    things a cache-only recompute can redo.

    Pass max_pages=1 to only extract each PDF's first page and ignore any
    additional pages entirely.

    Pass filter_style=False to skip style-based filtering entirely - every
    row that parsed cleanly goes into dataset.csv regardless of chart style
    (style/chart_colorfulness are still recorded in the output for
    reference). Use this once your source PDFs are already a single style
    (e.g. you've manually removed the other style) - the colorfulness
    heuristic was calibrated on very few examples and can misclassify some
    legitimate reports, so don't rely on it to filter a style that isn't
    present in your data anymore.

    Returns a summary dict with counts, and also writes summary.txt.
    """
    pdf_dir = Path(pdf_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found in {pdf_dir} (looked for *.pdf, case-sensitive). "
            "Check that this path is correct and that Google Drive has fully mounted/synced "
            "in this session - e.g. run `print(len(list(Path(pdf_dir).glob('*.pdf'))))` to confirm "
            "before calling build_dataset again."
        )
    for pdf_path in pdf_paths:
        try:
            all_records.extend(process_pdf(pdf_path, style_threshold=style_threshold, max_pages=max_pages))
        except Exception as exc:  # noqa: BLE001 - keep going on a bad file
            all_records.append(
                {
                    "source_file": pdf_path.name,
                    "page_number": None,
                    "site": None,
                    "date_tested": None,
                    "equipment_id": None,
                    "priority_raw": None,
                    "priority_num": None,
                    "recommendations": [],
                    "comments": None,
                    "chart_colorfulness": None,
                    "style": "unknown",
                    "spectrum_unit": None,
                    "spectrum_peak_amplitude": None,
                    "spectrum_peak_amplitude_raw": None,
                    "spectrum_priority_hint": None,
                    "spectrum_peak_error": None,
                    "chart_ocr_text": None,
                    "parse_ok": False,
                    "parse_notes": f"exception during processing: {exc}",
                }
            )

    rows = []
    for r in all_records:
        d = dataclasses.asdict(r) if dataclasses.is_dataclass(r) else dict(r)
        d["recommendations"] = REC_SEP.join(d.get("recommendations") or [])
        rows.append(d)
    df = pd.DataFrame(rows)
    df.insert(0, "report_id", [f"{d['source_file']}_p{d['page_number']}" for d in rows] if rows else [])

    summary = _split_and_write(df, out_dir, filter_style)
    summary["pdfs_found"] = len(pdf_paths)
    return summary


def recompute_dataset(out_dir: str | Path, filter_style: bool = True) -> dict:
    """Re-derive `style` from the chart_ocr_text already cached in
    out_dir's CSVs, re-derive the cross-report escalation signal from
    already-cached Spectrum readings, and re-split/rewrite dataset.csv,
    excluded_style.csv, and parse_errors.csv.

    What this CAN refresh without opening any PDFs or re-running OCR/pixel
    analysis (finishes in seconds regardless of how many reports you have):
      - style, from cached chart_ocr_text - use after changing
        classify_style_by_text()
      - escalation_flag / escalation_reason / escalation_priority_hint /
        prior_* columns, from each row's already-cached
        spectrum_peak_amplitude + spectrum_unit + spectrum_priority_hint +
        date_tested - use after changing add_escalation_signals() or
        ESCALATION_DROP_RATIO

    What this CANNOT refresh (needs a full build_dataset() re-run):
      - spectrum_peak_amplitude / spectrum_priority_hint themselves. Unlike
        the old Fund Amp text field, the Spectrum peak is read from the
        chart image's PIXELS (see graph_signals.read_spectrum_peak) - the
        cached chart_ocr_text alone isn't enough to redo that, since only
        the unit comes from OCR text; the amplitude comes from the image.
        A change to the peak-reading heuristics (graph_signals.py's
        _find_peak_pixel etc.) or to the in/s, gE, g threshold functions
        always needs the slow path.

    Only run this against an out_dir that build_dataset has already
    populated (i.e. after at least one full run with a version of the code
    that saves chart_ocr_text and spectrum_peak_amplitude).
    """
    out_dir = Path(out_dir)
    parts = []
    for name in ["dataset.csv", "excluded_style.csv", "parse_errors.csv"]:
        path = out_dir / name
        if path.exists():
            parts.append(pd.read_csv(path))
    if not parts:
        raise FileNotFoundError(
            f"No dataset.csv / excluded_style.csv / parse_errors.csv found in {out_dir} - "
            "run build_dataset first."
        )
    df = pd.concat(parts, ignore_index=True)
    # Concatenating with an empty CSV (no rows to infer a dtype from) can
    # upcast parse_ok from bool to object - and `~` on an object-dtype bool
    # does Python bitwise invert (~True == -2, ~False == -1), NOT logical
    # negation, silently matching every row in _split_and_write's masks.
    df["parse_ok"] = df["parse_ok"].astype(bool)

    if "chart_ocr_text" not in df.columns:
        raise ValueError(
            "chart_ocr_text column not found - these CSVs were written by an older version of "
            "build_dataset that didn't cache OCR text. Run build_dataset again (the slow way, once) "
            "to populate it; recompute_dataset can reuse it from then on."
        )
    if "spectrum_peak_amplitude" not in df.columns:
        raise ValueError(
            "spectrum_peak_amplitude column not found - these CSVs were written by an older version of "
            "build_dataset (pre pixel-based Spectrum reading, or still using spectrum_fund_amp). "
            "Run build_dataset again (the slow way, once) to populate it."
        )

    has_text = df["chart_ocr_text"].notna() & (df["chart_ocr_text"].astype(str).str.strip() != "")
    n_recomputable = int(has_text.sum())
    df.loc[has_text, "style"] = df.loc[has_text, "chart_ocr_text"].apply(classify_style_by_text)

    summary = _split_and_write(df, out_dir, filter_style)
    summary["rows_recomputed_from_cached_ocr"] = n_recomputable
    summary["rows_without_cached_ocr_text"] = len(df) - n_recomputable
    return summary
