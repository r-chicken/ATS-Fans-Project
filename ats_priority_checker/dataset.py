"""Build a flat dataset CSV from a folder of ATS vibration report PDFs."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd

from .extract import DEFAULT_STYLE_THRESHOLD, process_pdf

REC_SEP = " | "


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
    df.insert(0, "report_id", [f"{row.source_file}_p{row.page_number}" for row in all_records] if all_records else [])

    if filter_style:
        usable = df[(df["style"] == "waterfall") & (df["parse_ok"])].copy()
        excluded_style = df[df["style"] == "colored_spectrum"].copy()
        parse_errors = df[~df["parse_ok"] | (df["style"] == "unknown")].copy()
    else:
        usable = df[df["parse_ok"]].copy()
        excluded_style = df.iloc[0:0].copy()
        parse_errors = df[~df["parse_ok"]].copy()

    usable.to_csv(out_dir / "dataset.csv", index=False)
    excluded_style.to_csv(out_dir / "excluded_style.csv", index=False)
    parse_errors.to_csv(out_dir / "parse_errors.csv", index=False)

    summary = {
        "pdfs_found": len(pdf_paths),
        "pages_total": len(df),
        "usable_rows": len(usable),
        "excluded_colored_spectrum_rows": len(excluded_style),
        "parse_error_rows": len(parse_errors),
    }
    with open(out_dir / "summary.txt", "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
    return summary
