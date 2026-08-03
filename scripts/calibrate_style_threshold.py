"""Recalibrate the waterfall/colored-spectrum style-detection threshold.

The threshold in ats_priority_checker.extract.DEFAULT_STYLE_THRESHOLD was
set from a single example of each report style. Once you've manually
confirmed a handful of PDFs of each style, sort them into two folders and
run this to check whether the colorfulness heuristic still cleanly
separates them, and get a better threshold if not.

Usage:
    python scripts/calibrate_style_threshold.py \\
        --waterfall-dir data/labeled_styles/waterfall \\
        --colored-dir data/labeled_styles/colored_spectrum
"""
from __future__ import annotations

import argparse
from pathlib import Path

import fitz

from ats_priority_checker.extract import colorfulness, largest_embedded_image


def scores_for_dir(pdf_dir: Path) -> list[float]:
    scores = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        doc = fitz.open(pdf_path)
        try:
            for page in doc:
                img = largest_embedded_image(doc, page)
                if img is not None:
                    scores.append(colorfulness(img))
        finally:
            doc.close()
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waterfall-dir", required=True, type=Path)
    parser.add_argument("--colored-dir", required=True, type=Path)
    args = parser.parse_args()

    wf_scores = scores_for_dir(args.waterfall_dir)
    cs_scores = scores_for_dir(args.colored_dir)

    print(f"waterfall style:        n={len(wf_scores)}  scores={sorted(round(s, 1) for s in wf_scores)}")
    print(f"colored spectrum style: n={len(cs_scores)}  scores={sorted(round(s, 1) for s in cs_scores)}")

    if not wf_scores or not cs_scores:
        print("Need at least one scored PDF in each folder to suggest a threshold.")
        return

    wf_max, cs_min = max(wf_scores), min(cs_scores)
    if wf_max < cs_min:
        threshold = (wf_max + cs_min) / 2
        print(f"\nClasses are cleanly separated. Suggested DEFAULT_STYLE_THRESHOLD: {threshold:.1f}")
    else:
        print(
            f"\nWARNING: score ranges overlap (waterfall max={wf_max:.1f}, "
            f"colored-spectrum min={cs_min:.1f}). The colorfulness heuristic alone "
            "may misclassify some reports near the boundary - review those manually."
        )


if __name__ == "__main__":
    main()
