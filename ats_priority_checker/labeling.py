"""Helpers for building and merging back a human-labeled ground-truth set.

You can't measure (or usefully train) a mismatch detector without some
reports where a person has actually judged whether the stated priority
matches the write-up. These helpers turn dataset.csv into something easy
to hand-label in Google Sheets/Excel, then merge the labels back in.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

LABEL_COLUMN = "human_label"
NOTES_COLUMN = "human_notes"
VALID_LABELS = {"match", "mismatch", "unsure"}


def export_for_labeling(dataset_csv: str | Path, out_csv: str | Path, sample_n: int | None = None) -> pd.DataFrame:
    """Create a review sheet with the fields a person needs to judge
    match/mismatch, plus blank human_label / human_notes columns.

    Pass sample_n to label a random subset first (e.g. 100 of 500) rather
    than all of them - useful for building an initial validation set fast.
    """
    df = pd.read_csv(dataset_csv)
    cols = ["report_id", "site", "equipment_id", "priority_raw", "recommendations", "comments"]
    review = df[cols].copy()
    if sample_n is not None and sample_n < len(review):
        review = review.sample(n=sample_n, random_state=0).sort_index()
    review[LABEL_COLUMN] = ""
    review[NOTES_COLUMN] = ""
    review.to_csv(out_csv, index=False)
    return review


def merge_labels(dataset_csv: str | Path, labeled_csv: str | Path, out_csv: str | Path) -> pd.DataFrame:
    """Merge human labels back onto the full dataset by report_id.

    Rows in dataset.csv with no corresponding labeled row (or a blank
    label) come back with human_label = NaN - still usable for training
    the priority-prediction model, just not for measuring mismatch
    detection accuracy.
    """
    dataset = pd.read_csv(dataset_csv)
    labeled = pd.read_csv(labeled_csv)

    bad = set(labeled[LABEL_COLUMN].dropna().unique()) - VALID_LABELS - {""}
    if bad:
        raise ValueError(f"Unrecognized values in {LABEL_COLUMN}: {bad}. Expected one of {VALID_LABELS}.")

    merged = dataset.merge(
        labeled[["report_id", LABEL_COLUMN, NOTES_COLUMN]],
        on="report_id",
        how="left",
    )
    merged.to_csv(out_csv, index=False)
    n_labeled = merged[LABEL_COLUMN].notna().sum() - (merged[LABEL_COLUMN] == "").sum()
    print(f"{n_labeled} / {len(merged)} rows have a human label.")
    return merged
