"""Helpers for building and merging back a human-labeled ground-truth set.

You can't measure (or usefully train) a mismatch detector without some
reports where a person has actually judged whether the stated priority
matches the write-up. These helpers turn dataset.csv into something easy
to hand-label in Google Sheets/Excel, then merge the labels back in.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

LABEL_COLUMN = "human_label"
NOTES_COLUMN = "human_notes"
VALID_LABELS = {"match", "mismatch", "unsure"}

# Matches the "Priority should be N because ..." convention used in
# human_notes for mismatch rows, e.g. "Priority should be 2 because the
# newest scan shows new frequencies in the Waterfall compared to previous
# scans". Case-insensitive, tolerant of the exact wording after the number.
CORRECTED_PRIORITY_RE = re.compile(r"priority\s+should\s+be\s+(?P<priority>\d+(?:\.\d+)?)", re.IGNORECASE)


def parse_corrected_priority(notes) -> float | None:
    """Pull the human-corrected priority number out of a human_notes entry.

    Returns None if notes is blank/NaN or doesn't follow the
    "Priority should be N ..." convention - callers should treat that as
    "no correction available", not as an error.
    """
    if notes is None or (isinstance(notes, float) and pd.isna(notes)):
        return None
    match = CORRECTED_PRIORITY_RE.search(str(notes))
    return float(match.group("priority")) if match else None


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


def _count_filled(series: pd.Series) -> int:
    """Count cells that are actually filled in.

    pandas.read_csv turns a literal empty cell into NaN, so NaN and ""
    both mean "not labeled yet" and need to be treated the same way here.
    """
    return int((series.fillna("").astype(str).str.strip() != "").sum())


def merge_labels(dataset_csv: str | Path, labeled_csv: str | Path, out_csv: str | Path) -> pd.DataFrame:
    """Merge human labels back onto the full dataset by report_id.

    Rows in dataset.csv with no corresponding labeled row (or a blank
    label) come back with human_label = NaN - still usable for training
    the priority-prediction model, just not for measuring mismatch
    detection accuracy.

    Also derives two columns from human_notes, for rows labeled "mismatch"
    that follow the "Priority should be N because ..." convention:
      - corrected_priority: the N, or NaN if not present/not parseable
      - true_priority: corrected_priority where available, else the
        stated priority_num - this is the best available training target
        for "what priority should this text actually imply", since the
        stated priority is exactly what you're saying is wrong on
        mismatch rows and shouldn't be trained on as if it were correct.

    Prints diagnostics so a "0 labeled" result is never silent: it tells
    you whether the problem is the file you pointed at (no labels found in
    it - often a stale/cached copy in a Drive-mounted path right after an
    upload or rename, try Runtime > Disconnect and delete runtime, then
    re-mount Drive) vs. a report_id mismatch between the two files.
    """
    dataset = pd.read_csv(dataset_csv)
    labeled = pd.read_csv(labeled_csv)

    n_raw_labeled = _count_filled(labeled[LABEL_COLUMN]) if LABEL_COLUMN in labeled.columns else 0
    print(f"{labeled_csv}: {n_raw_labeled} / {len(labeled)} rows have a non-blank {LABEL_COLUMN} as read.")
    if n_raw_labeled == 0:
        print(
            "^ That file has no labels in it as read just now, before any merging happened. "
            "If you know you filled it in, this is almost always a stale cached copy "
            "(common right after uploading/renaming a file in a Drive-mounted path in Colab) "
            "rather than a bug in the merge - re-mount Drive with force_remount=True and retry."
        )

    overlap = set(dataset["report_id"]) & set(labeled["report_id"])
    print(f"{len(overlap)} / {len(labeled)} labeled report_id values matched a row in {dataset_csv}.")
    if overlap and n_raw_labeled and len(overlap) < len(labeled) * 0.5:
        print("^ Less than half matched - check report_id spelling/format wasn't altered when the sheet was edited.")

    bad = set(labeled[LABEL_COLUMN].dropna().unique()) - VALID_LABELS - {""}
    if bad:
        raise ValueError(f"Unrecognized values in {LABEL_COLUMN}: {bad}. Expected one of {VALID_LABELS}.")

    merged = dataset.merge(
        labeled[["report_id", LABEL_COLUMN, NOTES_COLUMN]],
        on="report_id",
        how="left",
    )

    merged["corrected_priority"] = merged[NOTES_COLUMN].apply(parse_corrected_priority)
    is_mismatch = merged[LABEL_COLUMN] == "mismatch"
    has_correction = merged["corrected_priority"].notna()
    merged["true_priority"] = merged["priority_num"]
    merged.loc[is_mismatch & has_correction, "true_priority"] = merged.loc[is_mismatch & has_correction, "corrected_priority"]

    merged.to_csv(out_csv, index=False)
    n_labeled = _count_filled(merged[LABEL_COLUMN])
    print(f"Result: {n_labeled} / {len(merged)} rows in the merged dataset have a human label.")

    n_mismatch = int(is_mismatch.sum())
    n_corrected = int((is_mismatch & has_correction).sum())
    n_uncorrected = n_mismatch - n_corrected
    print(f"{n_corrected} / {n_mismatch} mismatch rows had a parseable 'Priority should be N' correction in {NOTES_COLUMN}.")
    if n_uncorrected:
        print(
            f"{n_uncorrected} mismatch rows had no parseable correction - true_priority falls back to the "
            "stated priority for those, so they won't help train the model away from the error. Add a "
            "'Priority should be N because ...' note to fix that."
        )
    return merged
