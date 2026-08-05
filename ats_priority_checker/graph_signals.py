"""Supporting priority signals read from the embedded chart screenshot
(Spectrum / Waterfall / Trend), on top of the same OCR text already used
for style detection in extract.py.

Per the domain guidance behind this module: none of these signals is a
standalone verdict. Report writers sometimes under-state urgency for
equipment that's been high-priority before, so these exist to catch that -
but they're supporting evidence to combine with the text-based prediction
and with each other, not a replacement for it.

Status:
- Spectrum unit + Fund Amp threshold rule: implemented, validated against
  real reports' OCR text.
- Trend (escalation detection + priority hint): implemented below. Design
  note, because it took several failed attempts to land on: precisely
  locating "the second-to-last point" and "the last point" as exact pixel
  positions turned out to be unreliable (~75% ceiling across five
  different pixel-heuristic techniques, tested against 14 real reports
  with human-confirmed ground truth) - duplicate "current value" callout
  markers, variable point density, and small-magnitude changes near the
  noise floor all defeat precise point-matching. The design here sidesteps
  that: pixel analysis only has to answer a COARSER question (is the
  recent trend escalating at all - see trend_escalation_signal), and the
  actual priority NUMBER comes from the report's own printed "Amp: X,
  Date/Time: ..." reading (OCR text, already reliable) using the same
  velocity thresholds as Fund Amp - confirmed against a matched pair of
  real reports (one Priority 1, one Priority 2) where the relative jump
  size was nearly identical but the absolute landing value differed
  exactly at the >1 / 0.5-1 threshold boundary.
- Waterfall (new frequency vs. amplitude-only increase on an existing one,
  ignoring routine noise-floor drift) is NOT implemented yet - needs
  pixel-level trace segmentation by color (red=latest, blue=history) and
  peak-matching across scans, a harder problem than Trend. Don't wire a
  stub in here without validating it the same way.
"""
from __future__ import annotations

import re

import numpy as np
from PIL import Image

FUND_AMP_RE = re.compile(r"fund\s*amp[:\s]*(?P<amp>[\d.]+)", re.IGNORECASE)
TREND_OVERALL_RE = re.compile(r"trend\s+overall[:\s]*[\d.]+\s*(?P<unit>\S+)", re.IGNORECASE)
TREND_CURRENT_RE = re.compile(r"(?<!fund )amp[:\s]*(?P<amp>[\d.]+)\s*,\s*date", re.IGNORECASE)


def detect_spectrum_unit(ocr_text: str) -> str:
    """Best-effort unit detection for the Spectrum plot, read from the
    chart's title text (e.g. "...Trend Overall: 1.186 in/s") rather than
    the y-axis label itself, which is usually rendered rotated 90 degrees
    and OCRs unreliably. Returns "in/s", "g", "gE", or "unknown".

    Anchored specifically to the token right after "Trend Overall: N" -
    confirmed against real reports this is far more reliable than
    searching the whole OCR blob. "gE" is exactly 2 characters, and OCR
    sometimes misreads the italic "E" as something else entirely (seen for
    real as "g&") - since the whole unit token is only ever 1-2 characters
    here, matching its *exact* length distinguishes a genuine bare "g"
    (1 char) from a garbled "gE" (2 chars, second char unreliable) without
    needing to know what the garbled character actually is.
    """
    match = TREND_OVERALL_RE.search(ocr_text)
    if match:
        token = match.group("unit").strip(",;:.")
        if re.search(r"in\s*/\s*s", token, re.IGNORECASE):
            return "in/s"
        if re.fullmatch(r"g.", token, re.IGNORECASE):
            return "gE"
        if re.fullmatch(r"g", token, re.IGNORECASE):
            return "g"
        return "unknown"

    # No "Trend Overall" anchor found at all - fall back to a broader,
    # less precise search over the whole OCR text.
    if re.search(r"in\s*/\s*s", ocr_text, re.IGNORECASE):
        return "in/s"
    if re.search(r"\bgE\b", ocr_text):
        return "gE"
    if re.search(r"\bg\b", ocr_text):
        return "g"
    return "unknown"


def extract_fund_amp(ocr_text: str) -> float | None:
    """Pull the "Fund Amp: X" reading out of the Spectrum plot's OCR text.

    Some reports show more than one "Fund Amp" line - one per peak the
    analyst has marked, each at its own harmonic order. This takes
    whichever is listed FIRST in the text, by explicit choice - not by
    Order value. (An earlier version of this selected the line whose
    Order was closest to 1, i.e. the true fundamental; that was tried and
    then explicitly reverted back to first-by-position. Note this means a
    report that lists a non-fundamental order first, e.g. report_018
    lists Order 3.04 before Order 1, returns the Order-3.04 amplitude
    here, not the fundamental's.)
    """
    match = FUND_AMP_RE.search(ocr_text)
    return float(match.group("amp")) if match else None


def velocity_fund_amp_priority_hint(fund_amp: float) -> int | None:
    """Priority suggested by Spectrum Fund Amp, velocity (in/s) units only.

    Thresholds: 0.1-0.5 -> 3, 0.5-1 -> 2, >1 -> 1. Returns None below 0.1 -
    that's "no strong signal either way", not an implicit priority 4;
    plenty of real high-priority reports have a small Fund Amp because the
    real severity shows up in the Trend/Waterfall instead (see the
    conversation this module was built from for a concrete example).
    """
    if fund_amp > 1:
        return 1
    if fund_amp >= 0.5:
        return 2
    if fund_amp >= 0.1:
        return 3
    return None


def spectrum_priority_hint(ocr_text: str) -> dict:
    """Combine unit detection + Fund Amp extraction into one supporting
    signal from the Spectrum plot.

    For g/gE units, a high Spectrum amplitude alone doesn't indicate a
    problem (per domain guidance) - only Waterfall changes across scans
    do, which isn't implemented yet (see module docstring) - so
    priority_hint is only ever populated for in/s right now.
    """
    unit = detect_spectrum_unit(ocr_text)
    fund_amp = extract_fund_amp(ocr_text)
    priority_hint = None
    if unit == "in/s" and fund_amp is not None:
        priority_hint = velocity_fund_amp_priority_hint(fund_amp)
    return {
        "spectrum_unit": unit,
        "spectrum_fund_amp": fund_amp,
        "spectrum_priority_hint": priority_hint,
    }


def extract_trend_current_value(ocr_text: str) -> float | None:
    """Pull the Trend plot's own current-reading line, e.g.
    "...Amp: 1.155, Date/Time: 7/9/2026 7:29:04 AM" - this is the same
    value as the report's stated overall reading, already reliable via
    OCR text, so no pixel analysis is needed for the value itself.

    The negative lookbehind excludes "Fund Amp: X" (Spectrum plot,
    different quantity, no trailing ", Date/Time") - this pattern only
    matches when "amp:" is immediately followed by ", date", which Fund
    Amp lines never are.
    """
    match = TREND_CURRENT_RE.search(ocr_text)
    return float(match.group("amp")) if match else None


def _trend_mask_and_boundary(chart_image: Image.Image, min_count: int = 2) -> tuple[np.ndarray, int]:
    """Crop to the Trend plot, color-mask the line, and find the boundary
    before the duplicate "current value" callout (red/gray vertical
    cursor line, with a safety margin - see module docstring for why this
    boundary-finding logic exists rather than trying to locate exact
    marker positions).
    """
    w, h = chart_image.size
    x0, x1 = int(w * 0.02), int(w * 0.99)
    y0 = int(h * 0.64)
    region = np.asarray(chart_image.crop((x0, y0, x1, h)).convert("RGB")).astype(int)
    r, g, b = region[:, :, 0], region[:, :, 1], region[:, :, 2]
    mask = (b > 200) & (b - r > 40) & (b - g > 15) & (np.abs(r - g) < 60)

    red_mask = (r > 150) & (g < 100) & (b < 100)
    gray_mask = (np.abs(r - g) < 20) & (np.abs(g - b) < 20) & (r < 150)
    red_cols = np.where(red_mask.sum(axis=0) > region.shape[0] * 0.5)[0]
    gray_cols = np.where(gray_mask.sum(axis=0) > region.shape[0] * 0.5)[0]
    gray_cols = gray_cols[gray_cols > region.shape[1] * 0.5]
    candidates = list(red_cols) + list(gray_cols)
    raw_cutoff = int(min(candidates)) if candidates else mask.shape[1]
    return mask, raw_cutoff


def trend_escalation_signal(
    chart_image: Image.Image,
    near_window: int = 100,
    recent_window: int = 50,
    last5_window: int = 400,
    boundary_margin: int = 10,
    sharp_threshold_frac: float = 0.15,
    sustained_threshold_frac: float = 0.15,
    fade_threshold_frac: float = 0.10,
) -> str:
    """Coarse escalation check per the decision tree: look at second-to-
    last -> last first; if that's not a clear rise, fall back to whether
    the last-5 window shows a jump that's still holding.

    Returns one of:
      "sharp_recent_increase" - the immediate step up is itself large
      "sustained_increase"    - a jump happened within the last-5 window
                                 and hasn't faded back down since
      "stable_high"           - elevated within the window, but not new -
                                 already faded back, or no distinct jump
      "no_signal"              - nothing notable, or too few real data
                                 points to judge (e.g. only 1-2 points)

    This deliberately does NOT try to pinpoint exact point positions
    (see module docstring for why that proved unreliable) - it compares
    window averages, which is far more robust to exactly where a marker
    sits.
    """
    mask, raw_cutoff = _trend_mask_and_boundary(chart_image)
    boundary = max(raw_cutoff - boundary_margin, 1)

    def avg_height(x_start: int, x_end: int) -> float:
        x_start = max(x_start, 0)
        x_end = min(x_end, boundary)
        if x_end <= x_start:
            return float("nan")
        ys = np.where(mask[:, x_start:x_end])[0]
        return float(ys.mean()) if len(ys) else float("nan")

    all_ys = np.where(mask[:, :boundary])[0]
    if len(all_ys) == 0:
        return "no_signal"
    span = float(all_ys.max() - all_ys.min())
    if span <= 0:
        return "no_signal"

    recent_avg = avg_height(boundary - recent_window, boundary)
    near_avg = avg_height(boundary - recent_window - near_window, boundary - recent_window)
    baseline_avg = avg_height(0, max(boundary - recent_window - near_window - last5_window, 0))
    last5_avg = avg_height(boundary - recent_window - near_window - last5_window, boundary - recent_window)

    if np.isnan(recent_avg):
        return "no_signal"

    # Tier 1: sharp step up from the near-term (second-to-last-ish) level.
    # Smaller y = higher chart value, so a rise shows up as near_avg - recent_avg > 0.
    if not np.isnan(near_avg) and (near_avg - recent_avg) > sharp_threshold_frac * span:
        return "sharp_recent_increase"

    # Tier 2: did the wider last-5 window jump up from baseline, and is
    # that elevation still holding (recent hasn't faded back down)?
    if not np.isnan(baseline_avg) and not np.isnan(last5_avg):
        jumped = (baseline_avg - last5_avg) > sustained_threshold_frac * span
        still_elevated = (baseline_avg - recent_avg) > fade_threshold_frac * span
        if jumped and still_elevated:
            return "sustained_increase"
        if jumped:
            return "stable_high"

    return "no_signal"


def trend_priority_hint(chart_image: Image.Image, ocr_text: str) -> dict:
    """Combine escalation detection (pixel-based, coarse) with the
    report's own printed current-reading value (OCR text, reliable) into
    a supporting priority signal from the Trend plot.

    Only escalating cases ("sharp_recent_increase" / "sustained_increase")
    get a priority number, using the same velocity thresholds as Fund Amp
    applied to the Trend's current value - confirmed against a real
    Priority 1 vs Priority 2 pair with near-identical relative jump sizes
    but current values landing on either side of the >1 / 0.5-1 boundary.
    "stable_high" and "no_signal" don't get a number: an unchanging high
    reading isn't new information, and no clear pattern isn't a signal.

    Like the Spectrum hint, only populated for in/s right now - no
    threshold has been established yet for g/gE Trend readings.
    """
    unit = detect_spectrum_unit(ocr_text)
    current_value = extract_trend_current_value(ocr_text)
    escalation = trend_escalation_signal(chart_image)
    priority_hint = None
    if unit == "in/s" and current_value is not None and escalation in ("sharp_recent_increase", "sustained_increase"):
        priority_hint = velocity_fund_amp_priority_hint(current_value)
    return {
        "trend_current_value": current_value,
        "trend_escalation": escalation,
        "trend_priority_hint": priority_hint,
    }
