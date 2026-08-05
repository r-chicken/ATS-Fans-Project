"""Supporting priority signals read from the embedded chart screenshot
(Spectrum / Waterfall / Trend), on top of the same OCR text already used
for style detection in extract.py.

Per the domain guidance behind this module: none of these signals is a
standalone verdict. Report writers sometimes under-state urgency for
equipment that's been high-priority before, so these exist to catch that -
but they're supporting evidence to combine with the text-based prediction
and with each other, not a replacement for it.

Status:
- Spectrum unit + Fund Amp threshold rule (this module): implemented and
  validated against a real report's OCR text.
- Trend (last-point-vs-previous, and last-5-points overall direction) and
  Waterfall (new frequency vs. amplitude-only increase on an existing one,
  ignoring routine noise-floor drift) are NOT implemented yet. Both need
  pixel-level analysis of a raster screenshot rather than text search, and
  need validation against several real, diverse examples (different scan
  counts, both obvious and subtle cases) before they can be trusted -
  don't wire a stub in here without that.
"""
from __future__ import annotations

import re

FUND_AMP_RE = re.compile(r"fund\s*amp[:\s]*(?P<amp>[\d.]+)", re.IGNORECASE)
TREND_OVERALL_RE = re.compile(r"trend\s+overall[:\s]*[\d.]+\s*(?P<unit>\S+)", re.IGNORECASE)


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
