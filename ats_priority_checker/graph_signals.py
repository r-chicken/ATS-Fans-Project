"""Supporting priority signal read directly from the Spectrum plot's pixels
(top-left panel of the embedded chart screenshot), on top of the same OCR
text already used for style detection in extract.py.

Per the domain guidance behind this module: this is not a standalone
verdict. Report writers sometimes under-state urgency for equipment that's
been high-priority before, so this exists to catch that - but it's
supporting evidence to combine with the text-based prediction and with the
cross-report escalation signal (dataset.py), not a replacement for either.

Status / history:
- v1 (Fund Amp text field): read the printed "Fund Amp: X" line next to the
  Spectrum plot. Retired - Fund Amp is the amplitude at whichever harmonic
  order the analyst happened to list first, not the tallest peak in the
  plot, and the two can disagree a lot (e.g. a report with Fund Amp: 0.064
  whose actual tallest Spectrum peak is ~1.85 on a 0-2 scale). Do not bring
  Fund Amp extraction back.
- v2 (this version): reads the Spectrum plot's PIXELS directly - finds the
  tallest genuine data peak, reads it off the y-axis's own printed tick
  labels (OCR'd, calibrated with RANSAC so a misread digit or two doesn't
  wreck the whole scale), and floors it to the nearest labeled gridline at
  or below the peak. Validated by hand against 3 real reports spanning two
  different y-axis label spacings (0.5 and 0.05) and three different kinds
  of on-chart marker (a fixed-height magenta "harmonic flag" bar with a
  callout line down to the real peak; a cyan zoom-selection box with no
  relationship to data height at all; small red circle/number annotations
  sitting directly on top of genuine peaks). See _find_peak_pixel for how
  markers are told apart from real data. This is a best-effort pixel
  heuristic, not pixel-perfect - the plateau-width and run-length
  thresholds below are tuned against those 3 reports, not the full ~115
  labeled set, so if you have Spectrum peak readings that look visibly
  wrong once you run this against your real data, that's the first place
  to retune (see the constants just below the imports).
- Waterfall/Trend charts: intentionally not read anymore. The current
  project scope has one report per machine per date, spanning a real
  timeline, so escalation vs. the equipment's own recent history is now
  detected by comparing dated reports for the same equipment_id (see
  dataset.py's escalation signal) instead of pixel-reading the Waterfall
  or Trend panels. Do not re-add Waterfall/Trend pixel analysis without
  re-reading that discussion - the old Trend pixel heuristic here topped
  out around a 75% ceiling across five different techniques even for a
  narrower "is it escalating at all" question, well short of what precise
  point-reading would need.
- Future idea (not implemented): compare the FREQUENCIES of each report's
  peaks across dated reports for the same equipment, to catch a resonance
  shifting frequency even when its amplitude doesn't change much. Would
  need each peak's x-axis (frequency) calibration too, not just y-axis -
  same OCR+RANSAC approach as below would likely extend to it.
"""
from __future__ import annotations

import re

import numpy as np
from PIL import Image

TREND_OVERALL_RE = re.compile(r"trend\s+overall[:\s]*[\d.]+\s*(?P<unit>\S+)", re.IGNORECASE)

# --- Spectrum panel geometry -------------------------------------------
# The embedded screenshot is always the same 3-panel layout: Spectrum
# top-left, Waterfall top-right, Trend across the bottom. These fractions
# crop out just the Spectrum panel (with a little slack on the bottom/right
# so its own frame border is never clipped) and were checked against 3 real
# screenshots at three different pixel resolutions (1920x1080, 1902x1045,
# 1907x995) - all three cropped correctly with these fractions.
SPECTRUM_PANEL_WFRAC = 0.52
SPECTRUM_PANEL_HFRAC = 0.72

# Left margin (as a fraction of the panel width) that contains the y-axis
# tick number labels, for OCR.
Y_LABEL_STRIP_WFRAC = 0.10
OCR_UPSCALE = 4  # upscaling the label strip before OCR fixes most misreads

# UI chrome (toolbar/title bar) sits in the first ~10% of the panel height
# in every sample seen - the frame-border search below only has to look
# past that, which is what keeps it from mistaking a toolbar separator line
# for the plot's own top border.
CHROME_SKIP_FRAC = 0.12

# Peak-pixel classification (see _find_peak_pixel for what each guards
# against):
MAX_PLATEAU_WIDTH_PX = 8    # widest a genuine single-frequency peak should look
PLATEAU_ROW_TOL_PX = 2      # row jitter allowed while still calling two columns "level"
GAP_MERGE_PX = 3            # small antialiasing/letter gaps to bridge when measuring a run's height
MAX_RUN_HEIGHT_FRAC = 0.25  # fraction of plot height a genuine peak's own column may run solid
MIN_RUN_HEIGHT_PX = 2       # below this, treat it as compression/antialiasing noise, not a mark


def detect_spectrum_unit(ocr_text: str) -> str:
    """Best-effort unit detection for the Spectrum plot, read from the
    chart's title text (e.g. "...Trend Overall: 1.186 in/s") rather than
    the y-axis label itself, which is usually rendered rotated 90 degrees
    and OCRs unreliably. Returns "in/s", "g", "gE", or "unknown".

    This only reads the unit STRING off the title text - it has nothing to
    do with the Trend plot's pixels/history (see module docstring on why
    those aren't read anymore).

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


def _crop_spectrum_panel(chart_image: Image.Image) -> Image.Image:
    w, h = chart_image.size
    return chart_image.crop((0, 0, int(w * SPECTRUM_PANEL_WFRAC), int(h * SPECTRUM_PANEL_HFRAC)))


def _read_y_axis_ticks(panel: Image.Image) -> tuple[list[tuple[float, float]], float | None]:
    """OCR the y-axis tick number labels in the panel's left margin.

    Returns (points, label_right_edge) where points is a list of
    (pixel_row, value) pairs (not yet outlier-filtered - see
    _ransac_calibration for that) and label_right_edge is the x-position
    the tick labels are right-aligned against (used later to locate the
    plot frame's left border, which sits just past it).

    Two things this works around, found by testing against real reports:
    - Tesseract regularly drops the decimal point on these small chart
      fonts ("0.5" -> "05"). Rather than trying to guess where a missing
      point goes, this only keeps tokens that already look unambiguous
      ("\\d+\\.\\d+" or a bare "\\d+") and lets RANSAC calibration (next
      step) work from whatever legible subset that leaves.
    - The axis title ("gE - Peak"), a value baked into the title text
      above the plot, and other stray OCR fragments can each produce a
      spurious numeric-looking token. Real tick labels are right-aligned
      against the axis line, so this clusters candidates by their right
      edge and keeps only the largest cluster - reliably keeps the real
      ticks and drops the rest, even when a couple of them have very low
      OCR confidence (confidence alone was tried and wasn't reliable
      enough to use as the primary filter here).
    """
    import pytesseract

    cw, ch = panel.size
    strip = panel.crop((0, 0, int(cw * Y_LABEL_STRIP_WFRAC), ch))
    strip_up = strip.resize((strip.size[0] * OCR_UPSCALE, strip.size[1] * OCR_UPSCALE), Image.LANCZOS)
    data = pytesseract.image_to_data(
        strip_up,
        config="--psm 6 -c tessedit_char_whitelist=0123456789.",
        output_type=pytesseract.Output.DICT,
    )

    candidates = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text or not re.fullmatch(r"\d+\.\d+|\d+", text):
            continue
        row = (data["top"][i] + data["height"][i] / 2) / OCR_UPSCALE
        right_edge = (data["left"][i] + data["width"][i]) / OCR_UPSCALE
        candidates.append((row, float(text), right_edge))

    if not candidates:
        return [], None

    edges = np.array([c[2] for c in candidates])
    best_center, best_count = edges[0], 0
    for e in edges:
        count = int(np.sum(np.abs(edges - e) <= 4))
        if count > best_count:
            best_count, best_center = count, e

    points = sorted(
        {(row, val) for row, val, edge in candidates if abs(edge - best_center) <= 4},
        key=lambda p: p[1],
    )
    return points, float(best_center)


def _ransac_calibration(points: list[tuple[float, float]], row_tol: float = 6.0):
    """Fit pixel_row = a*value + b from OCR'd tick points, RANSAC-style.

    Needed because a single bad OCR token (most often two adjacent labels
    merged into one, e.g. "0.4" + "0.45" -> "20.45") is common enough that
    a plain least-squares fit (even with iterative residual trimming) can
    get pulled off badly enough to then reject the GOOD points instead.
    Trying every pair of points as a candidate line and keeping whichever
    line the most other points agree with sidesteps that.

    Returns (a, b, kept_points) or None if fewer than 2 usable points.
    """
    import itertools

    points = sorted(set(points), key=lambda p: p[1])
    if len(points) < 2:
        return None
    rows = np.array([p[0] for p in points])
    vals = np.array([p[1] for p in points])

    best_inliers = None
    for i, j in itertools.combinations(range(len(points)), 2):
        if vals[i] == vals[j]:
            continue
        a = (rows[j] - rows[i]) / (vals[j] - vals[i])
        if a >= 0:
            continue  # pixel row must decrease as the axis value increases
        b = rows[i] - a * vals[i]
        inliers = np.abs(rows - (a * vals + b)) <= row_tol
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers

    if best_inliers is None or best_inliers.sum() < 2:
        return None
    a, b = np.polyfit(vals[best_inliers], rows[best_inliers], 1)
    kept = list(zip(rows[best_inliers].tolist(), vals[best_inliers].tolist()))
    return a, b, kept


def _find_plot_frame(arr: np.ndarray, label_right_edge: float, a: float, b: float, max_tick_val: float):
    """Locate the plot's own black frame border: (left, right, top, bottom)
    in panel-pixel coordinates.

    Toolbar/title-bar chrome above the plot can be just as dark as the
    frame border, so top/bottom aren't found by "the darkest row" - they're
    found by anchoring to the y-axis calibration instead: bottom is the
    strong horizontal line closest to the calibration's own value=0 row,
    and top is the strong line closest to where the highest OCR'd tick
    should be. That estimate is enough even when OCR missed the very top
    label entirely (seen for real: a "2" tick rendered flush against a
    small triangle max-value marker, which tesseract failed to read even
    after upscaling) - the frame border pixel search recovers the exact
    row anyway.
    """
    H, W = arr.shape[0], arr.shape[1]
    r, g, bch = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    dark = (r < 140) & (g < 140) & (bch < 140)

    lo, hi = int(label_right_edge), int(label_right_edge) + 40
    col_frac = dark.mean(axis=0)
    left = lo + int(np.argmax(col_frac[lo:hi]))

    lo2, hi2 = int(W * 0.85), int(W * 0.995)
    right = hi2
    for i in range(lo2, hi2):
        if col_frac[i] > 0.5:
            right = i
            break

    row_frac = dark[:, left:right].mean(axis=1)
    strong_rows = np.where(row_frac > 0.5)[0]
    if len(strong_rows) == 0:
        return left, right, int(CHROME_SKIP_FRAC * H), H - 1

    bottom = int(strong_rows[np.argmin(np.abs(strong_rows - b))])
    est_top = a * max_tick_val + b
    below = strong_rows[strong_rows < bottom - 20]
    top = int(below[np.argmin(np.abs(below - est_top))]) if len(below) else int(max(est_top, 0))
    return left, right, top, bottom


def _ink_mask(region: np.ndarray) -> np.ndarray:
    """True where a pixel is plot "ink" (trace, marker, annotation - any
    color), False for plain white background or a gridline.

    Gridline gray is deliberately given some slack (up to a 20-point
    channel spread, not just near-equal RGB) - a plain equality check left
    a few antialiased near-gray pixels right next to the frame border
    classified as "ink", which produced a phantom 1-pixel-tall peak at the
    very top of the plot on a real report.
    """
    r, g, b = region[:, :, 0], region[:, :, 1], region[:, :, 2]
    white = (r > 225) & (g > 225) & (b > 225)
    gray = (~white) & (r > 140) & (np.abs(r - g) < 20) & (np.abs(g - b) < 20) & (np.abs(r - b) < 20)
    return ~(white | gray)



# A run whose pixels are (mostly) this saturated a blue is exempt from the
# tall-run marker/cursor-line cap in _find_peak_pixel - see that function's
# docstring point 2, and _is_saturated_blue below for why this is a fixed
# domain threshold rather than a per-report "dominant color" estimate.
SATURATED_BLUE_MAX_RG = 60
SATURATED_BLUE_MIN_B = 180
SATURATED_BLUE_MIN_FRAC = 0.5  # fraction of a run's own pixels that must qualify


def _is_saturated_blue(pixels: np.ndarray) -> np.ndarray:
    """True per-pixel for a real, richly-saturated blue (R and G both low,
    B clearly dominant) - calibrated against real reports' own baseline
    trace color (sampled from ordinary noise-floor ink, well away from any
    marker): (26, 6, 233), (30, 12, 228), (22, 10, 236) all comfortably
    clear SATURATED_BLUE_MAX_RG=60. A red cursor/order-marker line, e.g.
    (191, 0, 0), fails outright (R way over the cap). Two real reports use
    a lighter, washed-out blue for their trace instead - (128, 122, 245),
    (125, 128, 253) - which does NOT qualify here on purpose: on one of
    those two reports, a decorative marker-label text block happened to be
    rendered in almost exactly that same washed-out blue (126, 123, 255),
    close enough that no per-report color estimate (tried: most total
    pixels, then most distinct columns touched - both picked the text
    color, not the trace's, on that report) could tell real data and
    decorative text apart there. This fixed, deliberately narrow threshold
    sidesteps that: it only ever exempts a run when it's confident, and
    reports whose trace doesn't qualify just get the ordinary run-height
    cap applied to everything, same as before this exemption existed -
    which was already the correct answer on both washed-out-blue reports
    seen so far, since neither had a genuine peak tall enough to need the
    exemption.
    """
    r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    return (r < SATURATED_BLUE_MAX_RG) & (g < SATURATED_BLUE_MAX_RG) & (b > SATURATED_BLUE_MIN_B)


def _find_peak_pixel(arr: np.ndarray, left: int, right: int, top: int, bottom: int):
    """Find the (row, col) of the Spectrum plot's tallest genuine data
    peak, in panel-pixel coordinates - the core of "read the actual graph,
    not markings that get in the way".

    On these reports, on-chart markers/annotations come in two shapes that
    both need excluding, and one shape that's fine to leave in:

    1. Wide flat-topped blocks - a colored bar flagging a fault frequency,
       a zoom-selection box, etc. Many adjacent columns share (about) the
       same topmost ink row. A real spectral peak is one frequency wide,
       at most a couple of pixels - so any run of more than
       MAX_PLATEAU_WIDTH_PX columns at the same height is a block, not
       data, and every column in it is dropped. Applies regardless of
       color - a genuine trace essentially never forms a flat multi-column
       plateau (real spectral noise is jagged), so this is safe to apply
       universally.
    2. Tall thin marks that AREN'T richly-saturated blue - rotated
       marker-label text, a UI cursor/order-marker line (e.g. red, with a
       small square handle sitting right at/above the frame's top edge -
       confirmed against real reports; the small circle sitting just below
       that handle is part of the same marker, not a peak annotation) -
       only 1-2 columns wide, so the width check above doesn't catch them.
       What does: their own column runs solid ink for most of the plot's
       height (verified on a real report: 318-474px on a ~430px-tall
       plot). Antialiasing and letter-shaped gaps inside a marker can chop
       that long run into several short ones a few pixels apart -
       GAP_MERGE_PX bridges gaps that small before measuring, so the
       marker doesn't masquerade as a short (real-looking) run. This
       height cap is deliberately NOT applied to a run that's (mostly)
       richly-saturated blue - see _is_saturated_blue - a genuine sharp,
       narrow-bandwidth resonance can legitimately run near-vertically for
       most of the plot's height in a single column (confirmed on a real
       report: a real peak's own column ran ~300px on a 467px-tall plot,
       comfortably over what an earlier, color-blind version of this cap
       excluded as "too tall to be real data" - the taller and more severe
       the true peak, the more likely that mistake was to happen, which is
       exactly backwards), and the real peak is reliably THICKER (more ink
       per row) than a hairline cursor/order-marker line even where they
       sit close together in x. _is_saturated_blue uses a fixed threshold
       tuned from real reports' own baseline trace color rather than a
       per-report "what's the dominant color here" estimate - tried that
       first (twice), and both approaches picked a report's decorative
       marker-label text over its actual trace on a report where the two
       happen to be rendered in nearly the same (unsaturated) blue - see
       that function's docstring for the full story before changing this.
    3. Small annotations directly on a real peak (a circle, a number, a
       triangle) - a few pixels, sitting right at or just above the
       genuine tip. These pass every filter above and are left in on
       purpose: they shift the reading by at most a few pixels, and that's
       an acceptable trade for not needing a marker color palette that
       would have to be hand-maintained per report style.

    Returns (row, col) of the winning pixel, or (None, None) if the plot
    area has no ink at all.
    """
    y0, y1 = top + 1, bottom
    region = arr[y0:y1, left + 2 : right - 1, :]
    ink = _ink_mask(region)
    H, W = ink.shape
    run_cap = MAX_RUN_HEIGHT_FRAC * H

    topmost = np.full(W, H)
    for c in range(W):
        idx = np.where(ink[:, c])[0]
        if len(idx):
            topmost[c] = idx[0]

    # Drop wide flat-topped plateaus (marker bars/boxes) - color-blind by
    # design, see docstring point 1.
    c = 0
    width_ok_cols = []
    while c < W:
        if topmost[c] == H:
            c += 1
            continue
        c2 = c
        while c2 + 1 < W and topmost[c2 + 1] != H and abs(int(topmost[c2 + 1]) - int(topmost[c])) <= PLATEAU_ROW_TOL_PX:
            c2 += 1
        if c2 - c + 1 <= MAX_PLATEAU_WIDTH_PX:
            width_ok_cols.extend(range(c, c2 + 1))
        c = c2 + 1

    # Drop columns whose own (gap-merged) topmost run is implausibly tall
    # for a single spectral line UNLESS that run is (mostly) richly-
    # saturated blue - see docstring point 2 and _is_saturated_blue.
    valid_cols = []
    for c in width_ok_cols:
        idx = np.where(ink[:, c])[0]
        splits = np.where(np.diff(idx) > GAP_MERGE_PX)[0]
        first_run = np.split(idx, splits + 1)[0]
        run_len = first_run[-1] - first_run[0] + 1
        if run_len < MIN_RUN_HEIGHT_PX:
            continue
        if run_len <= run_cap:
            valid_cols.append(c)
            continue
        run_pixels = region[first_run, c]
        if _is_saturated_blue(run_pixels).mean() >= SATURATED_BLUE_MIN_FRAC:
            valid_cols.append(c)

    if not valid_cols:
        valid_cols = width_ok_cols or [c for c in range(W) if topmost[c] != H]
    if not valid_cols:
        return None, None

    best_col = min(valid_cols, key=lambda c: topmost[c])
    return y0 + int(topmost[best_col]), left + 2 + best_col


def _floor_to_axis_label(value: float, kept_points: list[tuple[float, float]]) -> float:
    """Snap value down to the largest y-axis tick label at or below it -
    "read the peak, match it to the closest label rounded down" - rather
    than reporting a continuous interpolated number. This deliberately
    trades a little precision for staying anchored to a number that's
    actually printed on the chart.
    """
    ticks = sorted({v for _, v in kept_points} | {0.0})
    floor_val = 0.0
    for t in ticks:
        if t <= value:
            floor_val = t
        else:
            break
    return floor_val


def read_spectrum_peak(chart_image: Image.Image) -> dict:
    """Read the Spectrum plot's tallest genuine peak off its own y-axis.

    Returns a dict with:
      peak_amplitude       the peak, floored to the nearest y-axis label
                            at or below it (float), or None if calibration
                            or peak-finding failed
      peak_amplitude_raw   the same reading before flooring (float or None) -
                            kept for debugging/inspection, not used for
                            priority thresholds
      y_axis_ticks         the (value) labels used for calibration, for
                            sanity-checking against the actual chart
      error                None on success, else a short string saying
                            what failed (e.g. "could not OCR enough y-axis
                            tick labels to calibrate")

    Never raises - any failure (OCR found <2 usable tick labels, frame
    border not found, empty plot area, ...) comes back as
    peak_amplitude=None with `error` explaining why, so one bad chart image
    doesn't take down a whole build_dataset() run. Callers should treat
    peak_amplitude=None the same as "no signal available", same as the old
    Fund Amp path did below its own noise floor.
    """
    try:
        panel = _crop_spectrum_panel(chart_image)
        points, label_right_edge = _read_y_axis_ticks(panel)
        if label_right_edge is None:
            return {"peak_amplitude": None, "peak_amplitude_raw": None, "y_axis_ticks": [], "error": "no y-axis tick labels OCR'd"}

        calibration = _ransac_calibration(points)
        if calibration is None:
            return {"peak_amplitude": None, "peak_amplitude_raw": None, "y_axis_ticks": [], "error": "could not calibrate y-axis (fewer than 2 consistent tick labels)"}
        a, b, kept = calibration

        arr = np.asarray(panel.convert("RGB")).astype(int)
        max_tick_val = max(v for _, v in kept)
        left, right, top, bottom = _find_plot_frame(arr, label_right_edge, a, b, max_tick_val)
        if right <= left + 4 or bottom <= top + 4:
            return {"peak_amplitude": None, "peak_amplitude_raw": None, "y_axis_ticks": sorted({v for _, v in kept}), "error": "plot frame border not found"}

        peak_row, _peak_col = _find_peak_pixel(arr, left, right, top, bottom)
        if peak_row is None:
            return {"peak_amplitude": None, "peak_amplitude_raw": None, "y_axis_ticks": sorted({v for _, v in kept}), "error": "no data ink found in plot area"}

        raw_value = (peak_row - b) / a
        floored = _floor_to_axis_label(raw_value, kept)
        return {
            "peak_amplitude": floored,
            "peak_amplitude_raw": raw_value,
            "y_axis_ticks": sorted({v for _, v in kept}),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - one bad image shouldn't kill a batch run
        return {"peak_amplitude": None, "peak_amplitude_raw": None, "y_axis_ticks": [], "error": f"unexpected error: {exc}"}


# --- Priority thresholds per unit ---------------------------------------
# All three read off the Spectrum peak amplitude (read_spectrum_peak above),
# never off Fund Amp - see module docstring.


def velocity_priority_hint(amp: float) -> int:
    """Velocity (in/s) peak amplitude -> priority. >1 -> 1, 0.5-1 -> 2,
    0.1-0.5 -> 3, <0.1 -> 4."""
    if amp > 1:
        return 1
    if amp >= 0.5:
        return 2
    if amp >= 0.1:
        return 3
    return 4


def acceleration_enveloping_priority_hint(amp: float) -> int:
    """Acceleration enveloping (gE) peak amplitude -> priority. >0.54 -> 1,
    0.3-0.54 -> 2, 0.09-0.3 -> 3, <0.09 -> 4."""
    if amp > 0.54:
        return 1
    if amp >= 0.3:
        return 2
    if amp >= 0.09:
        return 3
    return 4


def acceleration_priority_hint(amp: float) -> int:
    """Acceleration (g) peak amplitude -> priority. >2.5 -> 1, >2 -> 2,
    >1 -> 3, else -> 4. Few reports use plain g (most are gE) - thresholds
    here are as given, not independently re-derived from a large sample."""
    if amp > 2.5:
        return 1
    if amp > 2:
        return 2
    if amp > 1:
        return 3
    return 4


_UNIT_HINT_FNS = {
    "in/s": velocity_priority_hint,
    "gE": acceleration_enveloping_priority_hint,
    "g": acceleration_priority_hint,
}


def spectrum_priority_hint(chart_image: Image.Image, ocr_text: str) -> dict:
    """Combine unit detection (OCR text) with the pixel-read Spectrum peak
    (read_spectrum_peak) into one supporting priority signal.

    Needs the chart IMAGE now, not just its OCR text - unlike the old Fund
    Amp version, the peak reading is pixel analysis, not a text field.
    """
    unit = detect_spectrum_unit(ocr_text)
    peak = read_spectrum_peak(chart_image)
    amp = peak["peak_amplitude"]
    hint_fn = _UNIT_HINT_FNS.get(unit)
    priority_hint = hint_fn(amp) if (hint_fn is not None and amp is not None) else None
    return {
        "spectrum_unit": unit,
        "spectrum_peak_amplitude": amp,
        "spectrum_peak_amplitude_raw": peak["peak_amplitude_raw"],
        "spectrum_priority_hint": priority_hint,
        "spectrum_peak_error": peak["error"],
    }
