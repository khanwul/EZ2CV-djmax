"""
EZ2CV — Layer 4 / bpm_estimator : beats → piecewise-linear BPM
===============================================================================
Turns the POW LED beat stream (Layer 3's ``BeatEvent`` list) into a piecewise-
LINEAR BPM curve. Output is a ``TickClock`` (tick-bounded ``BPMSegment``\\s plus
their ms-anchors).

Algorithm
---------
1. **Active-window clip**. Beats outside ``[first_barline, last_barline]`` (the
   in-song range) are dropped from the BPM analysis. The intro fade-in often
   produces LED flashes that are not on a real musical beat; including them
   would skew the median used by the octave fold and spawn a fake first
   segment well below ``min_bpm``.
2. **Coarse change-points on a SMOOTHED interval series**. ``BeatEvent.frame_
   index`` is an integer, so individual inter-beat intervals jitter by ±1
   frame regardless of the true tempo — at 60 fps and ~215 BPM that is ±6 %,
   which would spawn a change-point every beat. We therefore difference a
   rolling-MEDIAN of intervals (window ``smooth_window``) rather than the raw
   series, then flag a break wherever the smoothed change exceeds ``±3 %``. A
   median (not mean) keeps sharp tempo steps sharp instead of ramping them.
3. **Per-segment octave fold**. Each segment's MEDIAN BPM is folded into
   ``[cal.min_bpm, cal.max_bpm]`` independently — a single global fold breaks
   on songs where the LED octave changes (e.g. half the song at 110 BPM
   flashing on beats, the other half at 220 BPM flashing on half-beats).
4. **Linear fit + endpoint clamp**. Least-squares BPM-vs-time over the
   segment's instantaneous BPMs, then clamp ``bpm_start`` / ``bpm_end`` to the
   segment's 5th..95th percentile of observed BPM so a short, noisy segment
   cannot extrapolate to physically impossible values (the symptom that gave
   JUSTITIA a "46 BPM" segment).
5. **Slope-zero collapse**. ``|Δbpm| < 0.1`` → constant segment.
6. **Constant-segment merge**. Adjacent constant segments at the same BPM
   collapse into one.

PELT 2nd-pass refinement is still future work — the coarse pass is sufficient
to clear the ±2 % combo-count gate on all three test songs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from layer3.beat import BeatEvent
from layer4.tick_clock import BPMDraft, TickClock


# =============================================================================
# Tunables
# =============================================================================

DEFAULT_CHANGE_RATE = 0.03         # ±3 % rate-change → segment break
DEFAULT_SMOOTH_WINDOW = 8          # rolling median span used to denoise intervals
SLOPE_ZERO_BPM = 0.1               # |Δbpm| < this  → collapse to constant
RELATIVE_SLOPE_TOL = 0.025         # |Δbpm|/level < this → also collapse to
                                   # constant. Kills fit-noise "ramps" like
                                   # 190.0→194.1 that the absolute gate leaves
                                   # as spurious accelerandos; real game
                                   # accelerandos exceed it.
MIN_BEATS_PER_SEGMENT = 16         # bigger than the smooth window; over-fit guard
ANCHOR_MIN_SPLIT_LEN = 3           # min split half-length when global anchor is
                                   # in effect (user range trusted → over-split
                                   # risk is bounded by hard endpoint clamps)
ENDPOINT_PERCENTILE = (5.0, 95.0)  # linear-fit clamp band per segment
ACTIVE_WINDOW_SLACK_MS = 200.0     # keep beats just outside the bar window
BIMODAL_RATIO = 1.4                # bpm_end / bpm_start ≥ this → octave switch
                                   # somewhere in the segment → split at the
                                   # variance-minimising point. 1.4 keeps a
                                   # genuine 40 % accelerando intact (single
                                   # segment) but catches the 1.8× and 2×
                                   # ramps that betray a mid-segment fold.


# =============================================================================
# Octave fold
# =============================================================================

def _fold_to_range(bpm: float, lo: float, hi: float) -> float:
    """Halve/double until ``bpm`` lies in ``[lo, hi]``. Bail after 4 steps."""
    if lo <= 0 or hi <= 0 or hi < lo:
        return bpm
    # generous slack — min_bpm/max_bpm in song.toml may be tight
    slo, shi = lo * 0.9, hi * 1.1
    for _ in range(4):
        if bpm < slo:
            bpm *= 2.0
        elif bpm > shi:
            bpm *= 0.5
        else:
            return bpm
    return bpm


def _octave_factor(median_bpm: float, lo: float, hi: float) -> float:
    """Power-of-two factor that maps ``median_bpm`` into [lo, hi]."""
    if median_bpm <= 0:
        return 1.0
    folded = _fold_to_range(median_bpm, lo, hi)
    return folded / median_bpm


# =============================================================================
# Global LED-multiplier anchor
# =============================================================================

ANCHOR_PERCENTILE = (0.5, 99.5)    # robust BPM extremes used for anchor check
ANCHOR_FIT_SLACK = 0.5             # scaled observed range must fit inside the
                                   # user range with this fractional slack on
                                   # each side (i.e. ≥ min_bpm*(1-slack) and
                                   # ≤ max_bpm*(1+slack)); slack absorbs rare
                                   # tempo outliers AND under-specified ranges

def _detect_led_multiplier(intervals: np.ndarray,
                           min_bpm: float, max_bpm: float) -> float | None:
    """Find the power-of-2 *interval* multiplier that places the observed BPM
    range inside the user-provided ``[min_bpm, max_bpm]`` (with slack).

    Hypothesis: the LED flashes at a fixed multiple of the beat throughout the
    song. Under this assumption, dividing every observed BPM by some 2^k yields
    a range that lies INSIDE the user's range (modulo a slack for outliers).

    Why containment, not endpoint matching: a song can spend 99 % of its time
    at the fast tempo (JUSTITIA does), so even an extreme low percentile (p0.5)
    won't actually reach the rare slow beats. Trying to PIN observed extremes
    to user extremes therefore fails. Instead we check that the observed range
    FITS WITHIN the user range — which is the actual physical requirement.

    Among accepting factors, return the one whose scaled extremes best match
    the user extremes (smallest log-ratio error). Returns ``None`` when no
    factor fits — caller falls back to per-segment octave fold.
    """
    if not (min_bpm > 0 and max_bpm > 0):
        return None
    valid = intervals[intervals > 0]
    if len(valid) < 16:
        return None
    bpms = 60_000.0 / valid
    obs_lo = float(np.percentile(bpms, ANCHOR_PERCENTILE[0]))
    obs_hi = float(np.percentile(bpms, ANCHOR_PERCENTILE[1]))
    if obs_lo <= 0 or obs_hi <= 0:
        return None

    lo_bound = min_bpm * (1.0 - ANCHOR_FIT_SLACK)
    hi_bound = max_bpm * (1.0 + ANCHOR_FIT_SLACK)
    best_f = None
    best_err = float('inf')
    for exp in range(-3, 4):
        f = 2.0 ** exp
        scaled_lo = obs_lo / f
        scaled_hi = obs_hi / f
        if scaled_lo < lo_bound or scaled_hi > hi_bound:
            continue                # this f doesn't contain the observed range
        # Among containing factors, prefer the one whose scaled extremes are
        # closest to the user extremes (log-domain error so 2× and 0.5× cost
        # equally).
        err = (abs(np.log2(scaled_lo / min_bpm))
               + abs(np.log2(scaled_hi / max_bpm)))
        if err < best_err:
            best_err = err
            best_f = f
    return best_f


# =============================================================================
# Change-point detection
# =============================================================================

def _rolling_median(x: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling median, edge-padded to keep length. Length-stable.

    Edge-PRESERVING, unlike a rolling mean: a sharp BPM step (a half-tempo
    section's 190→95 drop, or GEHENNA's per-measure staircase) survives as a
    sharp edge instead of being smeared across ``window`` beats into a fake
    ramp, so the change-point detector fires AT the step. The ±1-frame
    per-beat interval jitter is still removed.
    """
    if window <= 1 or len(x) <= 1:
        return x.astype(float, copy=True)
    w = min(window, len(x))
    pad_l, pad_r = w // 2, w - w // 2 - 1
    xp = np.pad(x.astype(float), (pad_l, pad_r), mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(xp, w)
    return np.median(win, axis=1)


def _change_points(intervals: np.ndarray,
                   *, threshold: float = DEFAULT_CHANGE_RATE,
                   smooth_window: int = DEFAULT_SMOOTH_WINDOW,
                   min_gap: int = MIN_BEATS_PER_SEGMENT,
                   ) -> list[int]:
    """Indices into ``intervals`` at which a new segment begins.

    Differences a SMOOTHED interval series — per-beat jitter at 60 fps is
    bigger than the change threshold, so differencing the raw series would
    spawn a spurious change-point at every beat. Also enforces ``min_gap``
    between change-points to keep segments fittable.
    """
    cps = [0]
    if len(intervals) >= 2:
        smoothed = _rolling_median(intervals, smooth_window)
        last_cp = 0
        for i in range(1, len(smoothed)):
            prev, cur = smoothed[i - 1], smoothed[i]
            if prev <= 0:
                continue
            if abs(cur - prev) / prev > threshold and i - last_cp >= min_gap:
                cps.append(i)
                last_cp = i
    cps.append(len(intervals))
    return cps


def _merge_short(cps: list[int], min_len: int) -> list[int]:
    """Drop change-points that produce sub-``min_len`` segments."""
    if len(cps) <= 2:
        return cps
    out = [cps[0]]
    for c in cps[1:-1]:
        if c - out[-1] >= min_len:
            out.append(c)
    out.append(cps[-1])
    return out


# =============================================================================
# Per-segment fit (with octave fold + clamp)
# =============================================================================

def _fit_segment(times: np.ndarray, intervals: np.ndarray,
                 i0: int, i1: int,
                 min_bpm: float, max_bpm: float,
                 *, global_anchor: bool = False) -> BPMDraft:
    """Fit BPM(t) linearly over intervals[i0:i1] → BPMDraft.

    Steps inside the fit:
    1. (anchor off) Octave-fold the segment's median BPM into [min_bpm,
       max_bpm]. Skipped when ``global_anchor=True``: a global LED-multiplier
       anchor was established upstream and already scaled the intervals.
    2. Linear fit BPM vs midpoint-time.
    3. Endpoint clamp — to ``[min_bpm, max_bpm]`` when anchored (the user's
       range is trusted as hard bounds, letting short slow sections reach the
       true min), otherwise to the segment's observed 5..95 percentile.
    4. Collapse to constant if |Δbpm| under SLOPE_ZERO_BPM.
    """
    seg_ints = intervals[i0:i1].copy()
    # midpoint time of each interval (used for the linear fit)
    mids = 0.5 * (times[i0:i1] + times[i0 + 1:i1 + 1])
    start_ms = float(times[i0])
    end_ms = float(times[i1])

    # (1) per-segment octave fold (skip when global anchor is in effect) ---
    if not global_anchor:
        raw_bpms = 60_000.0 / np.where(seg_ints > 0, seg_ints, np.nan)
        raw_bpms = raw_bpms[np.isfinite(raw_bpms)]
        if len(raw_bpms) and min_bpm > 0 and max_bpm > 0:
            factor = _octave_factor(float(np.median(raw_bpms)), min_bpm, max_bpm)
            if factor != 1.0:
                seg_ints = seg_ints / factor

    bpms = 60_000.0 / np.where(seg_ints > 0, seg_ints, np.nan)
    valid = np.isfinite(bpms)
    mids, bpms = mids[valid], bpms[valid]

    if len(bpms) < 2:
        b = float(np.mean(bpms)) if len(bpms) else 0.0
        return BPMDraft(start_ms, end_ms, b, b)

    # (2) linear fit -------------------------------------------------------
    slope, intercept = np.polyfit(mids, bpms, 1)
    b0 = float(slope * start_ms + intercept)
    b1 = float(slope * end_ms + intercept)

    # (3) endpoint clamp ---------------------------------------------------
    if global_anchor and min_bpm > 0 and max_bpm > 0:
        # Trust user range as hard bounds — lets a short slow sub-section pull
        # the endpoint down to min_bpm instead of being percentile-masked.
        clamp_lo, clamp_hi = min_bpm, max_bpm
    else:
        clamp_lo, clamp_hi = (float(v) for v in
                              np.percentile(bpms, ENDPOINT_PERCENTILE))
    b0 = float(np.clip(b0, clamp_lo, clamp_hi))
    b1 = float(np.clip(b1, clamp_lo, clamp_hi))

    # (4) near-constant collapse ------------------------------------------
    # Collapse to a single BPM when the fitted slope is within SLOPE_ZERO_BPM
    # (absolute) OR RELATIVE_SLOPE_TOL of the level (relative). The collapsed
    # value is itself clamped — without this, a noisy fast segment whose mean
    # sits above max_bpm leaked straight through here, ignoring the step-(3)
    # clamp (the bug that gave GEHENNA a 225 > max_bpm=222.22 segment).
    if abs(b1 - b0) < max(SLOPE_ZERO_BPM,
                          RELATIVE_SLOPE_TOL * max(abs(b0), abs(b1))):
        m = float(np.clip(np.median(bpms), clamp_lo, clamp_hi))
        return BPMDraft(start_ms, end_ms, m, m)
    return BPMDraft(start_ms, end_ms, b0, b1)


def _is_bimodal_ramp(draft: BPMDraft) -> bool:
    """A fit that "ramps" by ≥``BIMODAL_RATIO`` betrays an octave change."""
    a, b = sorted((draft.bpm_start, draft.bpm_end))
    return a > 0 and (b / a) >= BIMODAL_RATIO


def _segment_data_is_bimodal(intervals: np.ndarray, i0: int, i1: int,
                             *, min_len: int) -> bool:
    """True if per-beat BPMs in [i0, i1] span ≥ ``BIMODAL_RATIO``.

    Linear-fit endpoints can hide bimodality: a 30-beat segment with 25 beats
    at 290 BPM and 5 at 80 BPM regresses to a near-flat 150-BPM line. The
    per-beat BPM distribution (5..95 percentile) does not — it shows the real
    80↔290 spread and tells the splitter to recurse.

    ``min_len`` matches the splitter's minimum half-length; a segment too short
    to split usefully is reported as not-bimodal regardless of spread.
    """
    seg_ints = intervals[i0:i1]
    bpms = 60_000.0 / np.where(seg_ints > 0, seg_ints, np.nan)
    bpms = bpms[np.isfinite(bpms)]
    if len(bpms) < 2 * min_len:
        return False
    lo = float(np.percentile(bpms, 5))
    hi = float(np.percentile(bpms, 95))
    return lo > 0 and (hi / lo) >= BIMODAL_RATIO


def _best_split_point(bpms: np.ndarray, *, min_len: int) -> int | None:
    """Index that minimises within-half variance. None if no valid split."""
    n = len(bpms)
    if n < 2 * min_len:
        return None
    # cumulative moments → constant-time per-split variance computation
    csum = np.concatenate([[0.0], np.cumsum(bpms)])
    csqsum = np.concatenate([[0.0], np.cumsum(bpms * bpms)])

    def half_var(lo: int, hi: int) -> float:
        m = hi - lo
        if m <= 0:
            return 0.0
        s = csum[hi] - csum[lo]
        sq = csqsum[hi] - csqsum[lo]
        return float(sq / m - (s / m) ** 2)

    best_i = None
    best_cost = np.inf
    for i in range(min_len, n - min_len + 1):
        cost = half_var(0, i) * i + half_var(i, n) * (n - i)
        if cost < best_cost:
            best_cost = cost
            best_i = i
    return best_i


def _fit_segment_recursive(times: np.ndarray, intervals: np.ndarray,
                           i0: int, i1: int,
                           min_bpm: float, max_bpm: float,
                           *, global_anchor: bool = False,
                           depth: int = 0,
                           max_depth: int = 2) -> list[BPMDraft]:
    """``_fit_segment`` + split if the fit comes out bimodal.

    If the linear fit ramps by ≥``BIMODAL_RATIO`` (e.g. 125→225 BPM) the
    segment likely straddles an LED octave change (anchor off) OR a real
    sharp tempo transition (anchor on) — both better represented by splitting.
    ``max_depth=2`` keeps the worst case bounded.
    """
    draft = _fit_segment(times, intervals, i0, i1,
                        min_bpm, max_bpm, global_anchor=global_anchor)
    if depth >= max_depth:
        return [draft]
    # With a trusted user range (anchor), the hard endpoint clamp bounds the
    # over-split risk, so we may split smaller. Without anchor, keep the
    # original conservative min length.
    split_min_len = ANCHOR_MIN_SPLIT_LEN if global_anchor else MIN_BEATS_PER_SEGMENT
    # Split if EITHER the fit ramp OR the underlying data is bimodal — the fit
    # alone misses cases where a brief slow run is hidden in a fast segment.
    if not (_is_bimodal_ramp(draft)
            or _segment_data_is_bimodal(intervals, i0, i1,
                                        min_len=split_min_len)):
        return [draft]

    # Locate the split. With a global anchor the intervals are already scaled,
    # so we use raw per-beat BPMs as-is. Without anchor, fold per-segment first
    # so an octave jump shows as a real bimodal hop, not a 2× factor.
    seg_ints = intervals[i0:i1]
    raw_bpms = 60_000.0 / np.where(seg_ints > 0, seg_ints, np.nan)
    raw_bpms = raw_bpms[np.isfinite(raw_bpms)]
    if len(raw_bpms) < 2 * split_min_len:
        return [draft]
    if not global_anchor and min_bpm > 0 and max_bpm > 0:
        factor = _octave_factor(float(np.median(raw_bpms)), min_bpm, max_bpm)
        raw_bpms = raw_bpms * factor

    split_off = _best_split_point(raw_bpms, min_len=split_min_len)
    if split_off is None:
        return [draft]
    split = i0 + split_off
    return (_fit_segment_recursive(times, intervals, i0, split,
                                   min_bpm, max_bpm,
                                   global_anchor=global_anchor,
                                   depth=depth + 1, max_depth=max_depth)
            + _fit_segment_recursive(times, intervals, split, i1,
                                     min_bpm, max_bpm,
                                     global_anchor=global_anchor,
                                     depth=depth + 1, max_depth=max_depth))


# =============================================================================
# Active-window clipping
# =============================================================================

def _clip_to_active(times: np.ndarray,
                    window_ms: tuple[float, float] | None) -> np.ndarray:
    """Restrict beats to ``[window_lo − slack, window_hi + slack]``.

    The intro fade-in often emits LED flashes that aren't on a real musical
    beat; without this clip those polluted intervals leak into the median
    used by the octave fold and spawn a spurious sub-min_bpm first segment.
    """
    if window_ms is None:
        return times
    lo, hi = window_ms
    mask = (times >= lo - ACTIVE_WINDOW_SLACK_MS) & \
           (times <= hi + ACTIVE_WINDOW_SLACK_MS)
    return times[mask]


# =============================================================================
# Public API
# =============================================================================

def estimate_bpm_drafts(beats: list[BeatEvent], *,
                        min_bpm: float = 0.0,
                        max_bpm: float = 0.0,
                        active_window_ms: tuple[float, float] | None = None,
                        change_rate: float = DEFAULT_CHANGE_RATE,
                        smooth_window: int = DEFAULT_SMOOTH_WINDOW,
                        ) -> list[BPMDraft]:
    """Beats → list[BPMDraft] in ms-domain (TickClock not yet built)."""
    if len(beats) < 2:
        return []

    times = np.array([b.ms for b in beats], dtype=float)
    times = _clip_to_active(times, active_window_ms)
    if len(times) < 2:
        return []
    intervals = np.diff(times)

    # --- auto-derive a BPM range when the song config left it unset (0) ----
    # The octave-fold / anchor / endpoint-clamp machinery all need SOME range.
    # We pick one wide enough to span an octave each side of the robust median
    # BPM, so a genuine half- or double-tempo section stays IN range and is
    # NOT folded away (folding a real 95-BPM section up to 190 would erase it).
    # The range merely keeps the machinery from no-op'ing pathologically.
    if min_bpm <= 0 or max_bpm <= 0:
        valid = intervals[intervals > 0]
        if len(valid):
            med_bpm = 60_000.0 / float(np.median(valid))
            if min_bpm <= 0:
                min_bpm = med_bpm * 0.45   # just below half → 0.5× stays in
            if max_bpm <= 0:
                max_bpm = med_bpm * 2.1    # just above double → 2× stays in

    # --- global LED-multiplier anchor -------------------------------------
    # If the song's observed BPM median sits at a clean 2^k multiple of the
    # user's midpoint, the LED multiplier is consistent throughout. That
    # justifies using the user's range as hard endpoint bounds during fit
    # (instead of per-segment 5/95 percentile), which lets short slow sections
    # reach the true minimum.
    anchor_factor = _detect_led_multiplier(intervals, min_bpm, max_bpm)
    global_anchor = anchor_factor is not None
    if global_anchor and anchor_factor != 1.0:
        intervals = intervals * anchor_factor

    # --- segment the beat stream ------------------------------------------
    cps = _change_points(intervals, threshold=change_rate,
                         smooth_window=smooth_window,
                         min_gap=MIN_BEATS_PER_SEGMENT)
    cps = _merge_short(cps, MIN_BEATS_PER_SEGMENT)

    drafts: list[BPMDraft] = []
    for k in range(len(cps) - 1):
        i0, i1 = cps[k], cps[k + 1]
        if i1 - i0 < 1:
            continue
        drafts.extend(_fit_segment_recursive(
            times, intervals, i0, i1, min_bpm, max_bpm,
            global_anchor=global_anchor))

    # --- merge contiguous constant segments with same BPM ----------------
    merged: list[BPMDraft] = []
    for d in drafts:
        if (merged and merged[-1].is_constant and d.is_constant
                and abs(merged[-1].bpm_end - d.bpm_start) < SLOPE_ZERO_BPM):
            prev = merged[-1]
            merged[-1] = BPMDraft(prev.start_ms, d.end_ms,
                                  prev.bpm_start, d.bpm_end)
        else:
            merged.append(d)

    return merged


def build_tick_clock(beats: list[BeatEvent], *,
                     measure_zero_ms: float,
                     min_bpm: float = 0.0,
                     max_bpm: float = 0.0,
                     active_window_ms: tuple[float, float] | None = None,
                     tick_resolution: int = 192,
                     change_rate: float = DEFAULT_CHANGE_RATE,
                     smooth_window: int = DEFAULT_SMOOTH_WINDOW,
                     ) -> TickClock:
    """Beats → TickClock anchored at ``measure_zero_ms``."""
    drafts = estimate_bpm_drafts(beats, min_bpm=min_bpm, max_bpm=max_bpm,
                                 active_window_ms=active_window_ms,
                                 change_rate=change_rate,
                                 smooth_window=smooth_window)
    if not drafts:
        raise ValueError("not enough beats to build a TickClock")
    return TickClock.from_drafts(drafts, origin_ms=measure_zero_ms,
                                 tick_resolution=tick_resolution)


# =============================================================================
# CLI: python bpm_estimator.py [config/song.toml]
# =============================================================================

if __name__ == "__main__":
    from layer3 import Layer3Pipeline

    cfg = sys.argv[1] if len(sys.argv) > 1 else "config/song.toml"
    l3 = Layer3Pipeline.from_config(cfg).run(progress=False)
    window = ((l3.barlines[0].ms, l3.barlines[-1].ms)
              if l3.barlines else None)
    drafts = estimate_bpm_drafts(l3.beats,
                                 min_bpm=l3.cal.min_bpm,
                                 max_bpm=l3.cal.max_bpm,
                                 active_window_ms=window)
    print(f"=== BPM estimate: {len(drafts)} segment(s) "
          f"from {len(l3.beats)} beats "
          f"(active window: {window}) ===")
    for d in drafts:
        kind = "const" if d.is_constant else "ramp "
        print(f"  {kind}  {d.start_ms:9.1f}..{d.end_ms:9.1f}ms  "
              f"bpm {d.bpm_start:7.2f} → {d.bpm_end:7.2f}")
