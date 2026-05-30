"""
EZ2CV — Layer 4 / bpm_estimator (INTERVAL-DOMAIN variant)
===============================================================================
Drop-in replacement for ``bpm_estimator`` that removes the frame-quantisation
tempo bias. It reuses every helper from the original module and only overrides
the segment fit + its call chain (``_fit_segment`` → ``_fit_segment_recursive``
→ ``estimate_bpm_drafts`` → ``build_tick_clock``).

Why
---
POW-LED beat onsets are localised to whole frames, so an inter-beat interval is
an INTEGER number of frames. The original ``_fit_segment`` derives a constant
segment's BPM from ``mean(60000 / interval)`` — the mean of the per-beat BPMs.
Because ``BPM = 60000 / interval`` is CONVEX in the interval, Jensen's
inequality makes that mean biased HIGH:

    mean(60000 / iv)  >  60000 / mean(iv)

The bias is small but it accumulates as a forward tick-drift over the song
(e.g. Dream Walker fit 215.20 vs the true 215.0 → +64 ticks by the end). The
unbiased estimator is the INTERVAL-domain mean — identical to the drift-free
"span tempo" (n−1)·60000 / (t_last − t_first):

    bpm_const = 60000 / mean(interval)

This variant collapses every constant segment to that value. Verified against
the EZ2PATTERN ground truth:

    #BEYOND        128.0206  → 128.0146   (truth 128.0)
    Dream Walker   215.1989  → 215.0171   (truth 215.0)

Ramp segments are left to the original linear fit (the convexity bias there is
a second-order level offset); only the constant collapse and the degenerate
short-segment fallback are corrected.

Usage: swap the import in the orchestrator —
    from layer4.bpm_estimator_fixed import build_tick_clock, estimate_bpm_drafts
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from layer3.beat import BeatEvent
from layer4.tick_clock import BPMDraft, TickClock

# Reuse all leaf helpers + tunables from the original module unchanged.
from layer4.bpm_estimator import (
    SLOPE_ZERO_BPM, MIN_BEATS_PER_SEGMENT, ANCHOR_MIN_SPLIT_LEN,
    ENDPOINT_PERCENTILE, DEFAULT_CHANGE_RATE, DEFAULT_SMOOTH_WINDOW,
    _octave_factor, _detect_led_multiplier, _change_points, _merge_short,
    _clip_to_active, _is_bimodal_ramp, _segment_data_is_bimodal,
    _best_split_point,
)


# =============================================================================
# Per-segment fit — interval-domain constant level
# =============================================================================

def _fit_segment(times: np.ndarray, intervals: np.ndarray,
                 i0: int, i1: int,
                 min_bpm: float, max_bpm: float,
                 *, global_anchor: bool = False) -> BPMDraft:
    """``bpm_estimator._fit_segment`` with an UNBIASED constant level.

    Identical to the original except a constant segment's BPM is taken from the
    interval-domain mean (``60000 / mean(interval)``) instead of the convex,
    upward-biased ``mean(60000 / interval)``. See the module docstring.
    """
    seg_ints = intervals[i0:i1].copy()
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

    # Unbiased constant tempo: invert the MEAN INTERVAL (period domain).
    valid_ints = seg_ints[seg_ints > 0]
    bpm_const = (60_000.0 / float(np.mean(valid_ints))) if len(valid_ints) else 0.0

    bpms = 60_000.0 / np.where(seg_ints > 0, seg_ints, np.nan)
    valid = np.isfinite(bpms)
    mids, bpms = mids[valid], bpms[valid]

    if len(bpms) < 2:
        # too few beats to detect a ramp → constant at the unbiased level
        return BPMDraft(start_ms, end_ms, bpm_const, bpm_const)

    # (2) linear fit — used ONLY to detect a genuine ramp ------------------
    slope, intercept = np.polyfit(mids, bpms, 1)
    b0 = float(slope * start_ms + intercept)
    b1 = float(slope * end_ms + intercept)

    # (3) endpoint clamp ---------------------------------------------------
    if global_anchor and min_bpm > 0 and max_bpm > 0:
        b0 = float(np.clip(b0, min_bpm, max_bpm))
        b1 = float(np.clip(b1, min_bpm, max_bpm))
    else:
        p_lo, p_hi = np.percentile(bpms, ENDPOINT_PERCENTILE)
        b0 = float(np.clip(b0, p_lo, p_hi))
        b1 = float(np.clip(b1, p_lo, p_hi))

    # (4) slope-zero collapse → UNBIASED constant level --------------------
    if abs(b1 - b0) < SLOPE_ZERO_BPM:
        return BPMDraft(start_ms, end_ms, bpm_const, bpm_const)
    return BPMDraft(start_ms, end_ms, b0, b1)


# =============================================================================
# Recursive split (verbatim — resolves _fit_segment to the version above)
# =============================================================================

def _fit_segment_recursive(times: np.ndarray, intervals: np.ndarray,
                           i0: int, i1: int,
                           min_bpm: float, max_bpm: float,
                           *, global_anchor: bool = False,
                           depth: int = 0,
                           max_depth: int = 2) -> list[BPMDraft]:
    draft = _fit_segment(times, intervals, i0, i1,
                        min_bpm, max_bpm, global_anchor=global_anchor)
    if depth >= max_depth:
        return [draft]
    split_min_len = ANCHOR_MIN_SPLIT_LEN if global_anchor else MIN_BEATS_PER_SEGMENT
    if not (_is_bimodal_ramp(draft)
            or _segment_data_is_bimodal(intervals, i0, i1,
                                        min_len=split_min_len)):
        return [draft]

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
# Public API (verbatim — resolves the helpers above)
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

    anchor_factor = _detect_led_multiplier(intervals, min_bpm, max_bpm)
    global_anchor = anchor_factor is not None
    if global_anchor and anchor_factor != 1.0:
        intervals = intervals * anchor_factor

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
# CLI: python bpm_estimator_fixed.py [config/song.toml]
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
    print(f"=== BPM estimate (interval-domain): {len(drafts)} segment(s) "
          f"from {len(l3.beats)} beats (active window: {window}) ===")
    for d in drafts:
        kind = "const" if d.is_constant else "ramp "
        print(f"  {kind}  {d.start_ms:9.1f}..{d.end_ms:9.1f}ms  "
              f"bpm {d.bpm_start:7.4f} → {d.bpm_end:7.4f}")
