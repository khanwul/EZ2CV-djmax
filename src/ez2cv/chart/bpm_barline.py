"""
EZ2CV — chart conversion / bpm_estimator (BARLINE-DOMAIN variant)
===============================================================================
Derives the BPM curve from the **measure grid** (reconstructed barlines) instead
of the POW-LED beat stream. A barline is emitted once per musical measure, so a
measure's duration gives its tempo DIRECTLY:

    bpm[m] = beats_per_measure * 60000 / (barline[m+1] − barline[m])

Why this beats the beat-stream estimator (``bpm_estimator``)
-----------------------------------------------------------------
1. **No LED-multiplier ambiguity.** The beat estimator must guess whether the
   POW LED flashes on beats, half-beats or double-beats (the octave fold). The
   barline cadence is the musical measure itself — the derived value is the TRUE
   musical BPM, no power-of-two guess.
2. **Per-measure resolution resolves staircases.** The beat estimator's
   change-point detector enforces ``MIN_BEATS_PER_SEGMENT`` (≥16 beats) and so
   cannot represent a tempo that steps every measure (4 beats) — e.g. GEHENNA's
   111→122→…→222 staircase, which it smears into a single ramp. One BPM sample
   per measure sees every step.
3. **SV-immune** for the same reason the measure-line detector is.

Robustness
----------
* The barlines passed in are the *reconstructed* (gap-free) grid, so a missed
  on-screen barline — which would otherwise double a measure's duration and
  halve its BPM — is already inferred back in.
* Any residual ½/2× barline-detection error is absorbed by octave-folding each
  per-measure BPM into ``[min_bpm, max_bpm]`` and clamping to those bounds.
* Each constant segment's level is the drift-free INTERVAL-domain span tempo
  (total beats × 60000 / total span), not the convex, upward-biased mean of the
  per-measure BPMs — the same Jensen correction ``bpm_estimator`` applies.

Scope
-----
Needs a correct ``beats_per_measure`` per measure. For a 4/4 song (or correctly
detected variant runs) that is exact. When the time signature is WRONG (e.g.
JUSTITIA's 5/4·6/4 variants undetected → every measure assumed 3/4), the derived
BPM is garbage; ``build_tick_clock`` cross-checks against the beat estimator and
falls back to it when the two disagree.
"""

from __future__ import annotations

import numpy as np

from ez2cv.detection.beat import BeatEvent
from ez2cv.detection.barline import BarlineEvent
from ez2cv.chart.clock import BPMDraft, TickClock
from ez2cv.chart import bpm as beat_est
from ez2cv.chart.bpm import _fold_to_range, SLOPE_ZERO_BPM


# =============================================================================
# Tunables
# =============================================================================

SEGMENT_REL_TOL = 0.013       # relative side of the segment-break threshold: a
                              # measure stays in the current run while its
                              # ms-per-beat is within this fraction of the run's
                              # mean. Tight enough to resolve JUSTITIA's near-
                              # per-measure tempo steps; the lone-spike guard and
                              # the absolute floor keep a CONSTANT song from
                              # over-splitting on ±1-frame jitter.
SEGMENT_JITTER_FRAMES = 0.5   # absolute side of the segment-break threshold, in
                              # frames of measure-duration jitter. A barline ms
                              # is frame-quantised, so a measure's ms-per-beat
                              # jitters by ~frame/beats regardless of tempo;
                              # adding this absolute floor to the relative tol
                              # stops a SLOW song (long measures → tiny relative
                              # jitter) from over-splitting while still letting a
                              # FAST song (short measures) resolve real per-
                              # measure tempo steps.
MIN_BARLINES = 4              # need at least this many to derive a curve
FALLBACK_JUMP_FRAC = 0.25     # if more than this fraction of consecutive
                              # per-measure BPMs jump by ≥ FALLBACK_JUMP_RATIO,
                              # the assumed meters are probably wrong (a 6/4 read
                              # as 3/4 halves the BPM → pervasive ±2× zig-zag) →
                              # the barline curve is meaningless, use beats. A
                              # genuinely variable-tempo song (GEHENNA 111↔222)
                              # has only a FEW such jumps, well under this.
FALLBACK_JUMP_RATIO = 1.5


# =============================================================================
# Per-measure BPM → constant-run segments
# =============================================================================

def _per_measure_bpm(bar_ms: np.ndarray, beats_per_measure: np.ndarray,
                     min_bpm: float, max_bpm: float) -> np.ndarray:
    """Folded per-measure BPM. Length = len(bar_ms) − 1."""
    dur = np.diff(bar_ms)
    raw = np.where(dur > 0, beats_per_measure * 60_000.0 / dur, 0.0)
    if min_bpm > 0 and max_bpm > 0:
        return np.array([_fold_to_range(b, min_bpm, max_bpm) for b in raw])
    return raw


def _meter_zigzag_frac(barlines: list[BarlineEvent],
                       beats_per_measure: int | np.ndarray) -> float:
    """Fraction of consecutive per-measure BPMs that jump by ≥ the ratio.

    A wrong meter (e.g. a 6/4 measure scored as 3/4) halves that measure's
    derived BPM, so wrong meters show up as a pervasive ±2× zig-zag. A genuinely
    variable tempo steps smoothly and trips only a few. See FALLBACK_JUMP_FRAC.
    """
    bar_ms = np.array([b.ms for b in barlines], dtype=float)
    n_meas = len(bar_ms) - 1
    if n_meas < 2:
        return 0.0
    if np.isscalar(beats_per_measure):
        bpb = np.full(n_meas, float(beats_per_measure))
    else:
        bpb = np.asarray(beats_per_measure, dtype=float)[:n_meas]
        if len(bpb) < n_meas:
            bpb = np.concatenate([bpb, np.full(n_meas - len(bpb), bpb[-1])])
    dur = np.diff(bar_ms)
    pm = np.where(dur > 0, bpb * 60_000.0 / dur, 0.0)
    pm = pm[pm > 0]
    if len(pm) < 2:
        return 0.0
    r = pm[1:] / pm[:-1]
    return float(np.mean((r >= FALLBACK_JUMP_RATIO) | (r <= 1.0 / FALLBACK_JUMP_RATIO)))


def _segment_runs(period: np.ndarray, abs_tol: np.ndarray, *,
                  rel_tol: float = SEGMENT_REL_TOL) -> list[tuple[int, int]]:
    """Run-length split of a per-measure ms-per-beat series into constant runs.

    Works in the PERIOD domain (ms per beat) so the jitter floor is a plain ms
    quantity. A new run starts when a measure's period leaves the current run's
    running mean by more than ``rel_tol·mean`` (relative) OR ``abs_tol[i]``
    (absolute, the per-measure frame-jitter floor) — whichever is larger.
    Returns half-open ``[start, end)`` measure-index spans.
    """
    if len(period) == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = 0
    run_sum = period[0]
    run_n = 1
    n = len(period)
    for i in range(1, n):
        mean = run_sum / run_n
        thresh = max(rel_tol * mean, abs_tol[i])
        if mean > 0 and abs(period[i] - mean) > thresh:
            # Lone-spike guard: a single-measure JITTER spike (small deviation,
            # next measure returns to the run) is absorbed so a constant region
            # is not chopped by one noisy measure. Gated to SMALL deviations
            # (≤ 2× the break threshold) so a real isolated 1-measure tempo
            # change — which is large — still breaks into its own segment.
            if (i + 1 < n and abs(period[i] - mean) <= 2.0 * thresh
                    and abs(period[i + 1] - mean) <= max(rel_tol * mean,
                                                         abs_tol[i + 1])):
                run_sum += period[i]
                run_n += 1
                continue
            runs.append((start, i))
            start = i
            run_sum = period[i]
            run_n = 1
        else:
            run_sum += period[i]
            run_n += 1
    runs.append((start, len(period)))
    return runs


def estimate_bpm_drafts_from_barlines(
        barlines: list[BarlineEvent],
        beats_per_measure: int | np.ndarray,
        *, min_bpm: float = 0.0, max_bpm: float = 0.0,
        rel_tol: float | None = None,
        ) -> list[BPMDraft]:
    """Reconstructed barlines → constant ``BPMDraft`` per tempo step.

    ``beats_per_measure`` is either a scalar (uniform time signature) or an
    array aligned with the measures (one entry per ``barline[m]→barline[m+1]``
    gap) so detected variant runs use their own beat count.
    """
    if rel_tol is None:
        rel_tol = SEGMENT_REL_TOL
    bar_ms = np.array([b.ms for b in barlines], dtype=float)
    if len(bar_ms) < MIN_BARLINES:
        return []
    n_meas = len(bar_ms) - 1

    if np.isscalar(beats_per_measure):
        bpb = np.full(n_meas, float(beats_per_measure))
    else:
        bpb = np.asarray(beats_per_measure, dtype=float)[:n_meas]
        if len(bpb) < n_meas:                      # pad with the last value
            bpb = np.concatenate([bpb, np.full(n_meas - len(bpb), bpb[-1])])

    folded = _per_measure_bpm(bar_ms, bpb, min_bpm, max_bpm)

    # Effective per-measure duration AFTER folding (a folded ½/2× measure is
    # treated as if its barline had been detected correctly). Used so a segment
    # level is the drift-free span tempo, not a biased per-measure mean.
    eff_dur = np.where(folded > 0, bpb * 60_000.0 / folded, np.diff(bar_ms))

    # Segment in the PERIOD (ms-per-beat) domain with a relative + frame-jitter
    # absolute threshold. ms-per-frame is recovered from the barlines' frame
    # stamps; the absolute floor is SEGMENT_JITTER_FRAMES of jitter spread over
    # a measure's beats.
    period = np.where(folded > 0, 60_000.0 / folded, eff_dur / np.maximum(bpb, 1))
    cfs = np.array([b.cross_frame for b in barlines], dtype=float)
    dcf = float(cfs[-1] - cfs[0])
    frame_ms = ((bar_ms[-1] - bar_ms[0]) / dcf) if dcf > 0 else 0.0
    abs_tol = SEGMENT_JITTER_FRAMES * frame_ms / np.maximum(bpb, 1)

    drafts: list[BPMDraft] = []
    for a, b in _segment_runs(period, abs_tol, rel_tol=rel_tol):
        total_beats = float(np.sum(bpb[a:b]))
        total_span = float(np.sum(eff_dur[a:b]))
        bpm = (total_beats * 60_000.0 / total_span) if total_span > 0 else 0.0
        if min_bpm > 0 and max_bpm > 0:
            bpm = float(np.clip(bpm, min_bpm, max_bpm))
        drafts.append(BPMDraft(float(bar_ms[a]), float(bar_ms[b]), bpm, bpm))

    # merge adjacent runs that ended up at the same level (rel_tol can split a
    # noisy plateau then both halves round to the same span tempo)
    merged: list[BPMDraft] = []
    for d in drafts:
        if (merged and abs(merged[-1].bpm_end - d.bpm_start) < SLOPE_ZERO_BPM):
            prev = merged[-1]
            merged[-1] = BPMDraft(prev.start_ms, d.end_ms,
                                  prev.bpm_start, d.bpm_end)
        else:
            merged.append(d)
    return merged


# =============================================================================
# Public API — TickClock with beat-estimator fallback
# =============================================================================

def build_tick_clock(beats: list[BeatEvent],
                     barlines: list[BarlineEvent],
                     beats_per_measure: int | np.ndarray,
                     *, measure_zero_ms: float,
                     min_bpm: float = 0.0, max_bpm: float = 0.0,
                     active_window_ms: tuple[float, float] | None = None,
                     tick_resolution: int = 192,
                     ) -> TickClock:
    """Barline-derived TickClock, falling back to the beat estimator.

    Fallback fires when the barline curve is unavailable (too few barlines) or
    the per-measure BPMs zig-zag by ±2× across more than ``FALLBACK_JUMP_FRAC``
    of the song — the signature that the assumed ``beats_per_measure`` is wrong
    (undetected variants), which makes the barline curve meaningless. A genuine
    variable tempo (smooth steps) passes through and is kept.
    """
    drafts = estimate_bpm_drafts_from_barlines(
        barlines, beats_per_measure, min_bpm=min_bpm, max_bpm=max_bpm)

    if drafts and _meter_zigzag_frac(barlines, beats_per_measure) > FALLBACK_JUMP_FRAC:
        drafts = []                             # meters unreliable → use beats
    if not drafts:
        drafts = beat_est.estimate_bpm_drafts(
            beats, min_bpm=min_bpm, max_bpm=max_bpm,
            active_window_ms=active_window_ms)

    if not drafts:
        raise ValueError("not enough barlines or beats to build a TickClock")

    return TickClock.from_drafts(drafts, origin_ms=measure_zero_ms,
                                 tick_resolution=tick_resolution)
