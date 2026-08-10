"""Convert a reloadable millisecond-domain ``RawChart`` into tick data.

    1. normalize beat observations without inventing missing events
    2. infer arbitrary per-measure meters on the beat grid
    3. select denoised tempo anchors, including clear mid-measure steps
    4. ms → tick conversion for every note (head + tail)
    5. snap-to-grid (quantizer); longnote tails snap as RELATIVE lengths
    6. ``Chart`` ready for serialization

What chart conversion deliberately does not do
-----------------------------------------------
* No JSON I/O.
* No video re-decoding; it reads only ``RawChart``.
* No second-pass BPM refinement using already-snapped notes. Raw note times may
  choose between clocks already supported by beat/barline observations, but
  snapped chart notes never feed back into timing inference.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace

import numpy as np

from ez2cv.detection import RawChart, TrackMetadata
from ez2cv.chart.clock import BPMSegment, TickClock
from ez2cv.chart.meter import TimeSignature, TimeSigVariant
from ez2cv.chart.quantize import (MEASURE_GRID_LEVELS, SnapResult,
                                  snap_by_measure, snap_length,
                                  TRIPLET_DENOMS)
from ez2cv.chart.timeline import infer_timeline


TIMING_SIGMA_MULTIPLIER = 2.0


# =============================================================================
# ChartNote
# =============================================================================

@dataclass
class ChartNote:
    """Minimal tick-based note. color/type/snap/confidence are derivable."""
    lane: int
    start_tick: int
    end_tick: int | None       # None for taps
    off_grid: bool = False

# =============================================================================
# Chart
# =============================================================================

@dataclass
class Chart:
    song_name: str
    difficulty: str
    game: str
    key_mode: str
    tracks: tuple[TrackMetadata, ...]
    tick_resolution: int
    bpm_segments: list[BPMSegment]
    global_time_sig: TimeSignature
    variant_measures: list[TimeSigVariant]
    measure_zero_ms: float
    notes: list[ChartNote]
    barlines_tick: list[int]
    stats: dict

    @property
    def lane_colors(self) -> tuple[str, ...]:
        return tuple(track.color for track in self.tracks)

    def summary(self) -> str:
        bpm_lo = min(s.bpm_start for s in self.bpm_segments)
        bpm_hi = max(s.bpm_end for s in self.bpm_segments)
        lines = [
            f"=== Chart — {self.difficulty}, {len(self.notes)} notes, "
            f"{len(self.bpm_segments)} bpm seg(s), "
            f"{len(self.variant_measures)} variant run(s) ===",
            f"  time signature   : {self.global_time_sig}",
            f"  bpm range        : {bpm_lo:.2f} .. {bpm_hi:.2f}",
            f"  measure_zero_ms  : {self.measure_zero_ms:.1f}",
            f"  measures         : {self.stats['structure']['measure_count']}",
            f"  timing outliers  : "
            f"{self.stats['rhythm']['timing_outlier_ratio']*100:.1f}%",
            f"  fine-grid ratio  : "
            f"{self.stats['rhythm']['fine_grid_ratio']*100:.1f}%",
            f"  note count       : {self.stats['counts']['total_notes']}",
        ]
        return "\n".join(lines)


# =============================================================================
# Pipeline
# =============================================================================

def build_chart(raw: RawChart) -> Chart:
    if not raw.barlines:
        raise RuntimeError("chart conversion requires at least one barline — "
                           "no measure_zero anchor available")
    if len(raw.beats) < 2:
        raise RuntimeError("chart conversion requires at least 2 beats")

    # --- 1. joint meter grid + beat-anchored clock --------------------
    timeline = infer_timeline(raw)
    barlines = timeline.barlines
    global_ts, variants = timeline.global_time_sig, timeline.variants
    clock = timeline.clock
    measure_zero_ms = clock.tick_to_ms(0)
    bl_ticks = timeline.barline_ticks
    # --- 3. notes : raw ticks, then snap ------------------------------
    notes, snaps, grid_levels = _convert_and_snap_notes(raw, clock, bl_ticks)

    # --- 4. stats -----------------------------------------------------
    stats = _build_stats(notes, snaps, grid_levels, clock, raw, bl_ticks,
                         variants)

    return Chart(
        song_name=raw.song_name,
        difficulty=raw.difficulty,
        game=raw.game,
        key_mode=raw.key_mode,
        tracks=raw.tracks,
        tick_resolution=raw.tick_resolution,
        bpm_segments=clock.segments,
        global_time_sig=global_ts,
        variant_measures=variants,
        measure_zero_ms=measure_zero_ms,
        notes=notes,
        barlines_tick=bl_ticks,
        stats=stats,
    )

# ------------------------------------------------------------------ #

def _convert_and_snap_notes(raw: RawChart, clock: TickClock,
                            barline_ticks: list[int]
                            ) -> tuple[list[ChartNote], list[SnapResult],
                                       list[int]]:
    """RawNote → ChartNote with head snap + relative tail snap."""
    R = clock.tick_resolution

    # raw float ticks for every head
    raw_heads = [clock.ms_to_tick(n.trigger_ms) for n in raw.notes]

    # One shared vocabulary per measure; tails remain length-snapped relatively.
    snaps, grid_levels = snap_by_measure(
        raw_heads, barline_ticks, tick_resolution=R)
    snaps = [_apply_timing_uncertainty(note, snap, clock)
             for note, snap in zip(raw.notes, snaps, strict=True)]

    chart_notes: list[ChartNote] = []
    for raw_note, raw_t, snap in zip(raw.notes, raw_heads, snaps):
        end_tick: int | None = None
        if raw_note.type == "longnote" and raw_note.end_ms is not None:
            raw_end = clock.ms_to_tick(raw_note.end_ms)
            # snap LENGTH (relative), then offset from snapped head
            raw_len = raw_end - raw_t
            snapped_len = snap_length(raw_len, tick_resolution=R)
            if snapped_len > 0:
                end_tick = snap.tick + snapped_len
            else:
                end_tick = None        # collapsed to tap

        chart_notes.append(ChartNote(
            lane=raw_note.lane,
            start_tick=snap.tick,
            end_tick=end_tick,
            off_grid=snap.off_grid,
        ))

    # sort by (tick, lane) for stable downstream order
    order = sorted(range(len(chart_notes)),
                   key=lambda i: (chart_notes[i].start_tick,
                                  chart_notes[i].lane))
    chart_notes = [chart_notes[i] for i in order]
    return chart_notes, snaps, grid_levels


def _apply_timing_uncertainty(raw_note, snap: SnapResult,
                              clock: TickClock) -> SnapResult:
    """Separate measurement-compatible misses from real timing outliers."""
    residual_ms = abs(
        raw_note.trigger_ms - clock.tick_to_ms(snap.tick))
    uncertain = (snap.off_grid and raw_note.timing_sigma_ms > 0
                 and residual_ms <= (TIMING_SIGMA_MULTIPLIER
                                     * raw_note.timing_sigma_ms))
    return replace(
        snap,
        label="timing-uncertain" if uncertain else snap.label,
        off_grid=False if uncertain else snap.off_grid,
        timing_uncertain=uncertain,
        timing_residual_ms=residual_ms,
    )


# =============================================================================
# Stats
# =============================================================================

def _nps_peak(notes: list[ChartNote], clock: TickClock,
              window_ms: float = 4000.0) -> float:
    """Peak notes/sec over a sliding ``window_ms`` window."""
    if not notes:
        return 0.0
    times_ms = sorted(clock.tick_to_ms(n.start_tick) for n in notes)
    peak = 0
    j = 0
    for i in range(len(times_ms)):
        while times_ms[i] - times_ms[j] > window_ms:
            j += 1
        peak = max(peak, i - j + 1)
    return peak * 1000.0 / window_ms


def _build_stats(notes: list[ChartNote],
                 snaps: list[SnapResult],
                 grid_levels: list[int],
                 clock: TickClock,
                 raw: RawChart,
                 bl_ticks: list[int],
                 variants: list[TimeSigVariant]) -> dict:
    key_count = raw.key_count
    per_lane = [0] * key_count
    for n in notes:
        if 0 <= n.lane < key_count:
            per_lane[n.lane] += 1
    taps = sum(1 for n in notes if n.end_tick is None)
    lns = len(notes) - taps

    chord_count = sum(count >= 2 for count in Counter(
        n.start_tick for n in notes).values())
    snap_dist = dict(Counter(s.label for s in snaps))
    off_grid = sum(1 for n in notes if n.off_grid)
    fine_grid = sum(1 for s in snaps if s.fine_grid)
    timing_uncertain = sum(1 for s in snaps if s.timing_uncertain)
    base_grid_outliers = off_grid + fine_grid + timing_uncertain
    triplet = sum(1 for s in snaps if not s.off_grid and s.denom in TRIPLET_DENOMS)
    total_notes = len(notes) or 1

    # tempo
    bpm_lo = min(s.bpm_start for s in clock.segments)
    bpm_hi = max(s.bpm_end for s in clock.segments)
    has_ramp = any(not s.is_constant for s in clock.segments)

    # quality (from detection debug info)
    extrap = sum(1 for n in raw.notes if n.extrapolated)
    low_conf = sum(1 for n in raw.notes if n.confidence < 0.5)
    timing_sigmas = [n.timing_sigma_ms for n in raw.notes]
    timing_residuals = [s.timing_residual_ms for s in snaps]
    raw_total = len(raw.notes) or 1

    # structure
    duration_ms = raw.duration_ms
    measure_count = max(0, len(bl_ticks) - 1)
    longnote_hold = sum(max(0, (n.end_tick - n.start_tick))
                        for n in notes if n.end_tick is not None)

    nps_mean = (len(notes) / (duration_ms / 1000.0)) if duration_ms > 0 else 0.0
    nps_peak = _nps_peak(notes, clock)

    return {
        "counts": {
            "tap": taps,
            "longnote": lns,
            "total_notes": len(notes),
            "per_lane": per_lane,
        },
        "density": {
            "nps_mean": round(nps_mean, 3),
            "nps_peak_4s": round(nps_peak, 3),
            "chord_count": chord_count,
        },
        "tempo": {
            "bpm_min": round(bpm_lo, 3),
            "bpm_max": round(bpm_hi, 3),
            "bpm_segment_count": len(clock.segments),
            "has_linear_ramp": has_ramp,
        },
        "structure": {
            "measure_count": measure_count,
            "duration_ms": round(duration_ms, 3),
            "longnote_hold_ticks": int(longnote_hold),
            "time_signature_variant_count": len(variants),
        },
        "rhythm": {
            "snap_distribution": snap_dist,
            # Compatibility alias: off-grid now specifically means a timing
            # outlier after the measure's adaptive vocabulary is selected.
            "off_grid_ratio": round(off_grid / total_notes, 4),
            "timing_outlier_ratio": round(off_grid / total_notes, 4),
            "timing_uncertain_ratio": round(
                timing_uncertain / total_notes, 4),
            "fine_grid_ratio": round(fine_grid / total_notes, 4),
            "base_grid_outlier_ratio": round(
                base_grid_outliers / total_notes, 4),
            "adaptive_measure_count": sum(level > 0 for level in grid_levels),
            "measure_grid_max_denominator": [
                MEASURE_GRID_LEVELS[level][-1] for level in grid_levels],
            "measure_grid_distribution": dict(Counter(
                str(MEASURE_GRID_LEVELS[level][-1])
                for level in grid_levels)),
            "triplet_ratio": round(triplet / total_notes, 4),
            "timing_residual_ms_p50": round(
                float(np.median(timing_residuals)), 3)
                if timing_residuals else 0.0,
            "timing_residual_ms_p95": round(
                float(np.percentile(timing_residuals, 95)), 3)
                if timing_residuals else 0.0,
        },
        "quality": {
            "extrapolated_ratio": round(extrap / raw_total, 4),
            "low_confidence_ratio": round(low_conf / raw_total, 4),
            "timing_sigma_ms_p50": round(
                float(np.median(timing_sigmas)), 3)
                if timing_sigmas else 0.0,
            "timing_sigma_ms_p95": round(
                float(np.percentile(timing_sigmas, 95)), 3)
                if timing_sigmas else 0.0,
        },
    }
