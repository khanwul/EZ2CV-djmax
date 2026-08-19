"""Convert a millisecond-domain ``RawChart`` into a tick-domain ``Chart``.

Snapped notes do not feed back into timeline inference. DJMAX longnote tails
retain explicit release calibration and snap as relative lengths.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace

import numpy as np

from ez2cv.detection import RawChart, TrackMetadata
from ez2cv.chart.clock import BPMSegment, TickClock
from ez2cv.chart.meter import TimeSignature, TimeSigVariant
from ez2cv.chart.quantize import (MEASURE_GRID_LEVELS, SnapResult,
                                  choose_measure_grid, choose_measure_grids,
                                  snap_by_measure,
                                  snap_length,
                                  TRIPLET_DENOMS)
from ez2cv.chart.timeline import infer_timeline


TIMING_SIGMA_MULTIPLIER = 2.0
COARSE_HEAD_ALPHA = 2.0
FINE_HEAD_ALPHA = 0.4
FINE_GRID_COST_TOLERANCE = 3.0


# =============================================================================
# ChartNote
# =============================================================================

@dataclass
class ChartNote:
    """Minimal playable note; review evidence lives in Chart.diagnostics."""
    lane: int
    start_tick: int
    end_tick: int | None       # None for taps
    off_grid: bool = False
    needs_review: bool = False

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
    diagnostics: list[dict]

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
            f"  needs review     : "
            f"{self.stats['quality']['needs_review_ratio']*100:.1f}%",
            f"  repaired beats   : "
            f"+{self.stats['quality']['inserted_beat_count']} / "
            f"-{self.stats['quality']['deleted_beat_count']}",
            f"  fine-grid ratio  : "
            f"{self.stats['rhythm']['fine_grid_ratio']*100:.1f}%",
            f"  note count       : {self.stats['counts']['total_notes']}",
        ]
        return "\n".join(lines)


# =============================================================================
# Pipeline
# =============================================================================

def build_chart(raw: RawChart) -> Chart:
    if raw.tick_resolution != 192:
        raise ValueError(
            f"chart conversion requires 192 ticks per quarter "
            f"(got {raw.tick_resolution})")
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
    notes, snaps, grid_levels, diagnostics = _convert_and_snap_notes(
        raw, clock, bl_ticks)

    # --- 4. stats -----------------------------------------------------
    stats = _build_stats(notes, snaps, grid_levels, clock, raw, bl_ticks,
                         variants, timeline)

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
        diagnostics=diagnostics,
    )

# ------------------------------------------------------------------ #

def _convert_and_snap_notes(raw: RawChart, clock: TickClock,
                            barline_ticks: list[int]
                            ) -> tuple[list[ChartNote], list[SnapResult],
                                       list[int], list[dict]]:
    """RawNote → ChartNote with adaptive head and tail snapping."""
    R = clock.tick_resolution

    # Heads select per-measure vocabularies. DJMAX tail timing is already
    # measured at center crossing in detection, so longnote lengths stay
    # relative.
    raw_heads = [clock.ms_to_tick(n.trigger_ms) for n in raw.notes]
    head_grid_floor = choose_measure_grid(
        raw_heads, tick_resolution=R, cost_tolerance=1.0)
    if head_grid_floor:
        head_grid_floor = choose_measure_grid(
            raw_heads, tick_resolution=R,
            cost_tolerance=FINE_GRID_COST_TOLERANCE)
    corrected_heads = list(raw_heads)
    if head_grid_floor:
        fine_step = 4.0 * R / max(MEASURE_GRID_LEVELS[head_grid_floor])
        for segment in clock.segments:
            indices = [i for i, tick in enumerate(raw_heads)
                       if segment.start_tick <= tick < segment.end_tick]
            if len(indices) < 4:
                continue
            residuals = [raw_heads[i] - round(raw_heads[i] / fine_step) * fine_step
                         for i in indices]
            phase_bias = float(np.median(residuals))
            for i in indices:
                corrected_heads[i] -= phase_bias
    grid_levels = choose_measure_grids(
        corrected_heads, barline_ticks, tick_resolution=R)
    grid_levels = [max(level, head_grid_floor) for level in grid_levels]
    snaps, grid_levels = snap_by_measure(
        corrected_heads, barline_ticks, tick_resolution=R,
        grid_levels=grid_levels,
        alpha=COARSE_HEAD_ALPHA if head_grid_floor == 0 else FINE_HEAD_ALPHA)
    snaps = [_apply_timing_uncertainty(note, snap, clock)
             for note, snap in zip(raw.notes, snaps, strict=True)]

    chart_notes: list[ChartNote] = []
    diagnostics: list[dict] = []
    for note_index, (raw_note, raw_t, snap) in enumerate(
            zip(raw.notes, raw_heads, snaps, strict=True)):
        end_tick: int | None = None
        raw_end: float | None = None
        end_residual_ms: float | None = None
        tail_snap_source: str | None = None
        needs_review = (raw_note.pairing_status != "observed"
                        or (raw_note.type == "longnote"
                            and (snap.off_grid or snap.timing_uncertain)))
        if raw_note.type == "longnote" and raw_note.end_ms is not None:
            raw_end = clock.ms_to_tick(raw_note.end_ms)
            snapped_len = snap_length(raw_end - raw_t, tick_resolution=R)
            if snapped_len > 0:
                end_tick = snap.tick + snapped_len
                tail_snap_source = "relative-length"
            else:
                end_tick = max(snap.tick + 1, int(round(raw_end)))
                tail_snap_source = "raw-fallback"
                needs_review = True
            end_residual_ms = abs(
                raw_note.end_ms - clock.tick_to_ms(end_tick))

        chart_notes.append(ChartNote(
            lane=raw_note.lane,
            start_tick=snap.tick,
            end_tick=end_tick,
            off_grid=snap.off_grid,
            needs_review=needs_review,
        ))
        diagnostics.append({
            "source_raw_note": note_index,
            "confidence": round(float(raw_note.confidence), 4),
            "extrapolated": bool(raw_note.extrapolated),
            "pairing_status": raw_note.pairing_status,
            "start_sigma_ms": round(float(raw_note.start_sigma_ms), 3),
            "end_sigma_ms": (round(float(raw_note.end_sigma_ms), 3)
                             if raw_note.end_sigma_ms is not None else None),
            "raw_start_tick": round(float(raw_t), 4),
            "raw_end_tick": (round(float(raw_end), 4)
                             if raw_end is not None else None),
            "tail_bias_tick": None,
            "tail_grid_level": None,
            "start_residual_ms": round(float(snap.timing_residual_ms), 3),
            "end_residual_ms": (round(float(end_residual_ms), 3)
                                if end_residual_ms is not None else None),
            "tail_snap_source": tail_snap_source,
            "needs_review": needs_review,
        })

    order = sorted(range(len(chart_notes)),
                   key=lambda i: (chart_notes[i].start_tick,
                                  chart_notes[i].lane))
    chart_notes = [chart_notes[i] for i in order]
    diagnostics = [diagnostics[i] for i in order]
    return chart_notes, snaps, grid_levels, diagnostics


def _apply_timing_uncertainty(raw_note, snap: SnapResult,
                              clock: TickClock) -> SnapResult:
    """Separate measurement-compatible misses from real timing outliers."""
    return _apply_endpoint_uncertainty(
        raw_note.trigger_ms, raw_note.start_sigma_ms, snap, clock)


def _apply_endpoint_uncertainty(event_ms: float, sigma_ms: float,
                                snap: SnapResult,
                                clock: TickClock) -> SnapResult:
    residual_ms = abs(event_ms - clock.tick_to_ms(snap.tick))
    uncertain = bool(snap.off_grid and sigma_ms > 0
                     and residual_ms <= TIMING_SIGMA_MULTIPLIER * sigma_ms)
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
                 variants: list[TimeSigVariant],
                 timeline) -> dict:
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
    timing_sigmas = ([n.start_sigma_ms for n in raw.notes]
                     + [n.end_sigma_ms for n in raw.notes
                        if n.end_sigma_ms is not None])
    timing_residuals = [s.timing_residual_ms for s in snaps]
    raw_total = len(raw.notes) or 1
    beat_intervals = np.diff([beat.ms for beat in raw.beats])
    if len(beat_intervals):
        beat_period = float(np.median(beat_intervals))
        beat_interval_outliers = float(np.mean(
            np.abs(beat_intervals - beat_period)
            > max(1000.0 / raw.fps, beat_period * 0.2)))
    else:
        beat_interval_outliers = 0.0
    barline_phase_residuals = [
        min(abs(barline.ms - beat.ms) for beat in raw.beats)
        for barline in raw.barlines
    ] if raw.beats else []

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
            "bpm_bound_adjustment_count": len(clock.bpm_bound_adjustments),
            "inserted_beat_count": timeline.inserted_beats,
            "deleted_beat_count": timeline.deleted_beats,
            "beat_interval_outlier_ratio": round(
                beat_interval_outliers, 4),
            "barline_beat_phase_residual_ms_p95": round(
                float(np.percentile(barline_phase_residuals, 95)), 3)
                if barline_phase_residuals else 0.0,
            "reconstructed_barline_ratio": round(
                sum(barline.extrapolated for barline in timeline.barlines)
                / max(1, len(timeline.barlines)), 4),
            "orphan_tail_count": raw.orphan_tails,
            "needs_review_ratio": round(
                sum(note.needs_review for note in notes) / total_notes, 4),
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
