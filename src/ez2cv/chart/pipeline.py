"""Convert a reloadable millisecond-domain ``RawChart`` into tick data.

    1. ``measure_zero_ms``  = barlines[0].ms   (song-start anchor)
    2. ``TickClock``         from beats + measure_zero_ms       (bpm_estimator)
    3. ``TimeSignature``     from barlines + TickClock          (time_sig)
    4. ms → tick conversion for every note (head + tail)
    5. snap-to-grid (quantizer); longnote tails snap as RELATIVE lengths
    6. pickup-note policy: keep one measure pre-anacrusis, drop earlier
    7. ``Chart`` ready for serialization

What chart conversion deliberately does not do
-----------------------------------------------
* No JSON I/O.
* No video re-decoding; it reads only ``RawChart``.
* No second-pass BPM refinement using already-snapped notes. The note positions
  influence nothing about the tempo curve here — that ordering would form a
  feedback loop and was rejected during design.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from ez2cv.detection import RawChart
from ez2cv.chart.clock import BPMSegment, TickClock
from ez2cv.chart.bpm_barline import build_tick_clock
from ez2cv.chart.meter import TimeSignature, TimeSigVariant, barline_ticks
from ez2cv.chart.barline import reconstruct_barlines
from ez2cv.chart.quantize import (SnapResult, snap_with_local_context, snap_length,
                              TRIPLET_DENOMS)


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
    key_mode: str
    lane_colors: tuple[str, ...]
    tick_resolution: int
    bpm_segments: list[BPMSegment]
    global_time_sig: TimeSignature
    variant_measures: list[TimeSigVariant]
    measure_zero_ms: float
    notes: list[ChartNote]
    barlines_tick: list[int]
    stats: dict

    def summary(self) -> str:
        bpm_lo = min(s.bpm_start for s in self.bpm_segments)
        bpm_hi = max(s.bpm_end for s in self.bpm_segments)
        lines = [
            f"=== Chart — {len(self.notes)} notes, "
            f"{len(self.bpm_segments)} bpm seg(s), "
            f"{len(self.variant_measures)} variant run(s) ===",
            f"  time signature   : {self.global_time_sig}",
            f"  bpm range        : {bpm_lo:.2f} .. {bpm_hi:.2f}",
            f"  measure_zero_ms  : {self.measure_zero_ms:.1f}",
            f"  measures         : {self.stats['structure']['measure_count']}",
            f"  off-grid ratio   : "
            f"{self.stats['rhythm']['off_grid_ratio']*100:.1f}%",
            f"  predicted combo  : "
            f"{self.stats['counts']['predicted_max_combo']}",
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

    # --- 1. beat-phase barline reconstruction -------------------------
    # Recover a complete, FP-free measure grid (+ global time signature and
    # variant runs) from the INCOMPLETE detection barlines plus the robust
    # POW-LED beats. This runs BEFORE the clock: it is beat-count based and
    # needs no tick grid, and it gives a cleaned song-start anchor.
    rec = reconstruct_barlines(raw.barlines, raw.beats)
    barlines = rec.barlines
    global_ts, variants = rec.time_signature, rec.variants

    # --- 2. anchor & tick clock ---------------------------------------
    measure_zero_ms = float(barlines[0].ms)
    active_window = (float(barlines[0].ms), float(barlines[-1].ms))
    R = raw.tick_resolution
    # Barline-derived BPM (per-measure resolution → resolves staircases),
    # with an automatic fall back to the beat estimator when the assumed
    # time signature is wrong. Pass the PER-MEASURE beat counts from the
    # reconstruction so variant measures (5/4, 6/4, …) compute their BPM
    # from the right beats-per-bar instead of a wrong global value.
    clock = build_tick_clock(
        raw.beats,
        barlines,
        np.asarray(rec.measure_meters, dtype=float),
        measure_zero_ms=measure_zero_ms,
        active_window_ms=active_window,
        min_bpm=raw.min_bpm,
        max_bpm=raw.max_bpm,
        tick_resolution=R,
    )

    # --- 3. barline ticks on the reconstructed (gap-free) grid --------
    bl_ticks = barline_ticks(barlines, clock, global_ts, variants)
    ticks_per_global_measure = global_ts.ticks_per_measure(R)

    # --- 3. notes : raw ticks, then snap ------------------------------
    notes, snaps = _convert_and_snap_notes(raw, clock, ticks_per_global_measure)

    # --- 4. stats -----------------------------------------------------
    stats = _build_stats(notes, snaps, clock, raw, bl_ticks, variants)

    return Chart(
        song_name=raw.song_name,
        key_mode=raw.key_mode,
        lane_colors=raw.lane_colors,
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
                            ticks_per_global_measure: int
                            ) -> tuple[list[ChartNote], list[SnapResult]]:
    """RawNote → ChartNote with head snap + relative tail snap."""
    R = clock.tick_resolution

    # raw float ticks for every head
    raw_heads = [clock.ms_to_tick(n.trigger_ms) for n in raw.notes]

    # context-aware snap on heads only (tails are length-snapped relatively)
    snaps = snap_with_local_context(raw_heads, tick_resolution=R)

    chart_notes: list[ChartNote] = []
    anacrusis_floor = -ticks_per_global_measure       # 1-measure pickup zone

    for raw_note, raw_t, snap in zip(raw.notes, raw_heads, snaps):
        # pickup-note drop: pre-first-barline by > one measure
        if snap.tick < anacrusis_floor:
            continue

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
    snaps_filtered = [snaps[i] for i, rn in enumerate(raw.notes)
                      if snaps[i].tick >= anacrusis_floor]
    # NOTE: snaps_filtered keeps original order; re-sort to match chart_notes
    # (used only for stats and diagnostics — order doesn't have to be perfect)
    return chart_notes, snaps_filtered


# =============================================================================
# Stats
# =============================================================================

def _predicted_max_combo(notes: list[ChartNote]) -> int:
    """Per the algorithm memo: tap = 1, longnote = 1 + hold_ticks/48."""
    total = 0
    for n in notes:
        if n.end_tick is None:
            total += 1
        else:
            hold_ticks = max(0, n.end_tick - n.start_tick)
            total += 1 + hold_ticks // 48
    return total


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
    triplet = sum(1 for s in snaps if not s.off_grid and s.denom in TRIPLET_DENOMS)
    total_notes = len(notes) or 1

    # tempo
    bpm_lo = min(s.bpm_start for s in clock.segments)
    bpm_hi = max(s.bpm_end for s in clock.segments)
    has_ramp = any(not s.is_constant for s in clock.segments)

    # quality (from detection debug info)
    extrap = sum(1 for n in raw.notes if n.extrapolated)
    low_conf = sum(1 for n in raw.notes if n.confidence < 0.5)
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
            "predicted_max_combo": _predicted_max_combo(notes),
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
            "off_grid_ratio": round(off_grid / total_notes, 4),
            "triplet_ratio": round(triplet / total_notes, 4),
        },
        "quality": {
            "extrapolated_ratio": round(extrap / raw_total, 4),
            "low_confidence_ratio": round(low_conf / raw_total, 4),
        },
    }
