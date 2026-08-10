"""Infer an arbitrary per-measure meter sequence on the detected beat grid.

Observed on-beat measure lines are strong evidence and are preserved.  Dynamic
programming inserts only the boundaries hidden by notes/effects and chooses the
meter sequence with the fewest unsupported boundaries and gratuitous changes.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from ez2cv.chart.meter import TimeSignature, TimeSigVariant
from ez2cv.detection.barline import BarlineEvent
from ez2cv.detection.beat import BeatEvent


DEFAULT_NUMERATORS = tuple(range(1, 8))
RHO_MAX = 0.25
INFERRED_BOUNDARY_COST = 0.75
METER_CHANGE_COST = 0.60
SKIPPED_BARLINE_COST = 4.0
RHO_COST = 2.0


@dataclass
class ReconstructResult:
    barlines: list[BarlineEvent]
    time_signature: TimeSignature
    variants: list[TimeSigVariant]
    outliers: list[BarlineEvent]
    beat_indices: list[int]
    measure_meters: list[int]
    global_meter: int


def _index_barlines(barlines: list[BarlineEvent], beat_ms: np.ndarray
                    ) -> tuple[list[int], list[float]]:
    """Map each line to its nearest beat and phase residual."""
    ordinals: list[int] = []
    residuals: list[float] = []
    for barline in barlines:
        pos = int(np.searchsorted(beat_ms, barline.ms))
        candidates = [i for i in (pos - 1, pos) if 0 <= i < len(beat_ms)]
        i = min(candidates, key=lambda j: abs(beat_ms[j] - barline.ms))
        lo, hi = max(0, i - 1), min(len(beat_ms) - 1, i + 1)
        local_period = (beat_ms[hi] - beat_ms[lo]) / max(1, hi - lo)
        ordinals.append(i)
        residuals.append(abs(beat_ms[i] - barline.ms) /
                         max(1.0, local_period))
    return ordinals, residuals


def _last_active_ordinal(observed: list[int], beat_ms: np.ndarray,
                         last_event_ms: float | None) -> int:
    """End after the measure containing the final note, not empty outro bars."""
    if last_event_ms is None:
        return observed[-1]
    # Keep at least one measured span: it establishes phase even when every
    # note occurs before the first visible measure line.
    for ordinal in observed[1:]:
        lo, hi = max(0, ordinal - 1), min(len(beat_ms) - 1, ordinal + 1)
        half_beat = (beat_ms[hi] - beat_ms[lo]) / max(2, 2 * (hi - lo))
        if beat_ms[ordinal] > last_event_ms + half_beat:
            return ordinal
    return observed[-1]


def _infer_grid(start: int, end: int, observed: dict[int, float],
                numerators: tuple[int, ...]
                ) -> tuple[list[int], list[int]]:
    """Return boundary beat ordinals and the meter of every intervening measure."""
    # states[position][last meter] = (cost, previous position, previous meter)
    states: dict[int, dict[int, tuple[float, int, int]]] = {
        start: {0: (0.0, -1, -1)}
    }
    observed_ordinals = sorted(observed)

    for position in range(start, end):
        for previous_meter, (cost, _, _) in states.get(position, {}).items():
            for meter in numerators:
                nxt = position + meter
                if nxt > end:
                    continue
                skipped = sum(position < o < nxt for o in observed_ordinals)
                inferred = nxt != end and nxt not in observed
                new_cost = cost + skipped * SKIPPED_BARLINE_COST
                if inferred:
                    new_cost += INFERRED_BOUNDARY_COST
                else:
                    new_cost += RHO_COST * observed.get(nxt, 0.0)
                if previous_meter and previous_meter != meter:
                    new_cost += METER_CHANGE_COST

                current = states.setdefault(nxt, {}).get(meter)
                if current is None or new_cost < current[0]:
                    states[nxt][meter] = (new_cost, position, previous_meter)

    if end not in states:
        raise ValueError("measure grid cannot reach the final barline")

    meter = min(states[end], key=lambda m: states[end][m][0])
    boundaries = [end]
    meters: list[int] = []
    position = end
    while position != start:
        _, previous_position, previous_meter = states[position][meter]
        meters.append(meter)
        boundaries.append(previous_position)
        position, meter = previous_position, previous_meter
    boundaries.reverse()
    meters.reverse()
    return boundaries, meters


def _variants(meters: list[int], global_meter: int) -> list[TimeSigVariant]:
    variants: list[TimeSigVariant] = []
    for measure, meter in enumerate(meters):
        if meter == global_meter:
            continue
        time_sig = TimeSignature(meter, 4)
        if (variants and variants[-1].time_sig == time_sig
                and variants[-1].end_measure + 1 == measure):
            variants[-1].end_measure = measure
        else:
            variants.append(TimeSigVariant(measure, measure, time_sig))
    return variants


def reconstruct_barlines(barlines: list[BarlineEvent], beats: list[BeatEvent],
                         *, numerators=DEFAULT_NUMERATORS,
                         global_meter: int | None = None,
                         last_event_ms: float | None = None,
                         ) -> ReconstructResult:
    """Build a complete meter grid from on-beat observed measure lines."""
    if len(barlines) < 2 or len(beats) < 2:
        return ReconstructResult(list(barlines), TimeSignature(4, 4), [],
                                 [], [], [], 4)

    numerators = tuple(sorted(set(int(n) for n in numerators if n > 0)))
    if not numerators:
        raise ValueError("at least one positive time-signature numerator required")
    beat_ms = np.array(sorted(b.ms for b in beats), dtype=float)
    ordinals, residuals = _index_barlines(barlines, beat_ms)

    # One strongest line per beat. Off-beat lines are detector false positives.
    by_ordinal: dict[int, tuple[int, float]] = {}
    for source, (ordinal, residual) in enumerate(zip(ordinals, residuals)):
        if residual > RHO_MAX:
            continue
        previous = by_ordinal.get(ordinal)
        if previous is None or residual < previous[1]:
            by_ordinal[ordinal] = (source, residual)
    if len(by_ordinal) < 2:
        return ReconstructResult([], TimeSignature(4, 4), [],
                                 list(barlines), [], [], 4)

    observed_all = sorted(by_ordinal)
    start = observed_all[0]
    end = _last_active_ordinal(observed_all, beat_ms, last_event_ms)
    observed = {o: by_ordinal[o][1] for o in observed_all if start <= o <= end}
    boundaries, meters = _infer_grid(start, end, observed, numerators)

    if global_meter is None:
        counts = Counter(meters)
        global_meter = max(counts, key=lambda m: (counts[m], m == 4, -m))
    # The first observed line fixes phase even when the detector was blind in
    # the intro. DJMAX recordings need those early measures to retain notes.
    intro = list(range(start % global_meter, start, global_meter))
    boundaries = intro + boundaries
    meters = [global_meter] * len(intro) + meters

    chosen = set(boundaries)
    chosen_sources = {by_ordinal[o][0] for o in chosen if o in by_ordinal}
    outliers = [b for i, b in enumerate(barlines) if i not in chosen_sources]

    observed_lines = {o: barlines[source]
                      for o, (source, _) in by_ordinal.items() if o in chosen}
    if barlines[-1].ms > barlines[0].ms:
        frames_per_ms = ((barlines[-1].cross_frame - barlines[0].cross_frame)
                         / (barlines[-1].ms - barlines[0].ms))
        frame_intercept = barlines[0].cross_frame - frames_per_ms * barlines[0].ms
    else:
        frames_per_ms = frame_intercept = 0.0
    result_barlines: list[BarlineEvent] = []
    for ordinal in boundaries:
        ms = float(beat_ms[ordinal])
        if ordinal in observed_lines:
            b = observed_lines[ordinal]
            result_barlines.append(BarlineEvent(
                b.cross_frame, ms, b.strength, extrapolated=False))
        else:
            result_barlines.append(BarlineEvent(
                frame_intercept + frames_per_ms * ms, ms, 0.0,
                extrapolated=True))

    variants = _variants(meters, global_meter)
    return ReconstructResult(
        result_barlines, TimeSignature(global_meter, 4), variants, outliers,
        boundaries, meters, global_meter)
