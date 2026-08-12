"""Joint meter and tempo timeline inference from raw beat/barline events."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ez2cv.chart.barline import ReconstructResult, reconstruct_barlines
from ez2cv.chart.clock import BPMSegment, TickClock
from ez2cv.chart.meter import TimeSignature, TimeSigVariant
from ez2cv.chart.quantize import (ALLOWED_DENOMS, DEFAULT_ALPHA,
                                  DEFAULT_TRIPLET_CONTEXT_PENALTY,
                                  TRIPLET_DENOMS, grid_distances)
from ez2cv.detection import RawChart
from ez2cv.detection.barline import BarlineEvent
from ez2cv.detection.beat import BeatEvent


@dataclass
class Timeline:
    clock: TickClock
    barlines: list[BarlineEvent]
    barline_ticks: list[int]
    global_time_sig: TimeSignature
    variants: list[TimeSigVariant]
    measure_meters: list[int]
    inserted_beats: int = 0
    deleted_beats: int = 0


TEMPO_CHANGE_COST = 4.0
TEMPO_CHANGE_COSTS = (2.0, 4.0)
TEMPO_RAMP_COST = 3.0
NOTE_GRID_COST = 2.0
SUBBEAT_GRID_GAIN = 0.4
TEMPO_BPM_QUANTUM = 0.25
BOUNDED_CLOCK_RESIDUAL_FRAMES = 3.0


@dataclass(frozen=True)
class _TempoSpan:
    start: int
    end: int
    bpm_start: float
    bpm_end: float
    cost: float
    bound_adjustment_ms: float = 0.0


def _tempo_observations(beat_ms: np.ndarray, result: ReconstructResult,
                        tick_resolution: int) -> tuple[np.ndarray, np.ndarray]:
    """Align every active beat to measured barlines without changing order."""
    ordinals = np.asarray(result.beat_indices, dtype=int)
    active = np.arange(ordinals[0], ordinals[-1] + 1)
    bars = np.asarray([line.ms for line in result.barlines], dtype=float)
    corrections = bars - beat_ms[ordinals]
    corrected = beat_ms[active] + np.interp(active, ordinals, corrections)
    if np.any(np.diff(corrected) <= 0):
        raise ValueError("corrected beat observations must be monotonic")
    ticks = (active - ordinals[0]) * tick_resolution
    return corrected, ticks.astype(float)


def _fit_tempo_span(raw: RawChart, ms: np.ndarray, ticks: np.ndarray,
                    start: int, end: int) -> _TempoSpan | None:
    duration = float(ms[end] - ms[start])
    tick_span = float(ticks[end] - ticks[start])
    if duration <= 0 or tick_span <= 0:
        return None
    sigma = 1000.0 / raw.fps
    inferred_bpm = tick_span * 60_000.0 / (
        raw.tick_resolution * duration)
    average_bpm = float(np.clip(inferred_bpm, raw.min_bpm, raw.max_bpm))
    bounded_duration = tick_span * 60_000.0 / (
        raw.tick_resolution * average_bpm)
    bound_adjustment = abs(bounded_duration - duration)
    if bound_adjustment > 2000.0 / raw.fps:
        return None
    average_rate = raw.tick_resolution * average_bpm / 60_000.0

    local_ms = ms[start:end + 1]
    local_ticks = ticks[start:end + 1]
    elapsed = local_ms - ms[start]
    step_prediction = ticks[start] + average_rate * elapsed
    step_residual_ms = (local_ticks - step_prediction) / average_rate
    best = _TempoSpan(
        start, end, average_bpm, average_bpm,
        float(np.sum((step_residual_ms / sigma) ** 2)),
        bound_adjustment,
    )

    if end - start < 3 or bound_adjustment:
        return best
    curvature = elapsed * elapsed - duration * elapsed
    denominator = float(np.dot(curvature, curvature))
    if denominator <= 0:
        return best
    tick_residual = local_ticks - step_prediction
    coefficient = float(np.dot(curvature, tick_residual) / denominator)
    bpm_start = ((average_rate - coefficient * duration)
                 * 60_000.0 / raw.tick_resolution)
    bpm_end = ((average_rate + coefficient * duration)
               * 60_000.0 / raw.tick_resolution)
    if not (raw.min_bpm <= bpm_start <= raw.max_bpm
            and raw.min_bpm <= bpm_end <= raw.max_bpm):
        return best
    ramp_prediction = step_prediction + coefficient * curvature
    ramp_residual_ms = (local_ticks - ramp_prediction) / average_rate
    ramp_cost = float(np.sum((ramp_residual_ms / sigma) ** 2)) + TEMPO_RAMP_COST
    if ramp_cost < best.cost:
        return _TempoSpan(start, end, bpm_start, bpm_end, ramp_cost)
    return best


def _fit_tempo_clock(raw: RawChart, ms: np.ndarray,
                     ticks: np.ndarray,
                     change_cost: float = TEMPO_CHANGE_COST) -> TickClock:
    """Select step/ramp beat spans under one residual + complexity cost."""
    count = len(ms)
    costs = np.full(count, np.inf)
    previous = np.full(count, -1, dtype=int)
    chosen: list[_TempoSpan | None] = [None] * count
    costs[0] = 0.0
    for end in range(1, count):
        for start in range(end):
            span = _fit_tempo_span(raw, ms, ticks, start, end)
            if span is None or not np.isfinite(costs[start]):
                continue
            cost = costs[start] + span.cost
            if start:
                cost += change_cost
            if cost < costs[end]:
                costs[end] = cost
                previous[end] = start
                chosen[end] = span
    if chosen[-1] is None:
        raise ValueError("no in-range tempo model fits the beat observations")

    spans: list[_TempoSpan] = []
    end = count - 1
    while end:
        span = chosen[end]
        assert span is not None
        spans.append(span)
        end = previous[end]
    spans.reverse()
    segments = [BPMSegment(
        int(round(ticks[span.start])), int(round(ticks[span.end])),
        span.bpm_start, span.bpm_end,
    ) for span in spans]
    clock = TickClock(segments, tick_zero_ms=float(ms[0]),
                      tick_resolution=raw.tick_resolution)
    phase = float(np.mean([
        observed - clock.tick_to_ms(tick)
        for observed, tick in zip(ms, ticks, strict=True)
    ]))
    clock = TickClock(segments, tick_zero_ms=float(ms[0]) + phase,
                      tick_resolution=raw.tick_resolution)
    clock.bpm_bound_adjustments = [
        span.bound_adjustment_ms for span in spans
        if span.bound_adjustment_ms
    ]
    return _validate_bounded_clock(raw, clock, ms, ticks)


def _validate_bounded_clock(raw: RawChart, clock: TickClock,
                            ms: np.ndarray, ticks: np.ndarray) -> TickClock:
    """Reject bounded spans whose small local errors accumulate globally."""
    if clock.bpm_bound_adjustments:
        residual = max(abs(observed - clock.tick_to_ms(tick))
                       for observed, tick in zip(ms, ticks, strict=True))
        if residual > BOUNDED_CLOCK_RESIDUAL_FRAMES * 1000.0 / raw.fps:
            raise ValueError(
                "no in-range tempo model fits the beat observations: "
                f"bounded clock residual is {residual:.3f} ms")
    return clock


def _tempo_clock_score(raw: RawChart, clock: TickClock
                       ) -> tuple[int, float, int]:
    """Auxiliary note-grid score; beat/barline residual fit happens first."""
    note_ticks = np.asarray([
        clock.ms_to_tick(note.trigger_ms) for note in raw.notes
    ])
    distances = grid_distances(
        note_ticks, ALLOWED_DENOMS, tick_resolution=raw.tick_resolution)
    return (int(np.count_nonzero(distances > 6.0)),
            float(np.sum(np.minimum(distances, 24.0) ** 2)),
            len(clock.segments))


def _phase_fit_clock(clock: TickClock, result: ReconstructResult,
                     barline_ticks: np.ndarray) -> TickClock:
    phase = float(np.mean([
        line.ms - clock.tick_to_ms(tick)
        for line, tick in zip(result.barlines, barline_ticks, strict=True)
    ]))
    fitted = TickClock(
        clock.segments, tick_zero_ms=clock.tick_zero_ms + phase,
        tick_resolution=clock.tick_resolution)
    fitted.bpm_bound_adjustments = list(clock.bpm_bound_adjustments)
    return fitted


def _normalized_tempo_clock(raw: RawChart, clock: TickClock) -> TickClock:
    def normalized(bpm: float) -> float:
        rounded = round(bpm / TEMPO_BPM_QUANTUM) * TEMPO_BPM_QUANTUM
        return min(raw.max_bpm, max(raw.min_bpm, rounded))

    segments = [BPMSegment(
        segment.start_tick, segment.end_tick,
        normalized(segment.bpm_start), normalized(segment.bpm_end),
    ) for segment in clock.segments]
    normalized = TickClock(
        segments, tick_zero_ms=clock.tick_zero_ms,
        tick_resolution=clock.tick_resolution)
    normalized.bpm_bound_adjustments = list(clock.bpm_bound_adjustments)
    return normalized


def _expand_symmetric_subbeat_steps(raw: RawChart,
                                    clock: TickClock) -> TickClock:
    """Recover low-half/high/low-half steps hidden between beat flashes."""
    resolution = raw.tick_resolution
    expanded: list[BPMSegment] = []
    changed = False
    for index, segment in enumerate(clock.segments):
        bpm = segment.bpm_start
        high = 1.5 * bpm
        low = 0.75 * bpm
        neighbours = [clock.segments[i].bpm_start
                      for i in (index - 1, index + 1)
                      if 0 <= i < len(clock.segments)]
        eligible = (
            segment.is_constant
            and segment.end_tick - segment.start_tick == 2 * resolution
            and raw.min_bpm <= low <= high <= raw.max_bpm
            and any(abs(neighbour - high) <= 0.05 * high
                    for neighbour in neighbours)
        )
        if not eligible:
            expanded.append(segment)
            continue

        start_ms = clock.tick_to_ms(segment.start_tick)
        end_ms = clock.tick_to_ms(segment.end_tick)
        onsets: list[float] = []
        for event_ms in sorted(note.trigger_ms for note in raw.notes
                               if start_ms <= note.trigger_ms < end_ms):
            if not onsets or event_ms - onsets[-1] > 1000.0 / raw.fps:
                onsets.append(float(event_ms))
        if len(onsets) < 4:
            expanded.append(segment)
            continue

        half = resolution // 2
        candidate = [
            BPMSegment(0, half, low, low),
            BPMSegment(half, half + resolution, high, high),
            BPMSegment(half + resolution, 2 * resolution, low, low),
        ]
        local_clock = TickClock(
            candidate, tick_zero_ms=0.0, tick_resolution=resolution)
        observed_ticks = np.asarray([
            clock.ms_to_tick(event_ms) for event_ms in onsets
        ])
        candidate_ticks = segment.start_tick + np.asarray([
            local_clock.ms_to_tick(event_ms - start_ms)
            for event_ms in onsets
        ])

        def rhythm_cost(tick_values: np.ndarray) -> float:
            costs = []
            for tick in tick_values:
                costs.append(min(
                    abs(tick - round(tick / (4.0 * resolution / denominator))
                        * (4.0 * resolution / denominator))
                    + DEFAULT_ALPHA * np.log2(denominator)
                    + (DEFAULT_TRIPLET_CONTEXT_PENALTY
                       if denominator in TRIPLET_DENOMS else 0.0)
                    for denominator in ALLOWED_DENOMS))
            return float(np.mean(costs))

        if (rhythm_cost(candidate_ticks) + SUBBEAT_GRID_GAIN
                >= rhythm_cost(observed_ticks)):
            expanded.append(segment)
            continue

        expanded.extend(BPMSegment(
            segment.start_tick + part.start_tick,
            segment.start_tick + part.end_tick,
            part.bpm_start, part.bpm_end,
        ) for part in candidate)
        changed = True

    if not changed:
        return clock
    refined = TickClock(expanded, tick_zero_ms=clock.tick_zero_ms,
                        tick_resolution=resolution)
    refined.bpm_bound_adjustments = list(clock.bpm_bound_adjustments)
    return refined


def clean_beat_times(beats: list[BeatEvent]) -> np.ndarray:
    """Return a monotonic beat stream without inventing unobserved flashes.

    An isolated 2x interval can be either a missed flash or an intentional
    one-beat tempo change.  Timing data alone cannot distinguish them, so
    inserting a beat would make arbitrary speed changes impossible to retain.
    """
    return np.array(sorted(set(float(b.ms) for b in beats)), dtype=float)


def _meter_score(result: ReconstructResult) -> tuple[int, int]:
    if not result.measure_meters:
        return (10**9, 10**9)
    variants = sum(meter != result.global_meter
                   for meter in result.measure_meters)
    changes = sum(left != right for left, right in zip(
        result.measure_meters, result.measure_meters[1:]))
    return variants + changes, sum(line.extrapolated for line in result.barlines)


def _repair_score(result: ReconstructResult, beat_ms: np.ndarray, fps: float,
                  edits: int) -> float:
    """Shared meter/tempo/edit cost for observed-beat repair candidates."""
    meter_anomalies, inferred_lines = _meter_score(result)
    intervals = np.diff(beat_ms)
    sigma = 1000.0 / fps
    changes = np.abs(np.diff(intervals))
    excess = np.maximum(0.0, changes - sigma) / sigma
    tempo_cost = float(np.sum(np.minimum(excess * excess, 4.0)))
    return (20.0 * meter_anomalies + 2.0 * inferred_lines + tempo_cost
            + 12.0 * edits)


def _gap_note_grid_cost(raw: RawChart, start_ms: float, end_ms: float,
                        quarters: int) -> float:
    """Use raw onsets only as a tie-breaker for a beat edit inside one gap."""
    frame_ms = 1000.0 / raw.fps
    onsets: list[float] = []
    for event_ms in sorted(n.trigger_ms for n in raw.notes
                           if start_ms <= n.trigger_ms <= end_ms):
        if not onsets or event_ms - onsets[-1] > frame_ms:
            onsets.append(float(event_ms))
    if len(onsets) < 2 or end_ms <= start_ms:
        return 0.0
    resolution = getattr(raw, "tick_resolution", 192)
    ticks = ((np.asarray(onsets) - start_ms) / (end_ms - start_ms)
             * resolution * quarters)
    distances = grid_distances(
        ticks, ALLOWED_DENOMS, tick_resolution=resolution)
    return float(np.mean(np.minimum(distances, 24.0) ** 2))


def _repair_beats(raw: RawChart, beat_ms: np.ndarray
                  ) -> tuple[np.ndarray, ReconstructResult, int, int]:
    """Insert/delete only beat candidates that simplify the meter DP."""
    reconstruction = reconstruct_barlines(
        raw.barlines, _events(beat_ms, raw.fps),
        last_event_ms=_last_event_ms(raw))
    inserted = deleted = 0
    while len(beat_ms) >= 3:
        intervals = np.diff(beat_ms)
        best: tuple[float, np.ndarray, ReconstructResult,
                    int, int] | None = None
        baseline = _repair_score(
            reconstruction, beat_ms, raw.fps, inserted + deleted)

        # A spurious mid-beat flash splits one normal interval into two shorts.
        period = float(np.median(intervals))
        tolerance = max(1000.0 / raw.fps, period * 0.08)
        for i in range(1, len(beat_ms) - 1):
            left, right = intervals[i - 1], intervals[i]
            if (max(left, right) >= period * 0.8
                    or abs(left + right - period) > tolerance):
                continue
            candidate = np.delete(beat_ms, i)
            result = reconstruct_barlines(
                raw.barlines, _events(candidate, raw.fps),
                last_event_ms=_last_event_ms(raw))
            score = _repair_score(
                result, candidate, raw.fps, inserted + deleted + 1)
            score += NOTE_GRID_COST * (
                _gap_note_grid_cost(raw, beat_ms[i - 1], beat_ms[i + 1], 1)
                - _gap_note_grid_cost(raw, beat_ms[i - 1], beat_ms[i + 1], 2))
            if score < baseline and (best is None or score < best[0]):
                best = score, candidate, result, 0, 1

        for i, gap in enumerate(intervals):
            neighbours = np.delete(intervals[max(0, i - 3):i + 4],
                                   min(3, i))
            if not len(neighbours):
                continue
            period = float(np.median(neighbours))
            multiple = int(round(gap / period))
            if not 2 <= multiple <= 4:
                continue
            if abs(gap / multiple - period) > max(1000.0 / raw.fps, period * 0.08):
                continue
            additions = beat_ms[i] + gap * np.arange(1, multiple) / multiple
            candidate = np.insert(beat_ms, i + 1, additions)
            result = reconstruct_barlines(
                raw.barlines, _events(candidate, raw.fps),
                last_event_ms=_last_event_ms(raw))
            score = _repair_score(
                result, candidate, raw.fps,
                inserted + deleted + multiple - 1)
            score += NOTE_GRID_COST * (
                _gap_note_grid_cost(raw, beat_ms[i], beat_ms[i + 1], multiple)
                - _gap_note_grid_cost(raw, beat_ms[i], beat_ms[i + 1], 1))
            if score < baseline and (best is None or score < best[0]):
                best = score, candidate, result, multiple - 1, 0
        if best is None:
            break
        _, beat_ms, reconstruction, inserted_now, deleted_now = best
        inserted += inserted_now
        deleted += deleted_now
    return beat_ms, reconstruction, inserted, deleted


def _events(times: np.ndarray, fps: float) -> list[BeatEvent]:
    return [BeatEvent(int(round(ms * fps / 1000.0)), float(ms), 0.0)
            for ms in times]


def _last_event_ms(raw: RawChart) -> float | None:
    if not raw.notes:
        return None
    return max(n.end_ms if n.end_ms is not None else n.trigger_ms
               for n in raw.notes)


def infer_timeline(raw: RawChart) -> Timeline:
    beat_ms = clean_beat_times(raw.beats)
    if len(beat_ms) < 2:
        raise ValueError("not enough reliable beats to infer a timeline")

    beat_ms, reconstruction, inserted_beats, deleted_beats = _repair_beats(
        raw, beat_ms)
    if len(reconstruction.beat_indices) < 2:
        raise ValueError("not enough on-beat barlines to infer a timeline")

    origin = reconstruction.beat_indices[0]
    bar_ticks = np.array([
        (ordinal - origin) * raw.tick_resolution
        for ordinal in reconstruction.beat_indices
    ], dtype=float)
    observed_ms, observed_ticks = _tempo_observations(
        beat_ms, reconstruction, raw.tick_resolution)

    if abs(raw.max_bpm - raw.min_bpm) < 1e-9:
        # A configured fixed BPM is authoritative; fit only its absolute offset.
        bpm = raw.min_bpm
        ms_per_tick = 60_000.0 / (raw.tick_resolution * bpm)
        tick_zero_ms = float(np.mean(
            observed_ms - observed_ticks * ms_per_tick))
        clock = TickClock([BPMSegment(
            int(round(observed_ticks[0])), int(round(observed_ticks[-1])), bpm, bpm)],
            tick_zero_ms=tick_zero_ms, tick_resolution=raw.tick_resolution)
    else:
        clocks = []
        for change_cost in TEMPO_CHANGE_COSTS:
            try:
                clocks.append(_expand_symmetric_subbeat_steps(
                    raw, _fit_tempo_clock(
                        raw, observed_ms, observed_ticks, change_cost)))
            except ValueError:
                pass
        if not clocks:
            raise ValueError("no in-range tempo model fits the beat observations")
        clock = min(clocks, key=lambda candidate:
                    _tempo_clock_score(raw, candidate))
        clock = _validate_bounded_clock(
            raw, _phase_fit_clock(clock, reconstruction, bar_ticks),
            observed_ms, observed_ticks)
        normalized = _phase_fit_clock(
            _normalized_tempo_clock(raw, clock), reconstruction, bar_ticks)
        candidates = [clock]
        try:
            candidates.append(_validate_bounded_clock(
                raw, normalized, observed_ms, observed_ticks))
        except ValueError:
            pass
        clock = min(candidates, key=lambda candidate:
                    _tempo_clock_score(raw, candidate))
    barline_ticks = [int(tick) for tick in bar_ticks]
    return Timeline(
        clock, reconstruction.barlines, barline_ticks,
        reconstruction.time_signature, reconstruction.variants,
        reconstruction.measure_meters, inserted_beats, deleted_beats)
