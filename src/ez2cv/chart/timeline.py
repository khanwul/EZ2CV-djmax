"""Joint meter and tempo timeline inference from raw beat/barline events."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ez2cv.chart.barline import ReconstructResult, reconstruct_barlines
from ez2cv.chart.clock import TickClock
from ez2cv.chart.meter import TimeSignature, TimeSigVariant
from ez2cv.chart.quantize import ALLOWED_DENOMS, DEFAULT_MAX_TOLERANCE_TICK
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


TEMPO_RUN_REL_TOL = 0.013
SUBMEASURE_MIN_REL_CHANGE = 0.05
SUBMEASURE_MAX_RESIDUAL_RATIO = 0.10


def _tempo_runs(bar_ms: np.ndarray, meters: list[int], fps: float
                ) -> list[tuple[int, int]]:
    """Split measures only at clear changes in their measured beat period."""
    period = np.diff(bar_ms) / np.asarray(meters, dtype=float)
    if len(period) == 0:
        return []

    runs: list[tuple[int, int]] = []
    frame_floor = 500.0 / (fps * np.maximum(1, meters))
    start = 0
    run_sum = float(period[0])
    for i in range(1, len(period)):
        mean = run_sum / (i - start)
        threshold = max(TEMPO_RUN_REL_TOL * mean, frame_floor[i])
        if abs(period[i] - mean) > threshold:
            # An inferred line can move one isolated measure by a frame while
            # the following measure returns to the plateau. Absorb that pair,
            # but preserve a large one-measure tempo step.
            if (i + 1 < len(period)
                    and abs(period[i] - mean) <= 2.0 * threshold
                    and abs(period[i + 1] - mean)
                    <= max(TEMPO_RUN_REL_TOL * mean, frame_floor[i + 1])):
                run_sum += float(period[i])
                continue
            runs.append((start, i))
            start = i
            run_sum = float(period[i])
        else:
            run_sum += float(period[i])
    runs.append((start, len(period)))
    return runs


def _grid_score(note_ms: np.ndarray, anchor_ms: np.ndarray,
                anchor_ticks: np.ndarray, tick_resolution: int
                ) -> tuple[int, float]:
    """How naturally notes land on supported grids under one clock model."""
    if len(note_ms) == 0:
        return 0, 0.0
    ticks = np.interp(note_ms, anchor_ms, anchor_ticks)
    distances = np.full(len(ticks), np.inf)
    for denominator in ALLOWED_DENOMS:
        step = tick_resolution * 4.0 / denominator
        distances = np.minimum(distances,
                               np.abs(ticks - np.rint(ticks / step) * step))
    off_grid = int(np.count_nonzero(
        distances > DEFAULT_MAX_TOLERANCE_TICK))
    return off_grid, float(np.sum(np.minimum(distances, 24.0) ** 2))


def _select_tempo_anchors(raw: RawChart, bar_ms: np.ndarray,
                          bar_ticks: np.ndarray,
                          meters: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Denoise steady runs, retaining every bar in genuinely variable runs.

    Barline timing is frame-quantised.  A single line through a steady run
    removes that jitter; a changing run needs its intermediate observations.
    Raw note heads decide between those two measured models locally.
    """
    note_ms = np.array([n.trigger_ms for n in raw.notes], dtype=float)
    runs = _tempo_runs(bar_ms, meters, raw.fps)

    # A plateau's first and last detected lines each carry frame error. Fit all
    # of its lines, then reconcile the shared boundary with the neighbouring
    # fit. This removes per-measure BPM wobble without erasing real changes.
    fitted: list[tuple[float, float, int]] = []
    for start, end in runs:
        weight = end - start
        if weight >= 2:
            slope, intercept = np.polyfit(
                bar_ticks[start:end + 1], bar_ms[start:end + 1], 1)
            fitted.append((
                float(intercept + slope * bar_ticks[start]),
                float(intercept + slope * bar_ticks[end]),
                weight,
            ))
        else:
            fitted.append((float(bar_ms[start]), float(bar_ms[end]), weight))

    boundary_ms = [fitted[0][0]]
    for previous, following in zip(fitted, fitted[1:]):
        boundary_ms.append(
            (previous[1] * previous[2] + following[0] * following[2])
            / (previous[2] + following[2]))
    boundary_ms.append(fitted[-1][1])

    selected_ms: list[float] = []
    selected_ticks: list[float] = []
    for run_index, (start, end) in enumerate(runs):
        simple_ms = np.array(boundary_ms[run_index:run_index + 2])
        simple_ticks = bar_ticks[[start, end]]
        local_notes = note_ms[(bar_ms[start] <= note_ms)
                              & (note_ms < bar_ms[end])]

        observed = bar_ms[start:end + 1]
        fraction = ((bar_ticks[start:end + 1] - bar_ticks[start])
                    / (bar_ticks[end] - bar_ticks[start]))
        start_shift = simple_ms[0] - observed[0]
        end_shift = simple_ms[1] - observed[-1]
        detailed_ms = observed + start_shift + fraction * (
            end_shift - start_shift)
        detailed_ticks = bar_ticks[start:end + 1]

        simple = _grid_score(local_notes, simple_ms, simple_ticks,
                             raw.tick_resolution)
        detailed = _grid_score(local_notes, detailed_ms, detailed_ticks,
                               raw.tick_resolution)
        # Extra anchors must explain notes that would otherwise be off-grid.
        # Merely shaving sub-tolerance residuals is frame-jitter overfitting.
        if detailed[0] < simple[0]:
            chosen_ms, chosen_ticks = detailed_ms, detailed_ticks
        else:
            chosen_ms, chosen_ticks = simple_ms, simple_ticks
        if selected_ms:
            chosen_ms, chosen_ticks = chosen_ms[1:], chosen_ticks[1:]
        selected_ms.extend(chosen_ms)
        selected_ticks.extend(chosen_ticks)

    anchor_ms = np.asarray(selected_ms, dtype=float)
    anchor_ticks = np.asarray(selected_ticks, dtype=float)
    active_notes = note_ms[(bar_ms[0] <= note_ms) & (note_ms < bar_ms[-1])]
    slope, intercept = np.polyfit(bar_ticks, bar_ms, 1)
    global_ms = np.array([
        intercept + slope * bar_ticks[0],
        intercept + slope * bar_ticks[-1],
    ])
    global_ticks = bar_ticks[[0, -1]]
    global_score = _grid_score(active_notes, global_ms, global_ticks,
                               raw.tick_resolution)
    selected_score = _grid_score(active_notes, anchor_ms, anchor_ticks,
                                 raw.tick_resolution)
    direct_score = _grid_score(active_notes, bar_ms, bar_ticks,
                               raw.tick_resolution)
    return min(
        (global_score, global_ms, global_ticks),
        (selected_score, anchor_ms, anchor_ticks),
        (direct_score, bar_ms, bar_ticks),
        key=lambda candidate: candidate[0],
    )[1:]


def _add_submeasure_anchors(raw: RawChart, beat_ms: np.ndarray,
                            beat_ordinals: list[int], bar_ms: np.ndarray,
                            bar_ticks: np.ndarray, meters: list[int],
                            anchor_ms: np.ndarray, anchor_ticks: np.ndarray,
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a clear tempo step that occurs inside one measure.

    Measure averages cannot represent e.g. two beats at 200 BPM followed by
    two at 211 BPM. A split is accepted only when it nearly eliminates the
    beat-period residual and changes the period by at least 5%; ordinary
    60-fps alternation therefore remains a single denoised tempo.
    """
    additions: list[tuple[float, float]] = []
    note_ms = np.array([n.trigger_ms for n in raw.notes], dtype=float)
    for start, meter in enumerate(meters):
        if meter < 4:
            continue
        end = start + 1
        first = beat_ordinals[start]
        last = beat_ordinals[end]
        count = last - first
        if count < 4:
            continue

        start_ms = float(np.interp(bar_ticks[start], anchor_ticks, anchor_ms))
        end_ms = float(np.interp(bar_ticks[end], anchor_ticks, anchor_ms))
        ordinals = np.arange(first, last + 1)
        fraction = (ordinals - first) / count
        corrected = beat_ms[ordinals] + (start_ms - beat_ms[first]) + fraction * (
            (end_ms - beat_ms[last]) - (start_ms - beat_ms[first]))
        periods = np.diff(corrected)
        mean = float(np.mean(periods))
        unsplit_residual = float(np.sum((periods - mean) ** 2))
        if unsplit_residual <= 0:
            continue

        candidates: list[tuple[float, int, float, float]] = []
        for split in range(2, count - 1):
            left, right = periods[:split], periods[split:]
            left_mean, right_mean = float(np.mean(left)), float(np.mean(right))
            residual = (float(np.sum((left - left_mean) ** 2))
                        + float(np.sum((right - right_mean) ** 2)))
            candidates.append((residual, split, left_mean, right_mean))
        if not candidates:
            continue
        residual, split, left_mean, right_mean = min(candidates)
        relative_change = abs(left_mean - right_mean) / (
            0.5 * (left_mean + right_mean))
        if (relative_change < SUBMEASURE_MIN_REL_CHANGE
                or residual / unsplit_residual
                > SUBMEASURE_MAX_RESIDUAL_RATIO):
            continue
        candidate_ms = np.array([start_ms, corrected[split], end_ms])
        candidate_ticks = np.array([
            bar_ticks[start],
            (first + split - beat_ordinals[0]) * raw.tick_resolution,
            bar_ticks[end],
        ])
        local_notes = note_ms[(start_ms <= note_ms) & (note_ms < end_ms)]
        if (_grid_score(local_notes, candidate_ms, candidate_ticks,
                        raw.tick_resolution)[0]
                > _grid_score(local_notes, np.array([start_ms, end_ms]),
                              bar_ticks[[start, end]],
                              raw.tick_resolution)[0]):
            continue
        additions.extend(zip(candidate_ms, candidate_ticks, strict=True))

    if not additions:
        return anchor_ms, anchor_ticks
    by_tick = {float(tick): float(ms)
               for ms, tick in zip(anchor_ms, anchor_ticks, strict=True)}
    by_tick.update((float(tick), float(ms)) for ms, tick in additions)
    ticks = np.array(sorted(by_tick))
    return np.array([by_tick[tick] for tick in ticks]), ticks


def clean_beat_times(beats: list[BeatEvent]) -> np.ndarray:
    """Return a monotonic beat stream without inventing unobserved flashes.

    An isolated 2x interval can be either a missed flash or an intentional
    one-beat tempo change.  Timing data alone cannot distinguish them, so
    inserting a beat would make arbitrary speed changes impossible to retain.
    """
    return np.array(sorted(set(float(b.ms) for b in beats)), dtype=float)


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

    reconstruction: ReconstructResult = reconstruct_barlines(
        raw.barlines, _events(beat_ms, raw.fps),
        last_event_ms=_last_event_ms(raw))
    if len(reconstruction.beat_indices) < 2:
        raise ValueError("not enough on-beat barlines to infer a timeline")

    origin = reconstruction.beat_indices[0]
    bar_ms = np.array([b.ms for b in reconstruction.barlines], dtype=float)
    bar_ticks = np.array([
        (ordinal - origin) * raw.tick_resolution
        for ordinal in reconstruction.beat_indices
    ], dtype=float)
    anchor_ms, anchor_ticks = _select_tempo_anchors(
        raw, bar_ms, bar_ticks, reconstruction.measure_meters)
    anchor_ms, anchor_ticks = _add_submeasure_anchors(
        raw, beat_ms, reconstruction.beat_indices, bar_ms, bar_ticks,
        reconstruction.measure_meters, anchor_ms, anchor_ticks)

    clock = TickClock.from_anchors(
        anchor_ms, anchor_ticks,
        tick_resolution=raw.tick_resolution,
        max_error_ms=0.0,
        min_bpm=raw.min_bpm,
        max_bpm=raw.max_bpm)
    barline_ticks = [int(tick) for tick in bar_ticks]
    return Timeline(
        clock, reconstruction.barlines, barline_ticks,
        reconstruction.time_signature, reconstruction.variants,
        reconstruction.measure_meters)
