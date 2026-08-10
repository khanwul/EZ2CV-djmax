"""Measure-adaptive musical-grid snapping.

Ordinary measures use the base 1/4..1/32 vocabulary.  A measure may opt into
1/48, 1/64, 1/96, then 1/192 only when several distinct onsets support the
finer vocabulary and its lower residual pays for a complexity penalty.  This
keeps one noisy note from making every position valid while preserving charts
that genuinely use the game's native fine tick positions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Base 1/N note denominators and progressively finer per-measure vocabularies.
ALLOWED_DENOMS: tuple[int, ...] = (4, 8, 12, 16, 24, 32)
ADAPTIVE_DENOMS: tuple[int, ...] = (48, 64, 96, 192)
MEASURE_GRID_LEVELS: tuple[tuple[int, ...], ...] = tuple(
    ALLOWED_DENOMS + ADAPTIVE_DENOMS[:level]
    for level in range(len(ADAPTIVE_DENOMS) + 1)
)
TRIPLET_DENOMS: frozenset[int] = frozenset({12, 24})

# tunables
DEFAULT_ALPHA = 0.5                    # tick cost per bit of denom complexity
DEFAULT_MAX_TOLERANCE_TICK = 6.0       # 192-tick world: 6 ticks ≈ 1/128
DEFAULT_TRIPLET_CONTEXT_PENALTY = 1.5  # extra cost for a lone triplet
MEASURE_GRID_LEVEL_PENALTY = 4.0
MEASURE_GRID_MIN_OUTLIERS = 2
MEASURE_GRID_ONSET_MERGE_TICK = 1.0


# =============================================================================
# Result type
# =============================================================================

@dataclass(frozen=True)
class SnapResult:
    tick: int                # snapped tick
    label: str               # "1/16" etc., or "off-grid"
    denom: int               # the chosen denominator (matches label)
    off_grid: bool           # True if the snap distance exceeded tolerance
    raw_distance: float      # |raw − snapped|, in ticks
    fine_grid: bool = False  # rescued by a measure vocabulary finer than 1/32
    grid_level: int = 0      # index into MEASURE_GRID_LEVELS
    timing_uncertain: bool = False
    timing_residual_ms: float = 0.0


def grid_distances(raw_ticks, denominators: tuple[int, ...],
                   *, tick_resolution: int = 192) -> np.ndarray:
    """Distance from each raw tick to the nearest position in a vocabulary."""
    ticks = np.asarray(raw_ticks, dtype=float)
    distances = np.full(len(ticks), np.inf)
    for denominator in denominators:
        step = tick_resolution * 4.0 / denominator
        distances = np.minimum(
            distances, np.abs(ticks - np.rint(ticks / step) * step))
    return distances


def choose_measure_grid(raw_ticks, *, tick_resolution: int = 192,
                        max_tolerance_tick: float = DEFAULT_MAX_TOLERANCE_TICK,
                        ) -> int:
    """Return the coarsest well-supported adaptive grid level for one measure."""
    # Chord lanes cross a fraction of a tick apart, but are one timing onset.
    groups: list[list[float]] = []
    for tick in sorted(np.asarray(raw_ticks, dtype=float)):
        if (not groups
                or tick - groups[-1][-1] > MEASURE_GRID_ONSET_MERGE_TICK):
            groups.append([float(tick)])
        else:
            groups[-1].append(float(tick))
    ticks = np.array([float(np.mean(group)) for group in groups])
    if len(ticks) == 0:
        return 0
    base = grid_distances(ticks, ALLOWED_DENOMS,
                          tick_resolution=tick_resolution)
    if np.count_nonzero(base > max_tolerance_tick) < MEASURE_GRID_MIN_OUTLIERS:
        return 0

    choices: list[tuple[float, int]] = []
    for level, denominators in enumerate(MEASURE_GRID_LEVELS):
        distances = grid_distances(ticks, denominators,
                                   tick_resolution=tick_resolution)
        fit = float(np.mean(np.minimum(distances, 24.0) ** 2))
        choices.append((fit + MEASURE_GRID_LEVEL_PENALTY * level, level))
    return min(choices)[1]


def choose_measure_grids(raw_ticks, barline_ticks,
                         *, tick_resolution: int = 192) -> list[int]:
    """Choose one vocabulary level for every interval between barlines."""
    ticks = np.asarray(raw_ticks, dtype=float)
    bars = np.asarray(barline_ticks, dtype=float)
    return [
        choose_measure_grid(ticks[(start <= ticks) & (ticks < end)],
                            tick_resolution=tick_resolution)
        for start, end in zip(bars, bars[1:])
    ]


# =============================================================================
# Core snap
# =============================================================================

def snap_tick(raw_tick: float,
              *, tick_resolution: int = 192,
              alpha: float = DEFAULT_ALPHA,
              max_tolerance_tick: float = DEFAULT_MAX_TOLERANCE_TICK,
              neighbor_denoms: frozenset[int] | None = None,
              triplet_context_penalty: float = DEFAULT_TRIPLET_CONTEXT_PENALTY,
              allowed_denoms: tuple[int, ...] = ALLOWED_DENOMS,
              grid_level: int = 0,
              ) -> SnapResult:
    """Snap one tick. Pass ``neighbor_denoms`` for context-aware triplet bias.

    ``neighbor_denoms`` is the set of denominators chosen for nearby notes
    (e.g. within a measure). If it is empty / None, no context bias applies.
    """
    neighbours_have_triplet = (
        neighbor_denoms is not None
        and any(d in TRIPLET_DENOMS for d in neighbor_denoms)
    )

    best = None       # (cost, dist, denom, snapped_int)
    for n in allowed_denoms:
        grid_ticks = tick_resolution * 4.0 / n        # ticks per 1/n note
        k = round(raw_tick / grid_ticks)
        snapped = k * grid_ticks
        dist = abs(raw_tick - snapped)
        cost = dist + alpha * np.log2(n)
        if n in TRIPLET_DENOMS and not neighbours_have_triplet:
            cost += triplet_context_penalty
        cand = (cost, dist, n, int(round(snapped)))
        if best is None or cand < best:
            best = cand

    _, dist, denom, snapped_int = best
    off_grid = dist > max_tolerance_tick
    base_distance = float(grid_distances(
        [raw_tick], ALLOWED_DENOMS,
        tick_resolution=tick_resolution)[0])
    fine_grid = (grid_level > 0 and not off_grid
                 and base_distance > max_tolerance_tick)
    label = "off-grid" if off_grid else f"1/{denom}"
    return SnapResult(tick=snapped_int, label=label, denom=denom,
                      off_grid=off_grid, raw_distance=dist,
                      fine_grid=fine_grid, grid_level=grid_level)


# =============================================================================
# Helpers used by chart conversion conversion
# =============================================================================

def snap_with_local_context(raw_ticks: list[float],
                            *, tick_resolution: int = 192,
                            window_ticks: int | None = None,
                            **kwargs) -> list[SnapResult]:
    """Snap a sorted list of ticks, using a sliding-window neighbour set.

    The neighbour set for note ``i`` is the denominators chosen for the notes
    within ``window_ticks`` on either side. A first pass without context seeds
    the denominators; a second pass re-snaps each note using its neighbours.
    """
    if not raw_ticks:
        return []
    if window_ticks is None:
        window_ticks = tick_resolution * 2          # 2 beats by default

    # Pass 1: context-free seed
    first = [snap_tick(t, tick_resolution=tick_resolution, **kwargs)
             for t in raw_ticks]

    # Pass 2: re-snap using a neighbour denominator set
    ticks_sorted = list(raw_ticks)
    out: list[SnapResult] = []
    for i, t in enumerate(ticks_sorted):
        lo, hi = i, i
        while lo > 0 and ticks_sorted[i] - ticks_sorted[lo - 1] <= window_ticks:
            lo -= 1
        while (hi < len(ticks_sorted) - 1
               and ticks_sorted[hi + 1] - ticks_sorted[i] <= window_ticks):
            hi += 1
        neigh = frozenset(first[j].denom for j in range(lo, hi + 1) if j != i)
        out.append(snap_tick(t, tick_resolution=tick_resolution,
                             neighbor_denoms=neigh, **kwargs))
    return out


def snap_by_measure(raw_ticks: list[float], barline_ticks: list[int],
                    *, tick_resolution: int = 192
                    ) -> tuple[list[SnapResult], list[int]]:
    """Snap heads with one adaptive vocabulary shared by each measure."""
    ticks = np.asarray(raw_ticks, dtype=float)
    bars = np.asarray(barline_ticks, dtype=float)
    levels = choose_measure_grids(
        ticks, bars, tick_resolution=tick_resolution)
    snaps: list[SnapResult | None] = [None] * len(ticks)

    for measure, (start, end) in enumerate(
            zip(bars, bars[1:])):
        indices = np.flatnonzero((start <= ticks) & (ticks < end))
        if not len(indices):
            continue
        level = levels[measure]
        local = [float(ticks[index]) for index in indices]
        local_snaps = snap_with_local_context(
            local, tick_resolution=tick_resolution,
            allowed_denoms=MEASURE_GRID_LEVELS[level], grid_level=level)
        for index, snap in zip(indices, local_snaps, strict=True):
            snaps[int(index)] = snap

    # Pickup and post-grid notes retain the conservative base vocabulary.
    for index, snap in enumerate(snaps):
        if snap is None:
            snaps[index] = snap_tick(
                float(ticks[index]), tick_resolution=tick_resolution)
    return [snap for snap in snaps if snap is not None], levels


def snap_length(raw_length_ticks: float,
                *, tick_resolution: int = 192,
                **kwargs) -> int:
    """Snap a longnote length. Length 0 stays 0; we only need the tick count."""
    if raw_length_ticks <= 0:
        return 0
    return snap_tick(raw_length_ticks, tick_resolution=tick_resolution,
                     **kwargs).tick
