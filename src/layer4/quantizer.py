"""
EZ2CV — Layer 4 / quantizer : snap raw ticks to a musical grid
===============================================================================
A note's raw tick coordinate (from ms→tick conversion) is a real number near
— but rarely exactly on — one of the allowed grid positions. The quantiser
picks the grid that minimises a cost balancing PROXIMITY against COMPLEXITY:

    cost(grid) = |raw_tick − grid_tick|  +  α · log2(denom)
                                            └── coarse-first regulariser

A note close to two candidates (e.g. 1/16 vs 1/48) snaps to the COARSER one
because the log term penalises larger denominators. Context-awareness adds a
penalty to triplet denominators (1/12, 1/24, 1/48) when no neighbouring note
is on a triplet — a single off-grid note shouldn't drag the whole measure into
a triplet interpretation.

Allowed grid
------------
    {1/4, 1/8, 1/12, 1/16, 1/24, 1/32, 1/48, 1/64}

If the best candidate is still > ``max_tolerance_tick`` away, the note is
flagged as **off-grid** and kept at its rounded tick — the chart records the
fact so Layer 4's sanity check can act on the off-grid ratio (>5 % → re-try
octave / time signature; >5 % still → surface to the user).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# 1/N note denominators allowed for snapping
ALLOWED_DENOMS: tuple[int, ...] = (4, 8, 12, 16, 24, 32, 48, 64)
TRIPLET_DENOMS: frozenset[int] = frozenset({12, 24, 48})

# tunables
DEFAULT_ALPHA = 0.5                    # tick cost per bit of denom complexity
DEFAULT_MAX_TOLERANCE_TICK = 6.0       # 192-tick world: 6 ticks ≈ 1/128
DEFAULT_TRIPLET_CONTEXT_PENALTY = 1.5  # extra cost for a lone triplet


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


# =============================================================================
# Core snap
# =============================================================================

def snap_tick(raw_tick: float,
              *, tick_resolution: int = 192,
              alpha: float = DEFAULT_ALPHA,
              max_tolerance_tick: float = DEFAULT_MAX_TOLERANCE_TICK,
              neighbor_denoms: frozenset[int] | None = None,
              triplet_context_penalty: float = DEFAULT_TRIPLET_CONTEXT_PENALTY,
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
    for n in ALLOWED_DENOMS:
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
    label = "off-grid" if off_grid else f"1/{denom}"
    return SnapResult(tick=snapped_int, label=label, denom=denom,
                      off_grid=off_grid, raw_distance=dist)


# =============================================================================
# Helpers used by Layer4Pipeline
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


def snap_length(raw_length_ticks: float,
                *, tick_resolution: int = 192,
                **kwargs) -> int:
    """Snap a longnote length. Length 0 stays 0; we only need the tick count."""
    if raw_length_ticks <= 0:
        return 0
    return snap_tick(raw_length_ticks, tick_resolution=tick_resolution,
                     **kwargs).tick


# =============================================================================
# CLI: python quantizer.py — show how a few synthetic ticks snap
# =============================================================================

if __name__ == "__main__":
    R = 192
    # one of each denom, off by ~1 tick noise
    samples = [
        0.7,                      # near 1/4 (48 ticks)
        47.4,                     # near 1/4 (snap to 48)
        24.2,                     # near 1/8 (24)
        16.1,                     # near 1/12 (16)
        12.3,                     # near 1/16 (12)
        7.9,                      # near 1/24 (8)
        5.4,                      # near 1/32 (6)
        2.6,                      # near 1/64 (3)
        100.0,                    # exact 1/4 + a bit
    ]
    print(f"=== quantizer demo (R={R}) ===")
    for s in samples:
        r = snap_tick(s, tick_resolution=R)
        print(f"  raw {s:7.3f}  → {r.tick:4d}  {r.label:>9s}  "
              f"dist={r.raw_distance:.2f}  off_grid={r.off_grid}")
