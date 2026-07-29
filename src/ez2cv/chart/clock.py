"""
EZ2CV — chart conversion / tick_clock : BPMSegment + TickClock (ms ↔ tick)
===============================================================================
The chart world is tick-based; the raw world is ms-based. Everything that needs
to translate between the two goes through ``TickClock``. There is exactly one
TickClock per song, built from the BPM segments estimated by ``bpm_estimator``.

Why a piecewise-LINEAR model
----------------------------
Tempo can ramp gradually (e.g. a 4-bar accelerando). A constant-BPM model would
approximate the ramp by a stair-step, which leaks small ms→tick errors that
accumulate and push later notes off-grid. A linear BPM segment expresses the
ramp exactly in a closed form (the integral of a linear ticks-per-ms function
is a quadratic in time). Constant BPM is just the slope-zero case, so this
model is a strict superset.

Math (single segment of duration T_ms)
--------------------------------------
    ticks_per_ms(τ) = R · (b0 + (b1 − b0)·τ/T) / 60_000
    ticks(dt) = ∫_0^dt  ticks_per_ms(τ) dτ
              = R/60_000 · ( b0·dt + (b1−b0)·dt² / (2T) )

The constant case collapses to ``R · bpm · dt / 60_000``. We tag a segment with
|Δbpm| < 0.1 BPM as constant (``is_constant``) to dodge near-singular quadratic
inversion in ``tick_to_ms``.

Tick origin convention
----------------------
Tick 0 corresponds to ``measure_zero_ms`` (the moment the first measure line
crosses the judgment line). Pre-anacrusis notes therefore live at NEGATIVE tick.
Construction (``TickClock.from_drafts``) handles the shift internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


# =============================================================================
# BPMSegment
# =============================================================================

@dataclass
class BPMSegment:
    """One piecewise-linear BPM segment, tick-bounded.

    Bounds are in TICKS so the segment list is the canonical representation in
    chart.json. The ms coordinates of each segment live alongside in TickClock
    (the only piece that needs both worlds).
    """
    start_tick: int
    end_tick: int                  # exclusive
    bpm_start: float
    bpm_end: float

    @property
    def is_constant(self) -> bool:
        return abs(self.bpm_end - self.bpm_start) < 0.1

    @property
    def duration_ticks(self) -> int:
        return self.end_tick - self.start_tick

    def duration_ms(self, *, tick_resolution: int = 192) -> float:
        """Derive ms-length from tick-length + average BPM."""
        avg_bpm = 0.5 * (self.bpm_start + self.bpm_end)
        if avg_bpm <= 0:
            return 0.0
        return self.duration_ticks * 60_000.0 / (tick_resolution * avg_bpm)

    def tick_at(self, ms_from_start: float,
                *, tick_resolution: int = 192) -> float:
        """Tick at ``ms_from_start`` ms past the segment's ms-start.

        Closed-form integral. Accepts negative ms (linear continuation).
        """
        if self.is_constant:
            ticks = tick_resolution * self.bpm_start * ms_from_start / 60_000.0
        else:
            T = self.duration_ms(tick_resolution=tick_resolution)
            db = self.bpm_end - self.bpm_start
            ticks = tick_resolution * (
                self.bpm_start * ms_from_start
                + db * ms_from_start * ms_from_start / (2.0 * T)
            ) / 60_000.0
        return self.start_tick + ticks

    def ms_at(self, tick: float,
              *, tick_resolution: int = 192) -> float:
        """Inverse of tick_at: ms-from-start at ``tick``.

        For a linear segment this is a quadratic in ``ms_from_start``; we pick
        the root with the matching sign (positive when ``tick > start_tick``).
        """
        d_tick = tick - self.start_tick
        if self.is_constant:
            return d_tick * 60_000.0 / (tick_resolution * self.bpm_start)

        T = self.duration_ms(tick_resolution=tick_resolution)
        db = self.bpm_end - self.bpm_start
        # (db/(2T)) · t² + b0·t − 60000·d_tick/R = 0
        a = db / (2.0 * T)
        b = self.bpm_start
        c = -60_000.0 * d_tick / tick_resolution
        disc = b * b - 4.0 * a * c
        if disc < 0:
            disc = 0.0
        sqrt_disc = np.sqrt(disc)
        # pick the root continuous with the constant case (t≈c/-b for small a)
        if db > 0:
            return (-b + sqrt_disc) / (2.0 * a)
        return (-b - sqrt_disc) / (2.0 * a)


# =============================================================================
# Drafts
# =============================================================================

@dataclass
class BPMDraft:
    """Pre-tick representation used while building a TickClock."""
    start_ms: float
    end_ms: float
    bpm_start: float
    bpm_end: float

    @property
    def is_constant(self) -> bool:
        return abs(self.bpm_end - self.bpm_start) < 0.1


# =============================================================================
# TickClock
# =============================================================================

class TickClock:
    """Single ms ↔ tick gateway for the whole chart conversion pipeline.

    Construct via :meth:`from_drafts`. After construction, ``segments`` is a
    list of tick-bounded ``BPMSegment``\\s and ``segment_start_ms`` is their
    aligned ms-anchors. Tick 0 corresponds to ``origin_ms``.
    """

    def __init__(self, segments: list[BPMSegment],
                 segment_start_ms: list[float],
                 *, tick_resolution: int = 192):
        if len(segments) != len(segment_start_ms):
            raise ValueError("segments/segment_start_ms length mismatch")
        self.segments = segments
        self.segment_start_ms = segment_start_ms
        self.tick_resolution = tick_resolution

    # ------------------------------------------------------------------ #
    @classmethod
    def from_drafts(cls, drafts: Iterable[BPMDraft], *,
                    origin_ms: float,
                    tick_resolution: int = 192) -> "TickClock":
        """Build a TickClock from ms-domain drafts. Tick 0 = ``origin_ms``."""
        drafts = list(drafts)
        if not drafts:
            raise ValueError("at least one BPMDraft required")

        # Step 1: integrate ticks forward from the first draft (raw frame).
        raw_segments: list[BPMSegment] = []
        start_ms_list: list[float] = []
        cur_tick = 0.0
        for d in drafts:
            duration_ms = d.end_ms - d.start_ms
            if duration_ms <= 0:
                continue
            avg_bpm = 0.5 * (d.bpm_start + d.bpm_end)
            seg_ticks = tick_resolution * avg_bpm * duration_ms / 60_000.0
            raw_segments.append(BPMSegment(
                start_tick=int(round(cur_tick)),
                end_tick=int(round(cur_tick + seg_ticks)),
                bpm_start=d.bpm_start,
                bpm_end=d.bpm_end,
            ))
            start_ms_list.append(d.start_ms)
            cur_tick += seg_ticks

        if not raw_segments:
            raise ValueError("all drafts had non-positive duration")

        # Step 2: find raw tick at origin_ms (continuation OK for out-of-range).
        raw_clock = cls(raw_segments, start_ms_list,
                        tick_resolution=tick_resolution)
        origin_tick = raw_clock.ms_to_tick(origin_ms)

        # Step 3: shift so origin_ms ↔ tick 0.
        shift = int(round(origin_tick))
        shifted = [
            BPMSegment(start_tick=s.start_tick - shift,
                       end_tick=s.end_tick - shift,
                       bpm_start=s.bpm_start,
                       bpm_end=s.bpm_end)
            for s in raw_segments
        ]
        return cls(shifted, start_ms_list, tick_resolution=tick_resolution)

    # ------------------------------------------------------------------ #
    def _seg_for_ms(self, ms: float) -> int:
        """Index of the segment whose ms-range contains ``ms``.

        Out-of-range ms clamps to the first or last segment — the math
        continues linearly for negative ms_from_start (pickup notes).
        """
        sms = self.segment_start_ms
        if ms < sms[0]:
            return 0
        for i in range(len(sms) - 1):
            if sms[i] <= ms < sms[i + 1]:
                return i
        return len(sms) - 1

    def _seg_for_tick(self, tick: float) -> int:
        segs = self.segments
        if tick < segs[0].start_tick:
            return 0
        for i, s in enumerate(segs):
            if s.start_tick <= tick < s.end_tick:
                return i
        return len(segs) - 1

    # ------------------------------------------------------------------ #
    def ms_to_tick(self, ms: float) -> float:
        i = self._seg_for_ms(ms)
        seg = self.segments[i]
        seg_ms = self.segment_start_ms[i]
        return seg.tick_at(ms - seg_ms, tick_resolution=self.tick_resolution)

    def tick_to_ms(self, tick: float) -> float:
        i = self._seg_for_tick(tick)
        seg = self.segments[i]
        seg_ms = self.segment_start_ms[i]
        return seg_ms + seg.ms_at(tick, tick_resolution=self.tick_resolution)
