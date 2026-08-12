"""
EZ2CV — chart conversion / tick_clock : BPMSegment + TickClock (ms ↔ tick)
===============================================================================
The chart world is tick-based; the raw world is ms-based. Everything that needs
to translate between the two goes through ``TickClock``. There is exactly one
clock per song, built from the anchors selected by timeline inference.

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
Construction from selected anchors keeps every retained timing landmark on an
exact tick unless a configured BPM bound is applied within its explicit timing
tolerance. ``from_drafts`` remains for callers that already own a BPM model.
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
    chart.json. TickClock derives ms coordinates from these segments and the
    serialized tick-zero offset.
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
        # This stable form selects the root continuous with the constant case
        # for both increasing and decreasing ramps and avoids cancellation.
        return 2.0 * c / (-b - sqrt_disc)


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

    ``segments`` and ``tick_zero_ms`` are the only timing truth, exactly
    matching chart.json. Segment start milliseconds are derived from them.
    """

    def __init__(self, segments: list[BPMSegment], *, tick_zero_ms: float,
                 tick_resolution: int = 192):
        if not segments:
            raise ValueError("at least one BPM segment required")
        if any(s.end_tick <= s.start_tick or s.bpm_start <= 0 or s.bpm_end <= 0
               for s in segments):
            raise ValueError("invalid BPM segment")
        if any(left.end_tick != right.start_tick
               for left, right in zip(segments, segments[1:])):
            raise ValueError("BPM segments must be contiguous")
        self.segments = segments
        self.tick_zero_ms = float(tick_zero_ms)
        self.tick_resolution = tick_resolution
        self.bpm_bound_adjustments: list[float] = []

        zero_segment = self._seg_for_tick(0)
        starts = [0.0] * len(segments)
        starts[zero_segment] = (
            self.tick_zero_ms
            - segments[zero_segment].ms_at(0, tick_resolution=tick_resolution))
        for i in range(zero_segment + 1, len(segments)):
            starts[i] = starts[i - 1] + segments[i - 1].duration_ms(
                tick_resolution=tick_resolution)
        for i in range(zero_segment - 1, -1, -1):
            starts[i] = starts[i + 1] - segments[i].duration_ms(
                tick_resolution=tick_resolution)
        self.segment_start_ms = starts

    @classmethod
    def from_anchors(cls, anchor_ms: Iterable[float],
                     anchor_ticks: Iterable[float], *,
                     tick_resolution: int = 192,
                     max_error_ms: float = 20.0,
                     min_bpm: float = 0.0,
                     max_bpm: float = 0.0,
                     bound_tolerance_ms: float = 0.0) -> "TickClock":
        """Build a canonical piecewise-constant clock from measured anchors.

        Ramer-Douglas-Peucker removes redundant collinear anchors. Conversion
        and serialized BPM segments use the same retained knots, so tempo
        steps cannot accumulate timing drift. An out-of-range span is rejected
        unless the bounded replacement fits within ``bound_tolerance_ms``.
        """
        ms = np.asarray(list(anchor_ms), dtype=float)
        ticks = np.asarray(list(anchor_ticks), dtype=float)
        if len(ms) < 2 or len(ms) != len(ticks):
            raise ValueError("at least two aligned clock anchors required")
        if np.any(np.diff(ms) <= 0) or np.any(np.diff(ticks) <= 0):
            raise ValueError("clock anchors must be strictly increasing")

        spans: list[tuple[int, int]] = []
        stack = [(0, len(ms) - 1)]
        while stack:
            start, end = stack.pop()
            if end - start <= 1:
                spans.append((start, end))
                continue
            fraction = ((ticks[start:end + 1] - ticks[start])
                        / (ticks[end] - ticks[start]))
            fitted = ms[start] + fraction * (ms[end] - ms[start])
            residual = np.abs(ms[start:end + 1] - fitted)
            split = start + int(np.argmax(residual))
            if residual.max() <= max_error_ms or split in (start, end):
                spans.append((start, end))
            else:
                stack.extend(((split, end), (start, split)))
        spans.sort()

        segments: list[BPMSegment] = []
        starts: list[float] = []
        adjustments: list[float] = []
        for start, end in spans:
            start_tick, end_tick = int(round(ticks[start])), int(round(ticks[end]))
            bpm = ((end_tick - start_tick) * 60_000.0
                   / (tick_resolution * (ms[end] - ms[start])))
            bounded = max(min_bpm, bpm) if min_bpm > 0 else bpm
            bounded = min(max_bpm, bounded) if max_bpm > 0 else bounded
            if bounded != bpm:
                observed_ms = ms[end] - ms[start]
                bounded_ms = ((end_tick - start_tick) * 60_000.0
                              / (tick_resolution * bounded))
                residual_ms = abs(bounded_ms - observed_ms)
                if residual_ms > bound_tolerance_ms:
                    raise ValueError(
                        f"inferred BPM {bpm:.4f} outside configured range "
                        f"[{min_bpm:.4f}, {max_bpm:.4f}] by "
                        f"{residual_ms:.3f} ms")
                bpm = bounded
                adjustments.append(residual_ms)
            segments.append(BPMSegment(start_tick, end_tick, bpm, bpm))
            starts.append(float(ms[start]))

        tick_zero_ms = starts[0] + segments[0].ms_at(
            0, tick_resolution=tick_resolution)
        clock = cls(segments, tick_zero_ms=tick_zero_ms,
                    tick_resolution=tick_resolution)
        clock.bpm_bound_adjustments = adjustments
        return clock

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
            cur_tick += seg_ticks

        if not raw_segments:
            raise ValueError("all drafts had non-positive duration")

        # Step 2: find raw tick at origin_ms (continuation OK for out-of-range).
        raw_clock = cls(raw_segments, tick_zero_ms=drafts[0].start_ms,
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
        return cls(shifted, tick_zero_ms=raw_clock.tick_to_ms(shift),
                   tick_resolution=tick_resolution)

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
