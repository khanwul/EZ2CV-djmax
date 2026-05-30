"""
EZ2CV — Layer 3 / measureline : bar-line (measure-line) detection
===============================================================================
A measure line is the thin horizontal rule EZ2ON draws across the WHOLE
playfield at every measure boundary. Unlike a note it is:

  * full-width   — it spans every key lane at once (a note occupies one lane),
  * thin         — ~1px tall (a note/template is note_height ~22px),
  * marker-less  — a uniform mid-grey rule, no central judgment marker, and far
                   dimmer than a note body (~140 vs ~218 on this skin).

It scrolls at the note speed (so it is SV-affected). The instant its leading
edge reaches the judgment line is the measure boundary — which gives Layer 4
the grid PHASE / song-start offset and, crucially, the MEASURE phase. The POW
LED only gives BEAT phase: it flashes identically on every beat and cannot tell
a downbeat from an off-beat. The measure line can.

Why this is a separate path, not the note path
-----------------------------------------------
The note pipeline (Stage 1/2 + NoteTracker) is per-lane: every structure is
indexed by lane. A measure line is a single GLOBAL object, so forcing it
through the per-lane machinery would mean detecting five partial fragments and
stitching them. Instead this module gets its own detector + tracker that REUSE
the same algorithms — 1-D projection, run-finding, the directional stutter-
aware crossing logic — without the per-lane indexing.

It needs NO new Layer-2 ROI. Stage 1 already computes, per lane, the row-mean
projection (`Stage1Result.projection`). Stage 1 throws a 1px run away as
sub-`min_run_px` noise — its run finder is tuned for ~22px notes — but the
projection SIGNAL still carries the line. This module reads those projections
and looks for a thin row that is lit, at mid-grey brightness, in (almost) every
lane at once.

The "lit" test runs on the 2-row sliding ENERGY SUM of each projection, not the
raw row mean: a 1px line scrolling ~33px/frame straddles two pixel rows and
splits its energy below any single-row gate, so a raw-mean test flickers in and
out and drops whole measures. Summing adjacent rows recombines the split energy
regardless of straddle phase (see measureline_detection_findings).

Discriminators (a measure line vs. the things it could be confused with):
  * a full chord of notes  -> also full-width, but ~22px THICK     -> thickness
  * the judgment bar       -> full-width, ~4-6px, fixed at line_y  -> thickness
  * staggered note edges   -> a thin coincident sliver can occur,  -> brightness
                              but it is note-bright (~218), not       window
                              the line's dim grey (~140)

Output: BarlineEvent objects, ms-based. Tick/measure interpretation is Layer 4.

NOTE: the detector's tunables (min/max brightness, max thickness, lane slack)
live in skin.toml [measure_line] and are resolved by calibration.py into
``cal.measure_line`` — exactly like the note thresholds. Each is still exposed
as a constructor kwarg that OVERRIDES the config, so the module stays
self-contained and easy to sweep in a test.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from layer1.calibration import Calibration
from layer3.stage1 import Stage1Result, _find_runs
from layer3.tracking import ScrollSpeedEstimator


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class MeasureLineDetection:
    """One per-frame measure-line sighting (before temporal dedup)."""
    frame_index: int
    y_center: float            # full-frame y of the thin band's centre
    thickness: int             # band height in px
    strength: float            # mean brightness across lanes within the band


@dataclass
class TrackedLine:
    """One measure line followed across frames."""
    id: int
    trajectory: list           # [(frame, y_center, strength), ...]
    last_seen: int
    crossed: bool = False

    @property
    def last_y(self) -> float:
        return self.trajectory[-1][1]


@dataclass
class BarlineEvent:
    """A measure line crossing the judgment line — one measure boundary."""
    cross_frame: float         # fractional frame index of the crossing
    ms: float                  # crossing time in milliseconds
    strength: float
    extrapolated: bool = False


# =============================================================================
# Detector
# =============================================================================

class MeasureLineDetector:
    """Per-frame measure-line detector. Stateless across frames.

    Consumes the per-lane Stage 1 results (for their `projection` arrays) and
    reports any thin, mid-grey, full-width band.
    """

    def __init__(self, cal: Calibration, *,
                 lit_energy_threshold: float | None = None,
                 min_brightness: float | None = None,
                 max_brightness: float | None = None,
                 max_thickness: int | None = None,
                 min_lanes: int | None = None):
        """All tunables resolve from skin.toml [measure_line] via
        ``cal.measure_line`` (see that section for the rationale of each value).
        Pass a kwarg only to OVERRIDE the config — handy for a parameter sweep.

        min_lanes defaults to ``key_count - measure_line.lane_slack``: a band
        must span (almost) every lane, with `lane_slack` lanes of give for a
        split frame or a note-occluded lane.
        """
        ml = cal.measure_line
        self.cal = cal
        self.lit_energy_threshold = (ml.lit_energy_threshold
                                     if lit_energy_threshold is None
                                     else lit_energy_threshold)
        self.min_brightness = (ml.min_brightness if min_brightness is None
                               else min_brightness)
        self.max_brightness = (ml.max_brightness if max_brightness is None
                               else max_brightness)
        self.max_thickness = (ml.max_thickness if max_thickness is None
                              else max_thickness)
        self.min_lanes = (max(2, cal.key_count - ml.lane_slack)
                          if min_lanes is None else min_lanes)

    # ------------------------------------------------------------------ #
    def detect_frame(self, s1_results: list[Stage1Result]
                     ) -> list[MeasureLineDetection]:
        """Find every measure-line band in one frame's Stage 1 output.

        The "lit" test runs on the 2-row sliding energy SUM of each lane's
        projection, not the raw row mean. A 1px line scrolling fast straddles
        two pixel rows and splits its energy below any single-row gate (the
        flicker that dropped whole measures); summing adjacent rows recovers the
        full energy regardless of straddle phase. Long-note bodies survive the
        sum too but are still rejected downstream by ``max_thickness``.
        """
        if not s1_results:
            return []

        # per-lane row-mean projections, stacked: (n_lanes, H)
        projs = np.stack([r.projection for r in s1_results])
        # 2-row sliding energy sum recombines a sub-pixel-split thin line.
        # energy[:, i] covers projection rows i and i+1.
        energy = projs[:, :-1] + projs[:, 1:]              # (n_lanes, H-1)
        lit = energy > self.lit_energy_threshold           # (n_lanes, H-1)
        coincidence = lit.sum(axis=0)                      # (H-1,) lanes lit / row
        full_width = coincidence >= self.min_lanes

        origin = s1_results[0].roi_y_origin
        fidx = s1_results[0].frame_index

        out: list[MeasureLineDetection] = []
        for s, e in _find_runs(full_width):
            if (e - s) > self.max_thickness:
                continue                                   # too thick -> not a line
            # energy run [s, e) spans projection rows s .. e (inclusive).
            strength = float(projs[:, s:e + 1].mean())
            if not (self.min_brightness < strength < self.max_brightness):
                continue                                   # note-bright -> reject
            out.append(MeasureLineDetection(
                frame_index=fidx,
                y_center=(s + e) / 2.0 + origin,
                thickness=e - s + 1,
                strength=strength,
            ))
        return out


# =============================================================================
# Tracker
# =============================================================================

class MeasureLineTracker:
    """Temporal dedup + judgment-line crossing for measure lines.

    A measure line is detected on ~25 frames as it scrolls; this collapses
    those into ONE BarlineEvent at the sub-frame moment it reaches the line.
    Structurally a stripped-down NoteTracker: no lanes, no types, no longnote
    pairing, and (since spacing >> playfield height) usually 0-1 lines on
    screen at once. It reuses NoteTracker's directional, stutter-aware gate and
    its interpolate/extrapolate crossing logic.

    Crossing reference is `line_y`, not `trigger_template_y_top`: a note is
    tracked by its 22px template's top edge, but a 1px line simply crosses when
    its own centre reaches the judgment line.
    """

    def __init__(self, cal: Calibration, *,
                 max_stale_frames: int = 4,
                 up_jitter_px: float = 6.0,
                 down_jitter_px: float = 16.0):
        self.cal = cal
        self.line_y = cal.line_y
        self.fps = cal.fps
        self.max_stale = max_stale_frames
        self.up_jitter = up_jitter_px
        self.down_jitter = down_jitter_px

        self.speed = ScrollSpeedEstimator(cal)
        self._lines: list[TrackedLine] = []
        self._next_id = 0

    # ------------------------------------------------------------------ #
    def step(self, frame_index: int,
             detections: list[MeasureLineDetection]) -> list[BarlineEvent]:
        """Advance one frame; return any new measure-line crossings."""
        speed = self.speed.speed                   # speed from the PREVIOUS frame
        events: list[BarlineEvent] = []
        free = list(detections)

        # --- associate: all valid (dist, line, detection) pairs, nearest-first
        pairs = []
        for ln in self._lines:
            if ln.last_seen == frame_index:
                continue
            for d in free:
                dist = self._gate_dist(ln, d, frame_index, speed)
                if dist is not None:
                    pairs.append((dist, ln, d))
        pairs.sort(key=lambda p: p[0])
        used_l, used_d = set(), set()
        for dist, ln, d in pairs:
            if id(ln) in used_l or id(d) in used_d:
                continue
            ln.trajectory.append((frame_index, d.y_center, d.strength))
            ln.last_seen = frame_index
            used_l.add(id(ln))
            used_d.add(id(d))

        # --- unassociated detections -> new tracked lines -----------------
        for d in free:
            if id(d) in used_d:
                continue
            self._lines.append(TrackedLine(
                id=self._next_id,
                trajectory=[(frame_index, d.y_center, d.strength)],
                last_seen=frame_index))
            self._next_id += 1

        # --- crossings (interpolation) ------------------------------------
        for ln in self._lines:
            if ln.crossed:
                continue
            ev = self._check_crossing(ln)
            if ev is not None:
                ln.crossed = True
                events.append(ev)

        # --- prune; extrapolate near-line un-crossed lines ----------------
        survivors = []
        for ln in self._lines:
            if frame_index - ln.last_seen <= self.max_stale:
                survivors.append(ln)
            elif not ln.crossed:
                ev = self._extrapolate(ln)
                if ev is not None:
                    events.append(ev)
        self._lines = survivors

        # --- refresh global speed (only lines still above the line) -------
        moving = [ln for ln in self._lines if ln.last_y < self.line_y]
        self.speed.update(moving)
        return events

    # ------------------------------------------------------------------ #
    def flush(self) -> list[BarlineEvent]:
        """End-of-video: extrapolate any un-crossed line still near the line."""
        events = []
        for ln in self._lines:
            if not ln.crossed:
                ev = self._extrapolate(ln)
                if ev is not None:
                    events.append(ev)
        self._lines = []
        return events

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _gate_dist(self, ln: TrackedLine, d: MeasureLineDetection,
                   frame_index: int, speed: float) -> float | None:
        """Distance to prediction if d is inside ln's directional gate, else None.

        Measure lines only descend; the gate reaches from a little above the
        last position (jitter) down past one predicted step plus one stutter
        step — the same stutter-aware shape NoteTracker uses.
        """
        elapsed = frame_index - ln.last_seen
        lo = ln.last_y - self.up_jitter
        hi = ln.last_y + elapsed * speed + speed + self.down_jitter
        pred = ln.last_y + elapsed * speed
        if lo <= d.y_center <= hi:
            return abs(d.y_center - pred)
        return None

    def _check_crossing(self, ln: TrackedLine) -> BarlineEvent | None:
        """Emit if the trajectory straddles the judgment line (interpolation)."""
        traj = ln.trajectory
        for i in range(len(traj) - 1):
            fa, ya, sa = traj[i]
            fb, yb, sb = traj[i + 1]
            if ya < self.line_y <= yb and yb != ya:
                frac = (self.line_y - ya) / (yb - ya)
                cf = fa + frac * (fb - fa)
                return BarlineEvent(cf, cf / self.fps * 1000.0, (sa + sb) / 2)
        return None

    def _extrapolate(self, ln: TrackedLine) -> BarlineEvent | None:
        """Fallback: project a line that vanished into the judgment bar forward.

        A measure line merges with the (thicker) judgment bar right at the
        crossing and stops being detected as a thin band — exactly the way a
        tap merges with the bar. Project the last sighting forward on the
        global speed, but only if it died within reach of the line.
        """
        if len(ln.trajectory) < 2:
            return None
        fb, yb, sb = ln.trajectory[-1]
        if yb >= self.line_y:
            return None
        sp = self.speed.speed
        if sp <= 0 or (self.line_y - yb) > 3 * sp:         # died too far short
            return None
        cf = fb + (self.line_y - yb) / sp
        return BarlineEvent(cf, cf / self.fps * 1000.0, sb, extrapolated=True)


# =============================================================================
# CLI: python measureline.py [config/song.toml]
# =============================================================================

if __name__ == "__main__":
    from layer2.preprocessor import Preprocessor
    from layer3.stage1 import ProjectionDetector

    cfg = sys.argv[1] if len(sys.argv) > 1 else "config/song.toml"
    pre = Preprocessor.from_config(cfg)
    s1 = ProjectionDetector(pre.cal)
    mld = MeasureLineDetector(pre.cal)
    mlt = MeasureLineTracker(pre.cal)

    barlines: list[BarlineEvent] = []
    for pf in pre:
        s1r = s1.detect_frame(pf)
        barlines.extend(mlt.step(pf.frame_index, mld.detect_frame(s1r)))
    barlines.extend(mlt.flush())
    barlines.sort(key=lambda e: e.cross_frame)

    print(f"=== Layer 3 measure-line detection: {len(barlines)} bar lines ===")
    if len(barlines) >= 2:
        iv = np.diff([b.cross_frame for b in barlines])
        extr = sum(1 for b in barlines if b.extrapolated)
        print(f"interval (frames): median={np.median(iv):.1f}  "
              f"mean={iv.mean():.2f}  std={iv.std():.2f}  "
              f"min={iv.min():.1f}  max={iv.max():.1f}")
        print(f"extrapolated crossings: {extr}/{len(barlines)}")
    print(f"\nfirst 8 bar lines:")
    for b in barlines[:8]:
        flag = " [extrap]" if b.extrapolated else ""
        print(f"  f{b.cross_frame:8.2f}  {b.ms:9.1f}ms  "
              f"strength={b.strength:.0f}{flag}")
