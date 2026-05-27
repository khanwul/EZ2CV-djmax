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

COINCIDENCE_METHODS = ("lit", "in_window")
STRENGTH_METHODS    = ("lane_mean", "lane_median", "column_median")


class MeasureLineDetector:
    """Per-frame measure-line detector. Stateless across frames.

    Consumes the per-lane Stage 1 results (for their `projection` arrays) and
    reports any thin, mid-grey, full-width band.

    Two algorithm knobs (orthogonal — see the design discussion in
    ``doc/measure-line-detection.md`` and the inline notes below):

    ``coincidence_method`` — how a row qualifies as a CANDIDATE thin band:
      * ``"lit"``       — count lanes whose row-mean exceeds min_brightness.
                          Original behaviour. A note-bright lane counts as lit
                          (raising coincidence but also dragging strength up).
      * ``"in_window"`` — count lanes whose row-mean is INSIDE
                          [min_brightness, max_brightness]. A lane occluded by
                          a note no longer counts toward coincidence, so the
                          gate isn't "spent" by note-bright lanes.

    ``strength_method`` — how to score the candidate's brightness against the
    [min_brightness, max_brightness] mid-grey window:
      * ``"lane_mean"``     — mean over all (lane, row) cells in the band.
                              Original behaviour. ONE note-bright lane can drag
                              the whole strength past max_brightness.
      * ``"lane_median"``   — median across the 5 per-lane row-means of the
                              band. Robust to 1-2 note-occluded lanes.
      * ``"column_median"`` — median over per-column means inside the band,
                              computed on the FULL-WIDTH `measure_roi`
                              (~5*lane_width samples per band, vs 5 for the
                              lane variants). Robust to chord-occlusion at the
                              measure-line row provided notes don't fill the
                              lane width edge-to-edge.

    Defaults are the baseline (`lit` + `lane_mean`) — this preserves the
    pre-validation behaviour as the production path. Pass kwargs to A/B
    against alternative coincidence/strength rules; the validation harness in
    the CLI (`--compare`) sweeps the full grid.
    """

    def __init__(self, cal: Calibration, *,
                 min_brightness: float | None = None,
                 max_brightness: float | None = None,
                 max_thickness: int | None = None,
                 min_lanes: int | None = None,
                 coincidence_method: str = "lit",
                 strength_method: str = "lane_mean"):
        """All tunables resolve from skin.toml [measure_line] via
        ``cal.measure_line`` (see that section for the rationale of each value).
        Pass a kwarg only to OVERRIDE the config — handy for a parameter sweep.

        min_lanes defaults to ``key_count - measure_line.lane_slack``: a band
        must span (almost) every lane, with `lane_slack` lanes of give for a
        split frame or a note-occluded lane.
        """
        if coincidence_method not in COINCIDENCE_METHODS:
            raise ValueError(f"coincidence_method must be one of "
                             f"{COINCIDENCE_METHODS}, got {coincidence_method!r}")
        if strength_method not in STRENGTH_METHODS:
            raise ValueError(f"strength_method must be one of "
                             f"{STRENGTH_METHODS}, got {strength_method!r}")
        ml = cal.measure_line
        self.cal = cal
        self.min_brightness = (ml.min_brightness if min_brightness is None
                               else min_brightness)
        self.max_brightness = (ml.max_brightness if max_brightness is None
                               else max_brightness)
        self.max_thickness = (ml.max_thickness if max_thickness is None
                              else max_thickness)
        # min_lanes default depends on the coincidence semantics:
        #   * "lit"       — note-bright lanes count too, so the line still
        #                   reads (key_count - lane_slack)/key_count lanes lit.
        #   * "in_window" — note-bright lanes DROP OUT of the count, so a chord
        #                   of N notes on the measure-line row consumes N of
        #                   the available lanes. Use a majority gate
        #                   (ceil(key_count/2)) so up to floor(key_count/2)
        #                   lanes can be note-occluded and the candidate row
        #                   still passes. Without this auto-relax in_window is
        #                   strictly worse than lit at the same min_lanes.
        if min_lanes is None:
            if coincidence_method == "in_window":
                self.min_lanes = max(2, (cal.key_count + 1) // 2)
            else:
                self.min_lanes = max(2, cal.key_count - ml.lane_slack)
        else:
            self.min_lanes = min_lanes
        self.coincidence_method = coincidence_method
        self.strength_method = strength_method

    # ------------------------------------------------------------------ #
    def detect_frame(self, s1_results: list[Stage1Result],
                     measure_roi: np.ndarray | None = None
                     ) -> list[MeasureLineDetection]:
        """Find every measure-line band in one frame's Stage 1 output.

        ``measure_roi`` is the full-playfield-width single-channel crop from
        ``PreprocessedFrame.measure_roi``. Required when
        ``strength_method == "column_median"``; ignored otherwise. Its y axis
        must match the Stage 1 projection y axis (both are sliced at
        playfield_top:playfield_bottom — Layer 2 guarantees this).
        """
        if not s1_results:
            return []
        if self.strength_method == "column_median" and measure_roi is None:
            raise ValueError("strength_method='column_median' requires "
                             "measure_roi (pass pf.measure_roi)")

        # per-lane row-mean projections, stacked: (n_lanes, H)
        projs = np.stack([r.projection for r in s1_results])
        if self.coincidence_method == "lit":
            qualifies = projs > self.min_brightness        # (n_lanes, H)
        else:  # in_window — note-bright lanes are excluded from the count
            qualifies = ((projs >= self.min_brightness) &
                         (projs <= self.max_brightness))
        coincidence = qualifies.sum(axis=0)                # (H,) lanes / row
        full_width = coincidence >= self.min_lanes

        origin = s1_results[0].roi_y_origin
        fidx = s1_results[0].frame_index

        out: list[MeasureLineDetection] = []
        for s, e in _find_runs(full_width):
            if (e - s) > self.max_thickness:
                continue                                   # too thick -> not a line
            if self.strength_method == "lane_mean":
                strength = float(projs[:, s:e].mean())
            elif self.strength_method == "lane_median":
                # 5 lane row-means of the band; median picks the mid-grey
                # lanes even if one or two are dragged up by a note.
                strength = float(np.median(projs[:, s:e].mean(axis=1)))
            else:  # column_median — robust to chord-occlusion at this row
                band = measure_roi[s:e, :]                 # (band_h, field_w)
                strength = float(np.median(band.mean(axis=0)))
            if not (self.min_brightness < strength < self.max_brightness):
                continue                                   # note-bright -> reject
            out.append(MeasureLineDetection(
                frame_index=fidx,
                y_center=(s + e - 1) / 2.0 + origin,
                thickness=e - s,
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

        # Diagnostic counters — incremented as lines move through the tracker.
        # The 4 `dropped_*` reasons cover every path that loses a line without
        # emitting a BarlineEvent; their sum equals "lines we saw but lost",
        # which is the true Layer-3 miss count assuming the detector did see
        # them (vs. the line never being detected at all — separate counter).
        self.diagnostics = {
            "interpolated": 0,           # _check_crossing emitted (line crossed
                                         # judgment line within trajectory)
            "extrapolated": 0,           # _extrapolate emitted (projected from
                                         # last sighting near the line)
            "recovered_long_traj": 0,    # _extrapolate emitted because of the
                                         # relaxed (8*sp) gate when traj>=3 —
                                         # subset of `extrapolated`, would have
                                         # been dropped_too_far under the
                                         # strict 3*sp limit. Targets the
                                         # variable-BPM miss mode (GEHENNA).
            "dropped_short_traj": 0,     # died with len(trajectory) < 2 — only
                                         # one sighting, can't project
            "dropped_too_far": 0,        # died at (line_y - yb) > max_dist*sp
                                         # even with the relaxed gate
            "dropped_below_line": 0,     # already past the line, never crossed
            "dropped_zero_speed": 0,     # speed estimator returned <= 0
        }

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
                self.diagnostics["interpolated"] += 1

        # --- prune; extrapolate near-line un-crossed lines ----------------
        survivors = []
        for ln in self._lines:
            if frame_index - ln.last_seen <= self.max_stale:
                survivors.append(ln)
            elif not ln.crossed:
                ev = self._extrapolate(ln)
                if ev is not None:
                    events.append(ev)
                    self.diagnostics["extrapolated"] += 1
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
                    self.diagnostics["extrapolated"] += 1
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

    # Extrapolation distance multipliers (× scroll speed in px/frame). A line
    # whose last sighting is within `limit * sp` of the judgment line gets
    # projected forward; further than that we abandon it. The limit relaxes
    # with trajectory length:
    #
    #   * `_EXTRAP_DIST_SHORT` (2 sightings) — strict. A 2-frame trajectory
    #     could be a noise blob; only trust it if it died right next to the
    #     line.
    #   * `_EXTRAP_DIST_LONG`  (3+ sightings) — relaxed. A line tracked for 3+
    #     frames has confirmed its identity (consistent motion). Extrapolating
    #     further is justified, and it's the only way to recover the
    #     variable-BPM miss mode in songs like GEHENNA where slow-scroll
    #     sections push the strict gate below 1 frame of headroom.
    _EXTRAP_DIST_SHORT = 3.0
    _EXTRAP_DIST_LONG  = 8.0

    def _extrapolate(self, ln: TrackedLine) -> BarlineEvent | None:
        """Fallback: project a line that vanished into the judgment bar forward.

        A measure line merges with the (thicker) judgment bar right at the
        crossing and stops being detected as a thin band — exactly the way a
        tap merges with the bar. Project the last sighting forward on the
        global speed, but only if it died within reach of the line.

        Each early-return path bumps a diagnostic counter so the CLI can
        report WHY measure lines are being lost (single-sighting lines,
        died-too-far, etc.) — the baseline misses ~30% of true measures
        and this is the way to localise where they vanish.
        """
        if len(ln.trajectory) < 2:
            self.diagnostics["dropped_short_traj"] += 1
            return None
        fb, yb, sb = ln.trajectory[-1]
        if yb >= self.line_y:
            self.diagnostics["dropped_below_line"] += 1
            return None
        sp = self.speed.speed
        if sp <= 0:
            self.diagnostics["dropped_zero_speed"] += 1
            return None
        max_dist_mult = (self._EXTRAP_DIST_LONG if len(ln.trajectory) >= 3
                         else self._EXTRAP_DIST_SHORT)
        gap = self.line_y - yb
        if gap > max_dist_mult * sp:                       # died too far short
            self.diagnostics["dropped_too_far"] += 1
            return None
        if gap > self._EXTRAP_DIST_SHORT * sp:
            # Made it only because of the relaxed long-trajectory gate; record
            # for diagnostics so we can attribute recall gains to this change.
            self.diagnostics["recovered_long_traj"] += 1
        cf = fb + gap / sp
        return BarlineEvent(cf, cf / self.fps * 1000.0, sb, extrapolated=True)


# =============================================================================
# CLI: python measureline.py [config/song.toml] [--compare | --coincidence X --strength Y]
# =============================================================================

def _run_one(pre, s1, *, coincidence_method: str, strength_method: str
             ) -> tuple[list[BarlineEvent], dict]:
    """Run a fresh detector+tracker pass against an already-loaded video.

    Returns (barlines, tracker_diagnostics). The diagnostics dict explains
    where lost measure lines went — see MeasureLineTracker.diagnostics.

    Decodes the video once per call — the comparison harness wraps `pre` with
    a cache so the 4 configs share a single decode.
    """
    mld = MeasureLineDetector(pre.cal,
                              coincidence_method=coincidence_method,
                              strength_method=strength_method)
    mlt = MeasureLineTracker(pre.cal)
    barlines: list[BarlineEvent] = []
    for pf in pre:
        s1r = s1.detect_frame(pf)
        barlines.extend(mlt.step(
            pf.frame_index, mld.detect_frame(s1r, pf.measure_roi)))
    barlines.extend(mlt.flush())
    barlines.sort(key=lambda e: e.cross_frame)
    return barlines, dict(mlt.diagnostics)


def _summarise(barlines: list[BarlineEvent], true_measures: int | None = None
               ) -> dict:
    """Return summary stats used by the comparison table.

    `max_gap_ratio = max_interval / median_interval` is the strongest miss
    indicator on constant-BPM songs: a value near 2.0 means one whole measure
    was skipped, ~3.0 means two measures, etc. With variable BPM (e.g. GEHENNA
    111-222, JUSTITIA 80-290) the metric is much noisier — `recall` against
    a known true measure count is the authoritative score.
    """
    n = len(barlines)
    extr = sum(1 for b in barlines if b.extrapolated)
    out = {"n": n, "extr": extr, "med": float("nan"),
           "std": float("nan"), "max": float("nan"),
           "gap_ratio": float("nan"), "recall": float("nan")}
    if n >= 2:
        iv = np.diff([b.cross_frame for b in barlines])
        med = float(np.median(iv))
        out["med"] = med
        out["std"] = float(iv.std())
        out["max"] = float(iv.max())
        out["gap_ratio"] = float(iv.max() / med) if med > 0 else float("nan")
    if true_measures is not None and true_measures > 0:
        out["recall"] = n / true_measures
    return out


class _CachedPreprocessor:
    """Decode-once / replay-many wrapper around Preprocessor.

    The benchmark harness reuses the same video across 4 detector configs;
    PreprocessedFrame objects are pure data so we cache them on the first
    iteration and replay from memory after. Memory cost is O(frames) but the
    payload per frame is small (a few small ROIs), and the alternative is 4
    cv2 decode passes per song.
    """
    def __init__(self, pre):
        self._pre = pre
        self.cal = pre.cal
        self._cache: list | None = None

    def __iter__(self):
        if self._cache is None:
            self._cache = []
            for pf in self._pre:
                self._cache.append(pf)
                yield pf
        else:
            yield from self._cache


if __name__ == "__main__":
    from layer2.preprocessor import Preprocessor
    from layer3.stage1 import ProjectionDetector

    args = sys.argv[1:]
    cfg = "config/Dream Walker.toml"
    compare = False
    coincidence = "lit"
    strength = "lane_mean"
    true_measures: int | None = None
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--compare":
            compare = True
        elif a == "--coincidence":
            i += 1; coincidence = args[i]
        elif a == "--strength":
            i += 1; strength = args[i]
        elif a == "--true-measures":
            i += 1; true_measures = int(args[i])
        elif not a.startswith("--"):
            cfg = a
        i += 1

    pre = _CachedPreprocessor(Preprocessor.from_config(cfg))
    s1 = ProjectionDetector(pre.cal)

    def _fmt_diag(d: dict) -> str:
        return (f"interp={d['interpolated']:>3}  "
                f"extr={d['extrapolated']:>3}  "
                f"recov={d['recovered_long_traj']:>3}  "
                f"short={d['dropped_short_traj']:>3}  "
                f"far={d['dropped_too_far']:>3}  "
                f"below={d['dropped_below_line']:>3}  "
                f"sp0={d['dropped_zero_speed']:>3}")

    if compare:
        configs = [
            ("baseline",  "lit",       "lane_mean"),
            ("sol12",     "in_window", "lane_median"),
            ("sol123",    "in_window", "column_median"),
            ("sol3-only", "lit",       "column_median"),
        ]
        print(f"=== {cfg} ===")
        if true_measures is not None:
            print(f"true measures: {true_measures}")
        header = (f"{'config':<10}  {'coin':<10}  {'strength':<14}  "
                  f"{'N':>4}  {'recall':>7}  "
                  f"{'interp':>6}  {'extr':>4}  {'recov':>5}  "
                  f"{'short':>5}  {'far':>4}  "
                  f"{'below':>5}  {'sp0':>3}")
        print(header)
        for label, c, sgy in configs:
            barlines, diag = _run_one(pre, s1, coincidence_method=c,
                                      strength_method=sgy)
            st = _summarise(barlines, true_measures=true_measures)
            recall_s = (f"{st['recall']*100:>6.1f}%" if true_measures
                        else "    —  ")
            print(f"{label:<10}  {c:<10}  {sgy:<14}  "
                  f"{st['n']:>4}  {recall_s:>7}  "
                  f"{diag['interpolated']:>6}  {diag['extrapolated']:>4}  "
                  f"{diag['recovered_long_traj']:>5}  "
                  f"{diag['dropped_short_traj']:>5}  "
                  f"{diag['dropped_too_far']:>4}  "
                  f"{diag['dropped_below_line']:>5}  "
                  f"{diag['dropped_zero_speed']:>3}")
        sys.exit(0)

    # single-config mode
    barlines, diag = _run_one(pre, s1, coincidence_method=coincidence,
                              strength_method=strength)
    st = _summarise(barlines, true_measures=true_measures)
    print(f"=== Layer 3 measure-line detection: {st['n']} bar lines "
          f"(coincidence={coincidence}, strength={strength}) ===")
    if true_measures is not None:
        print(f"recall vs true ({true_measures}): {st['recall']*100:.1f}%")
    if st['n'] >= 2:
        print(f"interval (frames): median={st['med']:.1f}  "
              f"std={st['std']:.2f}  max={st['max']:.1f}  "
              f"gap_ratio={st['gap_ratio']:.2f}")
    print(f"tracker: {_fmt_diag(diag)}")
    print(f"\nfirst 8 bar lines:")
    for b in barlines[:8]:
        flag = " [extrap]" if b.extrapolated else ""
        print(f"  f{b.cross_frame:8.2f}  {b.ms:9.1f}ms  "
              f"strength={b.strength:.0f}{flag}")
