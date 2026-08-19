"""
EZ2CV — detection / tracking : scroll speed + NoteTracker + longnote pairing
===============================================================================
Stages 1 & 2 detect typed edges PER FRAME. A note is visible for ~25 frames, so
it is detected ~25 times. This module collapses those repeated detections into
ONE note, finds the sub-frame moment each note crosses the judgment line, and
pairs longnote head+tail edges into single longnote events.

Three pieces
------------
* ScrollSpeedEstimator — global adaptive px/frame, smoothed by an outlier-
  trimmed mean over a short window. SV is global, so it aggregates ALL lanes.
* NoteTracker — associates each frame's Stage 2 matches with tracked edges and
  emits ONE TriggerEvent per edge when it crosses the judgment line.
* LongnoteStateMachine — pairs the ordered TriggerEvent stream into RawNotes.

Real-video hazards this tracker is built around
------------------------------------------------
1. CAPTURE STUTTER. The recording repeats a frame every ~8 frames, so a note
   appears to move 0px then ~2x next frame. Fix: a DIRECTIONAL, stutter-aware
   gate — notes only move DOWN, by 0..~2*speed per frame.
2. DJMAX LONGNOTE ENDPOINTS. Unlike EZ2ON, both longnote endpoints keep
   descending through the judgment line. Head, tail, and tap crossings all use
   the same recent trajectory fit; only the tail's calibrated release offset
   differs.
3. MISSED POST-TRIGGER FRAME. If the stutter eats the one frame an endpoint is
   visible past the line, its crossing is extrapolated from that local fit,
   with global speed as the plausibility gate and fallback.

Output: RawNote objects, ms-based. Tick conversion / snapping is chart conversion.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ez2cv.config import RunConfig
# =============================================================================
# Data structures
# =============================================================================

@dataclass
class TrackedEdge:
    """One physical edge (note / lnhead / lntail) followed across frames."""
    id: int
    lane: int
    type: str
    trajectory: list                       # [(frame, y_top, score), ...]
    last_seen: int
    trigger_emitted: bool = False
    type_scores: dict[str, float] = field(default_factory=dict)

    @property
    def last_y(self) -> int:
        return self.trajectory[-1][1]


@dataclass
class TriggerEvent:
    """An edge crossing the judgment line."""
    lane: int
    type: str                  # "note" | "lnhead" | "lntail"
    cross_frame: float         # fractional frame index of the crossing
    ms: float                  # cross time in milliseconds
    score: float               # raw template-match strength
    extrapolated: bool = False
    confidence: float = 1.0    # composite reliability (see _trigger_confidence)
    timing_sigma_ms: float = 0.0


@dataclass
class RawNote:
    """A finished, ms-based note. chart conversion converts these to ticks."""
    lane: int
    type: str                  # "tap" | "longnote"
    trigger_ms: float          # tap hit, or longnote head (start)
    end_ms: float | None       # longnote tail (end); None for taps
    color: str
    confidence: float
    extrapolated: bool = False
    start_sigma_ms: float = 0.0
    end_sigma_ms: float | None = None
    pairing_status: str = "observed"

    @property
    def timing_sigma_ms(self) -> float:
        """Compatibility alias for callers that only inspect note heads."""
        return self.start_sigma_ms


# =============================================================================
# Scroll speed
# =============================================================================

class ScrollSpeedEstimator:
    """Global adaptive scroll-speed estimate in px/frame.

    Uses an outlier-trimmed MEAN over a pooled sample window — NOT a median.
    The capture stutters bimodally (a repeated frame gives 0px, the catch-up
    frame gives ~2x); in stretches where every other frame repeats, the
    displacement distribution is ~50/50 between 0 and 2x and its median
    collapses to 0. The mean correctly averages 0 and 2x back to the true
    speed. Implausible values (mis-tracks) are trimmed before averaging.
    """

    # plausible per-frame descent: 0 (stutter repeat) .. fast burst + catch-up
    _LO, _HI = -5.0, 160.0

    def __init__(self, cal: RunConfig, sample_window: int = 48):
        self._samples = deque(maxlen=sample_window)
        self._speed = cal.pixels_per_frame

    def update(self, edges) -> None:
        for e in edges:
            if len(e.trajectory) >= 2:
                (fa, ya, _), (fb, yb, _) = e.trajectory[-2], e.trajectory[-1]
                if fb > fa:
                    d = (yb - ya) / (fb - fa)
                    if self._LO <= d <= self._HI:    # trim mis-track outliers
                        self._samples.append(d)
        if self._samples:
            self._speed = float(np.mean(self._samples))

    @property
    def speed(self) -> float:
        return self._speed


# =============================================================================
# NoteTracker
# =============================================================================

def _trigger_confidence(traj_len: int, projection_ratio: float,
                        match_score: float) -> float:
    """Composite detection-reliability score in [0, 1].

    The bare template-match score is a poor reliability signal: a real note
    nearly always saturates it (~0.97). This score instead reflects how
    solidly the note was *tracked*:

      length      a note observed over many frames is reliable; one tracked
                  only ~2-4 frames (the bare minimum) is shaky. Reaches full
                  marks by 10 frames -- well under the ~25-frame norm of even
                  the fastest song tested.
      projection  an interpolated crossing (ratio 0) is bracketed by real
                  observations; an extrapolated one is projected forward, and
                  a longer projection (ratio -> 1) is a longer guess.
      match       the raw template-match strength, kept as a minor term.
    """
    length = max(0.0, min(1.0, (traj_len - 1) / 9.0))
    projection = 1.0 - 0.5 * max(0.0, min(1.0, projection_ratio))
    return length * projection * match_score


class NoteTracker:
    """Per-lane temporal dedup + trigger-crossing detection."""

    def __init__(self, cal: RunConfig, *,
                 max_stale_frames: int = 4,
                 up_jitter_px: float = 6.0,
                 down_jitter_px: float = 20.0,
                 local_fit_points: int = 8,
                 timing_sigma_floor_frames: float = 0.25):
        """
        max_stale_frames : drop a normal edge unseen this many frames.
        up/down_jitter_px : slack on the directional association gate.
        """
        self.cal = cal
        self.fps = cal.fps
        self.max_stale = max_stale_frames
        self.up_jitter = up_jitter_px
        self.down_jitter = down_jitter_px
        self.local_fit_points = local_fit_points
        self.timing_sigma_floor_frames = timing_sigma_floor_frames

        self.speed = ScrollSpeedEstimator(cal)
        self._lanes = {ln.index: [] for ln in cal.lanes}
        self._next_id = 0

    # ------------------------------------------------------------------ #
    def step(self, frame_index: int, s2_results):
        """Advance the tracker one frame; return any new trigger crossings."""
        events = []

        for res in s2_results:
            speed = self._lane_speed(res.lane_index)
            edges = self._lanes[res.lane_index]
            free = list(res.matches)

            # --- associate: build all valid (dist, edge, match) pairs, then
            #     assign nearest-first so the best edge wins each match -------
            # Association is PROXIMITY-ONLY: a per-frame type misclassification
            # (note <-> lnhead) would otherwise spawn a parallel same-position
            # edge of the new type. merge_duplicate_triggers keys on
            # (lane, type) and cannot collapse those, so one physical note
            # would split into a tap and a longnote open. The edge keeps the
            # type it was created with; the trajectory still tracks position.
            pairs = []
            for e in edges:
                if e.last_seen == frame_index:
                    continue
                for m in free:
                    # Tails are a distinct physical edge and must never steal
                    # a descending head. Normal tap/head templates may still
                    # trade labels frame-to-frame. Longnote-only side masks
                    # are authoritative; L/R masks overlap normal red notes
                    # and retain the normal type-flexible association.
                    lane = self.cal.lanes[e.lane]
                    role = getattr(lane, "role", "normal")
                    side_only = (role == "overlay"
                                 and "tap" not in lane.allowed_types)
                    if (e.type != m.type and (
                            side_only
                            or (role == "normal"
                                and "lntail" in (e.type, m.type)))):
                        continue
                    d = self._gate_dist(e, m, frame_index, speed)
                    if d is not None:
                        pairs.append((d, e, m))
            pairs.sort(key=lambda p: p[0])
            used_e, used_m = set(), set()
            for d, e, m in pairs:
                if id(e) in used_e or id(m) in used_m:
                    continue
                e.trajectory.append((frame_index, m.y_top, m.score))
                e.last_seen = frame_index
                e.type_scores[m.type] = e.type_scores.get(m.type, 0.0) + m.score
                e.type = max(e.type_scores, key=e.type_scores.get)
                used_e.add(id(e))
                used_m.add(id(m))

            # --- unassociated matches -> new edges -------------------------
            for m in free:
                if id(m) in used_m:
                    continue
                edges.append(TrackedEdge(
                    id=self._next_id, lane=res.lane_index, type=m.type,
                    trajectory=[(frame_index, m.y_top, m.score)],
                    last_seen=frame_index, type_scores={m.type: m.score}))
                self._next_id += 1

            # --- trigger crossings (interpolation) ------------------------
            for e in edges:
                if e.trigger_emitted:
                    continue
                ev = self._check_trigger(e)
                if ev is not None:
                    e.trigger_emitted = True
                    events.append(ev)

            # --- prune; extrapolate near-trigger un-emitted edges ---------
            survivors = []
            for e in edges:
                if self._keep(e, frame_index):
                    survivors.append(e)
                elif not e.trigger_emitted:
                    ev = self._extrapolate_trigger(e)
                    if ev is not None:
                        events.append(ev)
            self._lanes[res.lane_index] = survivors

        # --- refresh global speed -----------------------------------------
        # Only descending edges still above the judgment line count.
        moving = [e for lst in self._lanes.values() for e in lst
                  if self.cal.lanes[e.lane].include_in_consensus
                  and e.trajectory[-1][1]
                  < self.cal.lanes[e.lane].trigger_y_top]
        self.speed.update(moving)
        return events

    # ------------------------------------------------------------------ #
    def flush(self):
        """End-of-video: extrapolate any un-emitted edge still near the line."""
        events = []
        for edges in self._lanes.values():
            for e in edges:
                if not e.trigger_emitted:
                    ev = self._extrapolate_trigger(e)
                    if ev is not None:
                        events.append(ev)
        return events

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _lane_speed(self, lane_index: int) -> float:
        """Never track below the calibrated scroll speed."""
        lane = self.cal.lanes[lane_index]
        return (max(self.speed.speed, self.cal.pixels_per_frame)
                if lane.include_in_consensus else self.cal.pixels_per_frame)

    def _gate_dist(self, e, m, frame_index, speed):
        """Distance to prediction if m is inside e's directional gate, else None.

        Notes only descend. The gate reaches last_y + elapsed*speed plus one
        stutter step. Wide L/R tap masks retain one extra retry frame.
        """
        lane = self.cal.lanes[e.lane]
        trigger = lane.trigger_y_top
        elapsed = frame_index - e.last_seen
        role = getattr(lane, "role", "normal")
        down_jitter = self.down_jitter if role == "normal" else 16.0
        lo = e.last_y - self.up_jitter
        if elapsed <= self.max_stale:
            hi = e.last_y + elapsed * speed + speed + down_jitter
            pred = e.last_y + elapsed * speed
        elif (role == "overlay" and "tap" in lane.allowed_types
              and elapsed == self.max_stale + 1):
            # L/R's wide, sparse mask needs one retry. Normal tracks do not:
            # an expired edge can steal the next note in a dense lane.
            hi = trigger + lane.note_height + down_jitter
            pred = float(trigger)
        else:
            return None
        if lo <= m.y_top <= hi:
            return abs(m.y_top - pred)
        return None

    def _keep(self, e, frame_index) -> bool:
        """Whether an edge survives this frame's prune."""
        return frame_index - e.last_seen <= self.max_stale

    def _tail_lag_frames(self, e, speed: float | None = None) -> float:
        """Frames after center crossing before a calibrated tail release."""
        sp = self._lane_speed(e.lane) if speed is None else speed
        lane = self.cal.lanes[e.lane]
        return (lane.tail_release_offset_px / sp) if sp > 0 else 0.0

    def _timing_offset_frames(self, e, speed: float | None = None) -> float:
        """Advance bottom-touch tracking to centre judgment plus calibration."""
        sp = self._lane_speed(e.lane) if speed is None else speed
        lane = self.cal.lanes[e.lane]
        offset = (lane.note_height / 2.0
                  + getattr(lane, "timing_offset_px", 0))
        return (offset / sp) if sp > 0 else 0.0

    def _local_crossing(self, trajectory, target_y: float
                        ) -> tuple[float, float, float] | None:
        """Fit recent ``y(frame)`` and return crossing, speed, and sigma-ms."""
        points = trajectory[-self.local_fit_points:]
        if len(points) < 3:
            return None
        frames = np.array([point[0] for point in points], dtype=float)
        ys = np.array([point[1] for point in points], dtype=float)
        origin = frames[0]
        x = frames - origin
        speed, intercept = np.polyfit(x, ys, 1)
        if not (0.0 < speed <= ScrollSpeedEstimator._HI):
            return None

        cross_x = (target_y - intercept) / speed
        cross_frame = origin + cross_x
        projection = cross_frame - frames[-1]
        if projection < -0.5 or projection > 3.5:
            return None

        residual = ys - (intercept + speed * x)
        dof = len(points) - 2
        residual_std = float(np.sqrt(np.sum(residual ** 2) / dof))
        spread = float(np.sum((x - np.mean(x)) ** 2))
        if spread > 0:
            prediction_std = residual_std * np.sqrt(
                1.0 + 1.0 / len(points)
                + (cross_x - np.mean(x)) ** 2 / spread)
        else:
            prediction_std = residual_std
        sigma_frames = max(
            self.timing_sigma_floor_frames, prediction_std / speed)
        return (float(cross_frame), float(speed),
                sigma_frames / self.fps * 1000.0)

    def _fallback_sigma_ms(self, projection_frames: float) -> float:
        """Conservative uncertainty for the old global-speed fallback."""
        sigma_frames = max(0.5, 0.5 * max(0.0, projection_frames))
        return sigma_frames / self.fps * 1000.0

    def _check_trigger(self, e):
        """Emit if the trajectory straddles the trigger line (interpolation)."""
        trigger = self.cal.lanes[e.lane].trigger_y_top
        traj = e.trajectory
        for i in range(len(traj) - 1):
            fa, ya, sa = traj[i]
            fb, yb, sb = traj[i + 1]
            if ya < trigger <= yb and yb != ya:
                sp = self._lane_speed(e.lane)
                sigma_ms = self.timing_sigma_floor_frames / self.fps * 1000.0
                frac = (trigger - ya) / (yb - ya)
                cf = fa + frac * (fb - fa)
                local = self._local_crossing(traj[:i + 2], trigger)
                if local is not None:
                    _, sp, local_sigma_ms = local
                    sigma_ms = max(sigma_ms, local_sigma_ms)
                if e.type == "lntail":
                    cf += self._tail_lag_frames(e, sp)
                cf += self._timing_offset_frames(e, sp)
                match = (sa + sb) / 2
                conf = _trigger_confidence(len(traj), 0.0, match)
                return TriggerEvent(e.lane, e.type, cf,
                                    cf / self.fps * 1000.0, match,
                                    confidence=conf,
                                    timing_sigma_ms=sigma_ms)
        return None

    def _extrapolate_trigger(self, e):
        """Fallback: project a near-miss edge forward to the trigger line.

        Every DJMAX endpoint fits its recent trajectory; global speed remains
        the gate and fallback.
        """
        if len(e.trajectory) < 2:
            return None
        trigger = self.cal.lanes[e.lane].trigger_y_top
        fb, yb, sb = e.trajectory[-1]
        if yb >= trigger:
            return None
        global_speed = self._lane_speed(e.lane)
        lane = self.cal.lanes[e.lane]
        reach = (3 * global_speed
                 + (self.down_jitter if lane.role == "normal" else 0))
        if global_speed <= 0 or (trigger - yb) > reach:
            return None
        local = self._local_crossing(e.trajectory, trigger)
        if local is None:
            sp = global_speed
            projection = (trigger - yb) / sp
            cf = fb + projection
            sigma_ms = self._fallback_sigma_ms(projection)
        else:
            cf, sp, sigma_ms = local
        if e.type == "lntail":
            cf += self._tail_lag_frames(e, sp)
        cf += self._timing_offset_frames(e, sp)
        proj_ratio = (trigger - yb) / (3.0 * global_speed)
        conf = _trigger_confidence(len(e.trajectory), proj_ratio, sb)
        return TriggerEvent(e.lane, e.type, cf,
                            cf / self.fps * 1000.0, sb,
                            extrapolated=True, confidence=conf,
                            timing_sigma_ms=sigma_ms)


# =============================================================================
# Longnote state machine
# =============================================================================

class LongnoteStateMachine:
    """Pairs ordered TriggerEvents into RawNotes (per lane)."""

    # A second lnhead this close behind an open one (no tail between) is taken
    # to be that note's tail mis-typed as a head (see feed). Beyond this gap the
    # two are treated as unrelated notes and the older head is overwritten — the
    # window sits well above a short LN's body (~50 ms observed) yet far below
    # the spacing to a genuine next longnote head (>=280 ms observed).
    _MISTYPED_TAIL_MS = 150.0

    def __init__(self, cal: RunConfig):
        self.cal = cal
        self._colors = {ln.index: ln.color for ln in cal.lanes}
        self._allowed = {ln.index: ln.allowed_types for ln in cal.lanes}
        # a "longnote" whose head->tail gap is under min_longnote_px is a tap
        self._min_ln_ms = {
            ln.index: ln.min_longnote_px
            / cal.pixels_per_frame / cal.fps * 1000.0
            for ln in cal.lanes
        }
        self._open = {}                              # lane -> pending lnhead
        self.orphan_tails = 0                        # diagnostics

    def _close(self, head, end_ev):
        """Pair an open head with a closing edge into one RawNote."""
        color = self._colors[head.lane]
        extrap = head.extrapolated or end_ev.extrapolated
        conf = min(head.confidence, end_ev.confidence)   # weakest end governs
        if end_ev.ms - head.ms < self._min_ln_ms[head.lane]:
            if "tap" not in self._allowed[head.lane]:
                return None
            return RawNote(head.lane, "tap", head.ms, None, color,
                           conf, extrapolated=extrap,
                           start_sigma_ms=head.timing_sigma_ms,
                           pairing_status="short_pair")
        return RawNote(head.lane, "longnote", head.ms, end_ev.ms, color,
                       conf, extrapolated=extrap,
                       start_sigma_ms=head.timing_sigma_ms,
                       end_sigma_ms=end_ev.timing_sigma_ms,
                       pairing_status=("observed" if end_ev.type == "lntail"
                                       else "inferred_tail"))

    def feed(self, ev: TriggerEvent):
        """Consume one TriggerEvent; return a RawNote when one completes."""
        color = self._colors[ev.lane]

        if ev.type == "note":
            if "tap" not in self._allowed[ev.lane]:
                return None
            return RawNote(ev.lane, "tap", ev.ms, None, color, ev.confidence,
                           extrapolated=ev.extrapolated,
                           start_sigma_ms=ev.timing_sigma_ms)

        if ev.type == "lnhead":
            pending = self._open.get(ev.lane)
            min_ln_ms = self._min_ln_ms[ev.lane]
            if pending is not None and (
                    min_ln_ms <= ev.ms - pending.ms < self._MISTYPED_TAIL_MS):
                # A lane physically cannot hold two longnotes at once, so a
                # second lnhead arriving a short-LN's body-length after an open
                # one (no tail between) is the OPEN note's tail MIS-TYPED as a
                # head — common for short longnotes whose head and tail sprites
                # nearly touch (Stage 2 then picks the head template for the
                # tail). Close the open longnote with this event as its tail
                # rather than silently dropping the first head and losing the
                # whole note. The gap is bounded on BOTH sides: below min_ln_ms
                # the second head is a DUPLICATE re-detection of the same edge
                # (the genuine tail still arrives later, so overwrite and let it
                # pair); at/above the upper window the two are unrelated notes
                # and overwriting is benign when the displaced head was itself
                # spurious and a real head+tail pair follows.
                self._open.pop(ev.lane)
                return self._close(pending, ev)
            self._open[ev.lane] = ev                 # longnote opens
            if (pending is not None
                    and ev.ms - pending.ms >= self._MISTYPED_TAIL_MS
                    and getattr(self.cal.lanes[ev.lane], "role", "normal")
                    == "normal"
                    and "tap" in self._allowed[ev.lane]):
                # A later head proves the previous hold is over. If its tail
                # was missed, preserve the detected head as a tap instead of
                # silently discarding a real note while opening the new one.
                return RawNote(pending.lane, "tap", pending.ms, None, color,
                               pending.confidence,
                               extrapolated=pending.extrapolated,
                               start_sigma_ms=pending.timing_sigma_ms,
                               pairing_status="unclosed_head")
            return None

        if ev.type == "lntail":
            head = self._open.pop(ev.lane, None)
            if head is None:
                self.orphan_tails += 1               # tail with no head: drop
                return None
            return self._close(head, ev)
        return None

    def flush(self):
        """Longnotes whose tail was never seen — emit head-only as a tap."""
        out = []
        for lane, head in self._open.items():
            if "tap" in self._allowed[lane]:
                out.append(RawNote(lane, "tap", head.ms, None,
                                   self._colors[lane], head.confidence,
                                   extrapolated=head.extrapolated,
                                   start_sigma_ms=head.timing_sigma_ms,
                                   pairing_status="unclosed_head"))
        self._open.clear()
        return out


# =============================================================================
# Duplicate-trigger merge
# =============================================================================

def merge_duplicate_triggers(events, merge_window_ms: float = 10.0):
    """Drop spurious double-detections from a TriggerEvent stream.

    A single physical note occasionally spawns two tracked edges that each
    emit a trigger (e.g. a transient secondary template match -- the parasitic
    edge is short-lived and so scores low on `_trigger_confidence`). Two
    triggers in the same lane and of the same type within `merge_window_ms`
    are one note: keep the more confident one. The window sits well below the
    tightest real spacing (a 1/64 note at 300 BPM is 12.5 ms apart), so
    genuine consecutive notes are never merged.
    """
    kept: list = []
    last: dict = {}                       # (lane, type) -> index into `kept`
    for ev in sorted(events, key=lambda e: e.ms):
        key = (ev.lane, ev.type)
        j = last.get(key)
        if j is not None and ev.ms - kept[j].ms <= merge_window_ms:
            if ev.confidence > kept[j].confidence:    # duplicate: keep the best
                kept[j] = ev
        else:
            last[key] = len(kept)
            kept.append(ev)
    return kept
