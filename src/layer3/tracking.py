"""
EZ2CV — Layer 3 / tracking : scroll speed + NoteTracker + longnote pairing
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
2. LONGNOTE HEAD HOLD. A longnote head's template stops matching for ~10
   frames as it is "caught" at the line, then reappears HELD just past it.
   Fix: an un-emitted *lnhead* edge near the trigger gets an extended grace
   period so the post-gap (held) detection bridges the gap and the crossing
   interpolates. Grace is lnhead-ONLY: a tap or tail just vanishes, so letting
   it linger would make it steal a later note's detections.
3. MISSED POST-TRIGGER FRAME. If the stutter eats the one frame a tap is
   visible past the line, its crossing is EXTRAPOLATED from the global speed.

Output: RawNote objects, ms-based. Tick conversion / snapping is Layer 4.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from layer1.calibration import Calibration
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


@dataclass
class RawNote:
    """A finished, ms-based note. Layer 4 converts these to ticks."""
    lane: int
    type: str                  # "tap" | "longnote"
    trigger_ms: float          # tap hit, or longnote head (start)
    end_ms: float | None       # longnote tail (end); None for taps
    color: str
    confidence: float
    extrapolated: bool = False


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

    def __init__(self, cal: Calibration, sample_window: int = 48):
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

    def __init__(self, cal: Calibration, *,
                 max_stale_frames: int = 4,
                 max_stale_grace: int = 15,
                 up_jitter_px: float = 6.0,
                 down_jitter_px: float = 16.0):
        """
        max_stale_frames : drop a normal edge unseen this many frames.
        max_stale_grace  : an un-emitted *lnhead* edge near the trigger is kept
                           this many frames so the held detection can bridge.
        up/down_jitter_px : slack on the directional association gate.
        """
        self.cal = cal
        self.trigger = cal.trigger_template_y_top
        self.fps = cal.fps
        self.max_stale = max_stale_frames
        self.max_grace = max_stale_grace
        self.up_jitter = up_jitter_px
        self.down_jitter = down_jitter_px

        self.speed = ScrollSpeedEstimator(cal)
        self._lanes = {ln.index: [] for ln in cal.lanes}
        self._next_id = 0

    # ------------------------------------------------------------------ #
    def step(self, frame_index: int, s2_results):
        """Advance the tracker one frame; return any new trigger crossings."""
        speed = self.speed.speed                  # speed from the PREVIOUS frame
        events = []

        for res in s2_results:
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
                used_e.add(id(e))
                used_m.add(id(m))

            # --- unassociated matches -> new edges -------------------------
            for m in free:
                if id(m) in used_m:
                    continue
                edges.append(TrackedEdge(
                    id=self._next_id, lane=res.lane_index, type=m.type,
                    trajectory=[(frame_index, m.y_top, m.score)],
                    last_seen=frame_index))
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
                if self._keep(e, frame_index, speed):
                    survivors.append(e)
                elif not e.trigger_emitted:
                    ev = self._extrapolate_trigger(e)
                    if ev is not None:
                        events.append(ev)
            self._lanes[res.lane_index] = survivors

        # --- refresh global speed -----------------------------------------
        # Only DESCENDING edges (still above the judgment line) count. A held
        # longnote head sits motionless past the line; including its 0-px
        # "displacement" every frame would drag the median speed down, narrow
        # the gate, and fragment fast notes.
        moving = [e for lst in self._lanes.values() for e in lst
                  if e.trajectory[-1][1] < self.trigger]
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
    def _gate_dist(self, e, m, frame_index, speed):
        """Distance to prediction if m is inside e's directional gate, else None.

        Notes only descend. For a recently-seen edge the note kept moving, so
        the gate reaches last_y + elapsed*speed + one stutter step. For a
        grace-kept lnhead (stale beyond max_stale) the head is being HELD near
        the line, so its continuation can only be near the trigger — the gate
        is clamped there, which stops it grabbing a far-away later note.
        """
        elapsed = frame_index - e.last_seen
        lo = e.last_y - self.up_jitter
        if elapsed <= self.max_stale:
            hi = e.last_y + elapsed * speed + speed + self.down_jitter
            pred = e.last_y + elapsed * speed
        else:                                  # grace-kept lnhead, held at line
            hi = self.trigger + self.cal.note_height + self.down_jitter
            pred = float(self.trigger)
        if lo <= m.y_top <= hi:
            return abs(m.y_top - pred)
        return None

    def _keep(self, e, frame_index, speed) -> bool:
        """Whether an edge survives this frame's prune."""
        if frame_index - e.last_seen <= self.max_stale:
            return True
        # extended grace ONLY for a longnote head: it stops matching for ~10
        # frames as it is caught at the line, then reappears HELD. A tap or
        # tail just vanishes — letting it linger would steal a later note.
        if (e.type == "lnhead" and not e.trigger_emitted
                and e.last_y >= self.trigger - 3 * speed
                and e.last_y <= self.trigger + self.cal.note_height
                and frame_index - e.last_seen <= self.max_grace):
            return True
        return False

    def _tail_lag_frames(self) -> float:
        """Frames to ADD to a longnote tail's crossing time.

        A note/lnhead is hit the instant its tracked top reaches the line, and
        that is accurate. A longnote does not END at the tail-top crossing — the
        tail must descend until it has fully PASSED the line (its bottom, one
        ``note_height`` lower, reaches the line) AND a skin-specific release
        margin (``tail_release_offset_px``) of further descent before the hold
        actually lets go. So the release is ``(note_height + offset) / speed``
        frames later. Without the offset the tail-bottom model alone lands the
        end ~12-13px short, which snaps a longnote one 1/64 note too short
        (measured on Dream Walker 221 LN, GEHENNA 102 LN). Applied as a post-hoc
        lag on the COMPUTED crossing rather than by moving the trigger line, so
        the edge tracking, pairing and extrapolation gates are untouched (the
        bias-fix must not change which notes are detected, only an end time).
        """
        sp = self.speed.speed
        descent = self.cal.note_height + self.cal.tail_release_offset_px
        return (descent / sp) if sp > 0 else 0.0

    def _check_trigger(self, e):
        """Emit if the trajectory straddles the trigger line (interpolation)."""
        traj = e.trajectory
        for i in range(len(traj) - 1):
            fa, ya, sa = traj[i]
            fb, yb, sb = traj[i + 1]
            if ya < self.trigger <= yb and yb != ya:
                # A normal straddle spans consecutive frames, so linear
                # interpolation lands the crossing accurately. But a grace-kept
                # lnhead straddles across the HOLD gap: it reached the line
                # during its pre-hold descent (`fa, ya` just above), then sat
                # motionless for several frames and reappears HELD just past the
                # line (`fb, yb`). Interpolating across that stationary gap
                # smears the crossing over the whole hold and post-dates it by
                # several frames. The head actually crossed right after `fa`, so
                # project from there at the global descent speed instead.
                sp = self.speed.speed
                if fb - fa > self.max_stale and e.type == "lnhead" and sp > 0:
                    cf = fa + (self.trigger - ya) / sp
                else:
                    frac = (self.trigger - ya) / (yb - ya)
                    cf = fa + frac * (fb - fa)
                if e.type == "lntail":
                    cf += self._tail_lag_frames()
                match = (sa + sb) / 2
                conf = _trigger_confidence(len(traj), 0.0, match)
                return TriggerEvent(e.lane, e.type, cf,
                                    cf / self.fps * 1000.0, match,
                                    confidence=conf)
        return None

    def _extrapolate_trigger(self, e):
        """Fallback: project a near-miss edge forward to the trigger line.

        Uses the GLOBAL speed (robust, trimmed-mean smoothed) rather than the
        edge's own last segment, which the stutter can corrupt.
        """
        if len(e.trajectory) < 2:
            return None
        fb, yb, sb = e.trajectory[-1]
        if yb >= self.trigger:
            return None
        sp = self.speed.speed
        if sp <= 0 or (self.trigger - yb) > 3 * sp:    # died too far short
            return None
        cf = fb + (self.trigger - yb) / sp
        if e.type == "lntail":
            cf += self._tail_lag_frames()
        proj_ratio = (self.trigger - yb) / (3.0 * sp)  # 0 = clean .. 1 = max guess
        conf = _trigger_confidence(len(e.trajectory), proj_ratio, sb)
        return TriggerEvent(e.lane, e.type, cf,
                            cf / self.fps * 1000.0, sb,
                            extrapolated=True, confidence=conf)


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

    def __init__(self, cal: Calibration):
        self.cal = cal
        self._colors = {ln.index: ln.color for ln in cal.lanes}
        # a "longnote" whose head->tail gap is under min_longnote_px is a tap
        self._min_ln_ms = (cal.min_longnote_px
                           / cal.pixels_per_frame / cal.fps * 1000.0)
        self._open = {}                              # lane -> pending lnhead
        self.orphan_tails = 0                        # diagnostics

    def _close(self, head, end_ev):
        """Pair an open head with a closing edge into one RawNote."""
        color = self._colors[head.lane]
        extrap = head.extrapolated or end_ev.extrapolated
        conf = min(head.confidence, end_ev.confidence)   # weakest end governs
        if end_ev.ms - head.ms < self._min_ln_ms:        # too short -> it's a tap
            return RawNote(head.lane, "tap", head.ms, None, color,
                           conf, extrapolated=extrap)
        return RawNote(head.lane, "longnote", head.ms, end_ev.ms, color,
                       conf, extrapolated=extrap)

    def feed(self, ev: TriggerEvent):
        """Consume one TriggerEvent; return a RawNote when one completes."""
        color = self._colors[ev.lane]

        if ev.type == "note":
            return RawNote(ev.lane, "tap", ev.ms, None, color, ev.confidence,
                           extrapolated=ev.extrapolated)

        if ev.type == "lnhead":
            pending = self._open.get(ev.lane)
            if pending is not None and (
                    self._min_ln_ms <= ev.ms - pending.ms < self._MISTYPED_TAIL_MS):
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
            out.append(RawNote(lane, "tap", head.ms, None,
                               self._colors[lane], head.confidence,
                               extrapolated=head.extrapolated))
        self._open.clear()
        return out


# =============================================================================
# Duplicate-trigger merge
# =============================================================================

def merge_duplicate_triggers(events, merge_window_ms: float = 6.0):
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
