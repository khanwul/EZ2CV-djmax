"""
EZ2CV — detection / Stage 2 : constrained template matching
===============================================================================
Stage 2 turns Stage 1's brightness RUNS into typed, confirmed detections. For
each run it runs cv2.matchTemplate in a narrow +-band around the run's relevant
edge and keeps only matches whose similarity clears the per-lane threshold.

Role split (vs Stage 1)
-----------------------
* Stage 1 = recall (a loose brightness gate; over-detects on purpose).
* Stage 2 = precision. It is what REJECTS the judgment bar: the bar is bright
  (so Stage 1 emits a run for it) but it has no central marker, so it matches
  no template and is dropped here.

Why "constrained"
-----------------
Matching only inside a +-15px band around a Stage 1 candidate (not the whole
lane) makes Stage 2 cheap and unambiguous — every lane has many identical-
looking notes, so an unconstrained search would be hopelessly multi-modal.

Scope
-----
Stage 2 is STATELESS: it matches templates and reports what matched. It does
NOT pair longnote head+tail, suppress longnote bodies, or deduplicate across
frames — those belong to the longnote state machine and the NoteTracker.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ez2cv.config import RunConfig
from ez2cv.video import LaneFrame, PreprocessedFrame
from ez2cv.detection.stage1 import Run, Stage1Result


# =============================================================================
# Output structures
# =============================================================================

@dataclass
class Match:
    """A confirmed, typed detection produced by Stage 2."""
    lane_index: int
    type: str               # "note" | "lnhead" | "lntail"
    y_top: int              # full-frame y of the template's top edge
    x_offset: int           # in-ROI x of the best match (sanity / debug)
    score: float            # TM_CCOEFF_NORMED similarity
    source_run: Run         # the Stage 1 run this was confirmed from


@dataclass
class Stage2Result:
    """Stage 2 output for one lane of one frame."""
    frame_index: int
    lane_index: int
    color: str
    matches: list[Match] = field(default_factory=list)


# =============================================================================
# Template-selection policy
# =============================================================================

# A candidate group: templates competing for the SAME physical edge. Stage 2
# keeps the single best match per group (best-of). Different groups are
# independent edges and can each yield a match.
CandidateGroup = list[tuple[str, int]]   # [(template_key, center_y_roi), ...]


def _candidate_groups(run: Run, note_height: int,
                      allowed_types: frozenset[str]) -> list[CandidateGroup]:
    """Decide which template groups to try for a run.

    Returns a list of candidate GROUPS. Within a group the templates compete
    (best-of); across groups each may produce its own match.

    Design note — why this is NOT branched on run length:
    a regular note crossing the judgment line MERGES with the bright judgment
    bar, producing a run that can exceed short_run_max and get tagged "long".
    If "note" were only tried on short runs, those trigger-crossing notes would
    be lost. So the run's LEADING (top) edge always tries every template the
    track allows; length only decides whether to add a trailing lnhead probe.

    """
    leading = []
    if "tap" in allowed_types:
        leading.append(("note", run.y_start))
    if "longnote" in allowed_types:
        leading.extend((("lnhead", run.y_start), ("lntail", run.y_start)))
    groups = [leading]
    if run.kind == "long" and "longnote" in allowed_types:
        # a long run may also expose a longnote HEAD at its trailing edge
        groups.append([("lnhead", run.y_end - note_height)])
    return groups


# =============================================================================
# Matcher
# =============================================================================

class TemplateMatcher:
    """Stage 2 matcher. Stateless across frames."""

    def __init__(self, cal: RunConfig, *, search_band_px: int = 15):
        """
        Parameters
        ----------
        search_band_px : int
            Half-width (in px) of the y-band searched around each Stage 1
            candidate. Default 15 (per the design's +-15px rule).
        """
        self.cal = cal
        self.band = search_band_px
        self._th = {ln.index: ln.matching_threshold for ln in cal.lanes}

    # ------------------------------------------------------------------ #
    def match_lane(
        self,
        lane: LaneFrame,
        s1: Stage1Result,
    ) -> Stage2Result:
        """Confirm types for every Stage 1 run in one lane."""
        lane_config = self.cal.lanes[s1.lane_index]
        if lane_config.role == "overlay":
            return self._match_overlay(s1, lane_config)
        templates = lane_config.templates
        thr = self._th[s1.lane_index]
        roi = lane.matching_roi
        matches: list[Match] = []

        for run in s1.runs:
            for group in _candidate_groups(
                    run, lane_config.note_height, lane_config.allowed_types):
                # within a group the templates compete -> keep the best
                best: Match | None = None
                for key, center in group:
                    hit = self._match_in_band(roi, templates[key], center,
                                              s1.roi_y_origin)
                    if hit is None:
                        continue
                    score, y_top, x_off = hit
                    if score < thr:
                        continue
                    if not self._tail_gate_ok(key, y_top, lane_config):
                        continue
                    if best is None or score > best.score:
                        best = Match(s1.lane_index, key, y_top, x_off,
                                     score, run)
                if best is not None:
                    matches.append(best)

        return Stage2Result(
            frame_index=s1.frame_index,
            lane_index=s1.lane_index,
            color=s1.color,
            matches=matches,
        )

    def _match_overlay(self, s1: Stage1Result, lane_config) -> Stage2Result:
        """Turn reliable color-mask run edges directly into typed matches."""
        matches = []
        scan_y_max = lane_config.trigger_y_top - self.cal.playfield_top
        for run in s1.runs:
            score = run.mean_brightness
            if run.kind == "short" and "tap" in lane_config.allowed_types:
                # A partial LN entering at the top or leaving through the
                # judgment zone is not a new tap.
                if run.y_start > 0 and run.y_end < scan_y_max:
                    matches.append(Match(
                        s1.lane_index, "note", run.y_start + s1.roi_y_origin,
                        0, score, run))
                continue
            if "longnote" not in lane_config.allowed_types:
                continue
            tail_y = run.y_start + s1.roi_y_origin
            if self._tail_gate_ok("lntail", tail_y, lane_config):
                matches.append(Match(
                    s1.lane_index, "lntail", tail_y, 0, score, run))
            # Once the lower edge reaches the scan ceiling, the real head is
            # hidden by the judgment zone. Report the trigger position so the
            # tracker emits the start now instead of extrapolating it at tail
            # release. Before that point, use the visible endpoint geometry.
            head_y = (scan_y_max if run.y_end >= scan_y_max
                      else run.y_end - lane_config.note_height)
            matches.append(Match(
                s1.lane_index, "lnhead", head_y + s1.roi_y_origin,
                0, score, run))
        return Stage2Result(
            frame_index=s1.frame_index,
            lane_index=s1.lane_index,
            color=s1.color,
            matches=matches,
        )

    # ------------------------------------------------------------------ #
    def match_frame(
        self,
        pf: PreprocessedFrame,
        s1_results: list[Stage1Result],
    ) -> list[Stage2Result]:
        """Run Stage 2 on every lane of one frame."""
        by_lane = {r.lane_index: r for r in s1_results}
        return [
            self.match_lane(lane, by_lane[lane.index])
            for lane in pf.lanes
        ]

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _match_in_band(
        self,
        roi: np.ndarray,
        template: np.ndarray,
        center_y_roi: int,
        roi_y_origin: int,
    ) -> tuple[float, int, int] | None:
        """matchTemplate inside a +-band around center_y_roi.

        Returns (score, y_top_full_frame, x_offset) or None if the band is too
        small to host the template.
        """
        h = roi.shape[0]
        th = template.shape[0]
        # y_top can range over [center-band, center+band], clamped so the
        # template still fits inside the ROI.
        lo = max(0, center_y_roi - self.band)
        hi = min(h - th, center_y_roi + self.band)
        if hi < lo:
            return None
        search = roi[lo: hi + th]
        if search.shape[0] < th or search.shape[1] < template.shape[1]:
            return None
        res = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)          # loc = (x, y)
        y_top = lo + loc[1] + roi_y_origin
        return float(score), int(y_top), int(loc[0])

    @staticmethod
    def _tail_gate_ok(template_key: str, y_top: int, lane_config) -> bool:
        """Reject longnote-tail matches that fall in the judgment-bar zone.

        tail_search_y_max guards against the bright judgment bar being mistaken
        for a tail edge.
        """
        if template_key != "lntail":
            return True
        return y_top <= lane_config.tail_search_y_max
