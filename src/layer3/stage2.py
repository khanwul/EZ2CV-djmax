"""
EZ2CV — Layer 3 / Stage 2 : constrained template matching
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
`match_lane`/`match_frame` accept an optional `template_policy` — an extension
seam for restricting which templates Stage 2 tries. The 1-Pass always uses the
default: the longnote state machine runs AFTER Stage 2, so it cannot feed a
policy back within a frame. The hook exists for a future feedback/2-pass
experiment, not for the current pipeline.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from layer1.calibration import Calibration
from layer2.preprocessor import LaneFrame, PreprocessedFrame
from layer3.stage1 import Run, Stage1Result


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


def default_template_policy(run: Run, note_height: int) -> list[CandidateGroup]:
    """Decide which template groups to try for a run.

    Returns a list of candidate GROUPS. Within a group the templates compete
    (best-of); across groups each may produce its own match.

    Design note — why this is NOT branched on run length:
    a regular note crossing the judgment line MERGES with the bright judgment
    bar, producing a run that can exceed short_run_max and get tagged "long".
    If "note" were only tried on short runs, those trigger-crossing notes would
    be lost. So the run's LEADING (top) edge always gets a best-of-3
    {note, lnhead, lntail} group — length only decides whether we ADD a second
    group probing the trailing edge for a longnote head.

    `match_lane` accepts an alternate policy as an extension seam — e.g. a
    future feedback pass could drop "lnhead" from the leading group while a
    longnote is known to be mid-hold. The current 1-Pass uses only this
    default; nothing feeds a stricter policy back within a single frame.
    """
    top = run.y_start
    groups: list[CandidateGroup] = [
        [("note", top), ("lnhead", top), ("lntail", top)]   # leading edge
    ]
    if run.kind == "long":
        # a long run may also expose a longnote HEAD at its trailing edge
        groups.append([("lnhead", run.y_end - note_height)])
    return groups


# =============================================================================
# Matcher
# =============================================================================

class TemplateMatcher:
    """Stage 2 matcher. Stateless across frames."""

    def __init__(self, cal: Calibration, *, search_band_px: int = 15):
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
        *,
        policy=default_template_policy,
    ) -> Stage2Result:
        """Confirm types for every Stage 1 run in one lane."""
        templates = self.cal.lanes[s1.lane_index].templates
        thr = self._th[s1.lane_index]
        roi = lane.matching_roi
        matches: list[Match] = []

        for run in s1.runs:
            for group in policy(run, self.cal.note_height):
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
                    if not self._tail_gate_ok(key, y_top):
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

    # ------------------------------------------------------------------ #
    def match_frame(
        self,
        pf: PreprocessedFrame,
        s1_results: list[Stage1Result],
        *,
        policy=default_template_policy,
    ) -> list[Stage2Result]:
        """Run Stage 2 on every lane of one frame."""
        by_lane = {r.lane_index: r for r in s1_results}
        return [
            self.match_lane(lane, by_lane[lane.index], policy=policy)
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

    def _tail_gate_ok(self, template_key: str, y_top: int) -> bool:
        """Reject longnote-tail matches that fall in the judgment-bar zone.

        tail_search_y_max guards against the bright judgment bar being mistaken
        for a tail edge.
        """
        if template_key != "lntail":
            return True
        return y_top <= self.cal.tail_search_y_max


# =============================================================================
# CLI: python stage2.py [config/song.toml] [frame_index]
# =============================================================================

if __name__ == "__main__":
    from layer2.preprocessor import Preprocessor
    from layer3.stage1 import ProjectionDetector

    cfg = sys.argv[1] if len(sys.argv) > 1 else "config/Dream Walker.toml"
    want = int(sys.argv[2]) if len(sys.argv) > 2 else 7

    pre = Preprocessor.from_config(cfg)
    s1 = ProjectionDetector(pre.cal)
    s2 = TemplateMatcher(pre.cal)

    for pf in pre:
        if pf.frame_index != want:
            continue
        s1_res = s1.detect_frame(pf)
        s2_res = s2.match_frame(pf, s1_res)
        print(f"--- frame {pf.frame_index} @ {pf.timestamp_ms:.2f}ms ---")
        for r1, r2 in zip(s1_res, s2_res):
            kept = len(r2.matches)
            print(f"L{r2.lane_index+1} ({r2.color}): "
                  f"{len(r1.runs)} run(s) -> {kept} confirmed")
            for m in r2.matches:
                print(f"    {m.type:7s} y_top={m.y_top:3d} score={m.score:.3f} "
                      f"(from {m.source_run.kind} run len={m.source_run.length})")
            for run in r1.runs:
                fy0 = run.y_start + r1.roi_y_origin
                fy1 = run.y_end + r1.roi_y_origin
                confirmed = any(m.source_run is run for m in r2.matches)
                if not confirmed:
                    print(f"    REJECTED run frameY[{fy0},{fy1}] "
                          f"len={run.length} ({run.kind})")
        break
