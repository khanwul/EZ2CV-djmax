"""
EZ2CV — detection / Stage 1 : projection + run detection
===============================================================================
Stage 1 is the HARD GATE of the detector. It compresses each lane's detection
ROI into a 1D vertical signal and finds contiguous bright RUNS — not peaks.

Why runs, not find_peaks
------------------------
A longnote BODY is a wide bright plateau. find_peaks cannot localize a plateau
and instead emits spurious peaks at its noisy edges. A run (a maximal stretch
of "lit" rows) captures both a 22px regular note AND a 300px longnote body
correctly, and its two edges are exactly the longnote tail (top) and head
(bottom).

Stage 1 is tuned for RECALL: the threshold is loose. Anything Stage 1 misses,
Stage 2 can never recover. Precision is Stage 2's job (template similarity).
Run length here is only a PROVISIONAL hint — Stage 2 confirms the real type.

This module does projection + run-finding only. No template matching, no
tracking, no longnote pairing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ez2cv.config import RunConfig
from ez2cv.video import LaneFrame, PreprocessedFrame


# =============================================================================
# Output structures
# =============================================================================

@dataclass
class Run:
    """A maximal stretch of lit rows in one lane's projection.

    Coordinates are in ROI space (0 == top of the playfield). Add the owning
    Stage1Result.roi_y_origin to convert to full-frame y.
    """
    y_start: int          # top edge (smaller y)  -> longnote TAIL edge
    y_end: int            # bottom edge (larger y, exclusive) -> longnote HEAD edge
    kind: str             # "short" (regular-note hint) | "long" (longnote hint)
    peak_brightness: float
    mean_brightness: float

    @property
    def length(self) -> int:
        return self.y_end - self.y_start

    @property
    def y_center(self) -> float:
        return (self.y_start + self.y_end - 1) / 2.0


@dataclass
class Stage1Result:
    """Stage 1 output for one lane of one frame."""
    frame_index: int
    lane_index: int
    color: str
    roi_y_origin: int             # ROI y + this = full-frame y
    threshold: float              # the Stage 1 brightness gate used
    projection: np.ndarray        # normal: row mean 0..255; overlay: coverage 0..1
    runs: list[Run] = field(default_factory=list)


# =============================================================================
# Detector
# =============================================================================

class ProjectionDetector:
    """Stage 1 detector. Stateless across frames — pure per-frame projection."""

    def __init__(
        self,
        cal: RunConfig,
        *,
        min_run_px: int | None = None,
        short_run_max: int | None = None,
    ):
        """
        Parameters
        ----------
        min_run_px : int
            Runs shorter than this are discarded as noise. Kept small to stay
            recall-first (a real note is ~note_height px). Default: 5.
        short_run_max : int
            Runs at or below this length are tagged "short" (regular-note hint);
            longer runs are tagged "long" (longnote hint). This is only a hint
            for Stage 2. Default: note_height + 8.
        """
        self.cal = cal
        self.min_run_px = 5 if min_run_px is None else min_run_px
        self.short_run_max = short_run_max

    # ------------------------------------------------------------------ #
    def detect_lane(self, lane: LaneFrame, *, frame_index: int,
                    roi_y_origin: int) -> Stage1Result:
        """Run Stage 1 on a single lane."""
        lane_config = self.cal.lanes[lane.index]
        if lane_config.role == "overlay":
            proj = self._overlay_projection(lane, lane_config)
            thr = lane_config.coverage_threshold
        else:
            # Mean keeps the normal-lane signal on the configured 0-255 scale.
            proj = lane.detection_roi.mean(axis=1)
            thr = lane_config.stage1_threshold

        # --- binary lit mask + maximal runs --------------------------------
        # Suppress rows at/below this track's detection trigger. Event timing
        # later advances by half a note so the visual centre reaches the line.
        lit = proj > thr
        scan_y_max = max(
            0, lane_config.trigger_y_top - self.cal.playfield_top)
        if scan_y_max < lit.shape[0]:
            lit[scan_y_max:] = False
        runs: list[Run] = []
        short_run_max = (lane_config.note_height + 8
                         if self.short_run_max is None else self.short_run_max)
        for s, e in _find_runs(lit):
            if e - s < self.min_run_px:
                continue                       # noise — discard
            seg = proj[s:e]
            kind = "short" if (e - s) <= short_run_max else "long"
            runs.append(Run(
                y_start=int(s),
                y_end=int(e),
                kind=kind,
                peak_brightness=float(seg.max()),
                mean_brightness=float(seg.mean()),
            ))

        return Stage1Result(
            frame_index=frame_index,
            lane_index=lane.index,
            color=lane.color,
            roi_y_origin=roi_y_origin,
            threshold=thr,
            projection=proj,
            runs=runs,
        )

    def _overlay_projection(self, lane: LaneFrame, lane_config) -> np.ndarray:
        """Return per-row HSV-mask coverage for one fixed overlay ROI."""
        offset = lane_config.x_range[0] - lane_config.match_x_range[0]
        width = lane_config.x_range[1] - lane_config.x_range[0]
        hsv = cv2.cvtColor(lane.matching_roi[:, offset:offset + width],
                           cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        hue_match = np.zeros(hue.shape, dtype=bool)
        for low, high in lane_config.mask_hue_ranges:
            hue_match |= (hue >= low) & (hue <= high)
        mask = (hue_match
                & (saturation >= lane_config.mask_saturation_min)
                & (value >= lane_config.mask_value_min))
        return mask.mean(axis=1)

    # ------------------------------------------------------------------ #
    def detect_frame(self, pf: PreprocessedFrame) -> list[Stage1Result]:
        """Run Stage 1 on every lane of one PreprocessedFrame."""
        return [
            self.detect_lane(lane, frame_index=pf.frame_index,
                             roi_y_origin=pf.roi_y_origin)
            for lane in pf.lanes
        ]


# =============================================================================
# Helpers
# =============================================================================

def _find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return [(start, end_exclusive), ...] of every maximal True run."""
    if not mask.any():
        return []
    # sentinel-pad so edge runs produce clean rising/falling transitions
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return list(zip(starts.tolist(), ends.tolist()))
