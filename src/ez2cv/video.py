"""Stream a configured video as cropped, channel-selected frames.

Turns ``RunConfig + video`` into ``PreprocessedFrame`` objects. This module
does data shaping only:

  video decode -> per-frame channel prep -> two-ROI slicing -> packaging

It performs NO detection: no projection, no run-finding, no template matching,
no beat judgement. Those are detection. The Preprocessor is intentionally
skin/resolution/key-count agnostic — all that variability already lives inside
the RunConfig object produced by configuration.

Design invariants
-----------------
* Sequential decode only (no random seek) — 1-Pass walks the video once.
* Generator / streaming — memory stays flat regardless of video length.
* NO normalization (no histogram eq, no auto-exposure). RunConfig thresholds
  are absolute 0-255 values; normalizing would invalidate them.
* Coordinate convention: every ROI starts at row `playfield_top`, so a detection
  match's y_top maps to a full-frame y by `+ roi_y_origin`.

Usage
-----
    from ez2cv.config import load_config
    from ez2cv.video import Preprocessor
    pre = Preprocessor(load_config("config/song.toml"))
    for pf in pre:                       # one PreprocessedFrame per video frame
        for lane in pf.lanes:
            proj = lane.detection_roi.mean(axis=1)      # -> detection Stage 1
            ...                                          # -> detection Stage 2 etc.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from ez2cv.config import CHANNEL_EXTRACTORS, RunConfig


ALIGNMENT_DEADBAND_PX = 2


# =============================================================================
# Output data structures
# =============================================================================

@dataclass
class LaneFrame:
    """Preprocessed pixel data for one lane of one frame."""
    index: int                       # 0-based lane index -> cal.lanes[index]
    color: str                       # "white" | "cyan" | ...
    detection_roi: np.ndarray        # (H, lane_width) uint8, single channel
    matching_roi: np.ndarray         # (H, lane_width+2*margin, 3) uint8 BGR


@dataclass
class PreprocessedFrame:
    """Everything detection needs for a single video frame."""
    frame_index: int
    timestamp_ms: float
    roi_y_origin: int                # in-ROI y + this = full-frame y
    lanes: list[LaneFrame]
    beat_roi: np.ndarray             # (h, w) uint8, beat_channel of the POW LED


def _alignment_offset(frame: np.ndarray, cal: RunConfig) -> tuple[int, int]:
    """Locate the calibrated cyan judgment band and return panel translation."""
    band_y = getattr(cal, "alignment_band_y", None)
    limit = int(getattr(cal, "alignment_max_shift", 0))
    if band_y is None or limit <= 0:
        return 0, 0
    left = min(lane.x_range[0] for lane in cal.lanes)
    right = max(lane.x_range[1] for lane in cal.lanes)
    top, bottom = band_y
    y1, y2 = max(0, top - limit), min(frame.shape[0], bottom + limit)
    x1, x2 = max(0, left - limit), min(frame.shape[1], right + limit + 1)
    region = frame[y1:y2, x1:x2]
    blue, green, red = cv2.split(region)
    blue = blue.astype(np.int16)
    mask = ((blue > 100) & (blue > green.astype(np.int16) + 20)
            & (blue > red.astype(np.int16) + 40))

    def best_run(values: np.ndarray, threshold: float, expected_length: int,
                 expected_start: int) -> tuple[int, int] | None:
        indices = np.flatnonzero(values >= threshold)
        if not len(indices):
            return None
        breaks = np.flatnonzero(np.diff(indices) > 1) + 1
        runs = np.split(indices, breaks)
        candidates = [(int(run[0]), int(run[-1]) + 1) for run in runs
                      if abs(len(run) - expected_length) <= 2]
        if not candidates:
            return None
        return min(candidates, key=lambda run: abs(run[0] - expected_start))

    expected_width = right - left + 1
    expected_height = bottom - top
    row_run = best_run(mask.sum(axis=1), 0.8 * expected_width,
                       expected_height, top - y1)
    if row_run is None:
        raise RuntimeError("calibrated judgment band was not found")
    rows = mask[row_run[0]:row_run[1]]
    column_run = best_run(rows.sum(axis=0), 0.8 * len(rows),
                          expected_width, left - x1)
    if column_run is None:
        raise RuntimeError("calibrated judgment band width does not match")
    dx = x1 + column_run[0] - left
    dy = y1 + row_run[0] - top
    dx = 0 if abs(dx) <= ALIGNMENT_DEADBAND_PX else dx
    dy = 0 if abs(dy) <= ALIGNMENT_DEADBAND_PX else dy
    if abs(dx) > limit or abs(dy) > limit:
        raise RuntimeError(
            f"panel translation ({dx}, {dy}) exceeds {limit}px calibration limit")
    bx1, by1, bx2, by2 = cal.beat_roi
    if (cal.playfield_top + dy < 0
            or cal.playfield_bottom + dy > frame.shape[0]
            or left + dx < 0 or right + dx > frame.shape[1]
            or bx1 + dx < 0 or bx2 + dx > frame.shape[1]
            or by1 + dy < 0 or by2 + dy > frame.shape[0]):
        raise RuntimeError("aligned panel ROIs fall outside the video frame")
    return dx, dy


# =============================================================================
# Preprocessor
# =============================================================================

class Preprocessor:
    """Streaming video preprocessing component. Iterate it to get PreprocessedFrames."""

    def __init__(
        self,
        cal: RunConfig,
        *,
        precrop: bool = True,
        mask_regions: list[tuple[int, int, int, int]] | None = None,
        force: bool = False,
    ):
        """
        Parameters
        ----------
        cal : RunConfig
            Resolved configuration object.
        precrop : bool
            If True, run channel conversion on an X-cropped band covering all
            ROIs instead of the full 1920px width (a pure speed optimization;
            output ROIs are bit-identical either way).
        mask_regions : list of (x1, y1, x2, y2), optional
            Full-frame rectangles zeroed out on the detection channels before
            slicing — used to kill UI overlays (e.g. the "FAIL" text that sits
            over the top of lanes 2-3). Matching ROIs are left untouched since
            Stage 2 only looks at bands around Stage 1 candidates anyway.
        """
        self.cal = cal
        self.precrop = precrop
        self.mask_regions = mask_regions or []
        self.force = force

        # --- channel dedup: which detection channels are actually used ------
        self._channels = sorted({ln.detection_channel for ln in cal.lanes})

        # --- X-crop band (only if precrop) ----------------------------------
        bx1, by1, bx2, by2 = cal.beat_roi
        xs_lo = [ln.match_x_range[0] for ln in cal.lanes] + [bx1]
        xs_hi = [ln.match_x_range[1] for ln in cal.lanes] + [bx2]
        self._crop_x0 = min(xs_lo) if precrop else 0
        self._crop_x1 = max(xs_hi) if precrop else None  # None -> full width

        self._frame_count: int | None = None
        self.timestamp_source = "container_pts"
        self.alignment_offset = (0, 0)

    # ------------------------------------------------------------------ #
    # introspection
    # ------------------------------------------------------------------ #
    @property
    def frame_count(self) -> int:
        """Best-effort frame count (container metadata; may be approximate)."""
        if self._frame_count is None:
            cap = self._open()
            self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        return self._frame_count

    # ------------------------------------------------------------------ #
    # iteration
    # ------------------------------------------------------------------ #
    def __iter__(self) -> Iterator[PreprocessedFrame]:
        """Open a fresh capture and yield one PreprocessedFrame per frame."""
        cap = self._open()
        self._verify_stream(cap)
        cal = self.cal
        t, b = cal.playfield_top, cal.playfield_bottom
        base_x0 = self._crop_x0
        bx1, by1, bx2, by2 = cal.beat_roi
        beat_extract = CHANNEL_EXTRACTORS[cal.beat_channel]
        try:
            idx = 0
            pts_origin: float | None = None
            previous_ms = -1.0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if idx == 0:
                    try:
                        self.alignment_offset = _alignment_offset(frame, cal)
                    except RuntimeError as exc:
                        if not self.force:
                            raise
                        print(f"[preprocessor] WARNING: {exc} (--force)")
                align_x, align_y = self.alignment_offset

                pts = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                if idx == 0:
                    pts_origin = pts if math.isfinite(pts) and pts >= 0 else None
                timestamp_ms = (pts - pts_origin if pts_origin is not None
                                and math.isfinite(pts) else -1.0)
                if timestamp_ms <= previous_ms:
                    timestamp_ms = max(
                        idx / cal.fps * 1000.0,
                        previous_ms + 1000.0 / cal.fps)
                    self.timestamp_source = "configured_fps_fallback"
                previous_ms = timestamp_ms

                # working image: full frame, or X-cropped band
                physical_x0 = (0 if self._crop_x1 is None
                               else base_x0 + align_x)
                work = (frame if self._crop_x1 is None else
                        frame[:, physical_x0:self._crop_x1 + align_x])

                # --- channel dedup: convert each used channel once ----------
                chans: dict[str, np.ndarray] = {}
                for name in self._channels:
                    img = CHANNEL_EXTRACTORS[name](work)   # fresh array
                    for mx1, my1, mx2, my2 in self.mask_regions:
                        img[my1:my2,
                            max(0, mx1 - physical_x0):max(0, mx2 - physical_x0)] = 0
                    chans[name] = img

                # --- per-lane two-ROI slicing -------------------------------
                lanes: list[LaneFrame] = []
                for ln in cal.lanes:
                    dx1 = ln.x_range[0] + align_x - physical_x0
                    dx2 = ln.x_range[1] + align_x - physical_x0
                    mx1 = ln.match_x_range[0] + align_x - physical_x0
                    mx2 = ln.match_x_range[1] + align_x - physical_x0
                    det = np.ascontiguousarray(chans[ln.detection_channel][
                        t + align_y:b + align_y, dx1:dx2])
                    mat = np.ascontiguousarray(work[
                        t + align_y:b + align_y, mx1:mx2])
                    lanes.append(LaneFrame(
                        index=ln.index,
                        color=ln.color,
                        detection_roi=det,
                        matching_roi=mat,
                    ))

                # --- beat ROI -----------------------------------------------
                beat = beat_extract(np.ascontiguousarray(
                    work[by1 + align_y:by2 + align_y,
                         bx1 + align_x - physical_x0:
                         bx2 + align_x - physical_x0]))

                yield PreprocessedFrame(
                    frame_index=idx,
                    timestamp_ms=timestamp_ms,
                    roi_y_origin=t,
                    lanes=lanes,
                    beat_roi=beat,
                )
                idx += 1
            self._frame_count = idx
        finally:
            cap.release()

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _open(self) -> cv2.VideoCapture:
        path = self.cal.video_path
        if not Path(path).is_file():
            raise FileNotFoundError(f"video not found: {path}")
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"failed to open video: {path}")
        return cap

    def _verify_stream(self, cap: cv2.VideoCapture) -> None:
        """Re-check resolution/fps at decode time (the file may have changed)."""
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        exp_w, exp_h = self.cal.display_resolution
        if (w, h) != (exp_w, exp_h):
            raise RuntimeError(
                f"video is {w}x{h} but calibration expects {exp_w}x{exp_h} — "
                f"calibration is resolution-locked.")
        if abs(fps - self.cal.fps) > 0.5:
            message = (f"video fps {fps:.2f} != configured fps {self.cal.fps} "
                       f"— timing will drift")
            if not self.force:
                raise RuntimeError(message)
            print(f"[preprocessor] WARNING: {message} (--force)")
