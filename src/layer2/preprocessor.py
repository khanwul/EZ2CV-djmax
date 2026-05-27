"""
EZ2CV — Layer 2 : Preprocessor
===============================================================================
Turns "Calibration object + video file" into a STREAM of PreprocessedFrame
objects that Layer 3 consumes directly. This layer does data shaping only:

  video decode -> per-frame channel prep -> two-ROI slicing -> packaging

It performs NO detection: no projection, no run-finding, no template matching,
no beat judgement. Those are Layer 3. The Preprocessor is intentionally
skin/resolution/key-count agnostic — all that variability already lives inside
the Calibration object produced by Layer 1.

Design invariants
-----------------
* Sequential decode only (no random seek) — 1-Pass walks the video once.
* Generator / streaming — memory stays flat regardless of video length.
* NO normalization (no histogram eq, no auto-exposure). Calibration thresholds
  are absolute 0-255 values; normalizing would invalidate them.
* Coordinate convention: every ROI starts at row `playfield_top`, so a Layer 3
  match's y_top maps to a full-frame y by `+ roi_y_origin`; an in-ROI x maps by
  `+ lane.match_x_origin`.

Usage
-----
    from layer2.preprocessor import Preprocessor
    pre = Preprocessor.from_config("config/song.toml")
    for pf in pre:                       # one PreprocessedFrame per video frame
        for lane in pf.lanes:
            proj = lane.detection_roi.mean(axis=1)      # -> Layer 3 Stage 1
            ...                                          # -> Layer 3 Stage 2 etc.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from layer1.calibration import Calibration, CHANNEL_EXTRACTORS, resolve_calibration


# =============================================================================
# Output data structures
# =============================================================================

@dataclass
class LaneFrame:
    """Preprocessed pixel data for one lane of one frame."""
    index: int                       # 0-based lane index -> cal.lanes[index]
    color: str                       # "white" | "cyan" | ...
    detection_channel: str           # key into CHANNEL_EXTRACTORS
    detection_roi: np.ndarray        # (H, lane_width) uint8, single channel
    matching_roi: np.ndarray         # (H, lane_width+2*margin, 3) uint8 BGR
    match_x_origin: int              # in-ROI x  + this = full-frame x
    x_range: tuple[int, int]         # detection ROI x in full-frame coords
    match_x_range: tuple[int, int]   # matching  ROI x in full-frame coords


@dataclass
class PreprocessedFrame:
    """Everything Layer 3 needs for a single video frame."""
    frame_index: int
    timestamp_ms: float
    roi_y_origin: int                # in-ROI y + this = full-frame y
    lanes: list[LaneFrame]
    beat_roi: np.ndarray             # (h, w) uint8, beat_channel of the POW LED
    measure_roi: np.ndarray          # (H, full_playfield_width) uint8, single channel for measure line detection


# =============================================================================
# Preprocessor
# =============================================================================

class Preprocessor:
    """Streaming Layer 2 component. Iterate it to get PreprocessedFrames."""

    def __init__(
        self,
        cal: Calibration,
        *,
        precrop: bool = True,
        mask_regions: list[tuple[int, int, int, int]] | None = None,
    ):
        """
        Parameters
        ----------
        cal : Calibration
            Resolved Layer 1 object.
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

        # --- channel dedup: which detection channels are actually used ------
        self._channels = sorted(
            {ln.detection_channel for ln in cal.lanes} | {cal.measure_line_channel}
        )

        # --- X-crop band (only if precrop) ----------------------------------
        bx1, by1, bx2, by2 = cal.beat_roi
        mlx1, mlx2 = cal.measure_line_x_range
        xs_lo = [ln.match_x_range[0] for ln in cal.lanes] + [bx1, mlx1]
        xs_hi = [ln.match_x_range[1] for ln in cal.lanes] + [bx2, mlx2]
        self._crop_x0 = min(xs_lo) if precrop else 0
        self._crop_x1 = max(xs_hi) if precrop else None  # None -> full width

        self._frame_count: int | None = None

    # ------------------------------------------------------------------ #
    # constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, song_toml_path: str | Path, **kwargs) -> "Preprocessor":
        """Resolve a config and build a Preprocessor in one call."""
        return cls(resolve_calibration(song_toml_path), **kwargs)

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
        x0 = self._crop_x0
        bx1, by1, bx2, by2 = cal.beat_roi
        beat_extract = CHANNEL_EXTRACTORS[cal.beat_channel]
        mlx1, mlx2 = cal.measure_line_x_range
        ml_extract = CHANNEL_EXTRACTORS[cal.measure_line_channel]
        try:
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                # working image: full frame, or X-cropped band
                work = frame if self._crop_x1 is None \
                    else frame[:, x0:self._crop_x1]

                # --- channel dedup: convert each used channel once ----------
                chans: dict[str, np.ndarray] = {}
                for name in self._channels:
                    img = CHANNEL_EXTRACTORS[name](work)   # fresh array
                    for mx1, my1, mx2, my2 in self.mask_regions:
                        img[my1:my2, max(0, mx1 - x0):max(0, mx2 - x0)] = 0
                    chans[name] = img

                # --- per-lane two-ROI slicing -------------------------------
                lanes: list[LaneFrame] = []
                for ln in cal.lanes:
                    dx1, dx2 = ln.x_range[0] - x0, ln.x_range[1] - x0
                    mx1, mx2 = ln.match_x_range[0] - x0, ln.match_x_range[1] - x0
                    det = np.ascontiguousarray(chans[ln.detection_channel][t:b, dx1:dx2])
                    mat = np.ascontiguousarray(work[t:b, mx1:mx2])
                    lanes.append(LaneFrame(
                        index=ln.index,
                        color=ln.color,
                        detection_channel=ln.detection_channel,
                        detection_roi=det,
                        matching_roi=mat,
                        match_x_origin=ln.match_x_range[0],
                        x_range=ln.x_range,
                        match_x_range=ln.match_x_range,
                    ))

                # --- beat ROI -----------------------------------------------
                beat = beat_extract(np.ascontiguousarray(
                    work[by1:by2, bx1 - x0:bx2 - x0]))

                # --- measure line ROI (full playfield width, single channel) -
                measure = ml_extract(np.ascontiguousarray(
                    work[t:b, mlx1 - x0:mlx2 - x0]))

                yield PreprocessedFrame(
                    frame_index=idx,
                    timestamp_ms=idx / cal.fps * 1000.0,
                    roi_y_origin=t,
                    lanes=lanes,
                    beat_roi=beat,
                    measure_roi=measure,
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
            print(f"[preprocessor] WARNING: video fps {fps:.2f} != configured "
                  f"fps {self.cal.fps} — timing will drift.")


# =============================================================================
# CLI: python preprocessor.py [config/song.toml]
# =============================================================================

if __name__ == "__main__":
    import sys, time

    path = sys.argv[1] if len(sys.argv) > 1 else "config/song.toml"
    pre = Preprocessor.from_config(path)
    print(f"video      : {pre.cal.video_path}")
    print(f"frame_count: {pre.frame_count}")
    print(f"precrop    : {pre.precrop}  (X band "
          f"{pre._crop_x0}..{pre._crop_x1})")
    print(f"channels   : {pre._channels}")

    t0 = time.time()
    n = 0
    first = None
    for pf in pre:
        if first is None:
            first = pf
        n += 1
    dt = time.time() - t0

    print(f"\niterated {n} frames in {dt:.2f}s "
          f"({n / dt:.0f} fps preprocessing throughput)")
    if first:
        ln = first.lanes[0]
        print(f"frame 0 @ {first.timestamp_ms:.2f}ms  roi_y_origin="
              f"{first.roi_y_origin}")
        print(f"  L1 detection_roi {ln.detection_roi.shape} {ln.detection_roi.dtype}"
              f"  matching_roi {ln.matching_roi.shape}")
        print(f"  beat_roi {first.beat_roi.shape} {first.beat_roi.dtype}")
