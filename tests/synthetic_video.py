"""Procedural, copyright-free video fixture for the public end-to-end test."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ez2cv.config import LaneConfig, MeasureLineConfig, RunConfig


FPS = 60.0
SPEED = 4.0
TRIGGER = 120
LINE_Y = 130
NOTE_HEIGHT = 10
LANES = ((60, 90), (90, 120))
BEAT_FRAMES = tuple(range(20, 261, 30))
BARLINE_FRAMES = (20, 140, 260)


def _sprite(kind: str) -> np.ndarray:
    image = np.full((NOTE_HEIGHT, 30, 3), 180, dtype=np.uint8)
    if kind == "note":
        image[:, 12:18] = 250
        image[2:8, 14:16] = 40
    elif kind == "lnhead":
        image[:, 4:10] = 245
        image[2:8, 6:8] = 30
    else:
        image[:, 20:26] = 245
        image[2:8, 22:24] = 30
    return image


def _blit(frame: np.ndarray, sprite: np.ndarray, x: int, y: int) -> None:
    h, w = sprite.shape[:2]
    if 0 <= y <= frame.shape[0] - h:
        frame[y:y + h, x:x + w] = sprite


def make_fixture(path: Path, alignment_offset=(0, 0),
                 alignment_visible_from=0) -> tuple[RunConfig, dict]:
    align_x, align_y = alignment_offset
    templates = {kind: _sprite(kind) for kind in ("note", "lnhead", "lntail")}
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"FFV1"), FPS, (200, 160))
    if not writer.isOpened():
        raise RuntimeError("FFV1 video writer unavailable")

    taps = ((0, 50), (1, 80), (1, 230))
    longnote = (0, 110, 200)       # lane, head crossing, release frame
    try:
        for frame_index in range(285):
            frame = np.zeros((160, 200, 3), dtype=np.uint8)
            if frame_index >= alignment_visible_from:
                frame[140 + align_y:145 + align_y,
                      LANES[0][0] + align_x:
                      LANES[-1][1] + align_x + 1, 0] = 220
            if frame_index - 1 in BEAT_FRAMES:
                frame[10 + align_y:20 + align_y,
                      10 + align_x:20 + align_x] = 255
            for crossing in BARLINE_FRAMES:
                y = int(round(LINE_Y + SPEED * (frame_index - crossing)))
                if 10 <= y < 150:
                    frame[y + align_y,
                          LANES[0][0] + align_x:
                          LANES[-1][1] + align_x] = 80
            for lane, crossing in taps:
                y = int(round(TRIGGER + SPEED * (frame_index - crossing)))
                _blit(frame, templates["note"],
                      LANES[lane][0] + align_x, y + align_y)

            lane, head_crossing, release = longnote
            tail_crossing = release - NOTE_HEIGHT / SPEED
            head_y = int(round(TRIGGER + SPEED * (frame_index - head_crossing)))
            tail_y = int(round(TRIGGER + SPEED * (frame_index - tail_crossing)))
            top, bottom = max(10, tail_y), min(150, head_y + NOTE_HEIGHT)
            if top < bottom:
                frame[top + align_y:bottom + align_y,
                      LANES[lane][0] + align_x:
                      LANES[lane][1] + align_x] = 110
            _blit(frame, templates["lntail"],
                  LANES[lane][0] + align_x, tail_y + align_y)
            _blit(frame, templates["lnhead"],
                  LANES[lane][0] + align_x, head_y + align_y)
            writer.write(frame)
    finally:
        writer.release()

    lanes = [LaneConfig(
        index=i, name=f"K{i + 1}", role="normal", input_type="key",
        color="white", template_set="synthetic",
        allowed_types=frozenset(("tap", "longnote")),
        x_range=x_range, match_x_range=x_range,
        note_height=NOTE_HEIGHT, trigger_y_top=TRIGGER,
        timing_offset_px=0, tail_release_offset_px=0,
        tail_search_y_max=TRIGGER, min_longnote_px=10,
        include_in_consensus=True,
        detection_channel="gray", stage1_threshold=50.0,
        matching_threshold=0.75, templates=templates)
        for i, x_range in enumerate(LANES)]
    config = RunConfig(
        song_name="synthetic", difficulty="SC", game="synthetic",
        skin_name="synthetic", key_mode="2k", display_resolution=(200, 160),
        template_scale=1.0, video_path=path, fps=FPS, note_speed=1.0,
        tick_resolution=192, min_bpm=120.0, max_bpm=120.0,
        playfield_top=10, playfield_bottom=150, line_y=LINE_Y,
        normal_lane_count=2, lanes=lanes,
        beat_roi=(10, 10, 20, 20), beat_channel="gray",
        beat_diff_threshold=10.0,
        measure_line=MeasureLineConfig(20.0, 130.0, 3, 0, 50.0),
        pixels_per_frame=SPEED,
        alignment_band_y=(140, 145), alignment_max_shift=8,
        provenance={"fixture": "procedural"})

    ms = lambda frame: frame / FPS * 1000.0
    truth = {
        "notes": [
            {"lane": 0, "type": "tap", "start_ms": ms(50), "end_ms": None,
             "start_tick": 192, "end_tick": None},
            {"lane": 1, "type": "tap", "start_ms": ms(80), "end_ms": None,
             "start_tick": 384, "end_tick": None},
            {"lane": 0, "type": "longnote", "start_ms": ms(110),
             "end_ms": ms(200), "start_tick": 576, "end_tick": 1152},
            {"lane": 1, "type": "tap", "start_ms": ms(230), "end_ms": None,
             "start_tick": 1344, "end_tick": None},
        ],
        "beats_ms": [ms(frame) for frame in BEAT_FRAMES],
        "barlines_ms": [ms(frame) for frame in BARLINE_FRAMES],
        "tempo_segments": [{"start_tick": 0, "end_tick": 1536,
                             "bpm": 120.0, "interpolation": "step"}],
    }
    return config, truth
