"""
EZ2CV — Layer 1 : Calibration resolver
===============================================================================
Merges the 3-tier config (song.toml + skin.toml + profile.toml) into ONE flat,
validated `Calibration` object. Layer 2+ depend ONLY on this object and never
read TOML themselves — so the pipeline stays skin/resolution/key-count agnostic.

Resolution flow
---------------
    song.toml
      |-- setup.skin               -> skins/<skin>/skin.toml
      |-- setup.key_mode           -> lane_colors[mode], templates dir <mode>/
      |-- setup.display_resolution -> profiles/<resolution>/<key_mode>.toml
                |
                v
        resolve_calibration()  ->  Calibration

Usage
-----
    from layer1.calibration import resolve_calibration
    cal = resolve_calibration("config/song.toml")
    cal.summary()                       # human-readable dump
    for lane in cal.lanes:              # Layer 2 iterates here
        roi   = frame[cal.playfield_top:cal.playfield_bottom,
                      lane.x_range[0]:lane.x_range[1]]          # detection ROI
        mroi  = frame[cal.playfield_top:cal.playfield_bottom,
                      lane.match_x_range[0]:lane.match_x_range[1]]  # matching ROI
        chan  = CHANNEL_EXTRACTORS[lane.detection_channel](roi)     # Stage 1 channel
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# =============================================================================
# Detection-channel extractors
# -----------------------------------------------------------------------------
# Map a channel NAME (declared per-color in skin.toml [detection.channel]) to a
# function: BGR uint8 image -> single-channel uint8 image. Layer 2 calls these
# to build the Stage 1 projection input. Add new channels here only.
# =============================================================================

def _ch_gray(bgr):   return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
def _ch_blue(bgr):   return bgr[:, :, 0].copy()
def _ch_green(bgr):  return bgr[:, :, 1].copy()
def _ch_red(bgr):    return bgr[:, :, 2].copy()
def _ch_max_bg(bgr): return np.maximum(bgr[:, :, 0], bgr[:, :, 1])  # blue|green
def _ch_max_gr(bgr): return np.maximum(bgr[:, :, 1], bgr[:, :, 2])  # green|red

CHANNEL_EXTRACTORS = {
    "gray":   _ch_gray,
    "blue":   _ch_blue,
    "green":  _ch_green,
    "red":    _ch_red,
    "max_bg": _ch_max_bg,
    "max_gr": _ch_max_gr,
}


# =============================================================================
# Dataclasses — the resolved, flat Calibration interface
# =============================================================================

@dataclass
class MeasureLineConfig:
    """Per-skin measure-line detection tunables (Layer 3 consumes these).

    The Layer 3 detector looks for a thin, mid-grey, near-full-width band by
    stacking per-lane projections and counting lit lanes per row. Each field
    here corresponds to one filter:

    * min_brightness : a lane row must exceed this to count as "lit"
    * max_brightness : reject note-bright bands (~218 on the ez2on skin)
    * max_thickness  : reject thicker bands (a 22px note chord; the bar; glow)
    * lane_slack     : allow this many lanes to miss the coincidence (occlusion
                       or split-frame), so the gate is key_count - lane_slack
    """
    channel: str
    min_brightness: float
    max_brightness: float
    max_thickness: int
    lane_slack: int


@dataclass
class LaneCalibration:
    """Everything Layer 2 needs for a single lane."""
    index: int                          # 0-based lane index
    color: str                          # "white" | "cyan" | ...
    x_range: tuple[int, int]            # DETECTION ROI x (tight, == lane width)
    match_x_range: tuple[int, int]      # MATCHING ROI x (lane +/- roi_x_margin)
    detection_channel: str              # key into CHANNEL_EXTRACTORS
    stage1_threshold: float             # Stage 1 brightness gate (0-255 mean)
    matching_threshold: float           # Stage 2 template-match min score
    templates: dict[str, np.ndarray]    # {"note","lnhead","lntail"} -> BGR image

    @property
    def match_x_origin(self) -> int:
        """Add this to a match's in-ROI x to get a full-frame x."""
        return self.match_x_range[0]


@dataclass
class Calibration:
    """The single object the whole pipeline depends on."""
    # --- provenance ----------------------------------------------------------
    skin_name: str
    key_mode: str
    display_resolution: tuple[int, int]
    config_root: Path
    template_scale: float               # profile_w / skin_reference_w (1.0 here)

    # --- capture -------------------------------------------------------------
    video_path: Path
    fps: float
    note_speed: float

    # --- song ----------------------------------------------------------------
    tick_resolution: int
    min_bpm: float
    max_bpm: float

    # --- playfield / judgment ------------------------------------------------
    playfield_top: int
    playfield_bottom: int
    line_y: int
    oscillation_range: int
    note_height: int
    note_width: int
    trigger_template_y_top: int

    # --- lanes ---------------------------------------------------------------
    key_count: int
    lanes: list[LaneCalibration]

    # --- matching geometry ---------------------------------------------------
    roi_x_margin: int
    tail_search_y_max: int

    # --- beat indicator ------------------------------------------------------
    beat_roi: tuple[int, int, int, int]   # (x1, y1, x2, y2)
    beat_channel: str
    beat_detection_method: str
    beat_diff_threshold: float

    # --- measure line -------------------------------------------------------
    measure_line_x_range: tuple[int, int]   # (field_left, field_right) full playfield width
    measure_line_channel: str               # key into CHANNEL_EXTRACTORS
    measure_line: MeasureLineConfig         # Layer 3 detector tunables

    # --- measurements --------------------------------------------------------
    pixels_per_frame: float
    frames_per_traverse: int
    min_longnote_px: int                    # head-tail gap below this -> tap

    def lane_detection_roi(self, frame: np.ndarray) -> list[np.ndarray]:
        """Convenience: tight detection ROIs for every lane (Layer 2 helper)."""
        t, b = self.playfield_top, self.playfield_bottom
        return [frame[t:b, ln.x_range[0]:ln.x_range[1]] for ln in self.lanes]

    def summary(self) -> None:
        print(f"=== Calibration: {self.skin_name} / {self.key_mode} / "
              f"{self.display_resolution[0]}x{self.display_resolution[1]} ===")
        print(f"  video      : {self.video_path}")
        print(f"  fps/speed  : {self.fps} / {self.note_speed}   "
              f"tick_res={self.tick_resolution}  bpm=[{self.min_bpm},{self.max_bpm}]")
        print(f"  playfield  : y[{self.playfield_top},{self.playfield_bottom}]  "
              f"line_y={self.line_y}  trigger_y_top={self.trigger_template_y_top}")
        print(f"  note       : {self.note_width}x{self.note_height}px  "
              f"template_scale={self.template_scale:.4f}")
        print(f"  beat ROI   : {self.beat_roi}  ch={self.beat_channel} "
              f"{self.beat_detection_method} thr={self.beat_diff_threshold}")
        print(f"  measure ln : x{list(self.measure_line_x_range)}  "
              f"ch={self.measure_line_channel}  "
              f"bright=[{self.measure_line.min_brightness},"
              f"{self.measure_line.max_brightness}]  "
              f"max_thick={self.measure_line.max_thickness}  "
              f"slack={self.measure_line.lane_slack}")
        print(f"  min_ln_px  : {self.min_longnote_px}")
        print(f"  lanes ({self.key_count}):")
        for ln in self.lanes:
            print(f"    L{ln.index+1}: {ln.color:5s} det_x{list(ln.x_range)} "
                  f"match_x{list(ln.match_x_range)} ch={ln.detection_channel:6s} "
                  f"s1={ln.stage1_threshold} s2={ln.matching_threshold} "
                  f"tmpl={list(ln.templates)}")


# =============================================================================
# Helpers
# =============================================================================

class CalibrationError(Exception):
    """Raised on a fatal config inconsistency."""


def _load_toml(path: Path) -> dict:
    if not path.is_file():
        raise CalibrationError(f"config file not found: {path}")
    with open(path, "rb") as f:
        return tomllib.load(f)


def _warn(msg: str) -> None:
    print(f"[calibration] WARNING: {msg}")


# =============================================================================
# Main resolver
# =============================================================================

def resolve_calibration(song_toml_path: str | Path) -> Calibration:
    """Load song.toml and merge skin + profile into one validated Calibration."""
    song_path = Path(song_toml_path).resolve()
    config_root = song_path.parent
    song = _load_toml(song_path)

    # --- 1. read references --------------------------------------------------
    setup       = song["setup"]
    skin_name   = setup["skin"]
    key_mode    = setup["key_mode"]
    res_str     = setup["display_resolution"]

    skin_path    = config_root / "skins" / skin_name / "skin.toml"
    profile_path = config_root / "profiles" / res_str / f"{key_mode}.toml"
    skin    = _load_toml(skin_path)
    profile = _load_toml(profile_path)

    # --- 2. cross-file consistency checks -----------------------------------
    key_count   = profile["meta"]["key_count"]
    lane_colors = skin["lane_colors"].get(key_mode)
    if lane_colors is None:
        raise CalibrationError(
            f"skin '{skin_name}' has no lane_colors entry for key_mode '{key_mode}'")
    if len(lane_colors) != key_count:
        raise CalibrationError(
            f"key_count mismatch: profile says {key_count}, skin lane_colors"
            f"['{key_mode}'] has {len(lane_colors)} entries")

    prof_res = tuple(profile["meta"]["display_resolution"])
    if f"{prof_res[0]}x{prof_res[1]}" != res_str:
        _warn(f"profile meta resolution {prof_res} != song display_resolution "
              f"'{res_str}'")

    # --- 3. template scale (skin reference res -> profile res) --------------
    ref_res = tuple(skin["skin"]["reference_resolution"])
    template_scale = prof_res[0] / ref_res[0]
    if abs(template_scale - prof_res[1] / ref_res[1]) > 1e-6:
        _warn("non-uniform x/y scale between skin reference and profile "
              "resolution — aspect ratios differ; templates may not match.")

    # --- 4. judgment / trigger geometry -------------------------------------
    jd          = profile["judgment"]
    line_y      = jd["line_y"]
    note_height = jd["note_height"]
    trigger     = jd["trigger_template_y_top"]
    expected_trigger = line_y - note_height
    if trigger != expected_trigger:
        _warn(f"trigger_template_y_top={trigger} but line_y-note_height="
              f"{expected_trigger}; check the judgment geometry.")

    # --- 5. resolve templates & per-lane data -------------------------------
    pf          = profile["playfield"]
    lanes_cfg   = profile["lanes"]
    field_left  = lanes_cfg["field_left"]
    lane_width  = lanes_cfg["lane_width"]
    margin      = profile["matching"]["roi_x_margin"]

    thr               = skin["thresholds"]
    matching_default  = thr["matching"]
    stage1_default    = thr["stage1_brightness"]
    matching_by_color = thr.get("matching_by_color", {})
    stage1_by_color   = thr.get("stage1_by_color", {})
    channel_map       = skin["detection"]["channel"]
    templates_cfg     = skin["templates"]
    tmpl_dir          = config_root / "skins" / skin_name / key_mode

    frame_w, frame_h = prof_res
    template_cache: dict[str, np.ndarray] = {}   # color->type cached by filename

    def _load_template(color: str, ntype: str) -> np.ndarray:
        if color not in templates_cfg:
            raise CalibrationError(
                f"skin '{skin_name}' has no [templates.{color}] section "
                f"(needed by key_mode '{key_mode}')")
        fname = templates_cfg[color][ntype]
        if fname in template_cache:
            return template_cache[fname]
        fpath = tmpl_dir / fname
        if not fpath.is_file():
            raise CalibrationError(f"template image not found: {fpath}")
        img = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
        if img is None:
            raise CalibrationError(f"failed to decode template: {fpath}")
        if abs(template_scale - 1.0) > 1e-6:
            interp = cv2.INTER_AREA if template_scale < 1 else cv2.INTER_CUBIC
            img = cv2.resize(img, None, fx=template_scale, fy=template_scale,
                             interpolation=interp)
        template_cache[fname] = img
        return img

    lanes: list[LaneCalibration] = []
    for i, color in enumerate(lane_colors):
        x1 = field_left + i * lane_width
        x2 = x1 + lane_width
        mx1 = max(0, x1 - margin)
        mx2 = min(frame_w, x2 + margin)

        if color not in channel_map:
            raise CalibrationError(
                f"skin '{skin_name}' [detection.channel] has no entry for "
                f"color '{color}'")
        channel = channel_map[color]
        if channel not in CHANNEL_EXTRACTORS:
            raise CalibrationError(
                f"unknown detection channel '{channel}' for color '{color}' "
                f"(valid: {sorted(CHANNEL_EXTRACTORS)})")

        templates = {t: _load_template(color, t)
                     for t in ("note", "lnhead", "lntail")}

        lanes.append(LaneCalibration(
            index=i,
            color=color,
            x_range=(x1, x2),
            match_x_range=(mx1, mx2),
            detection_channel=channel,
            stage1_threshold=float(stage1_by_color.get(color, stage1_default)),
            matching_threshold=float(matching_by_color.get(color, matching_default)),
            templates=templates,
        ))

    # --- 6. validate template size against profile geometry -----------------
    th, tw = lanes[0].templates["note"].shape[:2]
    meas   = profile["measurements"]
    if th != note_height:
        _warn(f"template height {th}px != profile note_height {note_height}px")
    if tw != meas["note_width_px"]:
        _warn(f"template width {tw}px != profile note_width_px "
              f"{meas['note_width_px']}px")
    if tw != lane_width:
        _warn(f"template width {tw}px != lane_width {lane_width}px — "
              f"matchTemplate needs roi_x_margin>0 for slide room.")

    # --- 7. field-bounds sanity ---------------------------------------------
    field_right = field_left + key_count * lane_width
    if field_right > frame_w:
        raise CalibrationError(
            f"playfield right edge {field_right} exceeds frame width {frame_w}")

    # --- 8. capture / video --------------------------------------------------
    cap_cfg    = song["capture"]
    video_path = Path(cap_cfg["video_path"])
    if cap_cfg["video_path"] and not video_path.is_absolute():
        video_path = (config_root / video_path).resolve()
    if cap_cfg["video_path"]:
        _verify_video(video_path, frame_w, frame_h, cap_cfg["fps"])
    else:
        _warn("capture.video_path is empty — set it before running the pipeline.")

    # --- 9. assemble ---------------------------------------------------------
    bi_geom = profile["beat_indicator"]["roi"]
    bi_skin = skin["beat_indicator"]
    mt      = profile["matching"]

    ml_skin = skin["measure_line"]
    ml_channel = ml_skin["channel"]
    if ml_channel not in CHANNEL_EXTRACTORS:
        raise CalibrationError(
            f"unknown measure_line channel '{ml_channel}' "
            f"(valid: {sorted(CHANNEL_EXTRACTORS)})")
    measure_line_cfg = MeasureLineConfig(
        channel=ml_channel,
        min_brightness=float(ml_skin["min_brightness"]),
        max_brightness=float(ml_skin["max_brightness"]),
        max_thickness=int(ml_skin["max_thickness"]),
        lane_slack=int(ml_skin["lane_slack"]),
    )

    return Calibration(
        skin_name=skin_name,
        key_mode=key_mode,
        display_resolution=prof_res,
        config_root=config_root,
        template_scale=template_scale,
        video_path=video_path,
        fps=float(cap_cfg["fps"]),
        note_speed=float(cap_cfg["note_speed"]),
        tick_resolution=int(song["song"]["resolution"]),
        min_bpm=float(song["song"]["min_bpm"]),
        max_bpm=float(song["song"]["max_bpm"]),
        playfield_top=pf["top"],
        playfield_bottom=pf["bottom"],
        line_y=line_y,
        oscillation_range=jd["oscillation_range"],
        note_height=note_height,
        note_width=tw,
        trigger_template_y_top=trigger,
        key_count=key_count,
        lanes=lanes,
        roi_x_margin=margin,
        tail_search_y_max=mt["tail_search_y_max"],
        beat_roi=tuple(bi_geom),
        beat_channel=bi_skin["channel"],
        beat_detection_method=bi_skin["detection_method"],
        beat_diff_threshold=float(bi_skin["diff_threshold"]),
        measure_line_x_range=(field_left, field_right),
        measure_line_channel=ml_channel,
        measure_line=measure_line_cfg,
        pixels_per_frame=float(meas["pixels_per_frame"]),
        frames_per_traverse=int(meas["frames_per_traverse"]),
        min_longnote_px=int(meas["min_longnote_px"]),
    )


def _verify_video(path: Path, exp_w: int, exp_h: int, exp_fps: float) -> None:
    """Warn (don't crash) if the video does not match the calibrated profile."""
    if not path.is_file():
        _warn(f"video not found: {path}")
        return
    cap = cv2.VideoCapture(str(path))
    try:
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if (w, h) != (exp_w, exp_h):
            _warn(f"video is {w}x{h} but profile is {exp_w}x{exp_h} — "
                  f"calibration is resolution-locked.")
        if abs(fps - exp_fps) > 0.5:
            _warn(f"video fps {fps:.2f} != configured fps {exp_fps} — "
                  f"timing will be off.")
    finally:
        cap.release()


# =============================================================================
# CLI: python calibration.py [config/song.toml]
# =============================================================================

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "config/song.toml"
    cal = resolve_calibration(path)
    cal.summary()
