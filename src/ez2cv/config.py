"""Resolve song, skin, and geometry TOML into one validated ``RunConfig``.

Resolution flow
---------------
    song.toml
      |-- setup.skin               -> skins/<skin>/skin.toml
      |-- setup.key_mode           -> lane_colors[mode], templates dir <mode>/
      |-- setup.display_resolution -> profiles/<resolution>/<key_mode>.toml
                |
                v
        load_config()  ->  RunConfig

Usage
-----
    from ez2cv.config import load_config
    config = load_config("config/song.toml")
    config.summary()
    for lane in config.lanes:
        roi   = frame[config.playfield_top:config.playfield_bottom,
                      lane.x_range[0]:lane.x_range[1]]          # detection ROI
        mroi  = frame[config.playfield_top:config.playfield_bottom,
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
# function: BGR uint8 image -> single-channel uint8 image. video preprocessing calls these
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
# Dataclasses — the resolved, flat RunConfig interface
# =============================================================================

@dataclass
class MeasureLineConfig:
    """Per-skin measure-line detection tunables.

    The detection detector looks for a thin, mid-grey, near-full-width band by
    stacking per-lane projections and counting lit lanes per row. To survive
    sub-pixel motion (a 1px line scrolling ~33px/frame straddles two pixel rows
    and splits its energy below any single-row gate), the "lit" test runs on the
    2-row sliding energy SUM of each lane's projection, which recombines the
    split energy regardless of straddle phase. Each field is one filter:

    * lit_energy_threshold : a lane's 2-row energy sum must exceed this to count
                             as "lit" (recombined line energy; ~single-row line
                             brightness, since the sum collapses the split)
    * min_brightness : single-row band mean lower bound (rejects near-background)
    * max_brightness : reject note-bright bands (~218 on the ez2on skin)
    * max_thickness  : reject thicker bands (a 22px note chord; the bar; glow) —
                       this is what keeps long-note bodies from being mistaken
                       for the line even though the energy sum recovers dim rows
    * lane_slack     : allow this many lanes to miss the coincidence (occlusion
                       or split-frame), so the gate is key_count - lane_slack
    """
    min_brightness: float
    max_brightness: float
    max_thickness: int
    lane_slack: int
    lit_energy_threshold: float = 50.0


@dataclass
class LaneConfig:
    """Resolved settings for one normal or overlapping logical track."""
    index: int
    name: str
    role: str                           # "normal" | "overlay"
    input_type: str                     # "key" | "side" | "trigger"
    color: str
    template_set: str
    allowed_types: frozenset[str]       # "tap" | "longnote"
    x_range: tuple[int, int]
    match_x_range: tuple[int, int]
    note_height: int
    trigger_y_top: int
    timing_offset_px: int
    tail_release_offset_px: int
    tail_search_y_max: int
    min_longnote_px: int
    include_in_consensus: bool
    detection_channel: str
    stage1_threshold: float
    matching_threshold: float
    templates: dict[str, np.ndarray]
    mask_hue_ranges: tuple[tuple[int, int], ...] = ()
    mask_saturation_min: int = 0
    mask_value_min: int = 0
    coverage_threshold: float = 0.0


@dataclass
class RunConfig:
    """Resolved settings used only by video decoding and detection."""
    # --- provenance ----------------------------------------------------------
    song_name: str
    difficulty: str
    game: str
    skin_name: str
    key_mode: str
    display_resolution: tuple[int, int]
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

    # --- logical tracks ------------------------------------------------------
    normal_lane_count: int
    lanes: list[LaneConfig]

    # --- beat indicator ------------------------------------------------------
    beat_roi: tuple[int, int, int, int]   # (x1, y1, x2, y2)
    beat_channel: str
    beat_diff_threshold: float

    # --- measure line -------------------------------------------------------
    measure_line: MeasureLineConfig         # detection detector tunables

    # --- measurements --------------------------------------------------------
    pixels_per_frame: float

    @property
    def track_count(self) -> int:
        return len(self.lanes)

    @property
    def key_count(self) -> int:
        """Compatibility alias for consumers that mean all logical tracks."""
        return self.track_count

    def _normal_lane(self) -> LaneConfig:
        return next(lane for lane in self.lanes if lane.role == "normal")

    # ponytail: detector compatibility; remove after every detector reads its
    # owning LaneConfig instead of the old global normal-note geometry.
    @property
    def note_height(self) -> int:
        return self._normal_lane().note_height

    @property
    def note_width(self) -> int:
        return self._normal_lane().templates["note"].shape[1]

    @property
    def trigger_template_y_top(self) -> int:
        return self._normal_lane().trigger_y_top

    @property
    def tail_release_offset_px(self) -> int:
        return self._normal_lane().tail_release_offset_px

    @property
    def tail_search_y_max(self) -> int:
        return self._normal_lane().tail_search_y_max

    @property
    def min_longnote_px(self) -> int:
        return self._normal_lane().min_longnote_px

    def summary(self) -> None:
        print(f"=== Config: {self.skin_name} / {self.key_mode} / {self.difficulty} / "
              f"{self.display_resolution[0]}x{self.display_resolution[1]} ===")
        print(f"  video      : {self.video_path}")
        print(f"  fps/speed  : {self.fps} / {self.note_speed}   "
              f"tick_res={self.tick_resolution}  bpm=[{self.min_bpm},{self.max_bpm}]")
        print(f"  playfield  : y[{self.playfield_top},{self.playfield_bottom}]  "
              f"line_y={self.line_y}  trigger_y_top={self.trigger_template_y_top}")
        print(f"  note       : {self.note_width}x{self.note_height}px  "
              f"template_scale={self.template_scale:.4f}")
        print(f"  beat ROI   : {self.beat_roi}  ch={self.beat_channel} "
              f"diff thr={self.beat_diff_threshold}")
        print(f"  measure ln : bright=[{self.measure_line.min_brightness},"
              f"{self.measure_line.max_brightness}]  "
              f"max_thick={self.measure_line.max_thickness}  "
              f"slack={self.measure_line.lane_slack}")
        print(f"  min_ln_px  : {self.min_longnote_px}")
        print(f"  tracks ({self.track_count}; normal={self.normal_lane_count}):")
        for ln in self.lanes:
            print(f"    {ln.index}:{ln.name} [{ln.role}] {ln.color:5s} "
                  f"det_x{list(ln.x_range)} "
                  f"match_x{list(ln.match_x_range)} ch={ln.detection_channel:6s} "
                  f"s1={ln.stage1_threshold} s2={ln.matching_threshold} "
                  f"tmpl={list(ln.templates)}")


# =============================================================================
# Helpers
# =============================================================================

class ConfigError(Exception):
    """Raised on a fatal config inconsistency."""


def _load_toml(path: Path) -> dict:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with open(path, "rb") as f:
        return tomllib.load(f)


def _warn(msg: str) -> None:
    print(f"[config] WARNING: {msg}")


# =============================================================================
# Main resolver
# =============================================================================

def load_config(song_toml_path: str | Path) -> RunConfig:
    """Load song.toml and merge skin + profile into one validated RunConfig."""
    song_path = Path(song_toml_path).resolve()
    song_name = song_path.stem
    config_root = song_path.parent
    song = _load_toml(song_path)

    # --- 1. read references --------------------------------------------------
    setup       = song["setup"]
    skin_name   = setup["skin"]
    key_mode    = str(setup["key_mode"]).strip().lower()
    difficulty  = str(setup["difficulty"]).strip().upper()
    if difficulty not in {"NM", "HD", "MX", "SC"}:
        raise ConfigError(
            f"setup.difficulty must be NM, HD, MX, or SC (got {difficulty!r})")
    res_str     = setup["display_resolution"]

    skin_path    = config_root / "skins" / skin_name / "skin.toml"
    profile_path = config_root / "profiles" / res_str / f"{key_mode}.toml"
    skin    = _load_toml(skin_path)
    profile = _load_toml(profile_path)

    # --- 2. cross-file consistency checks -----------------------------------
    profile_mode = str(profile["meta"]["key_mode"]).strip().lower()
    if profile_mode != key_mode:
        raise ConfigError(
            f"profile key_mode '{profile_mode}' != song key_mode '{key_mode}'")

    game = str(profile["meta"]["game"]).strip().lower()
    if not game:
        raise ConfigError("profile game must not be empty")

    normal_lane_count = int(profile["meta"]["normal_lane_count"])
    lane_colors = skin["lane_colors"].get(key_mode)
    if lane_colors is None:
        raise ConfigError(
            f"skin '{skin_name}' has no lane_colors entry for key_mode '{key_mode}'")
    if len(lane_colors) != normal_lane_count:
        raise ConfigError(
            f"normal_lane_count mismatch: profile says {normal_lane_count}, "
            f"skin lane_colors"
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

    # --- 4. resolve templates & logical tracks -------------------------------
    pf          = profile["playfield"]
    line_y      = int(profile["judgment"]["line_y"])
    normal_cfg  = profile["normal_lanes"]
    field_left  = int(normal_cfg["field_left"])
    lane_width  = int(normal_cfg["lane_width"])
    index_start = int(normal_cfg.get("index_start", 1))
    normal_indices = [int(value) for value in normal_cfg.get(
        "indices", range(index_start, index_start + normal_lane_count))]
    if len(normal_indices) != normal_lane_count:
        raise ConfigError("normal lane indices must match normal_lane_count")
    margin      = int(profile["matching"].get("roi_x_margin", 0))
    meas        = profile["measurements"]

    thr               = skin["thresholds"]
    matching_default  = thr["matching"]
    stage1_default    = thr["stage1_brightness"]
    matching_by_set   = thr.get("matching_by_template_set", {})
    stage1_by_set     = thr.get("stage1_by_template_set", {})
    channel_map       = skin["detection"]["channel"]
    templates_cfg     = skin["templates"]
    overlay_masks     = skin.get("overlay_masks", {})
    tmpl_dir          = config_root / "skins" / skin_name / key_mode

    frame_w, frame_h = prof_res
    template_cache: dict[str, np.ndarray] = {}

    def _load_template(template_set: str, ntype: str) -> np.ndarray:
        if template_set not in templates_cfg:
            raise ConfigError(
                f"skin '{skin_name}' has no [templates.{template_set}] section "
                f"(needed by key_mode '{key_mode}')")
        if ntype not in templates_cfg[template_set]:
            raise ConfigError(
                f"template set '{template_set}' has no '{ntype}' template")
        fname = templates_cfg[template_set][ntype]
        if fname in template_cache:
            return template_cache[fname]
        fpath = tmpl_dir / fname
        if not fpath.is_file():
            raise ConfigError(f"template image not found: {fpath}")
        img = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
        if img is None:
            raise ConfigError(f"failed to decode template: {fpath}")
        if abs(template_scale - 1.0) > 1e-6:
            interp = cv2.INTER_AREA if template_scale < 1 else cv2.INTER_CUBIC
            img = cv2.resize(img, None, fx=template_scale, fy=template_scale,
                             interpolation=interp)
        template_cache[fname] = img
        return img

    def _common(template_set: str, allowed_types: frozenset[str],
                *, load_templates: bool = True):
        if not allowed_types or not allowed_types <= {"tap", "longnote"}:
            raise ConfigError(
                f"invalid allowed_types {sorted(allowed_types)} for '{template_set}'")
        if template_set not in channel_map:
            raise ConfigError(
                f"skin '{skin_name}' [detection.channel] has no entry for "
                f"template set '{template_set}'")
        channel = channel_map[template_set]
        if channel not in CHANNEL_EXTRACTORS:
            raise ConfigError(
                f"unknown detection channel '{channel}' for '{template_set}' "
                f"(valid: {sorted(CHANNEL_EXTRACTORS)})")
        keys = []
        if load_templates:
            if "tap" in allowed_types:
                keys.append("note")
            if "longnote" in allowed_types:
                keys.extend(("lnhead", "lntail"))
        return (
            channel,
            float(stage1_by_set.get(template_set, stage1_default)),
            float(matching_by_set.get(template_set, matching_default)),
            {key: _load_template(template_set, key) for key in keys},
        )

    lanes: list[LaneConfig] = []
    for i, color in enumerate(lane_colors):
        x1 = field_left + i * lane_width
        x2 = x1 + lane_width
        mx1 = max(0, x1 - margin)
        mx2 = min(frame_w, x2 + margin)
        template_set = f"normal_{color}"
        allowed = frozenset(("tap", "longnote"))
        channel, stage1_threshold, matching_threshold, templates = \
            _common(template_set, allowed)

        lanes.append(LaneConfig(
            index=normal_indices[i],
            name=f"K{i + 1}",
            role="normal",
            input_type="key",
            color=color,
            template_set=template_set,
            allowed_types=allowed,
            x_range=(x1, x2),
            match_x_range=(mx1, mx2),
            note_height=int(normal_cfg["note_height"]),
            trigger_y_top=int(normal_cfg["trigger_y_top"]),
            timing_offset_px=int(normal_cfg.get("timing_offset_px", 0)),
            tail_release_offset_px=int(normal_cfg.get("tail_release_offset_px", 0)),
            tail_search_y_max=int(normal_cfg["tail_search_y_max"]),
            min_longnote_px=int(normal_cfg["min_longnote_px"]),
            include_in_consensus=True,
            detection_channel=channel,
            stage1_threshold=stage1_threshold,
            matching_threshold=matching_threshold,
            templates=templates,
        ))

    for track in profile.get("overlay_tracks", []):
        template_set = str(track["template_set"])
        input_type = str(track["input_type"]).strip().lower()
        if input_type not in {"side", "trigger"}:
            raise ConfigError(
                f"invalid overlay input_type {input_type!r} for '{track['name']}'")
        allowed = frozenset(str(value) for value in track["allowed_types"])
        channel, stage1_threshold, matching_threshold, templates = \
            _common(template_set, allowed, load_templates=False)
        x1, x2 = (int(value) for value in track["x_range"])
        track_margin = int(track.get("roi_x_margin", margin))
        mask = overlay_masks.get(template_set)
        if mask is None:
            raise ConfigError(
                f"skin '{skin_name}' has no [overlay_masks.{template_set}] section")
        hue_ranges = tuple(tuple(int(v) for v in pair)
                           for pair in mask["hue_ranges"])
        lanes.append(LaneConfig(
            index=int(track["index"]),
            name=str(track["name"]),
            role="overlay",
            input_type=input_type,
            color=str(track["color"]),
            template_set=template_set,
            allowed_types=allowed,
            x_range=(x1, x2),
            match_x_range=(max(0, x1 - track_margin),
                           min(frame_w, x2 + track_margin)),
            note_height=int(track["note_height"]),
            trigger_y_top=int(track["trigger_y_top"]),
            timing_offset_px=int(track.get("timing_offset_px", 0)),
            tail_release_offset_px=int(track.get("tail_release_offset_px", 0)),
            tail_search_y_max=int(track["tail_search_y_max"]),
            min_longnote_px=int(track["min_longnote_px"]),
            include_in_consensus=bool(track.get("include_in_consensus", False)),
            detection_channel=channel,
            stage1_threshold=stage1_threshold,
            matching_threshold=matching_threshold,
            templates=templates,
            mask_hue_ranges=hue_ranges,
            mask_saturation_min=int(mask["saturation_min"]),
            mask_value_min=int(mask["value_min"]),
            coverage_threshold=float(mask["coverage_threshold"]),
        ))

    lanes.sort(key=lambda lane: lane.index)
    if [lane.index for lane in lanes] != list(range(len(lanes))):
        raise ConfigError("track indices must be unique and contiguous from 0")
    if sum(lane.role == "normal" for lane in lanes) != normal_lane_count:
        raise ConfigError("resolved normal track count does not match profile")

    # --- 5. geometry and template sanity ------------------------------------
    if not (0 <= pf["top"] < pf["bottom"] <= frame_h):
        raise ConfigError(
            f"playfield y-range [{pf['top']},{pf['bottom']}] is outside "
            f"frame height {frame_h}")
    for lane in lanes:
        x1, x2 = lane.x_range
        if not (0 <= x1 < x2 <= frame_w):
            raise ConfigError(
                f"track '{lane.name}' x-range {lane.x_range} is outside frame")
        if lane.trigger_y_top != line_y - lane.note_height:
            _warn(f"track '{lane.name}' trigger_y_top={lane.trigger_y_top} but "
                  f"line_y-note_height={line_y - lane.note_height}")
        if lane.role == "overlay":
            if not (0.0 < lane.coverage_threshold <= 1.0):
                raise ConfigError(
                    f"track '{lane.name}' has invalid coverage threshold")
            if not lane.mask_hue_ranges or any(
                    not (0 <= lo <= hi <= 179)
                    for lo, hi in lane.mask_hue_ranges):
                raise ConfigError(f"track '{lane.name}' has invalid HSV hue ranges")
        for key, template in lane.templates.items():
            th, tw = template.shape[:2]
            if th != lane.note_height:
                _warn(f"track '{lane.name}' {key} template height {th}px != "
                      f"note_height {lane.note_height}px")
            if tw > lane.match_x_range[1] - lane.match_x_range[0]:
                raise ConfigError(
                    f"track '{lane.name}' {key} template is wider than match ROI")

    # --- 6. capture / video --------------------------------------------------
    cap_cfg    = song["capture"]
    fps = float(cap_cfg["fps"])
    note_speed = float(cap_cfg["note_speed"])
    calibrated_speed = float(meas["note_speed"])
    if fps <= 0:
        raise ConfigError(f"capture.fps must be > 0 (got {fps})")
    if abs(note_speed - calibrated_speed) > 1e-6:
        raise ConfigError(
            f"capture.note_speed={note_speed} but profile is calibrated for "
            f"{calibrated_speed}; use a matching profile")
    video_path = Path(cap_cfg["video_path"])
    if cap_cfg["video_path"] and not video_path.is_absolute():
        video_path = (config_root / video_path).resolve()
    if cap_cfg["video_path"]:
        _verify_video(video_path, frame_w, frame_h, fps)
    else:
        _warn("capture.video_path is empty — set it before running the pipeline.")

    # --- 7. assemble ---------------------------------------------------------
    bi_geom = profile["beat_indicator"]["roi"]
    bi_skin = skin["beat_indicator"]
    beat_channel = bi_skin["channel"]
    if beat_channel not in CHANNEL_EXTRACTORS:
        raise ConfigError(
            f"unknown beat channel '{beat_channel}' "
            f"(valid: {sorted(CHANNEL_EXTRACTORS)})")
    bx1, by1, bx2, by2 = bi_geom
    if not (0 <= bx1 < bx2 <= frame_w and 0 <= by1 < by2 <= frame_h):
        raise ConfigError(f"beat ROI {bi_geom} is outside {frame_w}x{frame_h}")

    ml_skin = skin["measure_line"]
    measure_line_cfg = MeasureLineConfig(
        min_brightness=float(ml_skin["min_brightness"]),
        max_brightness=float(ml_skin["max_brightness"]),
        max_thickness=int(ml_skin["max_thickness"]),
        lane_slack=int(ml_skin["lane_slack"]),
        lit_energy_threshold=float(ml_skin.get("lit_energy_threshold", 50.0)),
    )

    min_bpm = float(song["song"]["min_bpm"])
    max_bpm = float(song["song"]["max_bpm"])
    tick_resolution = int(song["song"]["resolution"])
    if tick_resolution <= 0:
        raise ConfigError(
            f"song.resolution must be > 0 (got {tick_resolution})")
    if min_bpm <= 0.0 or max_bpm <= 0.0:
        raise ConfigError(
            f"song.min_bpm and song.max_bpm must be > 0 "
            f"(got min_bpm={min_bpm}, max_bpm={max_bpm})")
    if min_bpm > max_bpm:
        raise ConfigError(
            f"song.min_bpm must be <= song.max_bpm "
            f"(got min_bpm={min_bpm}, max_bpm={max_bpm})")

    return RunConfig(
        song_name=song_name,
        difficulty=difficulty,
        game=game,
        skin_name=skin_name,
        key_mode=key_mode,
        display_resolution=prof_res,
        template_scale=template_scale,
        video_path=video_path,
        fps=fps,
        note_speed=note_speed,
        tick_resolution=tick_resolution,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        playfield_top=pf["top"],
        playfield_bottom=pf["bottom"],
        line_y=line_y,
        normal_lane_count=normal_lane_count,
        lanes=lanes,
        beat_roi=tuple(bi_geom),
        beat_channel=beat_channel,
        beat_diff_threshold=float(bi_skin["diff_threshold"]),
        measure_line=measure_line_cfg,
        pixels_per_frame=float(meas["pixels_per_frame"]),
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
