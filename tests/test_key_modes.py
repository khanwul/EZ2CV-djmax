import contextlib
import io
import shutil
import tempfile
import textwrap
import tomllib
import unittest
from pathlib import Path

import cv2
import numpy as np

from ez2cv.config import load_config


ROOT = Path(__file__).parents[1]
MODES = {
    "4b": {
        "normal": 4,
        "tracks": 6,
        "pitch": 120,
        "height": 44,
        "timing_offset": 11,
        "names": ["SIDE_L", "K1", "K2", "K3", "K4", "SIDE_R"],
    },
    "5b": {
        "normal": 5,
        "tracks": 7,
        "pitch": 96,
        "height": 35,
        "timing_offset": 0,
        "names": ["SIDE_L", "K1", "K2", "K3", "K4", "K5", "SIDE_R"],
    },
    "6b": {
        "normal": 6,
        "tracks": 8,
        "pitch": 80,
        "height": 30,
        "timing_offset": 13,
        "names": ["SIDE_L", "K1", "K2", "K3", "K4", "K5", "K6", "SIDE_R"],
    },
    "8b": {
        "normal": 6,
        "tracks": 10,
        "pitch": 80,
        "height": 30,
        "timing_offset": 0,
        "names": ["SIDE_L", "K1", "K2", "K3", "L", "R", "K4", "K5", "K6", "SIDE_R"],
    },
}


class KeyModeConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.config_root = Path(cls._tmp.name) / "config"
        shutil.copytree(ROOT / "config/profiles", cls.config_root / "profiles")
        shutil.copytree(ROOT / "config/skins", cls.config_root / "skins")
        cls._write_placeholder_templates()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    @classmethod
    def _write_placeholder_templates(cls):
        skin_path = cls.config_root / "skins/ONGEKI/skin.toml"
        with skin_path.open("rb") as file:
            template_sets = tomllib.load(file)["templates"]

        for mode in MODES:
            profile_path = cls.config_root / f"profiles/1920x1080/{mode}.toml"
            with profile_path.open("rb") as file:
                profile = tomllib.load(file)
            normal_h = profile["normal_lanes"]["note_height"]
            normal_w = profile["measurements"]["note_width_px"]
            mode_dir = cls.config_root / f"skins/ONGEKI/{mode}"
            mode_dir.mkdir(parents=True, exist_ok=True)

            for template_set, filenames in template_sets.items():
                if template_set.startswith("normal_"):
                    height, width = normal_h, normal_w
                elif template_set == "side_cyan":
                    height, width = 4, 240
                else:
                    height, width = 26, 240
                image = np.zeros((height, width, 3), dtype=np.uint8)
                image[:, ::2] = 255
                for filename in filenames.values():
                    if not cv2.imwrite(str(mode_dir / filename), image):
                        raise RuntimeError(f"could not create {filename}")

    def _load(self, mode, difficulty="SC"):
        song_path = self.config_root / f"song_{mode}.toml"
        song_path.write_text(
            textwrap.dedent(f"""
            [setup]
            skin = "ONGEKI"
            key_mode = "{mode}"
            difficulty = "{difficulty}"
            display_resolution = "1920x1080"

            [capture]
            video_path = ""
            fps = 60.0
            note_speed = 6.0

            [song]
            resolution = 192
            min_bpm = 60.0
            max_bpm = 300.0
        """),
            encoding="utf-8",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            return load_config(song_path)

    def test_djmax_difficulties(self):
        for difficulty in ("NM", "HD", "MX", "SC"):
            with self.subTest(difficulty=difficulty):
                self.assertEqual(self._load("5b", difficulty.lower()).difficulty,
                                 difficulty)

    def test_every_mode_resolves_measured_normal_geometry(self):
        for mode, expected in MODES.items():
            with self.subTest(mode=mode):
                cal = self._load(mode)
                normals = [lane for lane in cal.lanes if lane.role == "normal"]

                self.assertEqual(cal.key_mode, mode)
                self.assertEqual(cal.game, "djmax_respect_v")
                self.assertEqual(cal.normal_lane_count, expected["normal"])
                self.assertEqual(cal.track_count, expected["tracks"])
                self.assertEqual([lane.name for lane in cal.lanes], expected["names"])
                self.assertEqual(
                    [lane.index for lane in cal.lanes], list(range(expected["tracks"]))
                )
                self.assertEqual(
                    [lane.x_range for lane in normals],
                    [
                        (720 + i * expected["pitch"], 720 + (i + 1) * expected["pitch"])
                        for i in range(expected["normal"])
                    ],
                )
                self.assertTrue(
                    all(lane.note_height == expected["height"] for lane in normals)
                )
                self.assertTrue(
                    all(
                        lane.timing_offset_px == expected["timing_offset"]
                        for lane in normals
                    )
                )
                self.assertTrue(
                    all(lane.allowed_types == {"tap", "longnote"} for lane in normals)
                )
                self.assertTrue(all(lane.include_in_consensus for lane in normals))
                self.assertTrue(
                    all(
                        set(lane.templates) == {"note", "lnhead", "lntail"}
                        for lane in normals
                    )
                )

    def test_side_tracks_are_overlapping_longnote_only_tracks(self):
        timing_offsets = {"4b": -6, "5b": -17, "6b": 0}
        for mode in timing_offsets:
            with self.subTest(mode=mode):
                cal = self._load(mode)
                left, right = cal.lanes[0], cal.lanes[-1]

                self.assertEqual(left.x_range, (720, 960))
                self.assertEqual(right.x_range, (960, 1200))
                for lane in (left, right):
                    self.assertEqual(lane.role, "overlay")
                    self.assertEqual(lane.input_type, "side")
                    self.assertEqual(lane.template_set, "side_cyan")
                    self.assertEqual(lane.allowed_types, {"longnote"})
                    self.assertEqual(lane.templates, {})
                    self.assertFalse(lane.include_in_consensus)
                    self.assertEqual(lane.coverage_threshold, 0.50)
                    self.assertEqual(lane.timing_offset_px, timing_offsets[mode])

    def test_8b_is_six_normal_tracks_plus_l_r_and_side_tracks(self):
        cal = self._load("8b")
        left, right = cal.lanes[4:6]

        self.assertEqual((left.name, right.name), ("L", "R"))
        for lane in (left, right):
            self.assertEqual(lane.role, "overlay")
            self.assertEqual(lane.input_type, "trigger")
            self.assertEqual(lane.x_range[1] - lane.x_range[0], 240)
            self.assertEqual(lane.template_set, "lr_red")
            self.assertEqual(lane.allowed_types, {"tap", "longnote"})
            self.assertEqual(lane.templates, {})
            self.assertEqual(lane.note_height, 26)
            self.assertEqual(lane.trigger_y_top, 729)
            self.assertEqual(lane.timing_offset_px, -5)
            self.assertFalse(lane.include_in_consensus)

        side_left, side_right = cal.lanes[0], cal.lanes[9]
        self.assertEqual((side_left.name, side_right.name), ("SIDE_L", "SIDE_R"))
        self.assertEqual(
            (side_left.x_range, side_right.x_range), ((720, 960), (960, 1200))
        )
        for lane in (side_left, side_right):
            self.assertEqual(lane.input_type, "side")
            self.assertEqual(lane.template_set, "side_cyan")
            self.assertEqual(lane.allowed_types, {"longnote"})
            self.assertEqual(lane.timing_offset_px, -17)
            self.assertFalse(lane.include_in_consensus)


if __name__ == "__main__":
    unittest.main()
