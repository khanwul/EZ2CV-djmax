import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
MODES = {"4k": 4, "5k": 5, "6k": 6, "8k": 8}


class KeyModeConfigTest(unittest.TestCase):
    def test_every_supported_mode_has_consistent_config(self):
        skin_path = ROOT / "config/skins/ez2on/skin.toml"
        with skin_path.open("rb") as file:
            colors = tomllib.load(file)["lane_colors"]

        for mode, key_count in MODES.items():
            with self.subTest(mode=mode):
                profile_path = ROOT / f"config/profiles/1920x1080/{mode}.toml"
                with profile_path.open("rb") as file:
                    profile = tomllib.load(file)

                self.assertEqual(profile["meta"]["key_mode"], mode)
                self.assertEqual(profile["meta"]["key_count"], key_count)
                self.assertEqual(len(colors[mode]), key_count)
                self.assertTrue((ROOT / f"config/skins/ez2on/{mode}").is_dir())
                self.assertEqual(profile["measurements"]["note_speed"], 8.0)

                left = profile["lanes"]["field_left"]
                width = profile["lanes"]["lane_width"]
                frame_width = profile["meta"]["display_resolution"][0]
                self.assertEqual(left, (frame_width - key_count * width) // 2)


if __name__ == "__main__":
    unittest.main()
