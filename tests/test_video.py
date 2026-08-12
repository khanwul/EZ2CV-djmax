import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace

import cv2
import numpy as np

from ez2cv.detection.pipeline import _frame_to_ms
from ez2cv.video import Preprocessor, _alignment_offset


class _Capture:
    def get(self, prop):
        return {
            cv2.CAP_PROP_FRAME_WIDTH: 100,
            cv2.CAP_PROP_FRAME_HEIGHT: 80,
            cv2.CAP_PROP_FPS: 30.0,
        }[prop]


def _config():
    return SimpleNamespace(
        lanes=[], beat_roi=(0, 0, 1, 1), display_resolution=(100, 80),
        fps=60.0)


class VideoValidationTest(unittest.TestCase):
    def test_judgment_band_alignment_finds_translation(self):
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        frame[46:54, 37:78, 0] = 220
        calibration = SimpleNamespace(
            lanes=[SimpleNamespace(x_range=(30, 70))],
            alignment_band_y=(50, 58), alignment_max_shift=10,
            playfield_top=10, playfield_bottom=90,
            beat_roi=(20, 60, 23, 63),
        )

        self.assertEqual(_alignment_offset(frame, calibration), (7, -4))

        with self.assertRaisesRegex(RuntimeError, "was not found"):
            _alignment_offset(np.zeros_like(frame), calibration)

    def test_fractional_frame_uses_presentation_timestamps(self):
        self.assertEqual(_frame_to_ms(1.5, [0.0, 10.0, 30.0], 60.0), 20.0)
        self.assertAlmostEqual(
            _frame_to_ms(3.0, [0.0, 10.0, 30.0], 60.0), 46.6666667)

    def test_fps_mismatch_requires_force(self):
        with self.assertRaisesRegex(RuntimeError, "video fps 30.00"):
            Preprocessor(_config())._verify_stream(_Capture())
        with redirect_stdout(StringIO()):
            Preprocessor(_config(), force=True)._verify_stream(_Capture())


if __name__ == "__main__":
    unittest.main()
