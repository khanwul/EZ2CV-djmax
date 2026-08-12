import tempfile
import unittest
from pathlib import Path

from ez2cv.chart import build_chart
from ez2cv.detection import DetectionPipeline
from ez2cv.io import serialize_chart
from synthetic_video import make_fixture


class SyntheticEndToEndTest(unittest.TestCase):
    def test_video_to_chart(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config, expected = make_fixture(root / "synthetic.avi")
            raw = DetectionPipeline(config).run(progress=False)
            chart = serialize_chart(build_chart(raw))

        self.assertEqual(raw.provenance["timestamp_source"], "container_pts")
        self.assertEqual(
            [(note.lane, note.type) for note in raw.notes],
            [(note["lane"], note["type"]) for note in expected["notes"]],
        )
        self.assertEqual(len(raw.beats), len(expected["beats_ms"]))
        self.assertEqual(len(raw.barlines), len(expected["barlines_ms"]))
        self.assertEqual(
            [(note["lane"], note["start_tick"], note["end_tick"])
             for note in chart["notes"]],
            [(note["lane"], note["start_tick"], note["end_tick"])
             for note in expected["notes"]],
        )
        self.assertEqual(chart["timing"]["tempo_segments"],
                         expected["tempo_segments"])

    def test_translated_panel_is_normalized(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config, expected = make_fixture(
                Path(temp_dir) / "translated.avi", alignment_offset=(5, -4),
                alignment_visible_from=3)
            raw = DetectionPipeline(config).run(progress=False)
            chart = serialize_chart(build_chart(raw))

        self.assertEqual(raw.provenance["alignment_offset_px"], [5, -4])
        self.assertEqual(
            [(note["lane"], note["start_tick"], note["end_tick"])
             for note in chart["notes"]],
            [(note["lane"], note["start_tick"], note["end_tick"])
             for note in expected["notes"]],
        )


if __name__ == "__main__":
    unittest.main()
