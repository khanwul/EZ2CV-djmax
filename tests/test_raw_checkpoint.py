import json
import tempfile
import unittest
from contextlib import chdir, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ez2cv.chart import build_chart
from ez2cv.cli import run
from ez2cv.detection import RawChart
from ez2cv.detection.barline import BarlineEvent
from ez2cv.detection.beat import BeatEvent
from ez2cv.detection.tracking import RawNote
from ez2cv.io import read_raw, serialize_raw, write_raw


def _raw_chart() -> RawChart:
    return RawChart(
        song_name="demo",
        skin_name="ez2on",
        key_mode="5k",
        lane_colors=("white", "cyan", "white", "cyan", "white"),
        display_resolution=(1920, 1080),
        video_path="demo.mp4",
        fps=60.0,
        note_speed=8.0,
        tick_resolution=192,
        min_bpm=120.0,
        max_bpm=120.0,
        frame_count=420,
        orphan_tails=0,
        notes=[
            RawNote(0, "tap", 500.0, None, "white", 0.9),
            RawNote(1, "longnote", 1_000.0, 1_500.0, "cyan", 0.8),
        ],
        beats=[BeatEvent(i, i * 500.0, 10.0) for i in range(13)],
        barlines=[
            BarlineEvent(i * 120.0, i * 2_000.0, 10.0)
            for i in range(4)
        ],
    )


class RawCheckpointTest(unittest.TestCase):
    def test_raw_json_round_trip_and_chart_rebuild(self):
        raw = _raw_chart()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_raw(raw, root=temp_dir)
            loaded = read_raw(path)

        self.assertEqual(loaded, raw)
        chart = build_chart(loaded)
        self.assertEqual(chart.song_name, "demo")
        self.assertEqual(len(chart.notes), 2)
        self.assertEqual(chart.notes[0].start_tick, 192)
        self.assertEqual(chart.notes[1].end_tick, 576)

    def test_invalid_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "raw.json"
            path.write_text('{"schema_version": "1.0"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported schema_version"):
                read_raw(path)

    def test_non_finite_numbers_are_rejected(self):
        payload = serialize_raw(_raw_chart())
        payload["meta"]["fps"] = float("nan")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "raw.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite number"):
                read_raw(path)

    def test_checkpoint_survives_chart_failure(self):
        raw = _raw_chart()
        config = SimpleNamespace(song_name="demo", summary=lambda: None)
        with tempfile.TemporaryDirectory() as temp_dir, chdir(temp_dir), \
                patch("ez2cv.cli.load_config", return_value=config), \
                patch("ez2cv.cli.DetectionPipeline") as pipeline, \
                patch("ez2cv.cli.build_chart", side_effect=RuntimeError("boom")):
            pipeline.return_value.run.return_value = raw
            with self.assertRaisesRegex(RuntimeError, "boom"):
                with redirect_stdout(StringIO()):
                    run("ignored.toml", chart_image=False, progress=False)
            self.assertTrue(Path("out/demo/demo_raw.json").is_file())


if __name__ == "__main__":
    unittest.main()
