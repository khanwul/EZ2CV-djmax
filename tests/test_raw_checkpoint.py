import json
import tempfile
import unittest
from contextlib import chdir, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ez2cv.chart import build_chart
from ez2cv.chart.meter import TimeSignature, TimeSigVariant
from ez2cv.cli import main, run
from ez2cv.detection import RawChart, TrackMetadata
from ez2cv.detection.barline import BarlineEvent
from ez2cv.detection.beat import BeatEvent
from ez2cv.detection.tracking import RawNote
from ez2cv.io import (read_chart, read_raw, serialize_chart, serialize_raw,
                      write_chart, write_raw)


def _raw_chart() -> RawChart:
    return RawChart(
        song_name="demo",
        difficulty="SC",
        game="djmax_respect_v",
        skin_name="ONGEKI",
        key_mode="5b",
        tracks=(
            TrackMetadata(0, "SIDE_L", "overlay", "side", "cyan"),
            TrackMetadata(1, "K1", "normal", "key", "red"),
            TrackMetadata(2, "K2", "normal", "key", "green"),
            TrackMetadata(3, "K3", "normal", "key", "red"),
            TrackMetadata(4, "K4", "normal", "key", "green"),
            TrackMetadata(5, "K5", "normal", "key", "cyan"),
            TrackMetadata(6, "SIDE_R", "overlay", "side", "cyan"),
        ),
        display_resolution=(1920, 1080),
        video_path="demo.mp4",
        fps=60.0,
        note_speed=6.0,
        tick_resolution=192,
        min_bpm=120.0,
        max_bpm=120.0,
        frame_count=420,
        orphan_tails=0,
        notes=[
            RawNote(1, "tap", 500.0, None, "red", 0.9),
            RawNote(2, "longnote", 1_000.0, 1_500.0, "green", 0.8),
        ],
        beats=[BeatEvent(i, i * 500.0, 10.0) for i in range(13)],
        barlines=[BarlineEvent(i * 120.0, i * 2_000.0, 10.0) for i in range(4)],
    )


class RawCheckpointTest(unittest.TestCase):
    def test_no_input_runs_every_song_config(self):
        with tempfile.TemporaryDirectory() as temp_dir, chdir(temp_dir):
            config = Path("config")
            (config / "profiles").mkdir(parents=True)
            for path in (config / "z.toml", config / "a.toml",
                         config / "song.toml", config / "profiles/p.toml"):
                path.write_text("", encoding="utf-8")
            with patch("ez2cv.cli.run") as run_all:
                self.assertEqual(main([]), 0)
            self.assertEqual([call.args[0] for call in run_all.call_args_list],
                             [config / "a.toml", config / "z.toml"])

    def test_raw_json_round_trip_and_chart_rebuild(self):
        raw = _raw_chart()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_raw(raw, root=temp_dir)
            loaded = read_raw(path)

        self.assertEqual(loaded, raw)
        self.assertEqual(loaded.tracks[0].name, "SIDE_L")
        self.assertEqual(loaded.tracks[0].role, "overlay")
        self.assertEqual(loaded.tracks[0].input_type, "side")
        chart = build_chart(loaded)
        self.assertEqual(chart.song_name, "demo")
        self.assertEqual(chart.difficulty, "SC")
        self.assertEqual(len(chart.notes), 2)
        self.assertEqual(chart.notes[0].start_tick, 192)
        self.assertEqual(chart.notes[1].end_tick, 576)
        self.assertIn("timing_outlier_ratio", chart.stats["rhythm"])
        self.assertIn("fine_grid_ratio", chart.stats["rhythm"])
        self.assertIn("base_grid_outlier_ratio", chart.stats["rhythm"])
        self.assertEqual(
            len(chart.stats["rhythm"]["measure_grid_max_denominator"]),
            chart.stats["structure"]["measure_count"])

    def test_legacy_raw_note_without_timing_sigma_is_supported(self):
        payload = serialize_raw(_raw_chart())
        for note in payload["notes"]:
            del note["timing_sigma_ms"]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "raw.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = read_raw(path)

        self.assertEqual([note.timing_sigma_ms for note in loaded.notes],
                         [0.0, 0.0])

    def test_v31_raw_gets_unknown_game_and_overlay_input_type(self):
        payload = serialize_raw(_raw_chart())
        payload["schema_version"] = "3.1"
        del payload["meta"]["game"]
        for track in payload["meta"]["tracks"]:
            del track["input_type"]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "raw.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = read_raw(path)

        self.assertEqual(loaded.game, "unknown")
        self.assertEqual(loaded.tracks[0].input_type, "unknown")
        self.assertEqual(loaded.tracks[1].input_type, "key")

    def test_negative_timing_sigma_is_rejected(self):
        payload = serialize_raw(_raw_chart())
        payload["notes"][0]["timing_sigma_ms"] = -1.0
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "raw.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "negative timing sigma"):
                read_raw(path)

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
        config = SimpleNamespace(song_name="demo", difficulty="SC",
                                 summary=lambda: None)
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            chdir(temp_dir),
            patch("ez2cv.cli.load_config", return_value=config),
            patch("ez2cv.cli.DetectionPipeline") as pipeline,
            patch("ez2cv.cli.build_chart", side_effect=RuntimeError("boom")),
        ):
            pipeline.return_value.run.return_value = raw
            with self.assertRaisesRegex(RuntimeError, "boom"):
                with redirect_stdout(StringIO()):
                    run("ignored.toml", chart_image=False, progress=False)
            self.assertTrue(Path("out/demo/SC/demo_raw.json").is_file())

    def test_chart_keeps_notes_before_first_observed_barline(self):
        raw = _raw_chart()
        raw.barlines = [
            BarlineEvent(240.0, 4_000.0, 10.0),
            BarlineEvent(360.0, 6_000.0, 10.0),
        ]

        chart = build_chart(raw)

        self.assertEqual(len(chart.notes), 2)
        self.assertEqual(chart.notes[0].start_tick, 192)

    def test_v3_chart_round_trip_and_meter_events(self):
        chart = build_chart(_raw_chart())
        chart.global_time_sig = TimeSignature(4, 4)
        chart.variant_measures = [
            TimeSigVariant(1, 1, TimeSignature(3, 4))]
        chart.barlines_tick = [0, 768, 1344, 2112]

        payload = serialize_chart(chart)
        self.assertEqual((payload["format"], payload["version"]),
                         ("ez2cv.chart", "3.2"))
        self.assertEqual(payload["meta"]["game"], "djmax_respect_v")
        self.assertEqual(payload["meta"]["difficulty"], "SC")
        self.assertEqual(payload["timing"]["meter_events"], [
            {"start_tick": 0, "numerator": 4, "denominator": 4},
            {"start_tick": 768, "numerator": 3, "denominator": 4},
            {"start_tick": 1344, "numerator": 4, "denominator": 4},
        ])
        self.assertEqual(payload["notes"][0]["id"], 0)
        self.assertIs(payload["notes"][0]["off_grid"], False)
        self.assertEqual(payload["meta"]["tracks"][0]["name"], "SIDE_L")
        self.assertEqual(payload["meta"]["tracks"][0]["input_type"], "side")

        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_chart(chart, root=temp_dir)
            loaded = read_chart(path)
        self.assertEqual(loaded, payload)

    def test_v31_chart_gets_unknown_game_and_overlay_input_type(self):
        payload = serialize_chart(build_chart(_raw_chart()))
        payload["version"] = "3.1"
        del payload["meta"]["game"]
        for track in payload["meta"]["tracks"]:
            del track["input_type"]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "chart.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = read_chart(path)

        self.assertEqual(loaded["meta"]["game"], "unknown")
        self.assertEqual(loaded["meta"]["tracks"][0]["input_type"], "unknown")
        self.assertEqual(loaded["meta"]["tracks"][1]["input_type"], "key")


if __name__ == "__main__":
    unittest.main()
