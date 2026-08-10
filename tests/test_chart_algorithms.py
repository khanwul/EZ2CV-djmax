import unittest
from types import SimpleNamespace

import numpy as np

from ez2cv.chart.barline import reconstruct_barlines
from ez2cv.chart.clock import BPMDraft, TickClock
from ez2cv.chart.pipeline import _apply_timing_uncertainty
from ez2cv.chart.quantize import snap_by_measure, snap_tick
from ez2cv.chart.timeline import (_add_submeasure_anchors,
                                  clean_beat_times)
from ez2cv.detection.barline import BarlineEvent
from ez2cv.detection.beat import BeatEvent


class ChartAlgorithmTest(unittest.TestCase):
    def test_tick_clock_round_trip_for_constant_and_ramp(self):
        for draft in (
            BPMDraft(0.0, 60_000.0, 120.0, 120.0),
            BPMDraft(0.0, 60_000.0, 120.0, 180.0),
        ):
            with self.subTest(draft=draft):
                clock = TickClock.from_drafts([draft], origin_ms=0.0)
                for ms in (0.0, 250.0, 15_000.0, 60_000.0):
                    self.assertAlmostEqual(
                        clock.tick_to_ms(clock.ms_to_tick(ms)), ms, places=6)

    def test_anchor_clock_preserves_a_tempo_step(self):
        clock = TickClock.from_anchors(
            [0.0, 1_000.0, 1_800.0], [0.0, 384.0, 768.0],
            max_error_ms=0.0)

        self.assertEqual([s.bpm_start for s in clock.segments], [120.0, 150.0])
        for ms in (0.0, 250.0, 1_000.0, 1_400.0, 1_800.0):
            self.assertAlmostEqual(clock.tick_to_ms(clock.ms_to_tick(ms)), ms)

    def test_quantizer_supported_grid_and_off_grid(self):
        self.assertEqual(snap_tick(47.4).tick, 48)
        self.assertEqual(snap_tick(64.2).denom, 12)
        self.assertTrue(snap_tick(16.1).off_grid)

    def test_timing_sigma_separates_measurement_uncertainty(self):
        clock = TickClock.from_drafts(
            [BPMDraft(0.0, 2_000.0, 120.0, 120.0)], origin_ms=0.0)
        raw_ms = 16.1 * 60_000.0 / (192 * 120)
        snap = snap_tick(16.1)

        uncertain = _apply_timing_uncertainty(
            SimpleNamespace(trigger_ms=raw_ms, timing_sigma_ms=12.0),
            snap, clock)
        precise = _apply_timing_uncertainty(
            SimpleNamespace(trigger_ms=raw_ms, timing_sigma_ms=1.0),
            snap, clock)

        self.assertTrue(uncertain.timing_uncertain)
        self.assertFalse(uncertain.off_grid)
        self.assertFalse(precise.timing_uncertain)
        self.assertTrue(precise.off_grid)

    def test_measure_grid_needs_repeated_fine_onsets(self):
        snaps, levels = snap_by_measure(
            [0.0, 16.0, 768.0 + 16.0, 768.0 + 80.0],
            [0, 768, 1536])

        self.assertEqual(levels, [0, 1])
        self.assertTrue(snaps[1].off_grid)       # one stray stays an outlier
        self.assertFalse(snaps[1].fine_grid)
        self.assertFalse(snaps[2].off_grid)
        self.assertTrue(snaps[2].fine_grid)
        self.assertTrue(snaps[3].fine_grid)
        self.assertEqual(
            sum(snap.off_grid or snap.fine_grid for snap in snaps), 3)

        _, chord_levels = snap_by_measure([16.0, 16.5], [0, 768])
        self.assertEqual(chord_levels, [0])

    def test_missing_barline_is_reconstructed_from_beats(self):
        beats = [BeatEvent(i, i * 500.0, 10.0) for i in range(13)]
        bars = [
            BarlineEvent(0.0, 0.0, 10.0),
            BarlineEvent(4.0, 2_000.0, 10.0),
            BarlineEvent(12.0, 6_000.0, 10.0),
        ]
        result = reconstruct_barlines(bars, beats)
        self.assertEqual(result.beat_indices, [0, 4, 8, 12])
        self.assertTrue(result.barlines[2].extrapolated)

    def test_observed_barline_time_snaps_to_its_pow_beat(self):
        beats = [BeatEvent(i, i * 500.0, 10.0) for i in range(9)]
        bars = [
            BarlineEvent(1.5, 25.0, 10.0),
            BarlineEvent(121.5, 2_025.0, 10.0),
            BarlineEvent(241.5, 4_025.0, 10.0),
        ]

        result = reconstruct_barlines(bars, beats)

        self.assertEqual([bar.ms for bar in result.barlines],
                         [0.0, 2_000.0, 4_000.0])

    def test_missing_intro_barlines_are_reconstructed_from_phase(self):
        beats = [BeatEvent(i, i * 500.0, 10.0) for i in range(13)]
        bars = [
            BarlineEvent(240.0, 4_000.0, 10.0),
            BarlineEvent(360.0, 6_000.0, 10.0),
        ]

        result = reconstruct_barlines(bars, beats)

        self.assertEqual(result.beat_indices, [0, 4, 8, 12])
        self.assertEqual(len(result.measure_meters), 3)

    def test_arbitrary_observed_meter_sequence_is_preserved(self):
        meters = [2, 3, 5, 6, 7, 1, 4]
        boundaries = [0]
        for meter in meters:
            boundaries.append(boundaries[-1] + meter)
        beats = [BeatEvent(i, i * 500.0, 10.0)
                 for i in range(boundaries[-1] + 1)]
        bars = [BarlineEvent(i, i * 500.0, 10.0)
                for i in boundaries]

        result = reconstruct_barlines(bars, beats)

        self.assertEqual(result.measure_meters, meters)
        self.assertEqual(result.beat_indices, boundaries)

    def test_beat_normalization_does_not_invent_missing_flashes(self):
        beats = [BeatEvent(i, ms, 10.0)
                 for i, ms in enumerate((0.0, 500.0, 1_500.0, 2_000.0))]
        np.testing.assert_array_equal(
            clean_beat_times(beats), [0.0, 500.0, 1_500.0, 2_000.0])

    def test_clear_mid_measure_tempo_step_gets_a_beat_anchor(self):
        raw = SimpleNamespace(fps=60.0, tick_resolution=192, notes=[])
        beat_ms = np.array([0.0, 300.0, 600.0, 880.0, 1_160.0])
        anchor_ms, anchor_ticks = _add_submeasure_anchors(
            raw, beat_ms, [0, 4],
            np.array([0.0, beat_ms[-1]]), np.array([0.0, 768.0]), [4],
            np.array([0.0, beat_ms[-1]]), np.array([0.0, 768.0]))

        self.assertEqual(anchor_ticks.tolist(), [0.0, 384.0, 768.0])
        self.assertAlmostEqual(anchor_ms[1], 600.0)

        raw.notes = [SimpleNamespace(trigger_ms=435.0)]
        _, rejected_ticks = _add_submeasure_anchors(
            raw, beat_ms, [0, 4],
            np.array([0.0, beat_ms[-1]]), np.array([0.0, 768.0]), [4],
            np.array([0.0, beat_ms[-1]]), np.array([0.0, 768.0]))
        self.assertEqual(rejected_ticks.tolist(), [0.0, 768.0])


if __name__ == "__main__":
    unittest.main()
