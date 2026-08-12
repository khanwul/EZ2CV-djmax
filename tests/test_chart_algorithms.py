import unittest
from types import SimpleNamespace

import numpy as np

from ez2cv.chart.barline import reconstruct_barlines
from ez2cv.chart.clock import BPMDraft, BPMSegment, TickClock
from ez2cv.chart.pipeline import (_apply_timing_uncertainty,
                                  _convert_and_snap_notes)
from ez2cv.chart.quantize import choose_measure_grid, snap_by_measure, snap_tick
from ez2cv.chart.timeline import (_expand_symmetric_subbeat_steps,
                                  _fit_tempo_clock, _repair_beats,
                                  _normalized_tempo_clock,
                                  _tempo_clock_score,
                                  clean_beat_times)
from ez2cv.detection.barline import BarlineEvent
from ez2cv.detection.beat import BeatEvent
from ez2cv.detection.tracking import RawNote


class ChartAlgorithmTest(unittest.TestCase):
    def test_tick_clock_round_trip_for_constant_and_ramp(self):
        for draft in (
            BPMDraft(0.0, 60_000.0, 120.0, 120.0),
            BPMDraft(0.0, 60_000.0, 120.0, 180.0),
            BPMDraft(0.0, 60_000.0, 180.0, 120.0),
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

    def test_anchor_clock_rejects_bpm_clamping(self):
        with self.assertRaisesRegex(ValueError, "outside configured range"):
            TickClock.from_anchors(
                [0.0, 1_000.0], [0.0, 768.0], max_bpm=200.0)

    def test_clock_state_matches_serialized_timing(self):
        clock = TickClock.from_anchors(
            [0.0, 1_000.0, 1_800.0], [0.0, 384.0, 768.0],
            max_error_ms=0.0)
        restored = TickClock(clock.segments, tick_zero_ms=clock.tick_zero_ms)
        for tick in (0.0, 192.0, 384.0, 576.0, 768.0):
            self.assertAlmostEqual(restored.tick_to_ms(tick),
                                   clock.tick_to_ms(tick))

    def test_quantizer_supported_grid_and_off_grid(self):
        self.assertEqual(snap_tick(47.4).tick, 48)
        self.assertEqual(snap_tick(64.2).denom, 12)
        off_grid = snap_tick(16.1).off_grid
        self.assertTrue(off_grid)
        self.assertIs(type(off_grid), bool)

        conservative, _ = snap_by_measure(
            [12.75], [0, 768], grid_levels=[0], alpha=2.0)
        self.assertEqual(conservative[0].tick, 0)

    def test_timing_sigma_separates_measurement_uncertainty(self):
        clock = TickClock.from_drafts(
            [BPMDraft(0.0, 2_000.0, 120.0, 120.0)], origin_ms=0.0)
        raw_ms = 16.1 * 60_000.0 / (192 * 120)
        snap = snap_tick(16.1)

        uncertain = _apply_timing_uncertainty(
            SimpleNamespace(trigger_ms=raw_ms, start_sigma_ms=12.0),
            snap, clock)
        precise = _apply_timing_uncertainty(
            SimpleNamespace(trigger_ms=raw_ms, start_sigma_ms=1.0),
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

    def test_tail_grid_can_prefer_a_statistically_tied_finer_level(self):
        ticks = [measure * 192.0 + phase
                 for measure in range(5)
                 for phase in (68.0, 77.0, 79.0, 122.0, 180.0)]

        self.assertEqual(choose_measure_grid(ticks), 1)
        self.assertEqual(
            choose_measure_grid(ticks, cost_tolerance=1.0), 2)

    def test_repeated_fine_heads_set_a_chart_wide_grid_floor(self):
        clock = TickClock.from_drafts(
            [BPMDraft(0.0, 8_000.0, 120.0, 120.0)], origin_ms=0.0)
        ticks = [68, 77, 79, 122, 180]
        raw = SimpleNamespace(notes=[RawNote(
            i % 5, "tap", clock.tick_to_ms(measure * 192 + tick), None,
            "white", 1.0)
            for measure in range(5) for i, tick in enumerate(ticks)])

        _, _, levels, _ = _convert_and_snap_notes(
            raw, clock, [0, 192, 384, 576, 768, 960])

        self.assertTrue(all(level >= 2 for level in levels))
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

    def test_missing_beat_is_inserted_only_when_meter_improves(self):
        bars = [BarlineEvent(ms / 500.0, ms, 10.0)
                for ms in (0.0, 2_000.0, 4_000.0, 6_000.0)]
        raw = SimpleNamespace(
            barlines=bars, fps=60.0, notes=[],
            beats=[BeatEvent(i, ms, 10.0) for i, ms in enumerate(
                (0.0, 500.0, 1_000.0, 1_500.0, 2_000.0, 3_000.0,
                 3_500.0, 4_000.0, 4_500.0, 5_000.0, 5_500.0, 6_000.0))])

        repaired, result, inserted, deleted = _repair_beats(
            raw, clean_beat_times(raw.beats))

        self.assertEqual(inserted, 1)
        self.assertEqual(deleted, 0)
        self.assertIn(2_500.0, repaired)
        self.assertEqual(result.measure_meters, [4, 4, 4])

        raw.barlines = [BarlineEvent(ms / 500.0, ms, 10.0)
                        for ms in (0.0, 2_500.0, 4_500.0)]
        raw.beats = [BeatEvent(i, ms, 10.0) for i, ms in enumerate(
            (0.0, 500.0, 1_000.0, 2_000.0, 2_500.0, 3_000.0,
             3_500.0, 4_000.0, 4_500.0))]
        repaired, _, inserted, deleted = _repair_beats(
            raw, clean_beat_times(raw.beats))
        self.assertEqual(inserted, 0)
        self.assertEqual(deleted, 0)
        self.assertNotIn(1_500.0, repaired)

    def test_slow_one_beat_gap_is_not_filled_when_note_grid_disagrees(self):
        beat_ms = [i * 500.0 for i in range(5)] + [3_000.0]
        raw = SimpleNamespace(
            fps=60.0, tick_resolution=192,
            notes=[SimpleNamespace(trigger_ms=2_490.0, end_ms=None),
                   SimpleNamespace(trigger_ms=2_990.0, end_ms=None)],
            beats=[BeatEvent(i, ms, 10.0)
                   for i, ms in enumerate(beat_ms)],
            barlines=[BarlineEvent(i, ms, 10.0)
                      for i, ms in enumerate((0.0, 2_000.0, 3_000.0))],
        )

        repaired, result, inserted, deleted = _repair_beats(
            raw, clean_beat_times(raw.beats))

        np.testing.assert_array_equal(repaired, beat_ms)
        self.assertEqual((inserted, deleted), (0, 0))
        self.assertEqual(result.measure_meters, [4, 1])

    def test_extra_midbeat_is_deleted_only_when_meter_improves(self):
        beat_ms = sorted([i * 500.0 for i in range(13)] + [2_250.0])
        raw = SimpleNamespace(
            fps=60.0, notes=[],
            beats=[BeatEvent(i, ms, 10.0)
                   for i, ms in enumerate(beat_ms)],
            barlines=[BarlineEvent(i * 120.0, i * 2_000.0, 10.0)
                      for i in range(4)],
        )

        repaired, result, inserted, deleted = _repair_beats(
            raw, clean_beat_times(raw.beats))

        np.testing.assert_array_equal(
            repaired, np.arange(13, dtype=float) * 500.0)
        self.assertEqual(inserted, 0)
        self.assertEqual(deleted, 1)
        self.assertEqual(result.measure_meters, [4, 4, 4])

    def test_beat_tempo_model_finds_a_step_without_frame_jitter_segments(self):
        raw = SimpleNamespace(fps=60.0, tick_resolution=192,
                              min_bpm=100.0, max_bpm=200.0)
        periods = [500.0] * 4 + [400.0] * 4
        beat_ms = np.concatenate(([0.0], np.cumsum(periods)))
        ticks = np.arange(len(beat_ms), dtype=float) * 192

        clock = _fit_tempo_clock(raw, beat_ms, ticks)

        self.assertEqual(len(clock.segments), 2)
        self.assertEqual(clock.segments[0].end_tick, 768)
        self.assertAlmostEqual(clock.segments[0].bpm_start, 120.0)
        self.assertAlmostEqual(clock.segments[1].bpm_start, 150.0)

        noisy = np.arange(17, dtype=float) * 500.0
        noisy[1:-1:2] += 4.0
        steady = _fit_tempo_clock(
            raw, noisy, np.arange(len(noisy), dtype=float) * 192)
        self.assertEqual(len(steady.segments), 1)

    def test_bpm_bounds_reject_accumulated_clock_drift(self):
        raw = SimpleNamespace(fps=60.0, tick_resolution=192,
                              min_bpm=100.0, max_bpm=200.0)
        ticks = np.arange(101, dtype=float) * 192
        beat_ms = ticks / 192 * 60_000.0 / 210.0

        with self.assertRaisesRegex(ValueError, "bounded clock residual"):
            _fit_tempo_clock(raw, beat_ms, ticks)

    def test_beat_tempo_model_can_select_one_linear_ramp(self):
        source = TickClock(
            [BPMSegment(0, 3072, 120.0, 180.0)],
            tick_zero_ms=0.0, tick_resolution=192)
        ticks = np.arange(17, dtype=float) * 192
        beat_ms = np.array([source.tick_to_ms(tick) for tick in ticks])
        raw = SimpleNamespace(fps=60.0, tick_resolution=192,
                              min_bpm=100.0, max_bpm=200.0)

        fitted = _fit_tempo_clock(raw, beat_ms, ticks)

        self.assertEqual(len(fitted.segments), 1)
        self.assertAlmostEqual(fitted.segments[0].bpm_start, 120.0, places=3)
        self.assertAlmostEqual(fitted.segments[0].bpm_end, 180.0, places=3)

    def test_tempo_candidate_score_uses_note_grid_as_auxiliary_evidence(self):
        raw = SimpleNamespace(
            tick_resolution=192,
            notes=[SimpleNamespace(trigger_ms=ms)
                   for ms in (0.0, 250.0, 500.0)],
        )
        aligned = TickClock(
            [BPMSegment(0, 384, 120.0, 120.0)], tick_zero_ms=0.0)
        shifted = TickClock(
            [BPMSegment(0, 384, 125.0, 125.0)], tick_zero_ms=0.0)

        self.assertLess(_tempo_clock_score(raw, aligned),
                        _tempo_clock_score(raw, shifted))

    def test_tempo_normalization_stays_inside_configured_bounds(self):
        raw = SimpleNamespace(min_bpm=111.11, max_bpm=222.22)
        clock = TickClock([
            BPMSegment(0, 192, 111.11, 222.22),
        ], tick_zero_ms=0.0)

        normalized = _normalized_tempo_clock(raw, clock)

        self.assertEqual(normalized.segments[0].bpm_start, 111.11)
        self.assertEqual(normalized.segments[0].bpm_end, 222.22)

    def test_note_grid_can_recover_symmetric_subbeat_tempo_steps(self):
        observed = TickClock([
            BPMSegment(0, 192, 150.0, 150.0),
            BPMSegment(192, 576, 100.0, 100.0),
            BPMSegment(576, 768, 150.0, 150.0),
        ], tick_zero_ms=0.0)
        expected = TickClock([
            BPMSegment(0, 192, 150.0, 150.0),
            BPMSegment(192, 288, 75.0, 75.0),
            BPMSegment(288, 480, 150.0, 150.0),
            BPMSegment(480, 576, 75.0, 75.0),
            BPMSegment(576, 768, 150.0, 150.0),
        ], tick_zero_ms=0.0)
        raw = SimpleNamespace(
            tick_resolution=192, min_bpm=75.0, max_bpm=150.0, fps=60.0,
            notes=[SimpleNamespace(trigger_ms=expected.tick_to_ms(tick))
                   for tick in range(192, 576, 48)],
        )

        refined = _expand_symmetric_subbeat_steps(raw, observed)

        self.assertEqual(
            [(s.start_tick, s.end_tick, s.bpm_start)
             for s in refined.segments],
            [(s.start_tick, s.end_tick, s.bpm_start)
             for s in expected.segments])


if __name__ == "__main__":
    unittest.main()
