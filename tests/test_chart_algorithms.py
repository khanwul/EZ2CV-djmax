import unittest

from ez2cv.chart.barline import reconstruct_barlines
from ez2cv.chart.clock import BPMDraft, TickClock
from ez2cv.chart.meter import TimeSignature, TimeSigVariant, barline_ticks
from ez2cv.chart.quantize import snap_tick
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

    def test_quantizer_supported_grid_and_off_grid(self):
        self.assertEqual(snap_tick(47.4).tick, 48)
        self.assertEqual(snap_tick(64.2).denom, 12)
        self.assertTrue(snap_tick(16.1).off_grid)

    def test_variant_aware_barline_ticks(self):
        bars = [BarlineEvent(0.0, 0.0, 0.0)] * 4
        variants = [TimeSigVariant(1, 1, TimeSignature(3, 4))]
        clock = TickClock.from_drafts(
            [BPMDraft(0.0, 10_000.0, 120.0, 120.0)], origin_ms=0.0)
        self.assertEqual(
            barline_ticks(bars, clock, TimeSignature(4, 4), variants),
            [0, 768, 1344, 2112],
        )

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


if __name__ == "__main__":
    unittest.main()
