import unittest
from types import SimpleNamespace

from ez2cv.detection.tracking import NoteTracker, TrackedEdge


class NoteTrackerTest(unittest.TestCase):
    @staticmethod
    def _calibration():
        lane = SimpleNamespace(
            index=0,
            role="normal",
            trigger_y_top=800.0,
            note_height=22.0,
            tail_release_offset_px=0.0,
            timing_offset_px=0.0,
            include_in_consensus=True,
        )
        return SimpleNamespace(
            fps=60.0,
            pixels_per_frame=30.0,
            lanes=[lane],
        )

    def test_extrapolation_uses_recent_local_trajectory(self):
        tracker = NoteTracker(self._calibration())
        edge = TrackedEdge(
            id=0,
            lane=0,
            type="note",
            trajectory=[(frame, 710.0 + 10.0 * frame, 0.9)
                        for frame in range(8)],
            last_seen=7,
        )

        event = tracker._extrapolate_trigger(edge)

        self.assertIsNotNone(event)
        self.assertAlmostEqual(event.cross_frame, 9.0)
        self.assertGreater(event.timing_sigma_ms, 0.0)
        self.assertTrue(event.extrapolated)

    def test_longnote_endpoint_keeps_global_projection(self):
        tracker = NoteTracker(self._calibration())
        edge = TrackedEdge(
            id=0,
            lane=0,
            type="lnhead",
            trajectory=[(frame, 710.0 + 10.0 * frame, 0.9)
                        for frame in range(8)],
            last_seen=7,
        )

        event = tracker._extrapolate_trigger(edge)

        self.assertAlmostEqual(event.cross_frame, 7.0 + 20.0 / 30.0)


if __name__ == "__main__":
    unittest.main()
