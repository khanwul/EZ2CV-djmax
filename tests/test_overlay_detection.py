import unittest
from types import SimpleNamespace

import numpy as np

from ez2cv.detection.barline import MeasureLineDetector
from ez2cv.detection.stage1 import ProjectionDetector, Run, Stage1Result
from ez2cv.detection.stage2 import (Match, Stage2Result, TemplateMatcher,
                                    _candidate_groups)
from ez2cv.detection.tracking import (LongnoteStateMachine, NoteTracker,
                                      TrackedEdge, TriggerEvent,
                                      merge_duplicate_triggers)
from ez2cv.video import LaneFrame, PreprocessedFrame


def _overlay(index, *, hue_ranges, allowed_types, color):
    return SimpleNamespace(
        index=index,
        role="overlay",
        color=color,
        x_range=(0, 240),
        match_x_range=(0, 240),
        note_height=4 if color == "cyan" else 26,
        trigger_y_top=70,
        stage1_threshold=60.0,
        coverage_threshold=0.50,
        mask_hue_ranges=hue_ranges,
        mask_saturation_min=120,
        mask_value_min=60,
        allowed_types=frozenset(allowed_types),
        include_in_consensus=False,
        matching_threshold=0.85,
        tail_search_y_max=70,
        templates={},
    )


def _frame(index, bgr):
    height, width = bgr.shape[:2]
    return LaneFrame(index, "overlay", np.zeros((height, width), np.uint8), bgr)


class OverlayDetectionTest(unittest.TestCase):
    def test_duplicate_triggers_inside_ten_ms_are_merged(self):
        weaker = TriggerEvent(0, "note", 0.0, 100.0, .9,
                              confidence=.6)
        stronger = TriggerEvent(0, "note", 0.5, 108.6, .9,
                                confidence=.9)

        self.assertEqual(merge_duplicate_triggers([weaker, stronger]),
                         [stronger])

    def test_side_coverage_survives_foreground_normal_note(self):
        side = _overlay(0, hue_ranges=((75, 105),),
                        allowed_types={"longnote"}, color="cyan")
        image = np.zeros((80, 240, 3), np.uint8)
        image[10:60] = (255, 255, 0)       # cyan side body
        image[25:35, :108] = (0, 255, 0)  # foreground normal note
        cal = SimpleNamespace(lanes=[side], playfield_top=0)

        result = ProjectionDetector(cal).detect_lane(
            _frame(0, image), frame_index=0, roi_y_origin=0)

        self.assertEqual([(run.y_start, run.y_end) for run in result.runs],
                         [(10, 60)])
        self.assertAlmostEqual(result.projection[25], 132 / 240)
        groups = _candidate_groups(result.runs[0], side.note_height,
                                   side.allowed_types)
        self.assertNotIn("note", {key for group in groups for key, _ in group})
        matches = TemplateMatcher(cal).match_lane(_frame(0, image), result).matches
        self.assertEqual({match.type for match in matches}, {"lnhead", "lntail"})

    def test_red_l_and_r_remain_independent_tracks(self):
        lanes = [
            _overlay(0, hue_ranges=((0, 5), (150, 179)),
                     allowed_types={"tap", "longnote"}, color="red"),
            _overlay(1, hue_ranges=((0, 5), (150, 179)),
                     allowed_types={"tap", "longnote"}, color="red"),
        ]
        left = np.zeros((80, 240, 3), np.uint8)
        right = np.zeros_like(left)
        left[8:34] = (0, 0, 255)       # hue 0
        right[8:34] = (255, 0, 255)    # hue 150
        pf = PreprocessedFrame(0, 0.0, 0,
                               [_frame(0, left), _frame(1, right)],
                               np.zeros((1, 1), np.uint8))

        results = ProjectionDetector(
            SimpleNamespace(lanes=lanes, playfield_top=0)).detect_frame(pf)

        self.assertEqual([[run.y_start, run.y_end]
                          for result in results for run in result.runs],
                         [[8, 34], [8, 34]])
        groups = _candidate_groups(results[0].runs[0], 26,
                                   lanes[0].allowed_types)
        self.assertEqual({key for group in groups for key, _ in group},
                         {"note", "lnhead", "lntail"})
        matches = TemplateMatcher(
            SimpleNamespace(lanes=lanes, playfield_top=0)).match_frame(pf, results)
        self.assertEqual([[match.type for match in result.matches]
                          for result in matches], [["note"], ["note"]])

    def test_clipped_overlay_head_reaches_detection_trigger(self):
        side = _overlay(0, hue_ranges=((75, 105),),
                        allowed_types={"longnote"}, color="cyan")
        cal = SimpleNamespace(lanes=[side], playfield_top=0)
        result = Stage1Result(
            0, 0, "cyan", 0, .5, np.ones(80),
            [Run(0, side.trigger_y_top, "long", 1.0, 1.0)],
        )

        matches = TemplateMatcher(cal).match_lane(
            _frame(0, np.zeros((80, 240, 3), np.uint8)), result).matches

        head = next(match for match in matches if match.type == "lnhead")
        self.assertEqual(head.y_top, side.trigger_y_top)

    def test_clipped_overlay_tail_is_not_a_new_tap(self):
        lane = _overlay(0, hue_ranges=((0, 5), (150, 179)),
                        allowed_types={"tap", "longnote"}, color="red")
        cal = SimpleNamespace(lanes=[lane], playfield_top=0)
        result = Stage1Result(
            0, 0, "red", 0, .5, np.ones(80),
            [Run(60, lane.trigger_y_top, "short", 1.0, 1.0)],
        )

        matches = TemplateMatcher(cal).match_lane(
            _frame(0, np.zeros((80, 240, 3), np.uint8)), result).matches

        self.assertEqual(matches, [])

    def test_measure_line_ignores_overlay_projection(self):
        normal = SimpleNamespace(include_in_consensus=True)
        overlay = SimpleNamespace(include_in_consensus=False)
        measure_line = SimpleNamespace(
            lit_energy_threshold=40.0,
            min_brightness=10.0,
            max_brightness=60.0,
            max_thickness=3,
            lane_slack=0,
        )
        cal = SimpleNamespace(lanes=[normal, normal, overlay],
                              normal_lane_count=2, measure_line=measure_line,
                              trigger_template_y_top=9)
        normal_projection = np.zeros(10)
        normal_projection[4] = 50
        results = [
            Stage1Result(0, 0, "red", 0, 60, normal_projection),
            Stage1Result(0, 1, "green", 0, 60, normal_projection),
            Stage1Result(0, 2, "cyan", 0, .5, np.ones(9)),
        ]

        detections = MeasureLineDetector(cal).detect_frame(results)

        self.assertEqual(len(detections), 1)

    def test_tracker_uses_each_tracks_trigger(self):
        lanes = [
            SimpleNamespace(index=0, trigger_y_top=10, note_height=4,
                            tail_release_offset_px=0,
                            include_in_consensus=False),
            SimpleNamespace(index=1, trigger_y_top=20, note_height=26,
                            tail_release_offset_px=0,
                            include_in_consensus=False),
        ]
        cal = SimpleNamespace(lanes=lanes, fps=60.0, pixels_per_frame=4.0)
        tracker = NoteTracker(cal)
        run = Run(0, 10, "short", 1.0, 1.0)

        for frame, y in ((0, 8), (1, 12)):
            results = [Stage2Result(frame, lane.index, "red", [
                Match(lane.index, "note", y, 0, 1.0, run)
            ]) for lane in lanes]
            events = tracker.step(frame, results)

        self.assertEqual([event.lane for event in events], [0])

    def test_overlay_tracker_uses_calibrated_speed(self):
        overlay = SimpleNamespace(
            index=0, trigger_y_top=20, note_height=4,
            tail_release_offset_px=0, include_in_consensus=False,
        )
        cal = SimpleNamespace(lanes=[overlay], fps=60.0,
                              pixels_per_frame=4.0)
        tracker = NoteTracker(cal)
        tracker.speed._speed = 40.0

        self.assertEqual(tracker._lane_speed(0), 4.0)

    def test_normal_extrapolation_keeps_calibrated_speed_and_jitter(self):
        normal = SimpleNamespace(
            index=0, role="normal", trigger_y_top=100, note_height=4,
            tail_release_offset_px=0, include_in_consensus=True,
        )
        cal = SimpleNamespace(lanes=[normal], fps=60.0,
                              pixels_per_frame=24.0)
        tracker = NoteTracker(cal)
        tracker.speed._speed = 5.0
        edge = TrackedEdge(0, 0, "note", [(0, 0, 1.0), (1, 12, 1.0)], 1)

        event = tracker._extrapolate_trigger(edge)

        self.assertEqual(tracker._lane_speed(0), 24.0)
        self.assertIsNotNone(event)

    def test_expired_tap_cannot_steal_the_next_note(self):
        normal = SimpleNamespace(
            index=0, role="normal", trigger_y_top=70, note_height=4,
            tail_release_offset_px=0, include_in_consensus=True,
        )
        cal = SimpleNamespace(lanes=[normal], fps=60.0,
                              pixels_per_frame=24.0)
        tracker = NoteTracker(cal)
        edge = TrackedEdge(0, 0, "note", [(0, 60, 1.0)], 0)
        match = SimpleNamespace(y_top=70)

        self.assertIsNone(tracker._gate_dist(edge, match, 5, 24.0))

    def test_lr_tap_keeps_its_one_frame_retry(self):
        lr = SimpleNamespace(
            index=0, role="overlay", trigger_y_top=70, note_height=26,
            tail_release_offset_px=0, include_in_consensus=False,
            allowed_types=frozenset({"tap", "longnote"}),
        )
        cal = SimpleNamespace(lanes=[lr], fps=60.0,
                              pixels_per_frame=24.0)
        tracker = NoteTracker(cal)
        edge = TrackedEdge(0, 0, "note", [(0, 60, 1.0)], 0)
        match = SimpleNamespace(y_top=70)

        self.assertIsNotNone(tracker._gate_dist(edge, match, 5, 24.0))

    def test_tracker_accepts_a_double_stutter_catch_up(self):
        normal = SimpleNamespace(
            index=0, role="normal", trigger_y_top=700, note_height=44,
            tail_release_offset_px=0, include_in_consensus=True,
        )
        cal = SimpleNamespace(lanes=[normal], fps=60.0,
                              pixels_per_frame=24.0)
        tracker = NoteTracker(cal)
        edge = TrackedEdge(0, 0, "note", [(0, 401, 1.0)], 0)
        match = SimpleNamespace(y_top=468)

        self.assertIsNotNone(tracker._gate_dist(edge, match, 1, 24.0))

    def test_tracker_applies_timing_offset_without_moving_trigger(self):
        overlay = SimpleNamespace(
            index=0, trigger_y_top=10, note_height=4,
            timing_offset_px=-1, tail_release_offset_px=0,
            include_in_consensus=False,
        )
        cal = SimpleNamespace(lanes=[overlay], fps=60.0,
                              pixels_per_frame=4.0)
        tracker = NoteTracker(cal)
        run = Run(0, 10, "short", 1.0, 1.0)

        tracker.step(0, [Stage2Result(0, 0, "red", [
            Match(0, "note", 8, 0, 1.0, run)
        ])])
        events = tracker.step(1, [Stage2Result(1, 0, "red", [
            Match(0, "note", 12, 0, 1.0, run)
        ])])

        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].cross_frame, 0.75)
        self.assertAlmostEqual(events[0].ms, 12.5)

    def test_overlay_tracker_does_not_swap_head_and_tail(self):
        overlay = SimpleNamespace(
            index=0, role="overlay", trigger_y_top=70, note_height=4,
            tail_release_offset_px=0, include_in_consensus=False,
            allowed_types=frozenset({"longnote"}),
        )
        cal = SimpleNamespace(lanes=[overlay], fps=60.0,
                              pixels_per_frame=24.0)
        tracker = NoteTracker(cal)
        run = Run(0, 70, "long", 1.0, 1.0)
        events = []

        for frame, head_y in enumerate((4, 29, 55, 70)):
            events.extend(tracker.step(frame, [Stage2Result(frame, 0, "cyan", [
                Match(0, "lntail", 0, 0, 1.0, run),
                Match(0, "lnhead", head_y, 0, 1.0, run),
            ])]))

        self.assertEqual([event.type for event in events], ["lnhead"])

    def test_normal_tracker_does_not_swap_tail_and_head(self):
        normal = SimpleNamespace(
            index=0, role="normal", trigger_y_top=70, note_height=4,
            tail_release_offset_px=0, include_in_consensus=False,
        )
        cal = SimpleNamespace(lanes=[normal], fps=60.0,
                              pixels_per_frame=24.0)
        tracker = NoteTracker(cal)
        run = Run(0, 70, "long", 1.0, 1.0)
        events = []

        for frame, head_y in enumerate((4, 29, 55, 70)):
            events.extend(tracker.step(frame, [Stage2Result(frame, 0, "red", [
                Match(0, "lntail", 0, 0, 1.0, run),
                Match(0, "lnhead", head_y, 0, 1.0, run),
            ])]))

        self.assertEqual([event.type for event in events], ["lnhead"])

    def test_side_never_falls_back_to_tap(self):
        side = SimpleNamespace(index=0, color="cyan",
                               allowed_types=frozenset({"longnote"}),
                               min_longnote_px=35)
        cal = SimpleNamespace(lanes=[side], fps=60.0, pixels_per_frame=24.0)
        state = LongnoteStateMachine(cal)
        head = TriggerEvent(0, "lnhead", 0.0, 0.0, 1.0)
        tail = TriggerEvent(0, "lntail", 1.0, 10.0, 1.0)

        self.assertIsNone(state.feed(head))
        self.assertIsNone(state.feed(tail))
        self.assertEqual(state.flush(), [])

    def test_lr_short_pair_is_a_tap(self):
        lr = SimpleNamespace(index=0, color="red",
                             allowed_types=frozenset({"tap", "longnote"}),
                             min_longnote_px=35)
        state = LongnoteStateMachine(SimpleNamespace(
            lanes=[lr], fps=60.0, pixels_per_frame=24.0))

        self.assertIsNone(state.feed(TriggerEvent(
            0, "lnhead", 0.0, 0.0, 1.0)))
        note = state.feed(TriggerEvent(0, "lntail", 1.0, 10.0, 1.0))

        self.assertEqual((note.type, note.pairing_status),
                         ("tap", "short_pair"))

    def test_later_head_preserves_a_head_whose_tail_was_missed(self):
        lane = SimpleNamespace(index=0, color="red",
                               allowed_types=frozenset({"tap", "longnote"}),
                               min_longnote_px=35)
        cal = SimpleNamespace(lanes=[lane], fps=60.0,
                              pixels_per_frame=24.0)
        state = LongnoteStateMachine(cal)
        first = TriggerEvent(0, "lnhead", 0.0, 0.0, 1.0)
        later = TriggerEvent(0, "lnhead", 30.0, 500.0, 1.0)

        self.assertIsNone(state.feed(first))
        recovered = state.feed(later)

        self.assertEqual(recovered.type, "tap")
        self.assertEqual(recovered.trigger_ms, 0.0)

if __name__ == "__main__":
    unittest.main()
