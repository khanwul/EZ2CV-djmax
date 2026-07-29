"""One-pass video detection orchestrator.

Runs the detector over a video in a single decode. The Preprocessor feeds
each frame down three paths at once:

  note path     :  Stage 1 (projection)  ->  Stage 2 (template match)
                   ->  NoteTracker  ->  LongnoteStateMachine
  beat path     :  BeatDetector (POW LED)
  bar-line path :  MeasureLineDetector  ->  MeasureLineTracker

The product is a RawChart — raw, MILLISECOND-based notes, beats and bar
lines held in memory.

The bar-line path reuses the note path's Stage 1 output (its per-lane
`projection` arrays) for free, so adding it costs no extra decode and almost no
extra compute. It exists because the POW LED gives only BEAT phase, while a
measure line gives MEASURE phase — chart conversion needs both.

Why detection stops at milliseconds
------------------------------------
Detection deliberately knows nothing about BPM, ticks, or the musical grid.
Keeping the expensive video pass grid-agnostic means it runs once and
every later experiment — a different BPM guess, a different snapping strategy —
reuses the same raw result instead of re-decoding 100k+ frames.

Usage
-----
    result = DetectionPipeline(load_config("config/song.toml")).run()
    print(result.summary())
    # result.notes    : list[RawNote]      (taps + longnotes, ms-based)
    # result.beats    : list[BeatEvent]    (POW LED beats, ms-based)
    # result.barlines : list[BarlineEvent] (measure boundaries, ms-based)
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ez2cv.config import RunConfig
from ez2cv.video import Preprocessor
from ez2cv.detection.stage1 import ProjectionDetector
from ez2cv.detection.stage2 import TemplateMatcher
from ez2cv.detection.tracking import (NoteTracker, LongnoteStateMachine, RawNote,
                             merge_duplicate_triggers)
from ez2cv.detection.beat import BeatDetector, BeatEvent
from ez2cv.detection.barline import MeasureLineDetector, MeasureLineTracker, BarlineEvent


# event type order for the chronological feed (a tie in ms must still open a
# longnote head before it is closed by a tail)
_EVENT_ORDER = {"lnhead": 0, "note": 1, "lntail": 2}


def _print_progress(done: int, total: int, *, width: int = 40,
                    label: str = "") -> None:
    """Render a single in-place progress bar (overwrites itself via ``\\r``)."""
    frac = done / total
    filled = int(width * frac)
    bar = "█" * filled + "░" * (width - filled)
    tag = f"[{label}] " if label else ""
    print(f"\r  {tag}[{bar}] {frac * 100:5.1f}%  {done}/{total} frames",
          end="", flush=True)


# =============================================================================
# Result
# =============================================================================

@dataclass
class RawChart:
    """Serializable ms-domain checkpoint produced by video detection."""
    song_name: str
    skin_name: str
    key_mode: str
    lane_colors: tuple[str, ...]
    display_resolution: tuple[int, int]
    video_path: str
    fps: float
    note_speed: float
    tick_resolution: int
    min_bpm: float
    max_bpm: float
    notes: list[RawNote]            # taps + longnotes, sorted by (ms, lane)
    beats: list[BeatEvent]          # POW LED beats, sorted by frame
    barlines: list[BarlineEvent]    # measure boundaries, sorted by frame
    frame_count: int
    orphan_tails: int               # tails with no matching head (dropped)

    @property
    def key_count(self) -> int:
        return len(self.lane_colors)

    @property
    def duration_ms(self) -> float:
        return self.frame_count / self.fps * 1000.0

    @property
    def taps(self) -> list[RawNote]:
        return [n for n in self.notes if n.type == "tap"]

    @property
    def longnotes(self) -> list[RawNote]:
        return [n for n in self.notes if n.type == "longnote"]

    def beat_interval_frames(self) -> float:
        """Median POW LED inter-beat gap in frames (0 if too few beats)."""
        if len(self.beats) < 2:
            return 0.0
        return float(np.median(np.diff([b.frame_index for b in self.beats])))

    def barline_interval_frames(self) -> float:
        """Median inter-bar-line gap in frames (0 if too few bar lines)."""
        if len(self.barlines) < 2:
            return 0.0
        return float(np.median(np.diff([b.cross_frame for b in self.barlines])))

    def summary(self) -> str:
        extr = sum(1 for n in self.notes if n.extrapolated)
        n = len(self.notes) or 1
        bi = self.beat_interval_frames()
        flash_bpm = (60.0 * self.fps / bi) if bi > 0 else 0.0
        mi = self.barline_interval_frames()
        # beats per measure, if the LED flashes once per beat: a sanity ratio
        beats_per_measure = (mi / bi) if bi > 0 else 0.0
        lines = [
            f"=== Detection result — {self.duration_ms / 1000:.1f}s, "
            f"{self.frame_count} frames ===",
            f"  notes        : {len(self.notes)}  "
            f"({len(self.taps)} tap, {len(self.longnotes)} longnote)",
            f"  crossings    : {extr} extrapolated ({extr * 100 // n}%), "
            f"{n - extr} interpolated",
            f"  orphan tails : {self.orphan_tails} (dropped)",
            f"  beats        : {len(self.beats)}  "
            f"(median interval {bi:.1f} frames)",
            f"  flash tempo  : {flash_bpm:.1f} BPM at 1 flash/beat  |  "
            f"{flash_bpm / 2:.1f} BPM at 1 flash/half-beat",
            f"  bar lines    : {len(self.barlines)}  "
            f"(median interval {mi:.1f} frames"
            + (f", ~{beats_per_measure:.1f} beats/measure)"
               if beats_per_measure else ")"),
        ]
        return "\n".join(lines)


# =============================================================================
# Pipeline
# =============================================================================

class DetectionPipeline:
    """Drive all detection components over one video pass."""

    def __init__(self, cal: RunConfig):
        self.cal = cal
        # per-frame POW LED brightness, retained after run() for visualization
        self.beat_signal: np.ndarray | None = None

    def run(self, *, progress: bool = True) -> RawChart:
        """Decode the video once; return the raw ms-based RawChart."""
        pre = Preprocessor(self.cal)
        s1 = ProjectionDetector(self.cal)
        s2 = TemplateMatcher(self.cal)
        tracker = NoteTracker(self.cal)
        lnsm = LongnoteStateMachine(self.cal)
        beat = BeatDetector(self.cal)
        mld = MeasureLineDetector(self.cal)
        mlt = MeasureLineTracker(self.cal)

        total = pre.frame_count             # best-effort, for the progress line
        events = []                         # all TriggerEvents, fed after sort
        barline_events: list[BarlineEvent] = []
        frame_count = 0

        for pf in pre:
            frame_count += 1
            # note path
            s1r = s1.detect_frame(pf)
            s2r = s2.match_frame(pf, s1r)
            events.extend(tracker.step(pf.frame_index, s2r))
            # bar-line path (reuses the note path's Stage 1 projections)
            barline_events.extend(mlt.step(pf.frame_index, mld.detect_frame(s1r)))
            # beat path (BeatDetector internally records pf.beat_roi.mean())
            beat.step(pf)
            if progress and total and (frame_count % 30 == 0
                                       or frame_count == total):
                _print_progress(frame_count, total)
        if progress and total:
            _print_progress(total, total)
            print()                         # end the in-place bar line
        events.extend(tracker.flush())      # project edges still mid-air
        barline_events.extend(mlt.flush())  # project bar lines still mid-air
        beat.finish()                       # confirm/drop a trailing candidate
        self.beat_signal = np.asarray(beat.signal)

        # drop spurious double-detections (one physical note can spawn two
        # tracked edges that each emit), then feed the longnote state machine
        # in global chronological order, so a head always precedes its tail
        events = merge_duplicate_triggers(events)
        events.sort(key=lambda e: (e.ms, e.lane, _EVENT_ORDER.get(e.type, 1)))
        notes: list[RawNote] = []
        for ev in events:
            note = lnsm.feed(ev)
            if note is not None:
                notes.append(note)
        notes.extend(lnsm.flush())          # longnotes whose tail never showed
        notes.sort(key=lambda n: (n.trigger_ms, n.lane))

        barline_events.sort(key=lambda e: e.cross_frame)

        return RawChart(
            song_name=self.cal.song_name,
            skin_name=self.cal.skin_name,
            key_mode=self.cal.key_mode,
            lane_colors=tuple(ln.color for ln in self.cal.lanes),
            display_resolution=self.cal.display_resolution,
            video_path=str(self.cal.video_path),
            fps=self.cal.fps,
            note_speed=self.cal.note_speed,
            tick_resolution=self.cal.tick_resolution,
            min_bpm=self.cal.min_bpm,
            max_bpm=self.cal.max_bpm,
            notes=notes,
            beats=beat.beats,
            barlines=barline_events,
            frame_count=frame_count,
            orphan_tails=lnsm.orphan_tails,
        )
