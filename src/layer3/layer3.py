"""
EZ2CV — Layer 3 orchestrator : the 1-Pass
===============================================================================
Runs the whole 1-Pass over a video in a SINGLE decode. The Preprocessor feeds
each frame down three paths at once:

  note path     :  Stage 1 (projection)  ->  Stage 2 (template match)
                   ->  NoteTracker  ->  LongnoteStateMachine
  beat path     :  BeatDetector (POW LED)
  bar-line path :  MeasureLineDetector  ->  MeasureLineTracker

The product is a Layer3Result — raw, MILLISECOND-based notes, beats and bar
lines held in memory.

The bar-line path reuses the note path's Stage 1 output (its per-lane
`projection` arrays) for free, so adding it costs no extra decode and almost no
extra compute. It exists because the POW LED gives only BEAT phase, while a
measure line gives MEASURE phase — Layer 4 needs both.

Why Layer 3 stops at milliseconds
---------------------------------
Layer 3 deliberately knows nothing about BPM, ticks, or the musical grid.
Converting ms -> ticks and snapping to the beat grid is Layer 4; serialising is
Layer 5. Keeping the expensive video pass grid-agnostic means it runs ONCE and
every later experiment — a different BPM guess, a different snapping strategy —
reuses the same raw result instead of re-decoding 100k+ frames.

Usage
-----
    result = Layer3Pipeline.from_config("config/song.toml").run()
    print(result.summary())
    # result.notes    : list[RawNote]      (taps + longnotes, ms-based)
    # result.beats    : list[BeatEvent]    (POW LED beats, ms-based)
    # result.barlines : list[BarlineEvent] (measure boundaries, ms-based)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from layer1.calibration import Calibration, resolve_calibration
from layer2.preprocessor import Preprocessor
from layer3.stage1 import ProjectionDetector
from layer3.stage2 import TemplateMatcher
from layer3.tracking import (NoteTracker, LongnoteStateMachine, RawNote,
                             merge_duplicate_triggers)
from layer3.beat import BeatDetector, BeatEvent
from layer3.measureline import MeasureLineDetector, MeasureLineTracker, BarlineEvent


# event type order for the chronological feed (a tie in ms must still open a
# longnote head before it is closed by a tail)
_EVENT_ORDER = {"lnhead": 0, "note": 1, "lntail": 2}


def _print_progress(done: int, total: int, *, width: int = 40) -> None:
    """Render a single in-place progress bar (overwrites itself via ``\\r``)."""
    frac = done / total
    filled = int(width * frac)
    bar = "█" * filled + "░" * (width - filled)
    print(f"\r  [{bar}] {frac * 100:5.1f}%  {done}/{total} frames",
          end="", flush=True)


# =============================================================================
# Result
# =============================================================================

@dataclass
class Layer3Result:
    """The raw, ms-based product of the 1-Pass — Layer 4's input."""
    cal: Calibration
    notes: list[RawNote]            # taps + longnotes, sorted by (ms, lane)
    beats: list[BeatEvent]          # POW LED beats, sorted by frame
    barlines: list[BarlineEvent]    # measure boundaries, sorted by frame
    frame_count: int
    orphan_tails: int               # tails with no matching head (dropped)

    @property
    def fps(self) -> float:
        return self.cal.fps

    @property
    def duration_ms(self) -> float:
        return self.frame_count / self.cal.fps * 1000.0

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
        flash_bpm = (60.0 * self.cal.fps / bi) if bi > 0 else 0.0
        mi = self.barline_interval_frames()
        # beats per measure, if the LED flashes once per beat: a sanity ratio
        beats_per_measure = (mi / bi) if bi > 0 else 0.0
        lines = [
            f"=== Layer 3 result — {self.duration_ms / 1000:.1f}s, "
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

class Layer3Pipeline:
    """Drives the Layer 3 components over one video pass."""

    def __init__(self, cal: Calibration):
        self.cal = cal
        # per-frame POW LED brightness, retained after run() for visualization
        self.beat_signal: np.ndarray | None = None

    @classmethod
    def from_config(cls, song_toml_path: str | Path) -> "Layer3Pipeline":
        return cls(resolve_calibration(song_toml_path))

    # ------------------------------------------------------------------ #
    def run(self, *, progress: bool = True) -> Layer3Result:
        """Decode the video once; return the raw ms-based Layer3Result."""
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

        return Layer3Result(cal=self.cal, notes=notes, beats=beat.beats,
                            barlines=barline_events,
                            frame_count=frame_count,
                            orphan_tails=lnsm.orphan_tails)


# =============================================================================
# CLI: python layer3.py [config/song.toml]
# =============================================================================

if __name__ == "__main__":
    import sys
    import time

    cfg = sys.argv[1] if len(sys.argv) > 1 else "config/song.toml"
    song = Path(cfg).stem
    t0 = time.time()
    pipeline = Layer3Pipeline.from_config(cfg)
    result = pipeline.run()
    print(f"\n1-Pass complete in {time.time() - t0:.0f}s\n")
    print(result.summary())

    # verification artifacts: the raw piano-roll and the POW LED trace
    try:
        from layer3.debug_viz import plot_raw_chart, plot_beat_signal
        raw_png = f"{song}_raw_chart.png"
        beat_png = f"{song}_beat_signal.png"
        plot_raw_chart(result.notes, result.cal, raw_png)
        plot_beat_signal(pipeline.beat_signal, result.beats,
                         beat_png, frame_range=(900, 1300),
                         barlines=result.barlines)
        print(f"\nwrote {raw_png}, {beat_png}")
    except Exception as exc:                # viz is optional, never fatal
        print(f"\n(visualization skipped: {exc})")
