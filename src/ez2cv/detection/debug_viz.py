"""
EZ2CV — debug visualization (READ-ONLY side channel)
===============================================================================
Renders what the detector "sees" so a human can sanity-check it. This module is
strictly an OBSERVER:

  * Every function draws on a COPY of the frame — never the array the detector
    holds (video preprocessing ROIs are numpy views; drawing on the original corrupts them).
  * It never feeds anything back into the pipeline. Pure data-in, image-out.
  * It is for a --debug mode only; keep it OUT of production runs (drawing +
    encoding is slow).

Two views are exposed:

  * STATIC per-frame PNGs (`annotate_stage1`, `plot_stage1_projection`,
    `annotate_stage2`) + post-run summaries (`plot_raw_chart`,
    `plot_beat_signal`). Cheap, used by the detection pipeline.
  * OVERLAY MP4 (`render_overlay_video`) for a frame range. Shows the model's
    per-frame decisions IN MOTION: typed Stage 2 matches, rejected Stage 1
    runs, tracked-edge trails, trigger crossings as they fire, the in-flight
    measure line, POW LED beats, and a running status panel. The right tool
    for checking "did the model actually recognise this segment correctly".
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from ez2cv.config import RunConfig
from ez2cv.detection.stage1 import Stage1Result
from ez2cv.detection.stage2 import Stage2Result

# BGR colors
_C_LANE    = (90, 90, 90)
_C_JUDGE   = (60, 60, 255)     # judgment line  (red)
_C_TRIGGER = (255, 60, 255)    # trigger y_top  (magenta)
_C_SHORT   = (90, 230, 90)     # short run      (green)  -> regular-note hint
_C_LONG    = (40, 170, 255)    # long  run      (orange) -> longnote hint
_C_TEXT    = (255, 255, 255)


def annotate_stage1(
    frame_bgr: np.ndarray,
    cal: RunConfig,
    results: list[Stage1Result],
    *,
    crop: bool = True,
) -> np.ndarray:
    """Draw Stage 1 runs over a frame. Returns a NEW annotated BGR image."""
    vis = frame_bgr.copy()                         # <-- never touch the original

    x_lo = cal.lanes[0].x_range[0]
    x_hi = cal.lanes[-1].x_range[1]
    for ln in cal.lanes:
        cv2.rectangle(vis, (ln.x_range[0], cal.playfield_top),
                      (ln.x_range[1], cal.playfield_bottom), _C_LANE, 1)
    cv2.line(vis, (x_lo - 30, cal.line_y), (x_hi + 30, cal.line_y), _C_JUDGE, 1)
    cv2.line(vis, (x_lo - 30, cal.trigger_template_y_top),
             (x_hi + 30, cal.trigger_template_y_top), _C_TRIGGER, 1)

    for res in results:
        ln = cal.lanes[res.lane_index]
        for r in res.runs:
            color = _C_SHORT if r.kind == "short" else _C_LONG
            fy0 = r.y_start + res.roi_y_origin
            fy1 = r.y_end + res.roi_y_origin
            cv2.rectangle(vis, (ln.x_range[0] + 1, fy0),
                          (ln.x_range[1] - 1, fy1), color, 2)
            label = f"{r.kind[0]}{r.length}"
            cv2.putText(vis, label, (ln.x_range[0] + 3, max(fy0 - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    if results:
        cap = (f"frame {results[0].frame_index}  Stage1: green=short(reg) "
               f"orange=long(LN)  thr={results[0].threshold:g}")
        cv2.putText(vis, cap, (x_lo - 25, cal.playfield_bottom + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _C_TEXT, 1, cv2.LINE_AA)

    if crop:
        y2 = min(vis.shape[0], cal.playfield_bottom + 40)
        x1 = max(0, x_lo - 60)
        x2 = min(vis.shape[1], x_hi + 60)
        vis = vis[0:y2, x1:x2]
    return vis


def plot_stage1_projection(
    results: list[Stage1Result],
    cal: RunConfig,
    save_path: str | Path,
) -> Path:
    """Plot the 1D projection signal of every lane with runs shaded."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 7), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        proj = res.projection
        ys = np.arange(len(proj))
        ax.plot(proj, ys, color="#222", lw=0.8)
        ax.axvline(res.threshold, color="#d33", lw=1.0, ls="--")
        for r in res.runs:
            color = "#5ce65c" if r.kind == "short" else "#ffaa28"
            ax.axhspan(r.y_start, r.y_end, color=color, alpha=0.5)
            ax.text(255, r.y_center, f"{r.kind[0]}{r.length}",
                    fontsize=7, va="center", ha="right")
        ax.axhline(cal.trigger_template_y_top - res.roi_y_origin,
                   color="#d33", lw=0.8, alpha=0.6)
        ax.set_title(f"L{res.lane_index+1} ({res.color})", fontsize=9)
        ax.set_xlim(0, 260)
        ax.set_xlabel("row mean (0-255)", fontsize=7)
        ax.invert_yaxis()
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("ROI y (0 = playfield top)", fontsize=8)
    fig.suptitle(f"Stage 1 projection — frame {results[0].frame_index} "
                 f"(dashed red = threshold)", fontsize=10)
    fig.tight_layout()

    out = Path(save_path)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


# type -> BGR color for Stage 2 matches
_C_MATCH = {
    "note":   (90, 230, 90),     # green
    "lnhead": (230, 200, 60),    # cyan-ish
    "lntail": (60, 200, 255),    # amber
}
_C_REJECT  = (95, 95, 95)        # gray — Stage 1 run that Stage 2 dropped
_C_BARLINE = (60, 220, 220)      # yellow-cyan — in-flight measure line
_C_LED_ON  = (0, 255, 255)
_C_LED_OFF = (90, 90, 90)
_C_EXTRAP  = (80, 80, 255)
_C_PANEL   = (240, 240, 240)


def annotate_stage2(
    frame_bgr: np.ndarray,
    cal: RunConfig,
    s1_results: list[Stage1Result],
    s2_results: list[Stage2Result],
    *,
    crop: bool = True,
) -> np.ndarray:
    """Draw Stage 2 confirmed matches + Stage 1 rejects. Returns a NEW image."""
    vis = frame_bgr.copy()
    x_lo = cal.lanes[0].x_range[0]
    x_hi = cal.lanes[-1].x_range[1]

    for ln in cal.lanes:
        cv2.rectangle(vis, (ln.x_range[0], cal.playfield_top),
                      (ln.x_range[1], cal.playfield_bottom), _C_LANE, 1)
    cv2.line(vis, (x_lo - 30, cal.line_y), (x_hi + 30, cal.line_y), _C_JUDGE, 1)
    cv2.line(vis, (x_lo - 30, cal.trigger_template_y_top),
             (x_hi + 30, cal.trigger_template_y_top), _C_TRIGGER, 1)

    confirmed_runs = {id(m.source_run) for r in s2_results for m in r.matches}

    for res in s1_results:
        ln = cal.lanes[res.lane_index]
        for run in res.runs:
            if id(run) in confirmed_runs:
                continue
            fy0 = run.y_start + res.roi_y_origin
            fy1 = run.y_end + res.roi_y_origin
            cv2.rectangle(vis, (ln.x_range[0] + 2, fy0),
                          (ln.x_range[1] - 2, fy1), _C_REJECT, 1)

    for res in s2_results:
        ln = cal.lanes[res.lane_index]
        for m in res.matches:
            color = _C_MATCH.get(m.type, (255, 255, 255))
            y0 = m.y_top
            y1 = m.y_top + cal.note_height
            cv2.rectangle(vis, (ln.x_range[0] + 1, y0),
                          (ln.x_range[1] - 1, y1), color, 2)
            cv2.putText(vis, f"{m.type} {m.score:.2f}",
                        (ln.x_range[0] + 2, max(y0 - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    if s2_results:
        cap = (f"frame {s2_results[0].frame_index}  Stage2: "
               f"green=note cyan=lnhead amber=lntail  gray=rejected")
        cv2.putText(vis, cap, (x_lo - 25, cal.playfield_bottom + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _C_TEXT, 1, cv2.LINE_AA)

    if crop:
        x1 = max(0, x_lo - 60)
        x2 = min(vis.shape[1], x_hi + 60)
        vis = vis[0:min(vis.shape[0], cal.playfield_bottom + 40), x1:x2]
    return vis


def plot_raw_chart(notes, cal, save_path: str | Path) -> Path:
    """Piano-roll of the detection raw output: lanes vs time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lane_color = {"white": "#d2d2d2", "cyan": "#1fb6d8"}
    fig, ax = plt.subplots(figsize=(14, 3.2))
    for n in notes:
        c = lane_color.get(n.color, "#888888")
        if n.type == "longnote" and n.end_ms is not None:
            ax.plot([n.trigger_ms, n.end_ms], [n.lane, n.lane], color=c,
                    lw=7, solid_capstyle="round", alpha=0.85, zorder=2)
        ring = "#e2554f" if n.extrapolated else "#303030"
        ax.scatter([n.trigger_ms], [n.lane], color=c, s=46,
                   edgecolors=ring, linewidths=1.3, zorder=3)

    ax.set_yticks(range(cal.key_count))
    ax.set_yticklabels([f"L{i + 1}" for i in range(cal.key_count)])
    ax.set_ylim(cal.key_count - 0.5, -0.5)
    ax.set_xlabel("time (ms)")
    ax.set_xlim(left=0)
    taps = sum(1 for n in notes if n.type == "tap")
    lns = sum(1 for n in notes if n.type == "longnote")
    ax.set_title(f"detection raw chart — {len(notes)} notes "
                 f"({taps} tap, {lns} longnote);  red ring = extrapolated")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    out = Path(save_path)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_beat_signal(signal, beats, save_path: str | Path,
                     frame_range: tuple[int, int] | None = None,
                     barlines=None) -> Path:
    """POW LED brightness signal with detected beat onsets marked.

    If `barlines` (list of BarlineEvent) is provided, each measure-line
    judgment-line crossing is overlaid as a dashed vertical so beat onsets
    and measure boundaries can be visually compared.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    signal = np.asarray(signal)
    f0, f1 = frame_range if frame_range else (0, len(signal))
    fig, ax = plt.subplots(figsize=(32, 3.4))
    ax.plot(range(f0, f1), signal[f0:f1], lw=0.9, color="#16a085",
            label="beat ROI brightness")

    beat_labelled = False
    for b in beats:
        if f0 <= b.frame_index < f1:
            ax.axvline(b.frame_index, color="#e2554f", lw=0.8, alpha=0.7,
                       label="beat onset" if not beat_labelled else None)
            beat_labelled = True

    n_bars_in_range = 0
    if barlines:
        bar_labelled = False
        for bar in barlines:
            if f0 <= bar.cross_frame < f1:
                ax.axvline(bar.cross_frame, color="#6c5ce7", lw=1.0,
                           ls="--", alpha=0.7,
                           label="measure line" if not bar_labelled else None)
                bar_labelled = True
                n_bars_in_range += 1

    ax.set_xlabel("frame")
    ax.set_ylabel("POW LED green mean")
    bars_part = (f", {n_bars_in_range} barlines (purple dashed)"
                 if barlines else "")
    ax.set_title(f"POW LED signal — {len(beats)} beats detected "
                 f"(red line = onset){bars_part}")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    out = Path(save_path)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


# =============================================================================
# Overlay video — per-frame composite of every detection decision
# =============================================================================
#
# Why this exists
# ---------------
# The static piano-roll tells you WHAT the model emitted; it does not tell you
# WHY. To check whether the model is correctly reading the gameplay you have to
# watch its per-frame state on top of the actual video: which Stage 1 runs were
# confirmed and which dropped, how tracked edges follow the descending notes
# through capture stutter, the moment a trigger crossing fires, the POW LED
# flashing on each beat, the lone thin band that becomes a measure line.
#
# Design notes
# ------------
# * Pure observer — runs the full pipeline like DetectionPipeline.run() but reads
#   the trackers' internal state (`tracker._lanes`, `mlt._lines`) for drawing
#   only. The pipeline outputs are not mutated.
# * Warm-up: pipeline runs from frame 0 through `frame_range[1]`, but only
#   frames in [start, end) are written to the mp4. This keeps the scroll-speed
#   estimator, beat-amplitude EMA, and longnote grace state correct at the
#   start of the visualised range. For a late-in-song range expect the warm-up
#   to dominate runtime.
# * Two decodes: the Preprocessor decodes for detection, a second VideoCapture
#   decodes the raw BGR for drawing. The second one stays closed until the
#   first in-range frame so warm-up costs only one decode.

_TRAIL_FADE = 0.12               # per-step color attenuation in tracked-edge trail
_FLASH_FRAMES = 6                # how long a fired event lingers on screen


def render_overlay_video(
    cal: RunConfig,
    save_path: str | Path,
    frame_range: tuple[int, int],
    *,
    trail_length: int = 8,
    progress: bool = True,
    force: bool = False,
) -> Path:
    """Render a frame-range mp4 of detection's per-frame decisions.

    Parameters
    ----------
    cal
        A resolved RunConfig (see `load_config`).
    save_path
        Output mp4 path. Parent directories are created if missing.
    frame_range
        `(start, end)` frame indices, end-exclusive.
    trail_length
        How many past positions of a tracked edge are drawn as fading dots.
    progress
        If True, show an in-place progress bar (warmup / render phases).

    Returns
    -------
    The save_path as a Path, after the writer has been released.
    """
    # imports kept local so the static viz functions stay usable without the
    # pipeline imports being satisfied
    from ez2cv.video import Preprocessor
    from ez2cv.detection.stage1 import ProjectionDetector
    from ez2cv.detection.stage2 import TemplateMatcher
    from ez2cv.detection.tracking import NoteTracker
    from ez2cv.detection.beat import BeatDetector
    from ez2cv.detection.barline import MeasureLineDetector, MeasureLineTracker
    from ez2cv.detection.pipeline import _frame_to_ms, _print_progress

    start, end = frame_range
    if not (0 <= start < end):
        raise ValueError(f"invalid frame_range {frame_range!r}")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    pre = Preprocessor(cal, force=force)
    s1 = ProjectionDetector(cal)
    s2 = TemplateMatcher(cal)
    tracker = NoteTracker(cal)
    beat = BeatDetector(cal)
    mld = MeasureLineDetector(cal)
    mlt = MeasureLineTracker(cal)

    raw_cap: cv2.VideoCapture | None = None
    writer: cv2.VideoWriter | None = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    recent_triggers: list = []          # (frame_seen, TriggerEvent)
    recent_barlines: list = []          # (frame_seen, BarlineEvent)
    recent_beats: list = []             # (frame_seen, BeatEvent)
    cum_triggers = 0
    cum_beats = 0
    cum_barlines = 0
    timestamps: list[float] = []

    for pf in pre:
        if pf.frame_index >= end:
            break
        timestamps.append(pf.timestamp_ms)

        s1r = s1.detect_frame(pf)
        s2r = s2.match_frame(pf, s1r)
        new_trigs = tracker.step(pf.frame_index, s2r)
        ml_dets = mld.detect_frame(s1r)
        new_bars = mlt.step(pf.frame_index, ml_dets)
        new_beat = beat.step(pf)
        beat_signal_val = float(pf.beat_roi.mean())
        for event in new_trigs:
            event.ms = _frame_to_ms(event.cross_frame, timestamps, cal.fps)
        for event in new_bars:
            event.ms = _frame_to_ms(event.cross_frame, timestamps, cal.fps)
        if new_beat is not None:
            new_beat.ms = _frame_to_ms(
                new_beat.frame_index, timestamps, cal.fps)

        cum_triggers += len(new_trigs)
        cum_barlines += len(new_bars)
        if new_beat is not None:
            cum_beats += 1
        for ev in new_trigs:
            recent_triggers.append((pf.frame_index, ev))
        for ev in new_bars:
            recent_barlines.append((pf.frame_index, ev))
        if new_beat is not None:
            recent_beats.append((pf.frame_index, new_beat))
        # expire stale flashes
        recent_triggers = [(f, e) for f, e in recent_triggers
                           if pf.frame_index - f < _FLASH_FRAMES]
        recent_barlines = [(f, e) for f, e in recent_barlines
                           if pf.frame_index - f < _FLASH_FRAMES]
        recent_beats = [(f, e) for f, e in recent_beats
                        if pf.frame_index - f < _FLASH_FRAMES]

        if progress and (pf.frame_index % 15 == 0 or pf.frame_index == end - 1):
            phase = "warmup" if pf.frame_index < start else "render"
            _print_progress(pf.frame_index + 1, end, label=phase)

        if pf.frame_index < start:
            continue

        # open the raw decoder on the first in-range frame, seek to start
        if raw_cap is None:
            raw_cap = cv2.VideoCapture(str(cal.video_path))
            if not raw_cap.isOpened():
                raise RuntimeError(f"cannot open video {cal.video_path}")
            raw_cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        ok, raw = raw_cap.read()
        if not ok:
            print(f"  raw decoder ran dry at frame {pf.frame_index}")
            break

        vis = _draw_overlay_frame(
            raw, cal, pf, s1r, s2r,
            tracker_lanes=tracker._lanes,
            tracked_barlines=mlt._lines,
            scroll_speed=tracker.speed.speed,
            recent_triggers=recent_triggers,
            recent_barlines=recent_barlines,
            recent_beats=recent_beats,
            beat_signal=beat_signal_val,
            cumulative=(cum_triggers, cum_beats, cum_barlines),
            trail_length=trail_length,
        )

        if writer is None:
            h, w = vis.shape[:2]
            writer = cv2.VideoWriter(str(save_path), fourcc, cal.fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(
                    f"VideoWriter failed to open: {save_path} "
                    f"(check codec availability for fourcc=mp4v)")
        writer.write(vis)

    if progress:
        print()                             # end the in-place bar line

    if raw_cap is not None:
        raw_cap.release()
    if writer is not None:
        writer.release()
    return save_path


def _draw_overlay_frame(
    raw_bgr: np.ndarray,
    cal: RunConfig,
    pf,
    s1_results: list[Stage1Result],
    s2_results: list[Stage2Result],
    *,
    tracker_lanes: dict,
    tracked_barlines: list,
    scroll_speed: float,
    recent_triggers: list,
    recent_barlines: list,
    recent_beats: list,
    beat_signal: float,
    cumulative: tuple[int, int, int],
    trail_length: int,
) -> np.ndarray:
    """Compose a single overlay frame. Pure data-in, image-out."""
    vis = raw_bgr.copy()
    x_lo = cal.lanes[0].x_range[0]
    x_hi = cal.lanes[-1].x_range[1]

    # --- reference geometry ---------------------------------------------------
    for ln in cal.lanes:
        cv2.rectangle(vis, (ln.x_range[0], cal.playfield_top),
                      (ln.x_range[1], cal.playfield_bottom), _C_LANE, 1)
    cv2.line(vis, (x_lo - 30, cal.line_y),
             (x_hi + 30, cal.line_y), _C_JUDGE, 1)
    cv2.line(vis, (x_lo - 30, cal.trigger_template_y_top),
             (x_hi + 30, cal.trigger_template_y_top), _C_TRIGGER, 1)

    # --- rejected Stage 1 runs (faint) ---------------------------------------
    confirmed_runs = {id(m.source_run) for r in s2_results for m in r.matches}
    for res in s1_results:
        ln = cal.lanes[res.lane_index]
        for run in res.runs:
            if id(run) in confirmed_runs:
                continue
            fy0 = run.y_start + res.roi_y_origin
            fy1 = run.y_end + res.roi_y_origin
            cv2.rectangle(vis, (ln.x_range[0] + 2, fy0),
                          (ln.x_range[1] - 2, fy1), _C_REJECT, 1)

    # --- Stage 2 confirmed matches -------------------------------------------
    for res in s2_results:
        ln = cal.lanes[res.lane_index]
        for m in res.matches:
            color = _C_MATCH.get(m.type, (255, 255, 255))
            y0 = m.y_top
            y1 = m.y_top + cal.note_height
            cv2.rectangle(vis, (ln.x_range[0] + 1, y0),
                          (ln.x_range[1] - 1, y1), color, 2)
            cv2.putText(vis, f"{m.type[:2]} {m.score:.2f}",
                        (ln.x_range[0] + 2, max(y0 - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)

    # --- tracked-edge trails -------------------------------------------------
    # The trail is the proof that the tracker is following ONE physical note
    # across frames rather than detecting fresh ones every time. Watch a fast
    # note: the trail forms a clean diagonal of dots.
    for edges in tracker_lanes.values():
        for e in edges:
            ln = cal.lanes[e.lane]
            cx = (ln.x_range[0] + ln.x_range[1]) // 2
            base = _C_MATCH.get(e.type, (255, 255, 255))
            trail = e.trajectory[-trail_length:]
            for i, (_, y, _) in enumerate(trail):
                age = len(trail) - 1 - i
                fade = max(0.25, 1.0 - age * _TRAIL_FADE)
                color = tuple(int(c * fade) for c in base)
                radius = max(1, 3 - age // 3)
                cv2.circle(vis, (cx, int(y) + cal.note_height // 2),
                           radius, color, -1, cv2.LINE_AA)
            last_y = int(e.trajectory[-1][1])
            cv2.putText(vis, f"#{e.id}",
                        (cx + 6, last_y + cal.note_height // 2 + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                        (220, 220, 220), 1, cv2.LINE_AA)

    # --- in-flight measure lines ---------------------------------------------
    for line in tracked_barlines:
        y = int(line.last_y)
        cv2.line(vis, (x_lo - 30, y), (x_hi + 30, y),
                 _C_BARLINE, 1, cv2.LINE_AA)
        cv2.putText(vis, f"bar #{line.id}", (x_hi + 35, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, _C_BARLINE, 1, cv2.LINE_AA)

    # --- POW LED region + beat indicator -------------------------------------
    bx1, by1, bx2, by2 = cal.beat_roi
    beat_fresh = any(pf.frame_index - f < 3 for f, _ in recent_beats)
    led_color = _C_LED_ON if beat_fresh else _C_LED_OFF
    cv2.rectangle(vis, (bx1, by1), (bx2, by2), led_color, 2)
    cv2.putText(vis, f"POW {beat_signal:.0f}", (bx1, max(by1 - 6, 16)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, led_color, 1, cv2.LINE_AA)
    if beat_fresh:
        cv2.putText(vis, "BEAT!", (bx1, min(by2 + 22, vis.shape[0] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, _C_LED_ON, 2, cv2.LINE_AA)

    # --- trigger-crossing flashes (just-emitted events) ----------------------
    # A crossing is interpolated AFTER the post-trigger frame is observed, so
    # the flash appears 1-2 frames AFTER the visible "passing the line". The
    # cross_frame stored in the event still carries the true sub-frame moment.
    for stack_i, (f, ev) in enumerate(recent_triggers):
        ln = cal.lanes[ev.lane]
        cx = (ln.x_range[0] + ln.x_range[1]) // 2
        color = _C_EXTRAP if ev.extrapolated else _C_MATCH.get(ev.type, (255,255,255))
        tag = ev.type + ("*" if ev.extrapolated else "")
        cv2.putText(vis, f"!{tag}",
                    (cx - 28, cal.line_y + 26 + stack_i * 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    for f, ev in recent_barlines:
        cv2.putText(vis, f"BARLINE @ {ev.ms:.0f}ms",
                    (x_hi + 35, cal.line_y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _C_BARLINE, 2, cv2.LINE_AA)

    # --- status panel (top-left) ---------------------------------------------
    lines = [
        f"frame {pf.frame_index}  t={pf.timestamp_ms:.0f}ms",
        f"scroll  {scroll_speed:5.2f} px/frame",
        f"triggers {cumulative[0]}   beats {cumulative[1]}   bars {cumulative[2]}",
    ]
    for i, text in enumerate(lines):
        cv2.putText(vis, text, (24, 38 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, _C_PANEL, 1, cv2.LINE_AA)
    return vis


# =============================================================================
# CLI: python debug_viz.py [config/song.toml] [frame_index | start:end] [out]
# =============================================================================

if __name__ == "__main__":
    from ez2cv.video import Preprocessor
    from ez2cv.detection.stage1 import ProjectionDetector
    from ez2cv.detection.stage2 import TemplateMatcher
    from ez2cv.config import load_config

    cfg = sys.argv[1] if len(sys.argv) > 1 else "config/song.toml"
    arg2 = sys.argv[2] if len(sys.argv) > 2 else "150"

    if ":" in arg2:
        # ---- overlay video mode ---------------------------------------------
        start_s, end_s = arg2.split(":", 1)
        start, end = int(start_s), int(end_s)
        cal = load_config(cfg)
        if len(sys.argv) > 3:
            out_path = Path(sys.argv[3])
        else:
            song = Path(cfg).stem
            out_path = Path("../out") / song / f"{song}_debug_overlay_{start}_{end}.mp4"
        print(f"rendering detection overlay video frames [{start}, {end}) "
              f"-> {out_path}")
        result = render_overlay_video(cal, out_path, (start, end))
        print(f"wrote {result}")
    else:
        # ---- single-frame PNG mode (existing behavior) ----------------------
        want = int(arg2)
        cal = load_config(cfg)
        pre = Preprocessor(cal)
        s1d = ProjectionDetector(cal)
        s2d = TemplateMatcher(cal)

        cap = cv2.VideoCapture(str(cal.video_path))
        s1r = s2r = raw_frame = None
        for pf in pre:
            if pf.frame_index == want:
                s1r = s1d.detect_frame(pf)
                s2r = s2d.match_frame(pf, s1r)
                cap.set(cv2.CAP_PROP_POS_FRAMES, want)
                _, raw_frame = cap.read()
                break
        cap.release()

        if raw_frame is None:
            print(f"frame {want} not reached")
            sys.exit(1)

        cv2.imwrite(f"stage1_overlay_f{want}.png",
                    annotate_stage1(raw_frame, cal, s1r))
        plot_stage1_projection(s1r, cal, f"stage1_projection_f{want}.png")
        cv2.imwrite(f"stage2_overlay_f{want}.png",
                    annotate_stage2(raw_frame, cal, s1r, s2r))
        print(f"wrote stage1_overlay_f{want}.png, stage1_projection_f{want}.png, "
              f"stage2_overlay_f{want}.png")
