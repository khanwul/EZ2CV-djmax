"""
EZ2CV — end-to-end entry point
===============================================================================
Takes a single song TOML, runs Layer 1 -> 2 -> 3 -> 4 -> 5 sequentially, and
produces the final artifacts (``out/<song>/raw.json``, ``chart.json``) along
with optional visualizations.

Each layer's ``__main__`` block is for standalone module debugging; this file
is the single entry point used to run the full pipeline in one shot.

Usage
-----
    uv run python src/main.py
    uv run python src/main.py "config/GEHENNA.toml"
    uv run python src/main.py "config/JUSTITIA.toml" --no-chart-image
    uv run python src/main.py "config/Dream Walker.toml" --debug-png
    uv run python src/main.py "config/GEHENNA.toml"   --debug-video
    uv run python src/main.py "config/GEHENNA.toml"   --debug-video 0:1000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from layer1.calibration import resolve_calibration
from layer2.preprocessor import Preprocessor
from layer3 import Layer3Pipeline
from layer4 import Layer4Result
from layer5 import song_name_from_config, output_dir, write_all


DEFAULT_CONFIG = "config/song.toml"


# =============================================================================
# pretty-print helpers
# =============================================================================

def _banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _parse_range(spec: str, *, max_end: int) -> tuple[int, int]:
    """Parse ``START:END`` (either side optional) into a clamped frame range."""
    if spec == "" or spec is None:
        return (0, max_end)
    if ":" not in spec:
        raise argparse.ArgumentTypeError(
            f"frame range must look like START:END (got {spec!r})")
    start_s, end_s = spec.split(":", 1)
    start = int(start_s) if start_s.strip() else 0
    end = int(end_s) if end_s.strip() else max_end
    if not (0 <= start < end):
        raise argparse.ArgumentTypeError(
            f"invalid frame range {spec!r}: need 0 <= start < end")
    return (start, min(end, max_end))


# =============================================================================
# pipeline
# =============================================================================

def run(config_path: str | Path, *,
        chart_image: bool = True,
        debug_png: bool = False,
        debug_video: str | None = None,
        progress: bool = True) -> tuple[Path, Path]:
    """Run the full L1 → L2 → L3 → L4 → L5 chain for one song TOML.

    Parameters
    ----------
    config_path
        Path to the per-song TOML.
    chart_image
        If True, render ``chart_visual.png`` from the final ``chart.json``.
    debug_png
        If True, also write ``raw_chart.png`` and ``beat_signal.png``
        (Layer 3 static debug overlays).
    debug_video
        If not None, render a Layer 3 overlay mp4. Pass ``""`` for the full
        clip (0..frame_count), or ``"START:END"`` for a custom frame range.
    progress
        If True, Layer 3 prints a heartbeat every 1000 frames.

    Returns
    -------
    (raw_json_path, chart_json_path)
    """
    cfg = str(config_path)
    song = song_name_from_config(cfg)
    out_root = output_dir(song)

    # --- Layer 1 ------------------------------------------------------------
    _banner(f"[1/5] Layer 1 — Calibration         ({song})")
    t0 = time.time()
    cal = resolve_calibration(cfg)
    cal.summary()
    print(f"[layer 1 done in {time.time() - t0:.2f}s]")

    # --- Layer 2 ------------------------------------------------------------
    _banner(f"[2/5] Layer 2 — Preprocessor        ({song})")
    t0 = time.time()
    pre = Preprocessor(cal)
    frame_count = pre.frame_count
    duration_s = frame_count / cal.fps if cal.fps else 0.0
    print(f"  video       : {cal.video_path}")
    print(f"  frames      : {frame_count}  ({duration_s:.1f}s @ {cal.fps} fps)")
    print(f"  lanes / mode: {cal.key_count} lanes ({cal.key_mode})")
    print(f"[layer 2 ready in {time.time() - t0:.2f}s] "
          f"(streaming begins inside Layer 3)")

    # --- Layer 3 ------------------------------------------------------------
    _banner(f"[3/5] Layer 3 — 1-Pass video decode ({song})")
    t0 = time.time()
    pipeline = Layer3Pipeline(cal)
    l3 = pipeline.run(progress=progress)
    print(f"\n[layer 3 done in {time.time() - t0:.0f}s]")
    print(l3.summary())

    # --- Layer 4 ------------------------------------------------------------
    _banner(f"[4/5] Layer 4 — ms → tick + snap    ({song})")
    t0 = time.time()
    l4 = Layer4Result.from_layer3(l3)
    print(f"[layer 4 done in {time.time() - t0:.2f}s]")
    print(l4.summary())

    # --- Layer 5 ------------------------------------------------------------
    _banner(f"[5/5] Layer 5 — JSON serialization  ({song})")
    t0 = time.time()
    raw_path, chart_path = write_all(l3, l4, song_name=song)
    print(f"  wrote {raw_path}")
    print(f"  wrote {chart_path}")
    print(f"[layer 5 done in {time.time() - t0:.2f}s]")

    # --- optional visualizations -------------------------------------------
    if debug_png or debug_video is not None or chart_image:
        _banner("Visualization")

    if debug_png:
        try:
            from layer3.debug_viz import plot_raw_chart, plot_beat_signal
            raw_png = out_root / "raw_chart.png"
            beat_png = out_root / "beat_signal.png"
            plot_raw_chart(l3.notes, l3.cal, raw_png)
            plot_beat_signal(pipeline.beat_signal, l3.beats, beat_png,
                             barlines=l3.barlines)
            print(f"  wrote {raw_png}")
            print(f"  wrote {beat_png}")
        except Exception as exc:
            print(f"  (debug PNG skipped: {exc})")

    if debug_video is not None:
        try:
            from layer3.debug_viz import render_overlay_video
            start, end = _parse_range(debug_video, max_end=frame_count)
            video_out = out_root / f"debug_overlay_{start}_{end}.mp4"
            print(f"  rendering Layer 3 overlay video [{start}, {end}) "
                  f"-> {video_out}")
            render_overlay_video(cal, video_out, (start, end))
            print(f"  wrote {video_out}")
        except Exception as exc:
            print(f"  (debug video skipped: {exc})")

    if chart_image:
        try:
            from layer5.visualize_chart import render
            chart_png = out_root / "chart_visual.png"
            render(chart_path, chart_png,
                   lane_colors=[ln.color for ln in cal.lanes])
            print(f"  wrote {chart_png}")
        except Exception as exc:
            print(f"  (chart image skipped: {exc})")

    return raw_path, chart_path


# =============================================================================
# CLI
# =============================================================================

_EPILOG = """\
example
----
  uv run src/main.py
  uv run src/main.py "config/GEHENNA.toml"
  uv run src/main.py "config/JUSTITIA.toml" --no-chart-image
  uv run src/main.py "config/Dream Walker.toml" --debug-png
  uv run src/main.py "config/GEHENNA.toml" --debug-video
  uv run src/main.py "config/GEHENNA.toml" --debug-video 500:2000

output
-----------
  out/<song>/raw.json
  out/<song>/chart.json
  out/<song>/chart_visual.png                  (disable via --no-chart-image)
  out/<song>/raw_chart.png                     (--debug-png)
  out/<song>/beat_signal.png                   (--debug-png)
  out/<song>/debug_overlay_<start>_<end>.mp4   (--debug-video)
"""


# argparse sentinel: flag present without an explicit range
_VIDEO_FULL = ""


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ez2cv",
        description="EZ2CV — chart converter for EZ2ON REBOOT : R",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("config", nargs="?", default=DEFAULT_CONFIG,
                   help=f"path to TOML file (default: {DEFAULT_CONFIG!r})")
    p.add_argument("--no-chart-image", dest="chart_image",
                   action="store_false",
                   help="disable rendering of the final chart.json to chart_visual.png"
                        "(default: enabled)")
    p.add_argument("--debug-png", action="store_true",
                   help="save two additional Layer 3 static debug PNGs"
                        "(raw_chart.png: millisecond-based raw piano roll, "
                        "beat_signal.png: POW LED brightness trace)")
    p.add_argument("--debug-video", nargs="?", const=_VIDEO_FULL, default=None,
                   metavar="START:END",
                   help="save Layer 3 overlay as an MP4 file."
                        "no arg: full vid(0..frame_count), "
                        "'START:END': specify range(END exclusive)")
    p.add_argument("--quiet", action="store_true",
                   help="disable process log.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if not Path(args.config).exists():
        print(f"config not found: {args.config}", file=sys.stderr)
        return 2

    t0 = time.time()
    run(args.config,
        chart_image=args.chart_image,
        debug_png=args.debug_png,
        debug_video=args.debug_video,
        progress=not args.quiet)
    print(f"\n[total {time.time() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
