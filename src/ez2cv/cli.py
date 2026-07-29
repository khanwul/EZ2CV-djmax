"""EZ2CV command-line entry point."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ez2cv.chart import build_chart
from ez2cv.config import ConfigError, load_config
from ez2cv.detection import DetectionPipeline
from ez2cv.io import output_dir, read_raw, write_chart, write_raw


DEFAULT_INPUT = "config/song.toml"
_VIDEO_FULL = ""


def _banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _parse_range(spec: str, *, max_end: int) -> tuple[int, int]:
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


def _render_chart(chart_path: Path) -> Path:
    from ez2cv.visualize import render

    song = chart_path.stem.removesuffix("_chart")
    image_path = chart_path.with_name(f"{song}_chart_visual.png")
    render(chart_path, image_path)
    return image_path


def run(config_path: str | Path, *,
        chart_image: bool = True,
        debug_png: bool = False,
        debug_video: str | None = None,
        progress: bool = True) -> tuple[Path, Path]:
    """Detect a video, checkpoint raw JSON, then build the tick chart."""
    started = time.monotonic()
    config = load_config(config_path)
    song = config.song_name
    out_root = output_dir(song)

    _banner(f"Detection — video → milliseconds ({song})")
    config.summary()
    pipeline = DetectionPipeline(config)
    raw = pipeline.run(progress=progress)
    print(raw.summary())

    # The expensive video result is durable before fallible chart inference.
    raw_path = write_raw(raw)
    print(f"  checkpoint {raw_path}")

    _banner(f"Chart — milliseconds → ticks ({song})")
    chart = build_chart(raw)
    chart_path = write_chart(chart)
    print(chart.summary())
    print(f"  wrote {chart_path}")

    if debug_png:
        from ez2cv.detection.debug_viz import plot_beat_signal, plot_raw_chart

        raw_png = out_root / f"{song}_raw_chart.png"
        beat_png = out_root / f"{song}_beat_signal.png"
        plot_raw_chart(raw.notes, config, raw_png)
        plot_beat_signal(pipeline.beat_signal, raw.beats, beat_png,
                         barlines=raw.barlines)
        print(f"  wrote {raw_png}")
        print(f"  wrote {beat_png}")

    if debug_video is not None:
        from ez2cv.detection.debug_viz import render_overlay_video

        start, end = _parse_range(debug_video, max_end=raw.frame_count)
        video_out = out_root / f"{song}_debug_overlay_{start}_{end}.mp4"
        render_overlay_video(config, video_out, (start, end))
        print(f"  wrote {video_out}")

    if chart_image:
        print(f"  wrote {_render_chart(chart_path)}")

    print(f"\n[total {time.monotonic() - started:.0f}s]")
    return raw_path, chart_path


def run_from_raw(raw_path: str | Path, *, chart_image: bool = True) -> Path:
    """Rebuild a chart directly from a saved detection checkpoint."""
    raw = read_raw(raw_path)
    chart = build_chart(raw)
    chart_path = write_chart(chart)
    print(chart.summary())
    print(f"  wrote {chart_path}")
    if chart_image:
        print(f"  wrote {_render_chart(chart_path)}")
    return chart_path


_EPILOG = """\
examples
--------
  ez2cv config/GEHENNA.toml
  ez2cv config/GEHENNA.toml --debug-video 500:2000
  ez2cv out/GEHENNA/GEHENNA_raw.json --from-raw
"""


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ez2cv",
        description="Extract an EZ2ON chart from gameplay video",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input", nargs="?", default=DEFAULT_INPUT,
        help=f"song TOML or raw JSON (default: {DEFAULT_INPUT!r})")
    parser.add_argument(
        "--from-raw", action="store_true",
        help="skip video detection and rebuild the chart from raw JSON")
    parser.add_argument(
        "--no-chart-image", dest="chart_image", action="store_false",
        help="do not render the final chart image")
    parser.add_argument(
        "--debug-png", action="store_true",
        help="render detection diagnostic images")
    parser.add_argument(
        "--debug-video", nargs="?", const=_VIDEO_FULL, default=None,
        metavar="START:END", help="render a detection overlay video")
    parser.add_argument("--quiet", action="store_true",
                        help="hide frame progress")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    source = Path(args.input)
    if not source.is_file():
        print(f"input not found: {source}", file=sys.stderr)
        return 2
    if args.from_raw and (args.debug_png or args.debug_video is not None):
        print("detection debug options cannot be used with --from-raw",
              file=sys.stderr)
        return 2
    try:
        if args.from_raw:
            run_from_raw(source, chart_image=args.chart_image)
        else:
            run(source, chart_image=args.chart_image,
                debug_png=args.debug_png, debug_video=args.debug_video,
                progress=not args.quiet)
    except (ConfigError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
