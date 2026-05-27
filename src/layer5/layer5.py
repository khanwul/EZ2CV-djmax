"""
EZ2CV — Layer 5 : JSON serialization
===============================================================================
Writes the canonical pipeline products:

  * ``out/<song>/raw.json``    — Layer 3 (ms-based, debug-rich; reproducible)
  * ``out/<song>/chart.json``  — Layer 4 (tick-based, minimal, external-tool
                                 friendly)

Why two files
-------------
``raw.json`` carries every digit Layer 3 produced (per-note confidence, the
``extrapolated`` flag, ms coordinates, beats, barlines). Anyone wanting to
re-run Layer 4 with a different snapping policy reads this file and skips the
1-Pass entirely. ``chart.json`` strips all derivable / debug fields and keeps
only what a chart editor or game engine needs.

Format rules (shared)
---------------------
* indent=2, ``sort_keys=False`` so the key order below is preserved.
* All ms values: ``round(x, 3)``. All ticks: ``int``.
* NaN / Infinity is a bug — raise rather than serialise.
* Top-level ``"schema_version": "1.0"``.

Output naming
-------------
Both files live under ``out/<song_stem>/``. ``song_stem`` is derived from the
song TOML's filename — Layer 5 doesn't peek inside the TOML. Optional debug
PNGs (``raw_chart.png``, ``beat_signal.png``) can be dropped in the same
directory by Layer 3's debug_viz.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layer3 import Layer3Result
from layer4 import Layer4Result, ChartNote
from layer4.time_sig import TimeSignature, TimeSigVariant
from layer4.tick_clock import BPMSegment


SCHEMA_VERSION = "1.0"


# =============================================================================
# Internal helpers
# =============================================================================

def _ms(value: float) -> float:
    """Round to microsecond precision; raise on NaN/Inf."""
    if not math.isfinite(value):
        raise ValueError(f"non-finite ms value: {value}")
    return round(float(value), 3)


def _tick(value: int) -> int:
    """Integer tick; ensure it is actually an integer."""
    if not math.isfinite(float(value)):
        raise ValueError(f"non-finite tick value: {value}")
    return int(value)


def _dump(obj: dict, path: Path) -> None:
    """Atomic-ish write — temp file + rename, so a crash never leaves a
    half-written JSON next to a successful run.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False,
                  sort_keys=False, allow_nan=False)
        f.write("\n")
    tmp.replace(path)


# =============================================================================
# raw.json — Layer 3
# =============================================================================

def serialize_raw(l3: Layer3Result, *,
                  song_name: str,
                  video_path: str | None = None) -> dict:
    """Build the ``raw.json`` payload from a Layer3Result."""
    cal = l3.cal
    res = cal.display_resolution
    return {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "song": song_name,
            "key_mode": cal.key_mode,
            "skin": cal.skin_name,
            "video": video_path or str(cal.video_path),
            "resolution": f"{res[0]}x{res[1]}",
            "fps": float(cal.fps),
            "frame_count": int(l3.frame_count),
            "duration_ms": _ms(l3.duration_ms),
        },
        "notes": [
            {
                "lane": int(n.lane),
                "color": n.color,
                "type": n.type,
                "trigger_ms": _ms(n.trigger_ms),
                "end_ms": _ms(n.end_ms) if n.end_ms is not None else None,
                "confidence": round(float(n.confidence), 4),
                "extrapolated": bool(n.extrapolated),
            }
            for n in l3.notes
        ],
        "beats": [
            {
                "frame_index": int(b.frame_index),
                "ms": _ms(b.ms),
                "strength": round(float(b.strength), 3),
            }
            for b in l3.beats
        ],
        "barlines": [
            {
                "cross_frame": round(float(b.cross_frame), 3),
                "ms": _ms(b.ms),
                "strength": round(float(b.strength), 3),
                "extrapolated": bool(b.extrapolated),
            }
            for b in l3.barlines
        ],
    }


# =============================================================================
# chart.json — Layer 4
# =============================================================================

def _note_payload(n: ChartNote) -> dict:
    """One ChartNote → dict. ``off_grid`` is omitted when False."""
    out: dict = {"lane": int(n.lane), "start_tick": _tick(n.start_tick),
                 "end_tick": _tick(n.end_tick) if n.end_tick is not None
                             else None}
    if n.off_grid:
        out["off_grid"] = True
    return out


def _bpm_segment_payload(s: BPMSegment) -> dict:
    return {
        "start_tick": _tick(s.start_tick),
        "end_tick": _tick(s.end_tick),
        "bpm_start": round(float(s.bpm_start), 4),
        "bpm_end": round(float(s.bpm_end), 4),
    }


def _time_sig_payload(global_ts: TimeSignature,
                      variants: list[TimeSigVariant]) -> dict:
    return {
        "global": [int(global_ts.numerator), int(global_ts.denominator)],
        "variants": [
            {
                "start_measure": int(v.start_measure),
                "end_measure": int(v.end_measure),
                "time_sig": [int(v.time_sig.numerator),
                             int(v.time_sig.denominator)],
            }
            for v in variants
        ],
    }


def serialize_chart(l4: Layer4Result, *, song_name: str) -> dict:
    """Build the ``chart.json`` payload from a Layer4Result."""
    cal = l4.cal
    return {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "song": song_name,
            "key_mode": cal.key_mode,
            "tick_resolution": int(cal.tick_resolution),
            "measure_zero_ms": _ms(l4.measure_zero_ms),
            "duration_ms": _ms(l4.stats["structure"]["duration_ms"]),
        },
        "bpm_segments": [_bpm_segment_payload(s) for s in l4.bpm_segments],
        "time_signature": _time_sig_payload(l4.global_time_sig,
                                            l4.variant_measures),
        "barlines_tick": [_tick(t) for t in l4.barlines_tick],
        "notes": [_note_payload(n) for n in l4.notes],
        "stats": l4.stats,
    }


# =============================================================================
# Orchestration
# =============================================================================

def song_name_from_config(song_toml_path: str | Path) -> str:
    """Derive a display name from the TOML filename (no extension)."""
    return Path(song_toml_path).stem


def output_dir(song_name: str, *, root: str | Path = "out") -> Path:
    return Path(root) / song_name


def write_all(l3: Layer3Result, l4: Layer4Result, *,
              song_name: str,
              root: str | Path = "out") -> tuple[Path, Path]:
    """Write ``raw.json`` + ``chart.json`` for one run. Returns both paths."""
    out_dir = output_dir(song_name, root=root)
    raw_path = out_dir / "raw.json"
    chart_path = out_dir / "chart.json"
    _dump(serialize_raw(l3, song_name=song_name), raw_path)
    _dump(serialize_chart(l4, song_name=song_name), chart_path)
    return raw_path, chart_path


# =============================================================================
# CLI: python layer5.py [config/song.toml] — runs the full L3+L4+L5 chain
# =============================================================================

if __name__ == "__main__":
    import time

    from layer3 import Layer3Pipeline

    cfg = sys.argv[1] if len(sys.argv) > 1 else "config/song.toml"
    song = song_name_from_config(cfg)

    t0 = time.time()
    l3 = Layer3Pipeline.from_config(cfg).run()
    print(f"\n[layer 3 done in {time.time() - t0:.0f}s]")
    print(l3.summary(), "\n")

    t0 = time.time()
    l4 = Layer4Result.from_layer3(l3)
    print(f"[layer 4 done in {time.time() - t0:.2f}s]")
    print(l4.summary(), "\n")

    raw_path, chart_path = write_all(l3, l4, song_name=song)
    print(f"[layer 5] wrote {raw_path}")
    print(f"[layer 5] wrote {chart_path}")
