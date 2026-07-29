"""JSON checkpoints and final chart serialization."""

from __future__ import annotations

import json
import math
from pathlib import Path

from ez2cv.chart import Chart, ChartNote
from ez2cv.chart.clock import BPMSegment
from ez2cv.chart.meter import TimeSignature, TimeSigVariant
from ez2cv.detection import RawChart
from ez2cv.detection.barline import BarlineEvent
from ez2cv.detection.beat import BeatEvent
from ez2cv.detection.tracking import RawNote


SCHEMA_VERSION = "2.0"


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite number {value}")


def _ms(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"non-finite ms value: {value}")
    return round(float(value), 3)


def _tick(value: int) -> int:
    if not math.isfinite(float(value)):
        raise ValueError(f"non-finite tick value: {value}")
    return int(value)


def _dump(obj: dict, path: Path) -> None:
    """Write through a temporary file so failed runs never leave partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as file:
        json.dump(obj, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")
    tmp.replace(path)


def serialize_raw(raw: RawChart) -> dict:
    """Return the complete, reloadable detection checkpoint payload."""
    return {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "song": raw.song_name,
            "skin": raw.skin_name,
            "key_mode": raw.key_mode,
            "lane_colors": list(raw.lane_colors),
            "display_resolution": list(raw.display_resolution),
            "video": raw.video_path,
            "fps": float(raw.fps),
            "note_speed": float(raw.note_speed),
            "tick_resolution": int(raw.tick_resolution),
            "min_bpm": float(raw.min_bpm),
            "max_bpm": float(raw.max_bpm),
            "frame_count": int(raw.frame_count),
            "duration_ms": _ms(raw.duration_ms),
            "orphan_tails": int(raw.orphan_tails),
        },
        "notes": [
            {
                "lane": int(note.lane),
                "color": note.color,
                "type": note.type,
                "trigger_ms": _ms(note.trigger_ms),
                "end_ms": _ms(note.end_ms) if note.end_ms is not None else None,
                "confidence": round(float(note.confidence), 4),
                "extrapolated": bool(note.extrapolated),
            }
            for note in raw.notes
        ],
        "beats": [
            {
                "frame_index": int(beat.frame_index),
                "ms": _ms(beat.ms),
                "strength": round(float(beat.strength), 3),
            }
            for beat in raw.beats
        ],
        "barlines": [
            {
                "cross_frame": round(float(barline.cross_frame), 3),
                "ms": _ms(barline.ms),
                "strength": round(float(barline.strength), 3),
                "extrapolated": bool(barline.extrapolated),
            }
            for barline in raw.barlines
        ],
    }


def read_raw(path: str | Path) -> RawChart:
    """Load a schema-2 raw checkpoint without video or template assets."""
    source = Path(path)
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
        )
        if payload["schema_version"] != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {payload['schema_version']!r}; "
                f"expected {SCHEMA_VERSION!r}")
        meta = payload["meta"]
        raw = RawChart(
            song_name=str(meta["song"]),
            skin_name=str(meta["skin"]),
            key_mode=str(meta["key_mode"]),
            lane_colors=tuple(str(color) for color in meta["lane_colors"]),
            display_resolution=tuple(int(v) for v in meta["display_resolution"]),
            video_path=str(meta["video"]),
            fps=float(meta["fps"]),
            note_speed=float(meta["note_speed"]),
            tick_resolution=int(meta["tick_resolution"]),
            min_bpm=float(meta["min_bpm"]),
            max_bpm=float(meta["max_bpm"]),
            frame_count=int(meta["frame_count"]),
            orphan_tails=int(meta["orphan_tails"]),
            notes=[RawNote(
                lane=int(note["lane"]),
                type=str(note["type"]),
                trigger_ms=float(note["trigger_ms"]),
                end_ms=(float(note["end_ms"])
                        if note["end_ms"] is not None else None),
                color=str(note["color"]),
                confidence=float(note["confidence"]),
                extrapolated=bool(note["extrapolated"]),
            ) for note in payload["notes"]],
            beats=[BeatEvent(
                frame_index=int(beat["frame_index"]),
                ms=float(beat["ms"]),
                strength=float(beat["strength"]),
            ) for beat in payload["beats"]],
            barlines=[BarlineEvent(
                cross_frame=float(barline["cross_frame"]),
                ms=float(barline["ms"]),
                strength=float(barline["strength"]),
                extrapolated=bool(barline["extrapolated"]),
            ) for barline in payload["barlines"]],
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid raw chart {source}: {exc}") from exc

    numeric = [raw.fps, raw.note_speed, raw.min_bpm, raw.max_bpm]
    numeric.extend(note.trigger_ms for note in raw.notes)
    numeric.extend(note.end_ms for note in raw.notes if note.end_ms is not None)
    numeric.extend(beat.ms for beat in raw.beats)
    numeric.extend(barline.ms for barline in raw.barlines)
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError(f"invalid raw chart {source}: non-finite number")
    if raw.fps <= 0 or raw.tick_resolution <= 0 or not raw.lane_colors:
        raise ValueError(f"invalid raw chart {source}: invalid timing or lanes")
    if raw.min_bpm <= 0 or raw.max_bpm < raw.min_bpm:
        raise ValueError(f"invalid raw chart {source}: invalid BPM range")
    if len(raw.display_resolution) != 2:
        raise ValueError(f"invalid raw chart {source}: invalid display resolution")
    if any(not 0 <= note.lane < raw.key_count for note in raw.notes):
        raise ValueError(f"invalid raw chart {source}: note lane out of range")
    if any(note.type not in {"tap", "longnote"} for note in raw.notes):
        raise ValueError(f"invalid raw chart {source}: unknown note type")
    if any(note.type == "longnote" and (
            note.end_ms is None or note.end_ms <= note.trigger_ms)
            for note in raw.notes):
        raise ValueError(f"invalid raw chart {source}: invalid longnote end")
    raw.notes.sort(key=lambda note: (note.trigger_ms, note.lane))
    raw.beats.sort(key=lambda beat: beat.frame_index)
    raw.barlines.sort(key=lambda barline: barline.cross_frame)
    return raw


def _note_payload(note: ChartNote) -> dict:
    out = {
        "lane": int(note.lane),
        "start_tick": _tick(note.start_tick),
        "end_tick": _tick(note.end_tick) if note.end_tick is not None else None,
    }
    if note.off_grid:
        out["off_grid"] = True
    return out


def _bpm_segment_payload(segment: BPMSegment) -> dict:
    return {
        "start_tick": _tick(segment.start_tick),
        "end_tick": _tick(segment.end_tick),
        "bpm_start": round(float(segment.bpm_start), 4),
        "bpm_end": round(float(segment.bpm_end), 4),
    }


def _time_sig_payload(global_ts: TimeSignature,
                      variants: list[TimeSigVariant]) -> dict:
    return {
        "global": [int(global_ts.numerator), int(global_ts.denominator)],
        "variants": [{
            "start_measure": int(variant.start_measure),
            "end_measure": int(variant.end_measure),
            "time_sig": [int(variant.time_sig.numerator),
                         int(variant.time_sig.denominator)],
        } for variant in variants],
    }


def serialize_chart(chart: Chart) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            "song": chart.song_name,
            "key_mode": chart.key_mode,
            "lane_colors": list(chart.lane_colors),
            "tick_resolution": int(chart.tick_resolution),
            "measure_zero_ms": _ms(chart.measure_zero_ms),
            "duration_ms": _ms(chart.stats["structure"]["duration_ms"]),
        },
        "bpm_segments": [_bpm_segment_payload(s) for s in chart.bpm_segments],
        "time_signature": _time_sig_payload(chart.global_time_sig,
                                            chart.variant_measures),
        "barlines_tick": [_tick(tick) for tick in chart.barlines_tick],
        "notes": [_note_payload(note) for note in chart.notes],
        "stats": chart.stats,
    }


def output_dir(song_name: str, *, root: str | Path = "out") -> Path:
    return Path(root) / song_name


def write_raw(raw: RawChart, *, root: str | Path = "out") -> Path:
    path = output_dir(raw.song_name, root=root) / f"{raw.song_name}_raw.json"
    _dump(serialize_raw(raw), path)
    return path


def write_chart(chart: Chart, *, root: str | Path = "out") -> Path:
    path = output_dir(chart.song_name, root=root) / f"{chart.song_name}_chart.json"
    _dump(serialize_chart(chart), path)
    return path
