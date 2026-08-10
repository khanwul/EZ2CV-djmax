"""JSON checkpoints and final chart serialization."""

from __future__ import annotations

import json
import math
from pathlib import Path

from ez2cv.chart import Chart, ChartNote
from ez2cv.chart.clock import BPMSegment
from ez2cv.chart.meter import TimeSignature, TimeSigVariant
from ez2cv.detection import RawChart, TrackMetadata
from ez2cv.detection.barline import BarlineEvent
from ez2cv.detection.beat import BeatEvent
from ez2cv.detection.tracking import RawNote


RAW_SCHEMA_VERSION = "3.1"
CHART_FORMAT = "ez2cv.chart"
CHART_VERSION = "3.1"


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
        "schema_version": RAW_SCHEMA_VERSION,
        "meta": {
            "song": raw.song_name,
            "difficulty": raw.difficulty,
            "skin": raw.skin_name,
            "key_mode": raw.key_mode,
            "normal_lane_count": raw.normal_lane_count,
            "tracks": [{
                "index": track.index,
                "name": track.name,
                "role": track.role,
                "color": track.color,
            } for track in raw.tracks],
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
                "timing_sigma_ms": _ms(note.timing_sigma_ms),
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
    """Load a schema-3 raw checkpoint without video or template assets."""
    source = Path(path)
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
        )
        if payload["schema_version"] != RAW_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {payload['schema_version']!r}; "
                f"expected {RAW_SCHEMA_VERSION!r}")
        meta = payload["meta"]
        if meta["difficulty"] not in {"NM", "HD", "MX", "SC"}:
            raise ValueError("invalid difficulty")
        raw = RawChart(
            song_name=str(meta["song"]),
            difficulty=str(meta["difficulty"]),
            skin_name=str(meta["skin"]),
            key_mode=str(meta["key_mode"]),
            tracks=tuple(TrackMetadata(
                index=int(track["index"]),
                name=str(track["name"]),
                role=str(track["role"]),
                color=str(track["color"]),
            ) for track in meta["tracks"]),
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
                timing_sigma_ms=float(note.get("timing_sigma_ms", 0.0)),
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
    numeric.extend(note.timing_sigma_ms for note in raw.notes)
    numeric.extend(beat.ms for beat in raw.beats)
    numeric.extend(barline.ms for barline in raw.barlines)
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError(f"invalid raw chart {source}: non-finite number")
    if raw.fps <= 0 or raw.tick_resolution <= 0 or not raw.tracks:
        raise ValueError(f"invalid raw chart {source}: invalid timing or lanes")
    if [track.index for track in raw.tracks] != list(range(raw.key_count)):
        raise ValueError(f"invalid raw chart {source}: invalid track indexes")
    if any(track.role not in {"normal", "overlay"} for track in raw.tracks):
        raise ValueError(f"invalid raw chart {source}: invalid track role")
    if int(meta["normal_lane_count"]) != raw.normal_lane_count:
        raise ValueError(f"invalid raw chart {source}: normal lane count mismatch")
    if raw.min_bpm <= 0 or raw.max_bpm < raw.min_bpm:
        raise ValueError(f"invalid raw chart {source}: invalid BPM range")
    if len(raw.display_resolution) != 2:
        raise ValueError(f"invalid raw chart {source}: invalid display resolution")
    if any(not 0 <= note.lane < raw.key_count for note in raw.notes):
        raise ValueError(f"invalid raw chart {source}: note lane out of range")
    if any(note.type not in {"tap", "longnote"} for note in raw.notes):
        raise ValueError(f"invalid raw chart {source}: unknown note type")
    if any(note.timing_sigma_ms < 0 for note in raw.notes):
        raise ValueError(f"invalid raw chart {source}: negative timing sigma")
    if any(note.type == "longnote" and (
            note.end_ms is None or note.end_ms <= note.trigger_ms)
            for note in raw.notes):
        raise ValueError(f"invalid raw chart {source}: invalid longnote end")
    raw.notes.sort(key=lambda note: (note.trigger_ms, note.lane))
    raw.beats.sort(key=lambda beat: beat.frame_index)
    raw.barlines.sort(key=lambda barline: barline.cross_frame)
    return raw


def _note_payload(note: ChartNote, note_id: int) -> dict:
    return {
        "id": note_id,
        "lane": int(note.lane),
        "start_tick": _tick(note.start_tick),
        "end_tick": _tick(note.end_tick) if note.end_tick is not None else None,
        "off_grid": bool(note.off_grid),
    }


def _bpm_segment_payload(segment: BPMSegment) -> dict:
    out = {
        "start_tick": _tick(segment.start_tick),
        "end_tick": _tick(segment.end_tick),
    }
    if segment.is_constant:
        out.update(bpm=round(float(segment.bpm_start), 4),
                   interpolation="step")
    else:
        out.update(bpm_start=round(float(segment.bpm_start), 4),
                   bpm_end=round(float(segment.bpm_end), 4),
                   interpolation="linear_time")
    return out


def _meter_events(global_ts: TimeSignature, variants: list[TimeSigVariant],
                  barlines: list[int]) -> list[dict]:
    meters = [global_ts] * max(0, len(barlines) - 1)
    for variant in variants:
        meters[variant.start_measure:variant.end_measure + 1] = [
            variant.time_sig] * (variant.end_measure - variant.start_measure + 1)

    events: list[dict] = []
    previous = None
    for measure, meter in enumerate(meters):
        if meter != previous:
            events.append({
                "start_tick": _tick(barlines[measure]),
                "numerator": int(meter.numerator),
                "denominator": int(meter.denominator),
            })
            previous = meter
    return events


def serialize_chart(chart: Chart) -> dict:
    return {
        "format": CHART_FORMAT,
        "version": CHART_VERSION,
        "meta": {
            "song": chart.song_name,
            "difficulty": chart.difficulty,
            "key_mode": chart.key_mode,
            "normal_lane_count": sum(track.role == "normal"
                                     for track in chart.tracks),
            "tracks": [{
                "index": track.index,
                "name": track.name,
                "role": track.role,
                "color": track.color,
            } for track in chart.tracks],
            "duration_ms": _ms(chart.stats["structure"]["duration_ms"]),
        },
        "timing": {
            "ticks_per_quarter": int(chart.tick_resolution),
            "tick_zero_ms": _ms(chart.measure_zero_ms),
            "tempo_segments": [
                _bpm_segment_payload(segment)
                for segment in chart.bpm_segments],
            "meter_events": _meter_events(
                chart.global_time_sig, chart.variant_measures,
                chart.barlines_tick),
            "barlines": [_tick(tick) for tick in chart.barlines_tick],
        },
        "notes": [
            _note_payload(note, note_id)
            for note_id, note in enumerate(chart.notes)],
        "analysis": {"stats": chart.stats},
    }


def read_chart(path: str | Path) -> dict:
    """Load and validate a v3 chart for renderers and external consumers."""
    source = Path(path)
    try:
        chart = json.loads(source.read_text(encoding="utf-8"),
                           parse_constant=_reject_constant)
        if chart["format"] != CHART_FORMAT or chart["version"] != CHART_VERSION:
            raise ValueError(
                f"unsupported chart {chart.get('format')!r} "
                f"version {chart.get('version')!r}")
        meta, timing, notes = chart["meta"], chart["timing"], chart["notes"]
        if meta["difficulty"] not in {"NM", "HD", "MX", "SC"}:
            raise ValueError("invalid difficulty")
        tracks = meta["tracks"]
        lane_count = len(tracks)
        resolution = int(timing["ticks_per_quarter"])
        barlines = timing["barlines"]
        tempos = timing["tempo_segments"]
        meters = timing["meter_events"]
        if lane_count <= 0 or resolution <= 0:
            raise ValueError("invalid lanes or tick resolution")
        if [track["index"] for track in tracks] != list(range(lane_count)):
            raise ValueError("invalid track indexes")
        if any(track["role"] not in {"normal", "overlay"}
               for track in tracks):
            raise ValueError("invalid track role")
        if meta["normal_lane_count"] != sum(
                track["role"] == "normal" for track in tracks):
            raise ValueError("normal lane count mismatch")
        if len(barlines) < 2 or any(
                left >= right for left, right
                in zip(barlines, barlines[1:])):
            raise ValueError("barlines must be strictly increasing")
        if not tempos or not meters:
            raise ValueError("tempo and meter timelines must not be empty")
        if meters[0]["start_tick"] != barlines[0]:
            raise ValueError("first meter must start at the first barline")
        for segment in tempos:
            if segment["end_tick"] <= segment["start_tick"]:
                raise ValueError("invalid tempo segment bounds")
            interpolation = segment["interpolation"]
            bpms = ([segment["bpm"]] if interpolation == "step" else
                    [segment["bpm_start"], segment["bpm_end"]]
                    if interpolation == "linear_time" else [])
            if not bpms or any(not math.isfinite(float(bpm)) or bpm <= 0
                               for bpm in bpms):
                raise ValueError("invalid tempo segment")
        for meter in meters:
            if meter["numerator"] <= 0 or meter["denominator"] <= 0:
                raise ValueError("invalid meter event")
        for note in notes:
            if not 0 <= note["lane"] < lane_count:
                raise ValueError("note lane out of range")
            if (note["end_tick"] is not None
                    and note["end_tick"] <= note["start_tick"]):
                raise ValueError("invalid longnote end")
            if not isinstance(note["off_grid"], bool):
                raise ValueError("off_grid must be boolean")
        numeric = [meta["duration_ms"], timing["tick_zero_ms"]]
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("non-finite chart metadata")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid chart {source}: {exc}") from exc
    return chart


def output_dir(song_name: str, difficulty: str, *,
               root: str | Path = "out") -> Path:
    return Path(root) / song_name / difficulty


def write_raw(raw: RawChart, *, root: str | Path = "out") -> Path:
    path = output_dir(raw.song_name, raw.difficulty, root=root) / f"{raw.song_name}_raw.json"
    _dump(serialize_raw(raw), path)
    return path


def write_chart(chart: Chart, *, root: str | Path = "out") -> Path:
    path = output_dir(chart.song_name, chart.difficulty, root=root) / f"{chart.song_name}_chart.json"
    _dump(serialize_chart(chart), path)
    return path
