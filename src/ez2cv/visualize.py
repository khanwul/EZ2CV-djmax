"""Render `out/<song>/<song>_chart.json` as a piano-roll style image.

Layout
- One lane per configured key, drawn bottom-up (early ticks at the bottom).
- Columns hold `MEASURES_PER_COL` musical measures each, laid left-to-right.
- BPM segment boundaries draw a horizontal marker with the BPM change.
- Pickup notes (negative ticks) and tail notes past the last detected barline
  are accommodated by extending the measure grid with virtual barlines using
  the global time signature.

CLI:
    python -m ez2cv.visualize [out/<song>/<song>_chart.json]
    # default: out/song/song_chart.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ───────────────────────────── styling knobs ──────────────────────────────

LANE_FILL = {
    "white": (235, 235, 235),
    "cyan": (70, 215, 240)
}
LANE_EDGE = {k: tuple(min(255, c + 30) for c in v) for k, v in LANE_FILL.items()}
LANE_BG = (28, 28, 32)
PAGE_BG = (16, 16, 20)
BARLINE_COLOR = (130, 130, 145)
BEAT_COLOR = (60, 60, 72)
SUBBEAT_COLOR = (38, 38, 46)   # 1/16 grid (quarter-of-a-beat) — dimmer than beats
TEXT = (215, 215, 220)
DIM_TEXT = (150, 150, 160)
BPM_MARK = (255, 195, 80)
BPM_RAMP = (255, 140, 220)

LANE_W = 26
LANE_GAP = 1
PX_PER_TICK = 0.36
MEASURES_PER_COL = 4
COL_PAD_LEFT = 38          # left padding for measure numbers
COL_PAD_RIGHT = 110        # right padding for BPM marker text
COL_GAP = 18
TOP_PAD = 70
BOTTOM_PAD = 30
SIDE_PAD = 24
TAP_H = 7                  # tap-note bar thickness (px)

# ──────────────────────────────── helpers ─────────────────────────────────


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _ticks_per_measure(time_sig: dict, tick_res: int) -> int:
    num, denom = time_sig["global"]
    return num * tick_res * 4 // denom


def _extend_barlines(barlines: list[int], tpm: int, min_tick: int, max_tick: int) -> list[int]:
    """Pad the barline grid so it covers every note (including pickup + tail)."""
    bl = list(barlines)
    while bl[0] - tpm >= min_tick - tpm + 1:  # prepend until we cover min_tick
        bl.insert(0, bl[0] - tpm)
    if bl[0] > min_tick:
        bl.insert(0, bl[0] - tpm)
    while bl[-1] < max_tick:
        bl.append(bl[-1] + tpm)
    return bl


# ─────────────────────────────── renderer ─────────────────────────────────


def render(chart_path: Path, out_path: Path) -> None:
    chart = json.loads(chart_path.read_text())
    meta = chart["meta"]
    lane_colors = meta.get("lane_colors")
    if not lane_colors:
        raise ValueError("chart meta has no lane_colors")
    n_lanes = len(lane_colors)
    tick_res = meta["tick_resolution"]
    tpm = _ticks_per_measure(chart["time_signature"], tick_res)

    notes = chart["notes"]
    segments = chart["bpm_segments"]
    barlines = chart["barlines_tick"]

    # Tick range covering every note + every barline
    note_ticks = [n["start_tick"] for n in notes] + [
        (n["end_tick"] or n["start_tick"]) for n in notes
    ]
    min_tick = min([barlines[0]] + note_ticks) if notes else barlines[0]
    max_tick = max([barlines[-1]] + note_ticks) if notes else barlines[-1]

    full_bars = _extend_barlines(barlines, tpm, min_tick, max_tick)

    # Build columns from consecutive groups of MEASURES_PER_COL barlines
    columns: list[tuple[int, int, int]] = []  # (start_tick, end_tick, start_measure_idx)
    for i in range(0, len(full_bars) - 1, MEASURES_PER_COL):
        end = min(i + MEASURES_PER_COL, len(full_bars) - 1)
        columns.append((full_bars[i], full_bars[end], i))

    # Geometry
    col_w = COL_PAD_LEFT + n_lanes * LANE_W + (n_lanes - 1) * LANE_GAP + COL_PAD_RIGHT
    col_inner_h = int(max(end - start for start, end, _ in columns) * PX_PER_TICK)
    img_h = TOP_PAD + col_inner_h + BOTTOM_PAD
    img_w = SIDE_PAD * 2 + len(columns) * col_w + (len(columns) - 1) * COL_GAP

    img = Image.new("RGB", (img_w, img_h), PAGE_BG)
    draw = ImageDraw.Draw(img)
    f_title = _load_font(15)
    f_meta = _load_font(11)
    f_lbl = _load_font(10)
    f_bpm = _load_font(11)

    # Header
    song = meta.get("song", "?")
    km = meta.get("key_mode", "?")
    draw.text((SIDE_PAD, 14), f"{song} — {km}", fill=TEXT, font=f_title)
    counts = chart.get("stats", {}).get("counts", {})
    cnt_txt = (
        f"notes={len(notes)}  taps={counts.get('tap','?')}  "
        f"longs={counts.get('longnote','?')}  "
        f"measures={len(barlines)-1}  segments={len(segments)}  "
        f"{MEASURES_PER_COL} measures / column"
    )
    draw.text((SIDE_PAD, 38), cnt_txt, fill=DIM_TEXT, font=f_meta)

    # Per-column drawing helpers
    def lane_x(col_x: int, lane: int) -> tuple[int, int]:
        x0 = col_x + COL_PAD_LEFT + lane * (LANE_W + LANE_GAP)
        return x0, x0 + LANE_W

    def tick_y(col_start: int, tick: int) -> int:
        # Bottom-up: greater tick → smaller y
        col_bot = TOP_PAD + col_inner_h
        return col_bot - int((tick - col_start) * PX_PER_TICK)

    for ci, (cs, ce, m_start) in enumerate(columns):
        col_x = SIDE_PAD + ci * (col_w + COL_GAP)
        col_inner_real = int((ce - cs) * PX_PER_TICK)
        col_bot = TOP_PAD + col_inner_h
        col_top = col_bot - col_inner_real
        is_last_col = ci == len(columns) - 1

        lane_l = lane_x(col_x, 0)[0]
        lane_r = lane_x(col_x, n_lanes - 1)[1]

        # lane background
        for ln in range(n_lanes):
            x0, x1 = lane_x(col_x, ln)
            draw.rectangle([x0, col_top, x1, col_bot], fill=LANE_BG)

        # grid — step in 1/16-notes (tick_res/4) so each beat is split into
        # quarters and 16th-note placement is readable at a glance.
        # Three tiers: barline > beat (quarter) > sub-beat (1/16).
        bars_set = set(full_bars)
        sub = max(1, tick_res // 4)
        t = ((cs // sub) + (1 if cs % sub else 0)) * sub
        while t <= ce:
            y = tick_y(cs, t)
            if t in bars_set:
                color = BARLINE_COLOR
            elif t % tick_res == 0:
                color = BEAT_COLOR
            else:
                color = SUBBEAT_COLOR
            draw.line([lane_l, y, lane_r, y], fill=color, width=1)
            t += sub

        # measure numbers (left of lanes)
        for mi in range(m_start, min(m_start + MEASURES_PER_COL + 1, len(full_bars))):
            bt = full_bars[mi]
            if not (cs <= bt <= ce):
                continue
            y = tick_y(cs, bt)
            # Number this measure if it's the START of a measure inside the column.
            if mi < len(full_bars) - 1:
                # Display 1-indexed measure number; pickup (pre-zero) shows as 0/-1/...
                # Align to the original chart's measure 1 being barlines[0].
                zero_idx = full_bars.index(barlines[0])
                label = str(mi - zero_idx + 1)
                draw.text((col_x + 4, y - 6), label, fill=DIM_TEXT, font=f_lbl)

        # BPM segment markers — anywhere a segment starts inside [cs, ce]
        for seg in segments:
            st = seg["start_tick"]
            if not (cs <= st <= ce):
                continue
            y = tick_y(cs, st)
            b0, b1 = seg["bpm_start"], seg["bpm_end"]
            is_ramp = abs(b1 - b0) >= 0.1
            line_color = BPM_RAMP if is_ramp else BPM_MARK
            draw.line([lane_l, y, lane_r, y], fill=line_color, width=2)
            # left arrow tick
            draw.polygon(
                [(lane_l - 6, y), (lane_l - 1, y - 4), (lane_l - 1, y + 4)],
                fill=line_color,
            )
            txt = f"♩ {b0:.1f}→{b1:.1f}" if is_ramp else f"♩ {b0:.1f}"
            draw.text((lane_r + 6, y - 6), txt, fill=line_color, font=f_bpm)

        # Notes — iterate everything overlapping the column.
        # Columns are half-open [cs, ce): a note (or longnote head) sitting
        # exactly on the upper barline belongs to the NEXT column's bottom, not
        # this column's top — otherwise it renders glued to the top edge,
        # duplicating the same note shown at the next column's base. The last
        # column has no successor, so it keeps the closing barline inclusive.
        for n in notes:
            st = n["start_tick"]
            et = n.get("end_tick")
            end_eff = et if et is not None else st
            beyond_top = st > ce if is_last_col else st >= ce
            if end_eff < cs or beyond_top:
                continue
            color_name = lane_colors[n["lane"]]
            fill = LANE_FILL.get(color_name, (200, 200, 200))
            edge = LANE_EDGE.get(color_name, (255, 255, 255))
            x0, x1 = lane_x(col_x, n["lane"])
            if et is None:
                y = tick_y(cs, st)
                draw.rectangle([x0, y - TAP_H + 1, x1, y], fill=fill, outline=edge)
            else:
                y_start = tick_y(cs, st)
                y_end = tick_y(cs, et)
                # body, clipped to the column so a longnote spanning the column
                # boundary never spills past the top/bottom edge
                body_fill = tuple(int(c * 0.55) for c in fill)
                body_top = max(col_top, y_end)
                body_bot = min(col_bot, y_start)
                draw.rectangle([x0, body_top, x1, body_bot], fill=body_fill, outline=edge)
                # head / tail caps only when their tick lands inside this column
                if cs <= st < ce or (is_last_col and st == ce):
                    draw.rectangle([x0, y_start - TAP_H + 1, x1, y_start], fill=fill, outline=edge)
                if cs <= et < ce or (is_last_col and et == ce):
                    draw.rectangle([x0, y_end, x1, y_end + TAP_H - 1], fill=fill, outline=edge)

        # column outline + footer label
        draw.rectangle([lane_l - 1, col_top, lane_r + 1, col_bot], outline=(70, 70, 82))
        m1 = m_start - full_bars.index(barlines[0]) + 1
        m2 = m1 + (min(m_start + MEASURES_PER_COL, len(full_bars) - 1) - m_start) - 1
        foot = f"m{m1}–{m2}"
        draw.text((col_x + COL_PAD_LEFT, col_bot + 6), foot, fill=DIM_TEXT, font=f_lbl)

    img.save(out_path)
    print(f"saved {out_path}  ({img_w}×{img_h})")


# ─────────────────────────────────── CLI ──────────────────────────────────

if __name__ == "__main__":
    # Usage: python -m ez2cv.visualize [<song>_chart.json]
    if len(sys.argv) >= 2:
        chart_path = Path(sys.argv[1])
    else:
        chart_path = Path("out/song/song_chart.json")
    if not chart_path.exists():
        raise SystemExit(f"chart not found: {chart_path}")

    # <song>_chart.json → <song>_chart_visual.png; fall back to parent dir name
    stem = chart_path.stem
    song = stem[:-len("_chart")] if stem.endswith("_chart") else chart_path.parent.name
    out_path = chart_path.with_name(f"{song}_chart_visual.png")
    render(chart_path, out_path)
