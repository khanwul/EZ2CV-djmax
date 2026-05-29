# EZ2CV

EZ2CV is a tool that analyzes gameplay videos of the rhythm game EZ2ON REBOOT : R and extracts chart data from them. It runs an OpenCV-based computer vision pipeline to recognize note patterns from the video and saves them as JSON files.

Created for chart visualization and analysis.

## Key Features

- Takes EZ2ON gameplay videos as input and analyzes them using OpenCV.
- Converts the analyzed chart data into a lightweight, easy-to-handle JSON file format.

## Requirements

- Python 3.14
- [uv](https://github.com/astral-sh/uv) for dependency management

## Setup

```bash
uv sync
```

## Usage

Configs live under `config/`. To analyze different songs, pass its TOML path as an argument. Use `config/song.toml` as a template for new songs.

Run the full pipeline (writes `out/<song>/{raw,chart}.json`):

```bash
uv run python src/main.py "config/<song>.toml"
```

## Config layout

| File | Purpose |
| ---- | ------- |
| `config/<song>.toml` | Per-song settings (video path, fps, scroll speed, BPM range, etc.) |
| `config/skins/<skin>/skin.toml` | Per-skin lane colors, templates, detection channels, thresholds |
| `config/profiles/<res>/<key_mode>.toml` | Per-resolution / key-mode playfield geometry |
| `config/skins/<skin>/<key_mode>/*.png` | Note template images |

Only the `5k` key mode at `1920x1080` is fully calibrated at the moment.

## Output

Two JSON files are written under `out/<song>/`:

- `<song>_raw.json` — Layer 3 result (ms-based notes, beats, barlines)
- `<song>_chart.json` — Layer 4 result (tick-based chart with BPM and time signature)

---

**Disclaimer**:
The copyright for the extracted chart data and gameplay videos belongs to NEONOVICE and the original creators. Please use this tool for personal analysis and learning purposes only. All responsibility arising from the use of this tool lies entirely with the user.
