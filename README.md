# EZ2CV

EZ2CV is a tool that analyzes gameplay videos of the rhythm game EZ2ON REBOOT : R and extracts chart data from them.  
EZ2CV runs an OpenCV-based computer vision pipeline to recognize note patterns from the video and saves them as JSON files.

**This project does not include any data (gameplay video, chart, note skins, etc.) from EZ2ON REBOOT: R. If you wish to run this project, please obtain the data yourself.**

## Key Features

- Takes EZ2ON gameplay videos as input and analyzes them using OpenCV.
- Converts the analyzed chart data into a lightweight, easy-to-handle JSON file format.

## Requirements

- Python 3.14
- [uv](https://github.com/astral-sh/uv) for dependency management
- Currently, only the **5K key mode at 1920×1080** resolution is fully calibrated.

## Setup

```bash
uv sync
```

#### Gameplay Video Setup
While most settings can be adjusted via the config file, there are many instances where you must manually calculate and enter values yourself, and these values have not been verified and cannot be trusted.  
Please configure the settings as instructed, if possible.  

video settings
- resolution: 1920*1080
- fps: 60 (The video must maintain a consistent frame rate. This has a significant impact on accuracy.)

ingame settings
- **No input should be provided during gameplay.** This project has been developed on the assumption that there is no input during gameplay, and this has a significant impact on accuracy. We recommend recording your gameplay using **LIVE CTRL** mode.
- Key Mode: 5K  

- Panel Skin: PG-RESPECT (recommended)  
Other skins can also be applied by modifying the config file(config/profiles), but the beat indicator must be clearly visible. In the case of PG-RESPECT, the beat indicator (POW) is clearly visible below the health bar, so we recommend using this skin.
- Note Skin: EZ2ON (recommended)  
Other skins can also be applied by modifying the config file(config/skins), but we recommend using a stick-shaped note skin with a consistent appearance(no glowing). This greatly affects image matching.
- Note Speed: 8.0 (recommended)  
Although the speed can be adjusted via the config file(config/song.toml), our tests showed that 8.0 achieved the highest recognition rate.
- other
    - judge line: old  
    The settings are based on the height of 'old'. You can change this to the height of 'new' in the config file.
    - judgement tracker: off
    - **Panel Opacity: 100%**
    - panel align: CENTER  
    You can change this by editing the coordinates in the config file to suit your location. The default setting is 'center'.
    - panel bg: none
    - judge height: 700 (max)

#### Note Template Setup

The pipeline requires note template images cropped from your gameplay recording. These are **not included** in this repository and must be prepared manually.

Place the following files under `config/skins/ez2on/5k/`:

| Filename | Description |
| -------- | ----------- |
| `note_cyan.png` | Cyan (active) note body |
| `note_cyan_lnhead.png` | Cyan long-note head |
| `note_cyan_lntail.png` | Cyan long-note tail |
| `note_white.png` | White (inactive) note body |
| `note_white_lnhead.png` | White long-note head |
| `note_white_lntail.png` | White long-note tail |

Crop each template from a clean frame of your gameplay video — the note must be fully visible, unobstructed, and captured at the exact resolution you intend to use (1920×1080).

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
| `config/skins/<skin>/<key_mode>/*.png` | Note template images (not included; see Note Template Setup) |

## Output

Two JSON files are written under `out/<song>/`:

- `<song>_raw.json` — Layer 3 result (ms-based notes, beats, barlines)
- `<song>_chart.json` — Layer 4 result (tick-based chart with BPM and time signature)

## License

This project is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE).  
This tool is intended for **personal, non-commercial use only**.

---

**Disclaimer**:  
This project is not affiliated with, endorsed by, or sponsored by NEONOVICE or any related parties. All copyrights to EZ2ON REBOOT: R game content belong to NEONOVICE and the original creators. This tool is intended for personal, non-commercial use only. All responsibility arising from the use of this tool lies entirely with the user.
