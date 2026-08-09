# EZ2CV-djmax

EZ2CV-djmax is an experimental fork of [EZ2CV](https://github.com/khanwul/EZ2CV) for extracting chart data from **DJMAX RESPECT V** gameplay videos. It uses an OpenCV-based pipeline and writes detected notes and inferred timing data as JSON.

> **Status:** the scoped 4B/5B/6B/8B pipeline runs end to end. All current samples match the rendered reference-marker counts, with 1.5-6.0% off-grid notes after timing calibration. Local note templates are still required.

This repository does not include gameplay videos, charts, note skins, or other game assets. Obtain any required material yourself and use it only where permitted.

## Reference recording setup

The current analysis assumes the following fixed capture setup:

- 1920x1080 at constant 60 fps; variable-frame-rate video is unsupported
- In-game note speed 6.0, calibrated independently from EZ2ON's speed values
- Centered gear with a fully opaque, black playfield
- A note skin with little or no brightness animation
- Fixed normal, side-track, and L/R colors
- No player input, key beams, judgment effects, or FEVER effects
- Key guide, rewind alert, and side-track alert disabled
- No pause, rewind, retry, or other timeline discontinuity
- Gameplay-only footage with enough pre-roll and post-roll to contain every note edge

Different resolutions, frame rates, note speeds, gear layouts, and skins require separate calibration.

## Development setup

Requirements:

- Python 3.14
- [uv](https://github.com/astral-sh/uv)

Install dependencies:

```bash
uv sync
```

Run the existing tests:

```bash
uv run python -m unittest discover -s tests -v
```

The inherited CLI remains:

```bash
uv run ez2cv "config/<song>.toml"
```

The configured skin's local template images must exist under `config/skins/djmax/<mode>/`.

## Output

The inherited pipeline writes two files under `out/<song>/`:

- `<song>_raw.json`: reloadable millisecond-domain detection checkpoint
- `<song>_chart.json`: quantized chart with BPM and time-signature data

EZ2ON-specific predicted combo statistics were removed because DJMAX scoring behavior has not been verified.

## Repository relationship

- Fork: <https://github.com/khanwul/EZ2CV-djmax>
- Upstream: <https://github.com/khanwul/EZ2CV>

## License

This project is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE) and is intended for personal, non-commercial use only.

This project is not affiliated with or endorsed by NEOWIZ or the DJMAX rights holders. All referenced game names and assets belong to their respective owners. Users are responsible for complying with applicable licenses, terms, and laws.
