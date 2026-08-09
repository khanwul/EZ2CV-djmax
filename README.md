# EZ2CV-djmax

EZ2CV-djmax is an experimental fork of [EZ2CV](https://github.com/khanwul/EZ2CV) for extracting chart data from **DJMAX RESPECT V** gameplay videos. It uses an OpenCV-based pipeline and writes detected notes and inferred timing data as JSON.

> **Status:** porting is in progress. The repository still contains the inherited EZ2ON detector and calibration profiles, so DJMAX videos are not yet supported end to end.

This repository does not include gameplay videos, charts, note skins, or other game assets. Obtain any required material yourself and use it only where permitted.

## Target scope

- 1920x1080, constant 60 fps recordings
- 4B, 5B, 6B, and 8B modes
- Normal tap and long notes
- Half-playfield side tracks, which use long notes only
- 8B L/R tracks, which use both tap and long notes
- POW-based beat detection and visible measure-line detection
- Raw millisecond-domain and quantized tick-domain JSON output

DJMAX 8B is modeled as six normal lanes plus the overlapping L and R tracks, not as eight equal-width lanes. In 5B, the center note is treated as one visual lane; controller-specific dual-input semantics are outside the current scope.

## Observed playfield structure

The reference recordings use a playfield about 480 pixels wide.

| Mode | Normal lanes | Normal-lane pitch | Overlapping tracks |
| --- | ---: | ---: | --- |
| 4B | 4 | about 120 px | Half-field side track, about 240 px, long-note only |
| 8B | 6 | about 80 px | L and R, about 240 px each, tap and long-note |

Side and L/R notes are rendered behind normal notes. Fixed colors make their masks easier to detect, but geometry is still authoritative because special and normal notes can share the same color.

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

## POW beat verification

The POW indicator was measured across the active portions of two constant-BPM reference recordings. In both, the signal jumps from roughly 25 to 195 in one frame and then decays gradually, matching the inherited beat detector's expected sawtooth waveform.

| Recording | POW interval | Implied tempo | Measure-line interval | Result |
| --- | ---: | ---: | ---: | --- |
| 4B reference | about 17.14 frames | about 210 BPM | 68-69 frames | 4 POW pulses per measure |
| 8B reference | about 21.82 frames | about 165 BPM | 87-88 frames | 4 POW pulses per measure |

The extrapolated measure-line crossing and POW event agree within 0-1 frame. Side and L/R notes did not interrupt the POW signal. These recordings do not contain a mid-song BPM change, so variable-BPM behavior remains unverified.

## Porting work still required

- Replace uniform, non-overlapping lane geometry with explicit normal and overlapping track ranges
- Detect special tracks separately and search normal-note templates independently inside overlapping regions
- Give each track its own allowed note types, note height, and judgment trigger
- Exclude side and L/R tracks from measure-line and scroll-speed consensus
- Store track names and roles in raw/chart metadata
- Recalibrate DJMAX profiles and provide DJMAX-specific templates

The decoder, per-ROI preprocessing, temporal tracker, long-note pairing, POW detector, BPM inference, and tick quantizer are intended to be reused where their existing assumptions still hold.

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

It will not process DJMAX correctly until the porting work and DJMAX calibration listed above are complete.

## Planned output

The inherited pipeline writes two files under `out/<song>/`:

- `<song>_raw.json`: reloadable millisecond-domain detection checkpoint
- `<song>_chart.json`: quantized chart with BPM and time-signature data

EZ2ON-specific predicted combo statistics must not be treated as DJMAX validation until DJMAX scoring behavior is verified.

## Repository relationship

- Fork: <https://github.com/khanwul/EZ2CV-djmax>
- Upstream: <https://github.com/khanwul/EZ2CV>

## License

This project is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE) and is intended for personal, non-commercial use only.

This project is not affiliated with or endorsed by NEOWIZ or the DJMAX rights holders. All referenced game names and assets belong to their respective owners. Users are responsible for complying with applicable licenses, terms, and laws.
