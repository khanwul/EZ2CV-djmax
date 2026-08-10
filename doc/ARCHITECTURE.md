# EZ2CV Architecture

EZ2CV has two computational phases separated by a durable JSON checkpoint:
video detection in milliseconds, followed by musical chart inference in ticks.

## Package structure

```text
src/ez2cv/
├── cli.py                 command orchestration
├── config.py              TOML resolution and validation
├── video.py               streaming OpenCV frame source
├── detection/
│   ├── pipeline.py        one-pass detector and RawChart
│   ├── stage1.py          row projection and run detection
│   ├── stage2.py          constrained template matching
│   ├── tracking.py        temporal tracking and long-note pairing
│   ├── beat.py            POW LED beat detection
│   └── barline.py         measure-line detection and tracking
├── chart/
│   ├── pipeline.py        RawChart → Chart
│   ├── timeline.py        joint meter/tempo inference
│   ├── barline.py         arbitrary meter-sequence reconstruction
│   ├── clock.py           anchor-based ms ↔ tick conversion
│   ├── meter.py           time-signature data
│   └── quantize.py        musical-grid snapping
├── io.py                  raw/chart JSON read and atomic write
└── visualize.py           final chart renderer
```

The package names describe responsibility. There are no numbered layers or
single-implementation interfaces.

## End-to-end flow

```mermaid
flowchart LR
    cfg["song.toml"] --> load["load_config()"]
    skin["skin.toml"] --> load
    profile["profile.toml"] --> load
    templates["template PNGs"] --> load
    load --> run["RunConfig"]

    video[["gameplay video"]] --> detect["DetectionPipeline"]
    run --> detect
    detect --> raw[("RawChart — milliseconds")]
    raw --> writeRaw["write_raw()"]
    writeRaw --> rawJson[/"*_raw.json"/]

    raw --> build["build_chart()"]
    rawJson --> read["read_raw()"]
    read --> build
    build --> chart[("Chart — ticks")]
    chart --> writeChart["write_chart()"]
    writeChart --> chartJson[/"*_chart.json"/]
```

`write_raw()` runs immediately after detection and before chart inference. If
BPM, meter, or quantization fails, the expensive video result is already safe
and can be retried with `ez2cv *_raw.json --from-raw`.

## Configuration boundary

`load_config()` merges three TOML tiers:

| Input | Responsibility |
| --- | --- |
| `config/<song>.toml` | video, difficulty, FPS, note speed, tick resolution, BPM range |
| `config/skins/<skin>/skin.toml` | colors, templates, channels, thresholds |
| `config/profiles/<resolution>/<mode>.toml` | pixel geometry and measured speed |

The resulting `RunConfig` contains decoded OpenCV templates and therefore stays
inside video/detection code. It validates key counts, channels, geometry, BPM,
FPS, and the profile's calibrated note speed.

The 1920×1080 profiles cover 4K, 5K, 6K, and 8K. Only 5K has been verified on
real recordings; the other modes are centered bootstrap geometry and must be
recalibrated when the panel differs.

## Detection phase

`video.Preprocessor` sequentially decodes the video and yields one
`PreprocessedFrame` at a time. Memory use therefore stays independent of video
length. Each frame fans out to three paths:

```mermaid
flowchart LR
    frame["PreprocessedFrame"] --> s1["projection"]
    s1 --> s2["template match"]
    s2 --> tracker["note tracking"]
    tracker --> notes["RawNote[]"]

    frame --> beat["beat detector"]
    beat --> beats["BeatEvent[]"]

    s1 --> bar["barline detector/tracker"]
    bar --> bars["BarlineEvent[]"]

    notes --> raw["RawChart"]
    beats --> raw
    bars --> raw
```

Important invariants:

- Detection thresholds use unnormalized 0–255 channel values.
- Stage 1 is recall-oriented; Stage 2 provides template precision.
- The barline detector reuses Stage 1 projections and uses two-row energy to
  tolerate sub-pixel line motion.
- Beat events report the rise edge of the POW LED flash.
- Tracking interpolates bracketed judgment-line crossings. Tap near misses use
  a regression over the latest local trajectory, with global scroll speed as a
  gate and fallback. Longnote endpoints keep their calibrated paired-duration
  projection. Each note records the resulting timing uncertainty.
- Detection stops at milliseconds and has no BPM or grid dependency.

## Raw checkpoint

`RawChart` contains only JSON-safe primitives and event dataclasses:

- song/skin/key-mode metadata and ordered normal/overlay tracks;
- FPS, frame count, video resolution, and source path;
- tick resolution and BPM bounds needed for chart reconstruction;
- raw notes, beats, barlines, confidence, extrapolation diagnostics, and
  `timing_sigma_ms` (legacy checkpoints default it to zero).

It does not retain `RunConfig`, OpenCV images, template arrays, or filesystem
configuration state. `serialize_raw()` and `read_raw()` form a schema-versioned
round trip.

## Chart phase

`build_chart(raw)` performs no video or TOML I/O:

1. Index detected barlines on the beat stream.
2. Use dynamic programming to drop false lines, infer hidden boundaries, and
   choose each measure's numerator from 1 through 7.
3. Fit stable tempo runs across all their barlines. Keep intermediate anchors
   only when the raw notes would otherwise leave the supported rhythm grid.
4. Add a beat anchor for a clear tempo step inside a single measure.
5. Build one piecewise-linear `TickClock` from those selected anchors.
6. Convert note times to ticks. Each measure starts with the
   `{1/4, 1/8, 1/12, 1/16, 1/24, 1/32}` vocabulary and opts into
   `1/48`, `1/64`, `1/96`, or `1/192` only when multiple distinct onsets
   support the finer grid. Near-simultaneous chord lanes count as one onset.
   `fine_grid_ratio` records notes rescued by that expansion. A remaining miss
   within twice its measured `timing_sigma_ms` is reported separately as
   `timing_uncertain_ratio`; `timing_outlier_ratio` is reserved for misses not
   explained by crossing uncertainty. `base_grid_outlier_ratio` preserves the
   sum of all three for before/after comparison, and
   `measure_grid_max_denominator` records the selected level per measure.
7. Snap long-note lengths relative to their snapped heads.

The output `Chart` owns only chart metadata, BPM segments, time signatures,
notes, barline ticks, and statistics. Neither `RawChart` nor `Chart` depends on
detection configuration objects. `write_chart()` serializes it as
`ez2cv.chart` 3.2: `meta.game` identifies the game, each track declares its
input type, timing lives under one `timing` object, fixed and ramped BPM
segments declare their interpolation, meter changes are tick-addressed events,
and derived statistics live under `analysis`.

## Verification

`tests/` contains standard-library `unittest` checks for configuration
consistency, clock round trips, grid snapping, arbitrary meters, missing
barline reconstruction, mid-measure tempo steps, schema rejection, and raw
checkpoint round trips followed by chart regeneration.
