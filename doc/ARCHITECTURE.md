# EZ2CV Architecture

A 5-layer computer-vision pipeline that takes an EZ2ON REBOOT : R gameplay video and reconstructs the chart (JSON). Every diagram below renders as Mermaid in GitHub/VSCode; the same diagrams are exported as static SVG under [doc/svg/](svg/).

> To re-export a diagram as SVG after editing: `mmdc -i doc/ARCHITECTURE.md -o doc/svg/diagram.svg -t neutral -b transparent` (requires system `mermaid-cli`).

---

## 1. End-to-end pipeline

Layer 3 decodes the video exactly once and produces ms-based raw data; everything after that is in-memory transformation (Layer 4) and serialization (Layer 5). Layers 1 and 2 are effectively setup. _Static SVG: [1-pipeline-overview.svg](svg/1-pipeline-overview.svg)_

```mermaid
flowchart LR
    %% Inputs
    cfg["config/&lt;song&gt;.toml"]:::input
    vid[["gameplay.mp4"]]:::input

    %% Layers
    subgraph L1["Layer 1 — Calibration (setup)"]
        direction TB
        l1["resolve_calibration()"]
        cal[("Calibration<br/>(flat)")]:::data
        l1 --> cal
    end

    subgraph L2["Layer 2 — Preprocessor (streaming)"]
        direction TB
        l2["Preprocessor<br/>(generator)"]
        pf[("PreprocessedFrame<br/>per frame")]:::data
        l2 -. yields .-> pf
    end

    subgraph L3["Layer 3 — 1-Pass detection (ms-based)"]
        direction TB
        l3["Layer3Pipeline.run()"]
        l3r[("Layer3Result<br/>notes / beats / barlines")]:::data
        l3 --> l3r
    end

    subgraph L4["Layer 4 — 2-Pass (ms → tick)"]
        direction TB
        l4["Layer4Result.from_layer3()"]
        l4r[("Layer4Result<br/>BPMSegments + ChartNotes")]:::data
        l4 --> l4r
    end

    subgraph L5["Layer 5 — Serialization"]
        direction TB
        l5["write_all()"]
        raw[/"out/&lt;song&gt;/raw.json"/]:::output
        chart[/"out/&lt;song&gt;/chart.json"/]:::output
        viz[/"chart_visual.png"/]:::output
        l5 --> raw
        l5 --> chart
        l5 --> viz
    end

    cfg --> l1
    cal --> l2
    vid --> l2
    cal --> l3
    pf -- per-frame stream --> l3
    l3r --> l4
    cal --> l4
    l3r --> l5
    l4r --> l5

    classDef input  fill:#fef3c7,stroke:#92400e,color:#000
    classDef data   fill:#dbeafe,stroke:#1e40af,color:#000
    classDef output fill:#dcfce7,stroke:#166534,color:#000
```

Key design decisions:

- **One decode, three paths.** Layer 3 decodes the video exactly once and extracts notes, beats, and barlines in parallel. Later experiments (different BPM assumptions, different snapping) reuse the raw result.
- **Layer 3 stops at milliseconds.** BPM, ticks, and grid snapping all belong to Layer 4, so the expensive video pass does not depend on any grid assumption.
- **Streaming.** Layer 2 is a generator, so memory usage stays flat regardless of video length.

---

## 2. Config resolution chain (Layer 1)

Three TOML tiers collapse into a single `Calibration` object. No layer below this one reads TOML directly. _Static SVG: [2-config-chain.svg](svg/2-config-chain.svg)_

```mermaid
flowchart TD
    song["config/&lt;song&gt;.toml<br/><i>per-run</i><br/>setup.skin, key_mode, display_resolution"]:::cfg

    skin["config/skins/&lt;skin&gt;/skin.toml<br/><i>skin-level</i><br/>lane colors, templates, channels, thresholds"]:::cfg
    profile["config/profiles/&lt;res&gt;/&lt;key_mode&gt;.toml<br/><i>resolution × mode</i><br/>playfield, lane x, judgment geometry"]:::cfg
    tpl["config/skins/&lt;skin&gt;/&lt;key_mode&gt;/*.png<br/><i>note / lnhead / lntail templates</i>"]:::asset

    resolve(["resolve_calibration()"]):::fn
    cal[/"Calibration<br/>───<br/>• video_path, fps, speed<br/>• min_bpm, max_bpm<br/>• lanes : list[LaneCalibration]<br/>• measure : MeasureLineConfig<br/>• tick_resolution = 192"/]:::out

    song -- "setup.skin"               --> skin
    song -- "setup.key_mode"           --> tpl
    song -- "setup.display_resolution" --> profile

    song    --> resolve
    skin    --> resolve
    profile --> resolve
    tpl     --> resolve
    resolve --> cal

    classDef cfg   fill:#fef3c7,stroke:#92400e,color:#000
    classDef asset fill:#fde2e2,stroke:#991b1b,color:#000
    classDef fn    fill:#e9d5ff,stroke:#6b21a8,color:#000
    classDef out   fill:#dbeafe,stroke:#1e40af,color:#000
```

| File | Change frequency | Purpose |
| --- | --- | --- |
| `config/<song>.toml` | per song | video path, FPS, speed, BPM range, time-signature candidates |
| `skins/<skin>/skin.toml` | when adding a skin | lane colors, template filenames, detection channels, thresholds |
| `profiles/<res>/<key_mode>.toml` | when calibrating a resolution × key mode | pixel-level geometry |

To calibrate a new resolution, add `profiles/<W>x<H>/<keymode>.toml`. Only `5k @ 1920x1080` is fully calibrated today.

---

## 3. Layer 2 → Layer 3 data fan-out

A single `PreprocessedFrame` branches into three independent paths. _Static SVG: [3-layer3-fanout.svg](svg/3-layer3-fanout.svg)_

```mermaid
flowchart LR
    subgraph PF["PreprocessedFrame (per frame, streamed)"]
        direction TB
        det["LaneFrame.detection_roi<br/><i>tight single-channel</i>"]
        match["LaneFrame.matching_roi<br/><i>wider BGR (±roi_x_margin)</i>"]
        beat_roi["beat_roi<br/><i>POW LED region</i>"]
        meas_roi["measure_roi<br/><i>full-playfield-width 1ch</i>"]
    end

    subgraph NOTE["note path"]
        s1["stage1.py<br/>ProjectionDetector<br/>row-mean projection<br/>+ contiguous-run finding"]
        s2["stage2.py<br/>TemplateMatcher<br/>cv2.matchTemplate<br/>{note, lnhead, lntail}"]
        tr["tracking.py<br/>ScrollSpeedEstimator<br/>NoteTracker<br/>LongnoteStateMachine"]
        s1 --> s2 --> tr
    end

    subgraph BEAT["beat path"]
        bd["beat.py<br/>BeatDetector<br/><i>POW LED sawtooth onset</i>"]
    end

    subgraph BAR["bar-line path"]
        md["measureline.py<br/>MeasureLineDetector<br/>+ MeasureLineTracker<br/><i>2-row energy-sum<br/>full-width thin band</i>"]
    end

    det     --> s1
    match   --> s2
    beat_roi --> bd
    meas_roi --> md
    s1 -. "reuses<br/>per-lane<br/>projection" .-> md

    tr --> notes[/"list[RawNote]"/]:::out
    bd --> beats[/"list[BeatEvent]"/]:::out
    md --> bars[/"list[BarlineEvent]"/]:::out

    notes --> L3R[("Layer3Result")]:::data
    beats --> L3R
    bars  --> L3R

    classDef out  fill:#dcfce7,stroke:#166534,color:#000
    classDef data fill:#dbeafe,stroke:#1e40af,color:#000
```

Design invariants:

- **No normalization.** Thresholds are absolute 0–255 values; normalizing would break them.
- **Coordinate convention.** `roi_y_origin + in-ROI y = full-frame y`, and `lane.match_x_origin + in-ROI x = full-frame x`.
- **Barline path free-rides Stage 1.** The measure-line detector reuses the per-lane projection that Stage 1 already computes, so it costs almost zero extra work.
- **Energy-sum lit test (sub-pixel robustness).** A 1px barline scrolling ~33px/frame straddles two pixel rows and splits its energy below any single-row brightness gate, so a raw row-mean test flickers and drops whole measures. The detector instead thresholds the **2-row sliding energy sum** (`proj[y] + proj[y+1]`), which recombines the split energy regardless of straddle phase (4-song recall 94–96%, FP~0). The `max_thickness` gate still rejects long-note bodies, so recovering dim rows does not let thick objects through.
- **Why POW LED.** The LED flicker is a tempo signal that is immune to scroll-speed changes (SV). Barlines give measure phase; LED gives beat phase — both are needed.
- **Beat onset = rise edge, not peak.** `BeatEvent.frame_index` is the **last dark frame** before the LED jumps (i.e. one frame before the detection frame), not the first bright frame. Verified against barline crossings: reporting the peak gives a systematic ~1.8 frame lag; reporting the rise edge halves it to ~0.8 frames. A residual ~0.8-frame lag (std 0.33) remains — this is the game itself rendering the LED flash ~1 frame after the visual barline crossing, and must be absorbed by Layer 4's LED-multiplier anchor rather than re-fought here.

---

## 4. Layer 4 — ms → tick algorithm flow

Layer 4 consumes Layer 3's ms-domain result, decides BPM / time signature / grid, and converts everything to ticks. Module decomposition is intentionally small — `bpm_estimator` and `quantizer` are pure-function modules. _Static SVG: [4-layer4-algorithm.svg](svg/4-layer4-algorithm.svg)_

```mermaid
flowchart TD
    L3R[("Layer3Result<br/>notes·beats·barlines (ms)")]:::data

    subgraph BPM["bpm_estimator.py"]
        a1["1. change-points<br/>(|Δt − Δt_prev| / Δt_prev > ±3%)"]
        a2["2. PELT re-segmentation<br/>(high-variance segments only, ruptures rbf)"]
        a3["3. linear fit per segment<br/>|Δbpm|&lt;0.1 → constant collapse"]
        anc["LED-multiplier anchor<br/>(global, containment)"]
        a1 --> a2 --> a3
        a3 <-->|"trigger ⇒ hard clamp"| anc
    end

    subgraph TS["time_sig.py"]
        t1["fit time-sig candidates to barlines<br/>pick min residual"]
        t2["variant measures = within ±5%<br/>group consecutive runs"]
        t1 --> t2
    end

    subgraph CLOCK["tick_clock.py"]
        clk["TickClock<br/>BPMSegment.tick_at()<br/>closed-form ms↔tick"]
    end

    subgraph Q["quantizer.py"]
        q1["allowed grids<br/>{1/4,1/8,1/12,1/16,1/24,1/32,1/48,1/64}"]
        q2["cost = |Δtick| + α·log₂(denom)<br/>+ context-aware penalty"]
        q3["longnote head : absolute snap<br/>tail : length-snap relative to head"]
        q1 --> q2 --> q3
    end

    L4R[/"Layer4Result<br/>bpm_segments<br/>global_time_sig + variants<br/>notes : list[ChartNote]<br/>barlines_tick · stats"/]:::out

    L3R -- "beats (ms)"     --> BPM
    L3R -- "barlines (ms)"  --> TS
    BPM -- "BPMSegments"    --> CLOCK
    BPM -- "stable segments"--> TS
    L3R -- "notes (ms)"     --> CLOCK
    CLOCK -- "ms→tick"      --> Q
    Q --> L4R
    TS --> L4R
    BPM --> L4R

    sanity{"off-grid > 5% ?"}:::warn
    L4R --> sanity
    sanity -- yes --> retry["retry an octave shift<br/>→ re-fit time signature<br/>→ raise error flag"]:::warn

    classDef data fill:#dbeafe,stroke:#1e40af,color:#000
    classDef out  fill:#dcfce7,stroke:#166534,color:#000
    classDef warn fill:#fde2e2,stroke:#991b1b,color:#000
```

Algorithm highlights:

- **Piecewise-linear BPM.** A gradual BPM change is expressed as a single segment in closed form; constant BPM automatically collapses to slope=0.
- **LED-multiplier anchor.** The POW LED can flash 0.5, 1, or 2 times per beat; the global `2^k` factor is decided up-front so the rest of the pipeline cannot mis-pick the octave.
- **Negative ticks.** Pickup notes that sit within one measure before the first barline are preserved with negative tick values.
- **Context-aware snapping.** If neighbors land on 1/16, candidate 1/24 / 1/48 positions for adjacent notes get a penalty — prevents triplet vs. duple confusion.
- **Determinism.** All stochastic steps (e.g., PELT) use a fixed seed. Same input → same output.

---

## 5. Core data types

The dataclasses that cross layer boundaries. All plain dataclasses — no inheritance, no ORM. _Static SVG: [5-data-model.svg](svg/5-data-model.svg)_

```mermaid
classDiagram
    class Calibration {
        +video_path: Path
        +fps: float
        +speed: float
        +min_bpm: float
        +max_bpm: float
        +key_mode: str
        +tick_resolution: int = 192
        +lanes: list~LaneCalibration~
        +measure: MeasureLineConfig
    }

    class LaneCalibration {
        +color: str
        +detection_channel: str
        +threshold: int
        +match_x_origin: int
        +templates: dict
    }

    class MeasureLineConfig {
        +channel: str
        +lit_energy_threshold: float
        +min_brightness: float
        +max_brightness: float
        +max_thickness: int
        +lane_slack: int
    }

    class PreprocessedFrame {
        +frame_index: int
        +ms: float
        +lanes: list~LaneFrame~
        +beat_roi: ndarray
        +measure_roi: ndarray
    }

    class LaneFrame {
        +detection_roi: ndarray
        +matching_roi: ndarray
        +roi_y_origin: int
    }

    class RawNote {
        +lane: int
        +ms: float
        +type: str    'tap' | 'longnote'
        +end_ms: float?
        +confidence: float
        +extrapolated: bool
    }

    class BeatEvent {
        +frame_index: int
        +ms: float
    }

    class BarlineEvent {
        +cross_frame: int
        +ms: float
    }

    class Layer3Result {
        +cal: Calibration
        +notes: list~RawNote~
        +beats: list~BeatEvent~
        +barlines: list~BarlineEvent~
        +frame_count: int
        +orphan_tails: int
    }

    class BPMSegment {
        +start_tick: int
        +end_tick: int
        +bpm_start: float
        +bpm_end: float
        +tick_at(ms_from_start) float
    }

    class TimeSignature {
        +numerator: int
        +denominator: int
    }

    class TimeSigVariant {
        +start_measure: int
        +end_measure: int
        +time_sig: TimeSignature
    }

    class ChartNote {
        +lane: int
        +start_tick: int
        +end_tick: int?
        +off_grid: bool
    }

    class TickClock {
        +ms_to_tick(ms) float
        +tick_to_ms(tick) float
    }

    class Layer4Result {
        +cal: Calibration
        +bpm_segments: list~BPMSegment~
        +global_time_sig: TimeSignature
        +variant_measures: list~TimeSigVariant~
        +measure_zero_ms: float
        +notes: list~ChartNote~
        +barlines_tick: list~int~
        +stats: dict
    }

    Calibration "1" *-- "many" LaneCalibration
    Calibration "1" *-- "1"    MeasureLineConfig
    PreprocessedFrame "1" *-- "many" LaneFrame
    Layer3Result "1" --> "1"    Calibration
    Layer3Result "1" *-- "many" RawNote
    Layer3Result "1" *-- "many" BeatEvent
    Layer3Result "1" *-- "many" BarlineEvent
    Layer4Result "1" --> "1"    Calibration
    Layer4Result "1" *-- "many" BPMSegment
    Layer4Result "1" --> "1"    TimeSignature
    Layer4Result "1" *-- "many" TimeSigVariant
    Layer4Result "1" *-- "many" ChartNote
    TickClock    "1" *-- "many" BPMSegment
    TimeSigVariant "1" --> "1"  TimeSignature
```

---

## 6. Call tree at a glance

The exact call order driven by `src/main.py`.

```text
main.py:run()
├── layer1.calibration.resolve_calibration(cfg)         → Calibration
├── layer2.preprocessor.Preprocessor(cal)               → streaming generator
├── layer3.Layer3Pipeline(cal).run()                    → Layer3Result
│   └─ for frame in preprocessor:                       (single decode pass)
│        ├── ProjectionDetector       (stage 1)
│        ├── TemplateMatcher          (stage 2)
│        ├── NoteTracker + LongnoteStateMachine
│        ├── BeatDetector             (POW LED)
│        └── MeasureLineDetector + Tracker
├── layer4.Layer4Result.from_layer3(l3)                 → Layer4Result
│   └─ bpm_estimator → time_sig → tick_clock → quantizer
└── layer5.write_all(l3, l4)                            → raw.json, chart.json
    └─ optional: visualize_chart.render(...)            → chart_visual.png
```

Each layer's `__main__` block is a standalone debug CLI (`uv run python src/layerN/*.py "config/*.toml"`). Production runs go through `src/main.py`.
