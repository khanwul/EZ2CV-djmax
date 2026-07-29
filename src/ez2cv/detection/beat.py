"""
EZ2CV — detection / beat : POW LED beat detection
===============================================================================
The EZ2ON playfield carries a "POW" indicator LED that flashes on every beat.
Captured, its brightness traces a clean SAWTOOTH: a one-frame spike UP on the
beat, then a roughly linear decay until the next beat. This module turns that
signal into a stream of BeatEvents — the musical beat grid.

Why a separate beat signal at all
---------------------------------
Note scroll speed is subject to SV (per-section visual speed gimmicks), so the
note stream alone cannot distinguish a real tempo change from an SV trick. The
POW LED flashes on the actual musical beat regardless of SV, so its interval is
the ground-truth tempo reference. chart conversion uses these beats — together with the
user-supplied BPM range — to build the tick grid and to tell a BPM change apart
from an SV change.

Detection
---------
* signal    s[f] = mean of the beat ROI's (green) channel for frame f.
* onset     a beat is a sharp RISE: d[f] = s[f] - s[f-1] crossing a threshold.
* spike     a real flash is a one-frame SPIKE — it rises, then DECAYS. When a
            stage loads, the whole screen (POW LED included) fades in, giving
            many frames of small positive d[f]; before the first real flash the
            adaptive threshold has nothing to learn from and falls back to the
            absolute floor, so a bare rise-test mistakes that fade-in ramp for
            ~7 beats. The fix enforces the sawtooth SHAPE: a candidate rise is
            CONFIRMED once the signal falls back BELOW the onset level, and
            REJECTED once it climbs ABOVE it (a sustained rise is a fade-in, not
            a flash). A flat frame — the capture stutter repeats a frame every
            ~8 frames, which can land right after a flash and give d[f]=0 —
            just defers the decision. Confirmation therefore lags emission by
            1..confirm_window frames.
* threshold ADAPTIVE — a fraction of the running flash amplitude — so it copes
            with different LED brightnesses; ``cal.beat_diff_threshold`` is the
            absolute floor (a noise gate) used before the first flash is seen.
* refractory a short window after each onset guards against a multi-frame rise
            being counted twice.

The one-frame rise localises each beat to within a single frame; every beat
carries the SAME sub-frame bias, so inter-beat INTERVALS — i.e. the tempo — are
exact even though the absolute phase is offset by up to half a frame.

Output: BeatEvent objects, ms-based. Tempo/tick interpretation is chart conversion.
"""

from __future__ import annotations

from dataclasses import dataclass

from ez2cv.config import RunConfig
from ez2cv.video import PreprocessedFrame


# =============================================================================
# Data structure
# =============================================================================

@dataclass
class BeatEvent:
    """One POW LED flash — a musical beat."""
    frame_index: int
    ms: float
    strength: float            # the onset diff magnitude (flash brightness jump)


# =============================================================================
# BeatDetector
# =============================================================================

class BeatDetector:
    """Streaming POW LED onset detector.

    Feed it one PreprocessedFrame per frame via :meth:`step`; collected beats
    accumulate in :attr:`beats`. Call :meth:`finish` once at end of stream.

    A rise is held as a CANDIDATE until a later frame proves the sawtooth shape
    (decay), so :meth:`step` emits each beat a few frames late. The returned
    event still carries the true onset frame/ms — only the moment of return
    shifts; the data is unaffected.
    """

    def __init__(self, cal: RunConfig, *,
                 refractory_frames: int = 4,
                 onset_fraction: float = 0.45,
                 amp_smoothing: float = 0.3,
                 confirm_window: int = 3):
        """
        refractory_frames : ignore onsets this many frames after a detected beat
                            (absorbs a 2-3 frame rise; safely below any beat gap).
        onset_fraction    : adaptive threshold = this fraction of the running
                            flash-amplitude estimate.
        amp_smoothing     : EMA weight for updating the flash-amplitude estimate.
        confirm_window    : a candidate may stay flat (stutter-repeated frames)
                            this many frames before a decay must be seen; past
                            it the candidate is dropped as a non-spike.
        """
        self.cal = cal
        self.refractory = refractory_frames
        self.onset_fraction = onset_fraction
        self.amp_smoothing = amp_smoothing
        self.confirm_window = confirm_window
        self.floor = cal.beat_diff_threshold        # absolute noise gate

        self._prev_signal: float | None = None
        self._flash_amp: float | None = None        # learned from real onsets
        self._last_onset = -(10 ** 9)
        # a rise awaiting proof of the sawtooth shape: (frame, ms, diff, signal)
        self._pending: tuple[int, float, float, float] | None = None
        self.beats: list[BeatEvent] = []
        # per-frame POW LED mean, kept for visualization (avoids re-meaning
        # pf.beat_roi in the orchestrator)
        self.signal: list[float] = []

    # ------------------------------------------------------------------ #
    def step(self, pf: PreprocessedFrame) -> BeatEvent | None:
        """Advance one frame; return a BeatEvent when an earlier rise is
        confirmed as a real flash (i.e. the sawtooth decay has been observed).
        """
        signal = float(pf.beat_roi.mean())
        self.signal.append(signal)

        if self._prev_signal is None:               # first frame: no diff yet
            self._prev_signal = signal
            return None

        diff = signal - self._prev_signal
        self._prev_signal = signal

        emitted: BeatEvent | None = None

        # --- 1) resolve a pending candidate -------------------------------
        # Real flash = a spike: a 1-3 frame rise, then decay. Fade-in ramp = a
        # sustained climb. So: decay below the tracked peak -> beat; still
        # rising/flat past the confirm window -> not a spike, drop it; a brief
        # climb inside the window -> track the new peak and keep waiting.
        if self._pending is not None:
            p_frame, p_ms, p_strength, p_signal = self._pending
            waited = pf.frame_index - p_frame
            if signal < p_signal:                   # decayed -> real beat
                # Report the rise EDGE — the last dark frame before the LED
                # jumped — instead of the first bright frame. Empirically
                # (verified against barline crossings) this halves the
                # systematic phase lag from ~1.8 to ~0.8 frames. All windowing
                # state above stays at the detection frame so timing semantics
                # (confirm_window, refractory) are unchanged.
                onset_frame = p_frame - 1
                onset_ms = onset_frame / self.cal.fps * 1000.0
                emitted = BeatEvent(onset_frame, onset_ms, p_strength)
                self.beats.append(emitted)
                self._last_onset = p_frame
                self._update_amp(p_strength)
                self._pending = None
            elif waited >= self.confirm_window:      # rising/flat too long -> not a spike
                self._pending = None
            elif signal > p_signal:                 # still climbing INSIDE the window:
                # a real flash can rise over 2-3 frames before it peaks, so
                # track the peak instead of dropping it as a fade-in. A genuine
                # fade-in keeps climbing and is dropped once `waited` expires.
                self._pending = (p_frame, p_ms, max(p_strength, diff), signal)
            # else: flat and still inside the window -> keep the candidate

        # --- 2) is THIS frame a new candidate onset? ----------------------
        if self._flash_amp is None:
            threshold = self.floor
        else:
            threshold = max(self.floor, self.onset_fraction * self._flash_amp)

        if (self._pending is None and emitted is None
                and diff > threshold
                and pf.frame_index - self._last_onset > self.refractory):
            self._pending = (pf.frame_index, pf.timestamp_ms, diff, signal)

        return emitted

    # ------------------------------------------------------------------ #
    def finish(self) -> BeatEvent | None:
        """Flush a still-pending candidate at end of stream.

        The final candidate may have no following frames to prove its decay.
        Emit it only if the adaptive gate is established AND its strength
        clears it — i.e. it is unmistakably a flash, not a trailing ramp frame.
        """
        if self._pending is None:
            return None
        p_frame, p_ms, p_strength, _ = self._pending
        self._pending = None
        if (self._flash_amp is not None
                and p_strength >= self.onset_fraction * self._flash_amp):
            onset_frame = p_frame - 1
            onset_ms = onset_frame / self.cal.fps * 1000.0
            ev = BeatEvent(onset_frame, onset_ms, p_strength)
            self.beats.append(ev)
            self._last_onset = p_frame
            return ev
        return None

    # ------------------------------------------------------------------ #
    def _update_amp(self, strength: float) -> None:
        """EMA the flash-amplitude estimate toward a confirmed onset's magnitude."""
        if self._flash_amp is None:
            self._flash_amp = strength
        else:
            a = self.amp_smoothing
            self._flash_amp = (1 - a) * self._flash_amp + a * strength
