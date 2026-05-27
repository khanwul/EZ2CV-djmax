"""
EZ2CV — Layer 4 / time_sig : barlines + BPM → TimeSignature (+ variants)
===============================================================================
A barline is a measure boundary; converted into ticks via ``TickClock``, each
inter-barline gap is one measure long. Dividing by ``tick_resolution`` gives
the measure length in BEATS — and that beat count is the natural unit for time
signature.

Why beat counts (not ticks)
---------------------------
The previous implementation scored candidates by ``median(|L − L_cand| / L_cand)``
in ticks. On songs with strong BPM variation the per-measure tick length
drifts (because the BPM curve is never perfectly fit), so every candidate
collects similar residuals and the wrong one can win. The first attempt on
JUSTITIA picked 6/4 with 16 variants — almost certainly wrong; the song is
4/4 with rapid BPM swings.

Beats per measure is robust to that drift: if a 4/4 measure's BPM is fit 4 %
high, the measure has 4 beats × 1.04 = 4.16 beats per measure — still rounds
to 4. The integer mode of ``round(measure_length_ticks / R)`` over all measures
is the global time signature numerator.

4/4 prior
---------
EZ2ON songs are overwhelmingly 4/4. To resist tie-breaking against 4/4 on
short songs or songs where one segment misfits, the vote for ``(4, 4)`` gets
an additive bonus before the argmax. A real 3/4 song still wins, because the
mode would be 3-beat-per-measure on most of its measures and the prior bonus
is constant.

Variant noise filter
--------------------
A measure flagged as a variant must:
* deviate from the global beats-per-measure by an integer (rounds to a
  different candidate);
* have its raw (float) beats-per-measure within ``±0.15`` of that integer
  (so a measure at "4.5 beats" — implying ~12 % BPM error somewhere — is
  rejected as BPM noise, not promoted to a fake 4.5/4 measure).

Low-confidence fallback
-----------------------
If the winning candidate (after voting + prior) has under
``TIME_SIG_MIN_CONFIDENCE`` of all measures as raw support, the barline data
is too noisy for a meaningful pick — JUSTITIA's measure-line stream has
``std / median = 0.77`` for its inter-barline intervals, with measures
ranging from 2 to 21 beats. Reporting 19 "variants" from such data is worse
than honestly defaulting. We fall back to ``4/4`` with NO variants and let
downstream sanity checks (off-grid ratio, combo count) flag any actual
problems.
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from layer3.measureline import BarlineEvent
from layer4.tick_clock import TickClock


DEFAULT_CANDIDATES = [(3, 4), (4, 4), (5, 4), (6, 4)]   # compound (6/8 etc.)
                                                        # is musically equivalent
                                                        # in beat count and is
                                                        # mapped through the
                                                        # same vote
PRIOR_44_BONUS = 4              # additive vote bonus for the 4/4 prior
VARIANT_BEAT_TOLERANCE = 0.15   # |raw beats − rounded| max for variant accept
TIME_SIG_MIN_CONFIDENCE = 0.5   # winner's raw vote share floor; below → fallback


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class TimeSignature:
    numerator: int
    denominator: int

    def ticks_per_measure(self, tick_resolution: int = 192) -> int:
        return self.numerator * tick_resolution * 4 // self.denominator

    def beats_per_measure(self) -> int:
        """In quarter-note beats, the unit used internally for voting."""
        return self.numerator * 4 // self.denominator

    def __str__(self) -> str:
        return f"{self.numerator}/{self.denominator}"


@dataclass
class TimeSigVariant:
    """A run of consecutive measures with a non-global time signature."""
    start_measure: int             # inclusive (0-based; measure 0 = barlines[0])
    end_measure: int               # inclusive
    time_sig: TimeSignature


# =============================================================================
# Helpers
# =============================================================================

def _candidates_by_beat(candidates) -> dict[int, TimeSignature]:
    """Map ``beats_per_measure → TimeSignature``.

    Multiple candidates can share a beat count (3/4 vs 6/8); the SIMPLEST
    (smallest denominator) wins the slot. Compound time is musically
    distinct but indistinguishable from POW LED flashes alone, so we collapse
    them to the simple form.
    """
    out: dict[int, TimeSignature] = {}
    for n, d in candidates:
        beats = n * 4 // d
        cand = TimeSignature(n, d)
        prev = out.get(beats)
        if prev is None or d < prev.denominator:
            out[beats] = cand
    return out


# =============================================================================
# Estimation
# =============================================================================

def estimate_time_signature(barlines: list[BarlineEvent],
                            clock: TickClock,
                            *, candidates=None,
                            variant_beat_tolerance: float = VARIANT_BEAT_TOLERANCE,
                            prior_44_bonus: int = PRIOR_44_BONUS,
                            ) -> tuple[TimeSignature, list[TimeSigVariant]]:
    """Pick a global time signature + list of variant runs."""
    if candidates is None:
        candidates = DEFAULT_CANDIDATES
    if len(barlines) < 2:
        return TimeSignature(4, 4), []

    R = clock.tick_resolution
    bl_ticks = np.array([clock.ms_to_tick(b.ms) for b in barlines])
    measure_lengths_ticks = np.diff(bl_ticks)

    # beats per measure: raw (real) + rounded (integer)
    beats_raw = measure_lengths_ticks / R
    beats_int = np.round(beats_raw).astype(int)

    # --- vote per candidate (collapse 6/8 → 3/4 via beats key) ------------
    by_beat = _candidates_by_beat(candidates)
    votes = Counter()
    for b_int in beats_int:
        ts = by_beat.get(int(b_int))
        if ts is not None:
            votes[ts] += 1

    # 4/4 prior — only when 4/4 is in the candidate set
    if TimeSignature(4, 4) in by_beat.values():
        votes[TimeSignature(4, 4)] += prior_44_bonus

    if not votes:
        return TimeSignature(4, 4), []
    global_ts = max(votes, key=votes.get)

    # --- confidence check: was the winner backed by enough RAW measures? --
    raw_votes = votes[global_ts]
    if global_ts == TimeSignature(4, 4):
        raw_votes -= prior_44_bonus
    total_measures = len(beats_int)
    if total_measures == 0 or raw_votes / total_measures < TIME_SIG_MIN_CONFIDENCE:
        # barline data too noisy for a reliable pick — default to 4/4 and
        # publish no variants. Off-grid ratio + combo count remain as the
        # downstream sanity gates.
        return TimeSignature(4, 4), []

    global_beats = global_ts.beats_per_measure()

    # --- variants: per-measure deviation that matches another candidate ---
    raw_variants: list[tuple[int, TimeSignature]] = []
    for i, b_int in enumerate(beats_int):
        b_int = int(b_int)
        if b_int == global_beats:
            continue
        if abs(beats_raw[i] - b_int) > variant_beat_tolerance:
            continue                       # BPM-noise, not a real variant
        ts = by_beat.get(b_int)
        if ts is not None and ts != global_ts:
            raw_variants.append((i, ts))

    # --- collapse consecutive same-signature variants into ranges ---------
    variants: list[TimeSigVariant] = []
    for m, ts in raw_variants:
        if (variants
                and variants[-1].time_sig == ts
                and variants[-1].end_measure + 1 == m):
            variants[-1].end_measure = m
        else:
            variants.append(TimeSigVariant(m, m, ts))

    return global_ts, variants


# =============================================================================
# Barline-tick computation (variant-aware)
# =============================================================================

def barline_ticks(barlines: list[BarlineEvent],
                  clock: TickClock,
                  global_ts: TimeSignature,
                  variants: list[TimeSigVariant]) -> list[int]:
    """Snap each barline's tick to a measure-grid that honours variants.

    The first barline always sits at tick 0 (that is how ``measure_zero_ms``
    is defined). Successive ticks accumulate the measure length implied by
    either the global signature or any matching variant.
    """
    if not barlines:
        return []
    R = clock.tick_resolution
    out = [0]
    cur = 0
    for i in range(1, len(barlines)):
        ts = global_ts
        for v in variants:
            if v.start_measure <= i - 1 <= v.end_measure:
                ts = v.time_sig
                break
        cur += ts.ticks_per_measure(R)
        out.append(cur)
    return out


# =============================================================================
# CLI: python time_sig.py [config/song.toml]
# =============================================================================

if __name__ == "__main__":
    from layer3 import Layer3Pipeline
    from layer4.bpm_estimator import build_tick_clock

    cfg = sys.argv[1] if len(sys.argv) > 1 else "config/Dream Walker.toml"
    l3 = Layer3Pipeline.from_config(cfg).run(progress=False)
    if not l3.barlines:
        print("no barlines detected"); raise SystemExit(0)

    measure_zero = l3.barlines[0].ms
    clock = build_tick_clock(l3.beats, measure_zero_ms=measure_zero,
                             min_bpm=l3.cal.min_bpm,
                             max_bpm=l3.cal.max_bpm,
                             active_window_ms=(l3.barlines[0].ms,
                                               l3.barlines[-1].ms),
                             tick_resolution=l3.cal.tick_resolution)
    ts, variants = estimate_time_signature(l3.barlines, clock)
    print(f"=== time signature: {ts}  "
          f"({len(l3.barlines)} barlines, {len(variants)} variant run(s)) ===")
    for v in variants:
        print(f"  variant: measures {v.start_measure}..{v.end_measure}  → {v.time_sig}")
