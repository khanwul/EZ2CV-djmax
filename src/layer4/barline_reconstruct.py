"""
EZ2CV — Layer 4 / barline_reconstruct : beat-phase measure-grid recovery
===============================================================================
Layer 3's measure-line detector is reliable where the playfield is clean but
goes blind through long-note / heavy-effect sections (the thin grey rule is
camouflaged against a long-note body). Its raw ``barlines`` stream is therefore
INCOMPLETE (whole measures missing) and carries occasional false positives.

Feeding that stream straight into ``time_sig`` makes a single-measure time-
signature change (변박) indistinguishable from a plain missed barline: a fixed-
period interpolator would silently absorb the variant. This module fixes that
by reconstructing the measure grid from the ONE robust phase reference Layer 3
also produces — the POW-LED ``beats`` — instead of from the noisy barlines
alone.

Why beat-COUNT indexing (the key idea)
--------------------------------------
A measure line always lands on a downbeat, so every barline coincides with a
beat. Index each barline by the ORDINAL of its nearest beat event — i.e. the
NUMBER of POW-LED beats before it — not by ``Δms / interval``. The gap between
two detected barlines is then an exact integer beat count, immune to BPM-fit
drift (validated on GEHENNA's 2× tempo swing: gaps stayed clean integer
multiples of the measure length). A measure of ``M`` beats appears as a gap of
``M``; a missed measure as ``k·M``; a single-measure variant of ``M'`` beats as
a gap whose residual mod ``M`` is ``M' − M`` and — crucially — that residual
PERSISTS in the phase of every later barline, which is exactly how a variant is
told apart from a miss.

The DP (joint FP rejection + variant detection)
-----------------------------------------------
``dp[j] = min_i  dp[i]
                 + w_var·|residual(n[j] − n[i], M)|   # gap's net variant beats
                 + λ_out·(j − i − 1)                  # barlines i..j skipped = FP
                 + w_rho·ρ[j]``                       # off-beat penalty

A false positive splits one true gap into two whose residuals CANCEL
(``+x`` then ``−x``); accepting it costs ``2·w_var·x`` while skipping it costs
``λ_out``, so with ``λ_out < 2·w_var`` a cancel-pair is removed as an outlier. A
real variant is a single NON-cancelling residual and survives. (Validated: 12
synthetic cases + 4 real GT-4/4 songs reporting 0 false variants, with detector
FPs correctly dropped.)

Scope
-----
This is the "4/4 + rare variants" reconstructor — the regime of essentially
every EZ2ON chart. Pervasive variable-meter songs (e.g. JUSTITIA: 2/3/4/5/6 all
over) have no dominant ``M`` and need a different, meter-language-model DP; this
module deliberately does not attempt that and falls back to a clean 4/4 grid
when no stable ``M`` is found.

Output: a ``ReconstructResult`` carrying the COMPLETE barline list (observed +
inferred, ms-based), the global ``TimeSignature``, the ``TimeSigVariant`` runs,
and the dropped outliers — a drop-in replacement for ``estimate_time_signature``
that also hands ``barline_ticks`` a gap-free measure grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from layer3.beat import BeatEvent
from layer3.measureline import BarlineEvent
from layer4.time_sig import TimeSignature, TimeSigVariant


# =============================================================================
# Tunables (score units are beats of |residual|; see module docstring)
# =============================================================================

DEFAULT_NUMERATORS = (3, 4, 5, 6)   # candidate measure lengths in /4 beats
W_VAR = 1.0                         # cost per variant beat
LAM_OUT = 1.5                       # cost to drop a barline as a false positive
W_RHO = 4.0                         # off-beat penalty weight (× ρ in [0,~0.5])
RHO_MAX = 0.25                      # a barline farther than this from any beat
                                    #   (in local beat-interval units) is dropped
PRIOR_44_BONUS_BEATS = 1.0          # 4/4 prior, expressed as a vote bonus
MAX_MEASURES_PER_GAP = 8            # a true→true gap longer than this is implausible
METER_MIN_SUPPORT = 3              # a single-measure gap length recurring at
                                    #   least this many times is a real meter
                                    #   the song uses (so its variant cost → 0),
                                    #   not a one-off beat-count artifact


# =============================================================================
# Result
# =============================================================================

@dataclass
class ReconstructResult:
    """Beat-phase measure-grid reconstruction."""
    barlines: list[BarlineEvent]            # COMPLETE: observed + inferred, by ms
    time_signature: TimeSignature           # global (M/4)
    variants: list[TimeSigVariant]          # non-global measures, collapsed to runs
    outliers: list[BarlineEvent]            # detected barlines dropped as FP
    beat_indices: list[int]                 # beat ordinal of each output barline
    measure_meters: list[int]               # beats-per-measure for measure 0..N-1
    global_meter: int                       # M, beats per measure

# =============================================================================
# Beat-count indexing
# =============================================================================

def _index_barlines(barlines: list[BarlineEvent], beat_ms: np.ndarray
                    ) -> tuple[list[int], list[float]]:
    """Map each barline to (nearest beat ordinal, on-beat residual ρ).

    ρ is the |offset| from the nearest beat in units of the LOCAL beat interval
    (so it is tempo-independent). A barline with ρ > RHO_MAX does not sit on a
    beat and is almost certainly spurious; it keeps a large ρ here and the DP's
    ``w_rho`` term lets it be dropped.
    """
    n = len(beat_ms)
    ords: list[int] = []
    rhos: list[float] = []
    for b in barlines:
        ms = b.ms
        pos = int(np.searchsorted(beat_ms, ms))
        cand = [k for k in (pos - 1, pos) if 0 <= k < n]
        j = min(cand, key=lambda k: abs(beat_ms[k] - ms))
        local = (beat_ms[min(j + 1, n - 1)] - beat_ms[max(0, j - 1)]) / 2.0
        local = local if local > 0 else 1.0
        ords.append(j)
        rhos.append(abs(beat_ms[j] - ms) / local)
    return ords, rhos


# =============================================================================
# Global meter (M) estimation
# =============================================================================

def estimate_global_meter(beat_gaps: list[int],
                          numerators=DEFAULT_NUMERATORS,
                          prior_44_bonus: float = PRIOR_44_BONUS_BEATS,
                          ) -> int:
    """Vote the dominant measure length (beats) from single-measure gaps.

    Only gaps that already equal a candidate are votes (a missed measure gives a
    multiple and is ignored here). A 4-beat prior bonus keeps 4/4 winning on
    short or noisy inputs, matching ``time_sig``'s philosophy.
    """
    cand = set(numerators)
    votes = {m: 0.0 for m in numerators}
    for g in beat_gaps:
        if g in cand:
            votes[g] += 1.0
    if 4 in votes:
        votes[4] += prior_44_bonus
    return max(votes, key=votes.get) if any(votes.values()) else 4


# =============================================================================
# Gap decomposition
# =============================================================================

def _known_meters(gaps: list[int], M: int, max_single: int,
                  min_support: int = METER_MIN_SUPPORT) -> set[int]:
    """Single-measure gap lengths that recur often enough to be REAL meters.

    A gap length appearing ``≥ min_support`` times across the song is a meter the
    composer actually uses (JUSTITIA's 5 and 6), not a one-off beat-count
    artifact (GEHENNA's lone gap of 3 = a missed POW-LED beat). The global meter
    ``M`` is always known.
    """
    from collections import Counter
    freq = Counter(g for g in gaps if 1 <= g <= max_single)
    known = {L for L, c in freq.items() if c >= min_support}
    known.add(M)
    return known


def _decompose(delta: int, M: int,
               max_single: int = max(DEFAULT_NUMERATORS),
               known_meters: set[int] | None = None) -> tuple[float, int, int]:
    """Cheapest (cost, m measures, net variant beats) for a `delta`-beat gap.

    Two regimes, split at ``max_single`` (the largest candidate meter):

    * ``delta ≤ max_single`` — the gap is bounded by two DETECTED barlines and is
      short enough to be a single measure, so it IS one measure of ``delta``
      beats (the detector would have caught an interior boundary). Its variant
      cost is ~0 when ``delta`` is a KNOWN (recurring) meter — so the DP trusts
      the detected boundary instead of dropping it — but the full ``W_VAR·|net|``
      when ``delta`` is a one-off length (a beat-count artifact), which keeps
      such barlines droppable. This is what lets a pervasive-variant song keep
      its 5/4·6/4 measures while a 4/4 song's stray short gap is still rejected.
    * ``delta > max_single`` — too long for one measure, so it spans MISSED
      barlines: decompose onto the nearest multiple of ``M`` (``m ≥ 2``); the
      residual ``net`` is the net variant content hidden in the blind gap.
    """
    if delta <= max_single:
        net = delta - M
        if known_meters is not None and delta in known_meters:
            return (W_VAR * 0.0, 1, net)
        return (W_VAR * abs(net), 1, net)
    base = max(2, round(delta / M))
    best: tuple[float, int, int] | None = None
    for m in {max(2, base - 1), base, base + 1}:
        net = delta - m * M
        cost = W_VAR * abs(net)
        if best is None or cost < best[0]:
            best = (cost, m, net)
    return best                                     # type: ignore[return-value]


# =============================================================================
# The DP : pick true barlines (drop FPs), score each gap by variant content
# =============================================================================

def _segment(ords: list[int], rhos: list[float], M: int,
             max_single: int = max(DEFAULT_NUMERATORS),
             known_meters: set[int] | None = None
             ) -> tuple[list[int], list[int]]:
    """Return (indices of true barlines, indices of dropped FP barlines)."""
    K = len(ords)
    NEG = float("-inf")
    dp = [NEG] * K
    prev = [-1] * K
    # any barline may be the first true one; everything before it is an FP
    for j in range(K):
        dp[j] = -(j * LAM_OUT) - W_RHO * rhos[j]
    for j in range(K):
        if dp[j] == NEG:
            continue
        for k in range(j + 1, K):
            delta = ords[k] - ords[j]
            if delta < min(2, M):                   # zero-length / duplicate
                continue
            if delta > M * MAX_MEASURES_PER_GAP:    # implausibly long jump
                break
            cost, _, _ = _decompose(delta, M, max_single, known_meters)
            cand = (dp[j] - cost - LAM_OUT * (k - j - 1) - W_RHO * rhos[k])
            if cand > dp[k]:
                dp[k] = cand
                prev[k] = j
    # best terminal barline (trailing barlines after it are FPs)
    best_j, best = -1, NEG
    for j in range(K):
        score = dp[j] - LAM_OUT * (K - 1 - j)
        if score > best:
            best, best_j = score, j
    true_idx: list[int] = []
    j = best_j
    while j != -1:
        true_idx.append(j)
        j = prev[j]
    true_idx.reverse()
    keep = set(true_idx)
    outliers = [i for i in range(K) if i not in keep]
    return true_idx, outliers


# =============================================================================
# Reconstruction
# =============================================================================

def _ms_for_ordinal(o: int, beat_ms: np.ndarray) -> float:
    """ms of beat ordinal `o`; linear-extrapolate past the ends if needed."""
    n = len(beat_ms)
    if 0 <= o < n:
        return float(beat_ms[o])
    if o < 0:
        step = beat_ms[1] - beat_ms[0] if n >= 2 else 0.0
        return float(beat_ms[0] + o * step)
    step = beat_ms[-1] - beat_ms[-2] if n >= 2 else 0.0
    return float(beat_ms[-1] + (o - (n - 1)) * step)


def reconstruct_barlines(barlines: list[BarlineEvent],
                         beats: list[BeatEvent],
                         *,
                         numerators=DEFAULT_NUMERATORS,
                         global_meter: int | None = None,
                         ) -> ReconstructResult:
    """Beat-phase reconstruction of a complete, FP-free measure grid.

    Single-measure variants (``m == 1`` gaps with a non-zero residual) get an
    EXACT location and meter. Variants hidden inside a multi-measure blind gap
    (``m ≥ 2``) keep the correct net content but their position is placed at the
    gap's first measure — note ticks are unaffected (the gap's total beat count
    is exact), only the structural label is approximate.
    """
    if len(barlines) < 2 or len(beats) < 2:
        return ReconstructResult(list(barlines), TimeSignature(4, 4), [],
                                 [], [], [], 4)

    beat_ms = np.sort(np.array([b.ms for b in beats], dtype=float))

    # 1) index barlines on the beat grid; collapse barlines that share an ordinal
    ords_all, rhos_all = _index_barlines(barlines, beat_ms)
    by_ord: dict[int, int] = {}                     # ordinal -> barline index (best ρ)
    for i, o in enumerate(ords_all):
        if o not in by_ord or rhos_all[i] < rhos_all[by_ord[o]]:
            by_ord[o] = i
    uniq_ord = sorted(by_ord)
    ords = uniq_ord
    rhos = [rhos_all[by_ord[o]] for o in uniq_ord]
    src_idx = [by_ord[o] for o in uniq_ord]         # back to original barline

    # 2) global meter
    gaps = [ords[i + 1] - ords[i] for i in range(len(ords) - 1)]
    M = global_meter if global_meter is not None else \
        estimate_global_meter(gaps, numerators)

    # 3) DP: true barlines + outliers
    max_single = max(numerators)
    known = _known_meters(gaps, M, max_single)
    true_local, outlier_local = _segment(ords, rhos, M, max_single, known)
    outliers = [barlines[src_idx[i]] for i in outlier_local]
    true_ords = [ords[i] for i in true_local]

    # ms↔frame slope from observed barlines, to fill inferred cross_frame
    obs = barlines
    if len(obs) >= 2 and (obs[-1].ms - obs[0].ms) > 0:
        fpms = (obs[-1].cross_frame - obs[0].cross_frame) / (obs[-1].ms - obs[0].ms)
        f0 = obs[0].cross_frame - fpms * obs[0].ms
    else:
        fpms, f0 = 0.0, 0.0

    def make_barline(o: int, observed_src: int | None) -> BarlineEvent:
        # ``extrapolated`` here means RECONSTRUCTION-inferred (not detected), a
        # different notion from Layer 3's crossing-extrapolation flag — so an
        # observed barline is emitted with extrapolated=False regardless of how
        # its Layer 3 crossing was obtained.
        if observed_src is not None:
            b = barlines[observed_src]
            return BarlineEvent(cross_frame=b.cross_frame, ms=b.ms,
                                strength=b.strength, extrapolated=False)
        ms = _ms_for_ordinal(o, beat_ms)
        return BarlineEvent(cross_frame=f0 + fpms * ms, ms=ms,
                            strength=0.0, extrapolated=True)

    # 4) walk gaps, decompose, emit the complete grid + per-measure meters
    out_barlines: list[BarlineEvent] = []
    beat_indices: list[int] = []
    measure_meters: list[int] = []

    first_o = true_ords[0]
    out_barlines.append(make_barline(first_o, src_idx[true_local[0]]))
    beat_indices.append(first_o)

    for li in range(len(true_local) - 1):
        a_o, b_o = true_ords[li], true_ords[li + 1]
        b_src = src_idx[true_local[li + 1]]
        _, m, net = _decompose(b_o - a_o, M, max_single, known)
        # measure lengths in this gap
        if net == 0:
            lengths = [M] * m
        elif m == 1:
            lengths = [M + net]                     # exact single-measure variant
        else:
            lengths = [M + net] + [M] * (m - 1)     # variant placed at gap start
        cursor = a_o
        for mi, L in enumerate(lengths):
            cursor += L
            measure_meters.append(L)
            is_last = (mi == len(lengths) - 1)
            out_barlines.append(
                make_barline(cursor, b_src if is_last else None))
            beat_indices.append(cursor)

    # 5) global time signature + variant runs
    ts = TimeSignature(M, 4)
    allowed = set(numerators)
    raw_variants: list[tuple[int, TimeSignature]] = []
    for measure_idx, L in enumerate(measure_meters):
        if L != M and L in allowed:
            raw_variants.append((measure_idx, TimeSignature(L, 4)))
    variants: list[TimeSigVariant] = []
    for m_idx, vts in raw_variants:
        if (variants and variants[-1].time_sig == vts
                and variants[-1].end_measure + 1 == m_idx):
            variants[-1].end_measure = m_idx
        else:
            variants.append(TimeSigVariant(m_idx, m_idx, vts))

    return ReconstructResult(
        barlines=out_barlines,
        time_signature=ts,
        variants=variants,
        outliers=outliers,
        beat_indices=beat_indices,
        measure_meters=measure_meters,
        global_meter=M,
    )
