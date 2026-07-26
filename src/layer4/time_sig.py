"""Time-signature data and reconstructed barline tick placement."""

from __future__ import annotations

from dataclasses import dataclass

from layer3.measureline import BarlineEvent
from layer4.tick_clock import TickClock


@dataclass(frozen=True)
class TimeSignature:
    numerator: int
    denominator: int

    def ticks_per_measure(self, tick_resolution: int = 192) -> int:
        return self.numerator * tick_resolution * 4 // self.denominator

    def __str__(self) -> str:
        return f"{self.numerator}/{self.denominator}"


@dataclass
class TimeSigVariant:
    start_measure: int
    end_measure: int
    time_sig: TimeSignature


def barline_ticks(barlines: list[BarlineEvent],
                  clock: TickClock,
                  global_ts: TimeSignature,
                  variants: list[TimeSigVariant]) -> list[int]:
    """Place reconstructed barlines on the variant-aware measure grid."""
    if not barlines:
        return []
    ticks = [0]
    for measure in range(len(barlines) - 1):
        ts = next((v.time_sig for v in variants
                   if v.start_measure <= measure <= v.end_measure), global_ts)
        ticks.append(ticks[-1] + ts.ticks_per_measure(clock.tick_resolution))
    return ticks


if __name__ == "__main__":
    from types import SimpleNamespace
    bars = [BarlineEvent(0.0, 0.0, 0.0)] * 4
    variants = [TimeSigVariant(1, 1, TimeSignature(3, 4))]
    assert barline_ticks(bars, SimpleNamespace(tick_resolution=192),
                         TimeSignature(4, 4), variants) == [0, 768, 1344, 2112]
