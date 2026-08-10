"""Time-signature data."""

from __future__ import annotations

from dataclasses import dataclass

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
