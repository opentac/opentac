"""High dynamic range (HDR) on the 60-pin board: each taxel read at the highest gain
that does not saturate it.

The firmware scans every taxel at GAIN_SW, then, in the same frame, re-reads at the
next gain in HDR_GAINS only the taxels whose reading reached HDR_CLIP, and so on down
the passes. A re-read value carries its pass in bits 12-15 (value = raw + 4096 x pass),
so a frame with nothing re-read is identical to one read without HDR. In Xampling
the passes re-read only the taxels the frame sampled; values from pass 0 are counts
above the board's own zero, as without HDR, and a re-read value is the raw reading
at its gain.

`HdrTable` turns those values into one unit, counts above the zero at the highest
gain, from the zero measured at every gain with nothing touching (`Sensor.hdr()`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: GAIN_SW of each pass, highest gain first: 10 kOhm, 3197, 991, 263 and 67.5 Ohm of
#: feedback, gains 1, 1/3, 1/10, 1/38 and 1/148. The switch's other settings sit
#: within 1-21 % of these, so a re-read there would saturate again.
LADDER = (0, 1, 2, 4, 8)
CLIP = 3800             # the firmware's default HDR_CLIP
PASS_SHIFT = 12
RAW_MASK = 0x0FFF
ZERO_FRAMES = 30        # frames averaged into the zero at each gain
ZERO_SKIP = 3           # frames dropped after each gain change while zeroing


@dataclass(eq=False)
class HdrTable:
    """What converts an HDR frame: the gain and the zero of each pass.

    `gain_sw[p]` is the GAIN_SW of pass p and `gain[p]` its gain relative to pass 0.
    `zero[p]` and `noise[p]` are the mean and standard deviation of each taxel's raw
    reading at that gain with nothing touching. `clip` is HDR_CLIP.
    """

    gain_sw: tuple
    gain: np.ndarray
    zero: np.ndarray
    noise: np.ndarray
    clip: int = CLIP

    @property
    def code(self) -> int:
        """HDR_GAINS for these passes: the GAIN_SW of each re-read pass, one per nibble,
        lowest nibble first."""
        return sum(sw << (4 * i) for i, sw in enumerate(self.gain_sw[1:]))

    def passes(self, values):
        """(pass, 12-bit reading) of each value as the board sent it; a taxel not read
        (NaN) gives pass 0, reading 0."""
        v = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0).astype(np.int64)
        return np.minimum(v >> PASS_SHIFT, len(self.gain_sw) - 1), v & RAW_MASK

    def decode(self, values, board_zero: bool = False):
        """(counts above the zero at the highest gain, the gain each taxel was read at),
        NaN where a taxel was not read. `board_zero`: an Xampling frame, whose pass-0
        values are already counts above the board's zero at the highest gain."""
        read = ~np.isnan(np.asarray(values, dtype=float))
        p, raw = self.passes(values)
        k = np.arange(raw.shape[-1])
        counts = (raw - self.zero[p, k]) / self.gain[p]
        if board_zero:
            counts = np.where(p == 0, raw, counts)
        return np.where(read, counts, np.nan), np.where(read, self.gain[p], np.nan)

    def saturated(self, values, at: int | None = None):
        """Taxels at or above `at` (default HDR_CLIP) raw counts at the lowest gain,
        which no further pass can re-read."""
        p, raw = self.passes(values)
        return (p == len(self.gain_sw) - 1) & (raw >= (self.clip if at is None else at))

    def labels(self) -> list[str]:
        """Each pass's gain as text: '1', '1/3', ... '1/148'."""
        return ["1" if g >= 0.999 else f"1/{1 / g:.0f}" for g in self.gain]
