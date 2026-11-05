"""One reading of a sensor, from the hardware or the simulator alike.

Code written against a Frame runs unchanged on a real board (opentac.device)
and on the simulator (opentac.sim): both return this type.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Frame:
    """`taxels[k]` is taxel k in manifest `pixels` order.

    A taxel not read this frame (xampling reads a subset) is NaN, and `mask`
    is False there. `t` is seconds on the host clock. `imu` holds whatever the
    board reported alongside the frame (accelerometer, orientation), or is empty.
    `node` is the bus node the frame came from and `t_us` the bus controller's
    clock in microseconds, shared by all nodes; both None for a board on its
    own USB port and for the simulator.

    `gain` is set for a frame read in HDR (`Sensor.hdr()`): the gain each taxel
    was read at, relative to the highest (1 down to 1/148), and `taxels` are
    then counts above the zero at the highest gain. None otherwise.
    """

    taxels: np.ndarray
    t: float
    mask: np.ndarray | None = None
    imu: dict = field(default_factory=dict)
    units: str = "counts"
    node: int | None = None
    t_us: int | None = None
    gain: np.ndarray | None = None

    def __post_init__(self):
        self.taxels = np.asarray(self.taxels, dtype=float)
        if self.mask is None:
            self.mask = ~np.isnan(self.taxels)

    def image(self, manifest) -> np.ndarray:
        """The frame as a 2-D array laid out like the sensor (row 0 at the top),
        NaN where there is no taxel or it was not read."""
        rows, cols = manifest.grid_index()
        out = np.full((rows.max() + 1, cols.max() + 1), np.nan)
        out[rows, cols] = self.taxels
        return out
