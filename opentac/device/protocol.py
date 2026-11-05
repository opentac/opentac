"""The readout board's text protocol: parsing what it sends.

Every line is a record whose first field says what it is:

    F,<frame>,<gen>,<mode>,<dt_us>,<dropped>,<N values>,<13 IMU fields>    raster frame
    X,<frame>,<gen>,<mode>,<dt_us>,<dropped>,<n>,<mask hex>,<n values>,<13 IMU fields>
    V,<seq>,<ax>,<ay>,<az>          one accelerometer sample (mg)
    M,<gen>,<mode>,<tac>,<vib>,<ori>,<active>,<shear>    status
    R,...                           reply to a command; R,ERR,<reason> is a refusal
    # ...                           a notice for people

The 13 IMU fields are accel (mg, 3), gyro (mdps, 3), quaternion (w, x, y, z)
and Euler angles (roll, pitch, yaw, degrees). The whole command set is at
https://opentac.github.io/reference/protocol/.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from opentac.frame import Frame

IMU_FIELDS = 13

#: Friendly names for the two enumerated parameters.
TAC_MODES = {"raster": 0, "xamp_fixed": 1, "xamp_grow": 2}
LED_MODES = {"off": 0, "manual": 1, "identity": 2, "heartbeat": 3, "sensor": 4, "binary": 5}
ENUM_NAMES = {"TAC_MODE": TAC_MODES, "LED_MODE": LED_MODES}

#: Board refusals (R,ERR,<reason>) that need a sentence to act on.
REFUSALS = {
    "NOBASE": "xampling needs a baseline first: call zero() (or send Z) with nothing touching the skin",
    "GROUPS": "this skin cannot be xampled: a shear group is too spread out to fill",
    "NOCFG": "the board holds no skin: run `opentac setup <manifest>` first",
    "NOTIMPL": "this firmware does not implement that mode",
    "NOSHEAR": "that mode needs a skin with shear bumps",
    "GAIN": "the gain switch did not take the setting (no answer at I2C 0x4C)",
    "NONODE": "the bus controller has no such node (`opentac info` lists them)",
    "OFFLINE": "that node stopped answering the bus controller; check its cable and power",
    "BUSY": "the bus controller is still sending that node earlier commands",
}

G4_DEV_IDS = {0x468: "STM32G431/G441", 0x469: "STM32G47x/G48x", 0x479: "STM32G491/G4A1"}
FLASH_STAGES = {1: "erase", 2: "program", 3: "verify"}


class DeviceError(Exception):
    """The board refused a command, or did not answer."""


def explain(reason: str) -> str:
    """A refusal reason as a sentence; FLASH,... is decoded field by field."""
    if reason in REFUSALS:
        return f"{reason}: {REFUSALS[reason]}"
    f = reason.split(",")
    if f[0] != "FLASH" or len(f) != 7:
        return reason
    stage, hal, sr, addr, dev, kb = (int(f[1]), int(f[2], 16), int(f[3], 16),
                                     int(f[4], 16), int(f[5], 16), int(f[6]))
    chip = G4_DEV_IDS.get(dev, f"unknown G4 device id {dev:#x}")
    return (f"flash {FLASH_STAGES.get(stage, stage)} failed at {addr:#010x}: HAL error {hal:#x}, "
            f"FLASH->SR {sr:#x}; chip {chip}, {kb} KB flash")


@dataclass
class BoardFrame(Frame):
    """A Frame from a readout board, with the board's own bookkeeping.

    `index` counts frames on the board, `generation` counts accepted settings
    changes since boot (it going backwards means the board rebooted), `mode` is
    the TAC_MODE that produced it (0 raster, 1 xamp_fixed, 2 xamp_grow), `dt_us`
    is the time since the previous frame, `dropped` counts IMU samples the
    board had to drop. `node` and `t_us` (from Frame) are set for a bus node.

    In HDR (`Sensor.hdr()`), `hdr_raw` holds the values as the board sent them
    (the pass in bits 12-15) and `hdr` the opentac.device.hdr.HdrTable that
    converted them; both None otherwise.
    """

    index: int = 0
    generation: int = 0
    mode: int = 0
    dt_us: int = 0
    dropped: int = 0
    hdr_raw: np.ndarray | None = None
    hdr: object | None = None

    @property
    def sparse(self) -> bool:
        """True for a xampling frame: only `mask` taxels were read."""
        return self.mode != 0


def _imu(fields) -> dict:
    return {"accel_mg": [int(v) for v in fields[0:3]],
            "gyro_mdps": [int(v) for v in fields[3:6]],
            "quat_wxyz": [float(v) for v in fields[6:10]],
            "euler_deg": [float(v) for v in fields[10:13]]}


def parse_frame(line: str, active: int, t: float | None = None) -> BoardFrame | None:
    """An F or X line as a BoardFrame; None for any other line.

    `active` is the skin's taxel count (the board's R,CFG). Raster values are
    raw 12-bit counts. Xampling values are counts above the board's own zero
    (`Z`), NaN for taxels not read this frame. Raises ValueError for a frame
    line whose fields do not add up, rather than put values on the wrong taxels.
    """
    sparse = line.startswith("X,")
    if not (sparse or line.startswith("F,")):
        return None
    f = line.split(",")[1:]
    if len(f) < 6:
        raise ValueError(f"frame record too short: {len(f)} fields")
    head = [int(v) for v in f[:5]]
    if sparse:
        n = int(f[5])
        expected = 7 + n + IMU_FIELDS
        if len(f) != expected:
            raise ValueError(f"X record has {len(f)} fields, expected {expected}")
        mask_bytes = bytes.fromhex(f[6])
        if len(mask_bytes) != (active + 7) // 8:
            raise ValueError(f"X mask is {len(mask_bytes)} bytes, a {active}-taxel skin needs "
                             f"{(active + 7) // 8}")
        bits = np.unpackbits(np.frombuffer(mask_bytes, dtype=np.uint8), bitorder="little")
        if bits[active:].any():
            raise ValueError("X mask marks a taxel past the end of the skin")
        mask = bits[:active].astype(bool)
        if int(mask.sum()) != n:
            raise ValueError(f"X mask marks {int(mask.sum())} taxels, record says {n}")
        taxels = np.full(active, np.nan)
        taxels[mask] = np.asarray(f[7:7 + n], dtype=float)
        imu = f[7 + n:]
        units = "counts above zero"
    else:
        expected = 5 + active + IMU_FIELDS
        if len(f) != expected:
            raise ValueError(f"F record has {len(f)} fields, expected {expected} "
                             f"for a {active}-taxel skin")
        taxels = np.asarray(f[5:5 + active], dtype=float)
        mask = np.ones(active, dtype=bool)
        imu = f[5 + active:]
        units = "counts"
    return BoardFrame(taxels=taxels, t=time.time() if t is None else t, mask=mask,
                      imu=_imu(imu), units=units, index=head[0], generation=head[1],
                      mode=head[2], dt_us=head[3], dropped=head[4])


def parse_caps(line: str) -> dict:
    """R,CAPS,<api>,<steps>,<channels>,<max taxels>,<modes hex>,<led hex>,<params>,<need shear hex>,<board>"""
    f = line.split(",")
    return {"api_version": int(f[2]), "steps": int(f[3]), "channels": int(f[4]),
            "max_taxels": int(f[5]), "tactile_modes": int(f[6], 16), "led_modes": int(f[7], 16),
            "modes_need_shear": int(f[9], 16) if len(f) > 9 else 0,
            # Builds older than the field were all 70-pin readouts.
            "board": f[10] if len(f) > 10 else "READOUT32"}


def parse_param(line: str) -> dict:
    """R,PARAM,<id>,<name>,<lo>,<hi>,<value>"""
    f = line.split(",")
    return {"id": int(f[2]), "name": f[3], "lo": int(f[4]), "hi": int(f[5]), "value": int(f[6])}


def parse_config(line: str) -> dict | None:
    """R,CFG,<version>,<rows>,<cols>,<active>,<pitch x um>,<pitch y um>,<shear>,<id hex>,<crc hex>,<bytes>; R,NOCFG -> None."""
    if line == "R,NOCFG":
        return None
    f = line.split(",")
    return {"version": int(f[2]), "rows": int(f[3]), "cols": int(f[4]), "active": int(f[5]),
            "pitch_x_um": int(f[6]), "pitch_y_um": int(f[7]), "shear": int(f[8]),
            "config_id": int(f[9], 16), "crc": int(f[10], 16), "bytes": int(f[11])}


def parse_zero(line: str) -> dict:
    """R,OK,Z,<frames>,<baseline mean>,<baseline max>,<noise median>,<noise max>,<noisy taxels>"""
    keys = ("frames", "baseline_mean", "baseline_max", "noise_median", "noise_max", "noisy")
    return dict(zip(keys, (int(v) for v in line.split(",")[3:])))


def parse_stats(line: str) -> dict:
    """R,STAT,<frames>,<vib dropped>,<scan us max>,<loop us max>,<generation>"""
    f = [int(v) for v in line.split(",")[2:]]
    return dict(zip(("frames", "vib_dropped", "scan_us_max", "loop_us_max", "generation"), f))
