"""A readout board as a Python object.

    from opentac.device import Sensor

    with Sensor() as s:            # finds the board on USB
        s.setup("my_design.zip")   # once per skin: the board stores it, this computer caches it
        frame = s.read()           # an opentac.frame.Frame, taxels in manifest `pixels` order

After `setup` the board remembers its skin across power cycles, and this
computer keeps a copy of the manifest under ~/.opentac/manifests/, named by the
id the board stores. `Sensor()` reads that id and finds the copy, so a script
never needs the manifest path again.

The parameter table is discovered from the board (`?`), never copied here, so a
firmware with more parameters needs no change on this side.
"""
from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import numpy as np

from opentac import manifest as manifest_mod
from opentac.device import config as cfg
from opentac.device import hdr as hdr_mod
from opentac.device.link import BusLink, UsbLink
from opentac.device.protocol import (ENUM_NAMES, DeviceError, parse_caps, parse_config,
                                     parse_frame, parse_param, parse_stats, parse_zero)

CHUNK_BYTES = 64        # config bytes per D line: 128 hex characters

#: Session settings Sensor() applies unless told otherwise. MUX_SETTLE_US 8:
#: untouched taxels read high at the firmware's default of 1 (on one skin with
#: nothing touching it, 62 taxels exceeded 300 counts at 1 us and 19 at 8 us),
#: and the force calibration was made at 8. IMU off: on one board the IMU's FIFO reads spiked the taxels
#: read just after them; pass imu=True when you need orientation.
DEFAULT_MUX_SETTLE_US = 8


def home() -> Path:
    """Where this computer keeps OpenTac state: $OPENTAC_HOME, else ~/.opentac."""
    return Path(os.environ.get("OPENTAC_HOME", "~/.opentac")).expanduser()


def cache_path(config_id: int) -> Path:
    return home() / "manifests" / f"{config_id:08X}.json"


def cache_store(m: manifest_mod.Manifest) -> Path:
    """Keep a manifest under the id a board stores for it. Returns the path."""
    path = cache_path(m.config_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_bytes() != m.source_bytes:
        path.write_bytes(m.source_bytes)
    return path


def cache_lookup(config_id: int) -> manifest_mod.Manifest | None:
    path = cache_path(config_id)
    return manifest_mod.load(path) if path.exists() else None


class Sensor:
    """One readout board, on its own USB port or as node `node` behind a bus controller.

    This is `opentac.Sensor`, the real hardware. The simulator has a function
    of the same name, `opentac.sim.Sensor(manifest)`, which returns a simulated
    sensor with `press()`; both give the same `opentac.frame.Frame`.

    On open it asserts a known session: raster mode with HDR off, MUX_SETTLE_US 8,
    IMU off (each a keyword argument; None or False leaves that setting as the
    board has it), plus any `settings` {name: value}. Session settings live on the
    board until it reboots; the skin it holds persists.
    """

    def __init__(self, port=None, *, node=None, raster=True,
                 mux_settle_us=DEFAULT_MUX_SETTLE_US, imu=False, settings=None, timeout=5.0,
                 _link=None):
        self._owns_link = _link is None
        if _link is not None:
            self.link = _link
        elif node is not None:
            self.link = BusLink(port, timeout)
        else:
            self.link = UsbLink(port, timeout)
        self.node = node
        self.rebooted = False
        self._hdr_table, self._hdr_on, self._gain_before_hdr = None, False, None
        try:
            self.refresh()
            self.stored = self.config()
            self.manifest = cache_lookup(self.stored["config_id"]) if self.stored else None
            if raster:
                self.set("TAC_MODE", "raster")
                if self.params.get("HDR_GAINS", {}).get("value"):
                    self._set("HDR_GAINS", 0)    # left on by a program that did not close
            if mux_settle_us is not None:
                self.set("MUX_SETTLE_US", mux_settle_us)
            if imu is not None:
                self.set_imu(imu)
            for name, value in (settings or {}).items():
                self.set(name, value)
        except Exception:
            self.close()
            raise

    # --- plumbing -------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._leave_hdr()
        if self._owns_link:
            self.link.close()

    def _leave_hdr(self):
        """Turn HDR off if this object turned it on, so other programs get plain frames."""
        if self._hdr_on:
            try:
                self.hdr(False)
            except Exception:              # an unplugged board: nothing to leave on
                self._hdr_on = False

    def _command(self, text, until=None, match=None):
        return self.link.command(text, self.node, until, match)

    def __repr__(self):
        where = f"node {self.node}" if self.node is not None else getattr(self.link, "port", "?")
        skin = f"skin {self.stored['config_id']:08X}" if self.stored else "no skin"
        return f"Sensor({self.board_name} on {where}, {skin})"

    # --- what the board is ----------------------------------------------------

    def refresh(self) -> dict:
        """Read the capability line and the whole parameter table from the board."""
        self.caps, self.params = {}, {}
        for line in self._command("?", until="R,END"):
            kind = line.split(",")[1]
            if kind == "MCAPS":
                raise DeviceError("this port is a bus controller, not a board: pass node=N")
            if kind == "CAPS":
                self.caps = parse_caps(line)
            elif kind == "PARAM":
                p = parse_param(line)
                self.params[p["name"]] = p
        return self.params

    @property
    def board_name(self) -> str:
        return self.caps.get("board", "?")

    @property
    def board(self) -> cfg.Board:
        """This board's wiring table (opentac/device/boards/*.json)."""
        return cfg.board(self.board_name)

    def config(self) -> dict | None:
        """The skin config the board holds (R,CFG fields), or None."""
        return parse_config(self._command("G")[0])

    def stats(self) -> dict:
        return parse_stats(self._command("t")[0])

    def info(self) -> dict:
        """Board, firmware capabilities, stored skin, statistics and every parameter."""
        return {"board": self.board_name, "caps": dict(self.caps), "skin": self.config(),
                "manifest": str(cache_path(self.stored["config_id"])) if self.manifest else None,
                "stats": self.stats(), "params": {n: p["value"] for n, p in self.params.items()}}

    # --- parameters ------------------------------------------------------------

    def _resolve(self, name, value) -> int:
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                names = ENUM_NAMES.get(name, {})
                if value not in names:
                    raise DeviceError(f"{name} has no setting {value!r}; try one of "
                                      f"{sorted(names)} or a number") from None
                return names[value]
        return int(value)

    #: Set by hdr() while HDR is on: a change there would make the frames' conversion wrong.
    HDR_SETTINGS = ("GAIN_SW", "HDR_GAINS", "HDR_CLIP")

    def set(self, name: str, value) -> int:
        """Set one parameter. Returns what the board stored (it clamps to range).

        While HDR is on, GAIN_SW, HDR_GAINS and HDR_CLIP belong to hdr(); HDR_GAINS is
        turned on only by hdr(), which measures the zero every pass needs.
        """
        if name in self.HDR_SETTINGS and name in self.params:
            if self._hdr_on:
                raise DeviceError(f"{name} is set by HDR while it is on: call hdr(False) first, "
                                  f"or hdr(gains=..., clip=...)")
            if name == "HDR_GAINS" and self._resolve(name, value) != 0:
                raise DeviceError("turn HDR on with hdr(): it measures the zero at every gain "
                                  "and converts the frames")
        return self._set(name, value)

    def _set(self, name: str, value) -> int:
        if name not in self.params:
            raise DeviceError(f"no parameter {name!r} on this board; it has {sorted(self.params)}")
        p = self.params[name]
        prefix = f"R,PARAM,{p['id']},"
        reply = self._command(f"p {p['id']} {self._resolve(name, value)}",
                              match=lambda ln: ln.startswith(prefix))[-1]
        p["value"] = parse_param(reply)["value"]
        return p["value"]

    def get(self, name: str) -> int:
        p = self.params[name]
        prefix = f"R,PARAM,{p['id']},"
        reply = self._command(f"P {p['id']}", match=lambda ln: ln.startswith(prefix))[-1]
        p["value"] = parse_param(reply)["value"]
        return p["value"]

    __getitem__ = get
    __setitem__ = set

    def set_imu(self, on: bool) -> bool:
        """Both IMU streams on or off (VIB_ENABLE, ORIENT_ENABLE)."""
        for name in ("VIB_ENABLE", "ORIENT_ENABLE"):
            if self.set(name, int(bool(on))) != int(bool(on)):
                raise DeviceError(f"board refused {name} {int(bool(on))}")
        return bool(on)

    # --- gain (60-pin board) ---------------------------------------------------

    def _gain(self) -> dict:
        gain = self.board.gain
        if not gain or gain["parameter"] not in self.params:
            raise DeviceError(f"{self.board_name} has no gain switch; the 60-pin board does")
        return gain

    def feedback_ohm(self, switches: int | None = None) -> float:
        """The amplifier's feedback resistance for a GAIN_SW value (default: the current one)."""
        gain = self._gain()
        if switches is None:
            switches = self.params[gain["parameter"]]["value"]
        conductance = 1.0 / gain["base_ohm"] + sum(
            1.0 / r for k, r in enumerate(gain["switch_ohm"]) if switches >> k & 1)
        return 1.0 / conductance

    def set_feedback(self, ohm: float) -> float:
        """Set the gain to the switch setting nearest `ohm` of feedback. Returns the ohms set.

        Higher feedback, higher gain: 10000 is the most sensitive (the default),
        68 the least. Changing it drops the xampling zero.
        """
        import math
        gain = self._gain()
        options = range(1 << len(gain["switch_ohm"]))
        best = min(options, key=lambda s: abs(math.log(self.feedback_ohm(s) / float(ohm))))
        self.set(gain["parameter"], best)
        return self.feedback_ohm(best)

    # --- the skin --------------------------------------------------------------

    def setup(self, source) -> dict:
        """Tell the board which skin is plugged in, and keep the manifest on this computer.

        `source` is a manifest.json, a design export .zip, a dict or a Manifest.
        If the board already holds exactly this config nothing is written to
        its flash. Returns the board's stored config.
        """
        m = manifest_mod.load(source, scan=True)
        for note in m.warnings:
            warnings.warn(note, stacklevel=2)
        board = self.board
        spec = cfg.extract(m.raw, board)
        for note in spec["warnings"]:
            warnings.warn(note, stacklevel=2)
        if not board.hardware_checked:
            warnings.warn(f"{board.name}'s wiring comes from its netlist and has not yet been "
                          f"checked by pressing a skin: press one corner and check that the "
                          f"right taxel lights up", stacklevel=2)
        blob = cfg.build_config(spec, m.config_id)
        crc = int.from_bytes(blob[-4:], "little")
        stored = self.config()
        if stored is None or stored["crc"] != crc:
            stored = self._provision(blob)
        if stored["crc"] != crc or stored["config_id"] != m.config_id:
            raise DeviceError(f"stored config does not match: sent crc {crc:08X} id "
                              f"{m.config_id:08X}, board has crc {stored['crc']:08X} id "
                              f"{stored['config_id']:08X}")
        cache_store(m)
        self.stored, self.manifest = stored, m
        return stored

    def _provision(self, blob: bytes) -> dict:
        self._command(f"S{len(blob)}")
        try:
            for offset in range(0, len(blob), CHUNK_BYTES):
                self._command(f"D{offset},{blob[offset:offset + CHUNK_BYTES].hex().upper()}")
            # Commit erases and programs flash, stalling the board for tens of ms.
            saved, self.link.timeout = self.link.timeout, max(self.link.timeout, 10.0)
            try:
                self._command("E")
            finally:
                self.link.timeout = saved
        except DeviceError:
            try:
                self._command("A")
            except DeviceError:
                pass
            raise
        stored = self.config()
        if stored is None:
            raise DeviceError("board reports no config after the commit")
        return stored

    def erase(self):
        """Forget the stored skin (the cached manifest stays on this computer)."""
        self._command("X")
        self.stored, self.manifest = self.config(), None
        if self.stored is not None:
            raise DeviceError("board still reports a config after erase")

    # --- frames ------------------------------------------------------------------

    def _active(self) -> int:
        if self.stored is None:
            self.stored = self.config()
        if self.stored is None:
            raise DeviceError("the board holds no skin: run `opentac setup <manifest>` first")
        return self.stored["active"]

    def frames(self, limit=None):
        """Yield every frame the board sends (BoardFrame, an opentac.frame.Frame)."""
        active, seen, last_gen = self._active(), 0, None
        while limit is None or seen < limit:
            line = self.link.readline(self.node)
            if not line:
                continue
            try:
                frame = parse_frame(line, active)
            except ValueError as exc:
                warnings.warn(f"skipping a malformed frame: {exc}", stacklevel=2)
                continue
            if frame is None:
                continue
            frame = self._hdr_frame(frame)
            if frame is None:
                continue
            frame.node = self.node
            frame.t_us = getattr(self.link, "last_t_us", {}).get(self.node)
            if last_gen is not None and frame.generation < last_gen:
                self.rebooted = True        # the board reset; its session settings went with it
            last_gen = frame.generation
            seen += 1
            yield frame

    def read(self, timeout: float = 2.0):
        """The next frame the board sends after this call."""
        active = self._active()
        self.link.reset_input_buffer()
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.link.readline(self.node)
            try:
                frame = parse_frame(line, active)
            except ValueError:
                continue                    # the tail of a line cut by the flush
            if frame is not None:
                frame = self._hdr_frame(frame)
            if frame is not None:
                frame.node = self.node
                frame.t_us = getattr(self.link, "last_t_us", {}).get(self.node)
                return frame
        raise DeviceError(f"no frame in {timeout:.0f} s: is TAC_ENABLE 1, and for xampling, "
                          f"has the board been zeroed?")

    # --- modes ---------------------------------------------------------------------

    def zero(self) -> dict:
        """Measure the board's per-taxel zero for xampling. Nothing may touch the skin.

        The board scans XAMP_ZERO_FRAMES raster frames: each taxel's zero is the
        mean of the first 16, its noise the highest reading above that. Lost on
        reboot, and when the config, a settle time or the gain changes.
        """
        frames = self.params.get("XAMP_ZERO_FRAMES", {}).get("value", 64)
        saved, self.link.timeout = self.link.timeout, max(self.link.timeout, 10.0 + frames * 0.1)
        try:
            reply = self._command("Z", match=lambda ln: ln.startswith("R,OK,Z"))[-1]
        finally:
            self.link.timeout = saved
        return parse_zero(reply)

    def xampling(self, budget: int = 128, grow: bool = True) -> int:
        """Switch to Xampling: each frame reads part of the sensor, chosen by the board.

        grow=True (the default, TAC_MODE xamp_grow): `budget` (XAMP_M) search reads,
        then growth around every contact for as far as it reaches, so a large contact
        is read whole. grow=False (xamp_fixed): exactly `budget` reads per frame,
        growth included, for a steady frame rate. Call zero() first, with nothing
        touching. Values are counts above that zero; unread taxels NaN.
        Returns the budget the board stored.
        """
        stored = self.set("XAMP_M", budget)
        self.set("TAC_MODE", "xamp_grow" if grow else "xamp_fixed")
        return stored

    def raster(self):
        """Back to reading every taxel every frame (TAC_MODE raster)."""
        self.set("TAC_MODE", "raster")

    # --- HDR (60-pin board) ----------------------------------------------------------

    def hdr(self, on: bool = True, *, zero: bool = True, clip: int | None = None,
            gains=hdr_mod.LADDER):
        """High dynamic range: each taxel read at the highest gain that does not saturate
        it (60-pin board), in raster and in Xampling.

        on=True first measures the zero at every gain, so nothing may touch the skin
        (about 1 s; zero=False reuses the last one). Frames then carry `taxels` in
        counts above that zero at the highest gain, whatever gain each taxel was read
        at, and `gain`, the gain of each taxel (1 to 1/148 with the default `gains`,
        GAIN_SW of each pass, highest first). `clip` sets HDR_CLIP (the board's,
        3800, if None). on=False: one fixed gain again, frames as without HDR, and
        GAIN_SW as it was before. Returns the HdrTable in use, or None when off.
        close() turns HDR off.

        In Xampling the zero at every gain is measured in raster, then the board's own
        zero (`zero()`) again at the gain HDR scans first, and Xampling resumes; both
        need nothing touching. Switching HDR off and on with zero=False needs neither
        while GAIN_SW stays the same.
        """
        gain = self._gain()
        param = gain["parameter"]
        if "HDR_GAINS" not in self.params:
            raise DeviceError("this board's firmware has no HDR: `opentac flash` installs "
                              "firmware that has it")
        mode = self.params["TAC_MODE"]["value"]
        if not on:
            self._set("HDR_GAINS", 0)
            was_on, self._hdr_on = self._hdr_on, False
            restore = self._gain_before_hdr
            if was_on and restore is not None and restore != self.params[param]["value"]:
                self._set(param, restore)
                if mode != 0:
                    self.zero()             # Xampling's zero was measured at the other gain
            return None
        gains = tuple(int(g) for g in gains)
        if not self._hdr_on:
            self._gain_before_hdr = self.params[param]["value"]
        table = self._hdr_table
        fresh = (zero or table is None or table.gain_sw != gains
                 or table.zero.shape[1] != self._active())
        if fresh:
            if mode != 0:
                self._set("TAC_MODE", 0)    # the zero at every gain is measured in raster
            table = self._hdr_zero(gains)
        if clip is not None:
            self._set("HDR_CLIP", clip)
        table.clip = self.params["HDR_CLIP"]["value"]
        moved = self.params[param]["value"] != gains[0]
        self._set(param, gains[0])
        if self._set("HDR_GAINS", table.code) != table.code:
            raise DeviceError(f"board refused HDR_GAINS 0x{table.code:04X}")
        self._hdr_table, self._hdr_on = table, True
        if mode != 0:
            if fresh or moved:
                self.zero()                 # the board's zero, at the gain HDR scans first
            if self.params["TAC_MODE"]["value"] != mode:
                self._set("TAC_MODE", mode)
        return table

    @property
    def hdr_on(self) -> bool:
        return self._hdr_on

    def _hdr_zero(self, gains) -> hdr_mod.HdrTable:
        """The zero and noise of every taxel at each gain in `gains`, HDR off meanwhile."""
        param = self._gain()["parameter"]
        self._set("HDR_GAINS", 0)
        self._hdr_on = False
        active = self._active()
        zeros, noise = [], []
        for sw in gains:
            self._set(param, sw)
            self.link.reset_input_buffer()
            acc, deadline = [], time.time() + 10.0
            while len(acc) < hdr_mod.ZERO_SKIP + hdr_mod.ZERO_FRAMES:
                if time.time() > deadline:
                    raise DeviceError("no frames while zeroing for HDR: is TAC_ENABLE 1?")
                try:
                    frame = parse_frame(self.link.readline(self.node), active)
                except ValueError:
                    continue                # the tail of a line cut by the flush
                if frame is not None and frame.mode == 0:
                    acc.append(frame.taxels)
            acc = np.asarray(acc[hdr_mod.ZERO_SKIP:], dtype=float)
            zeros.append(acc.mean(axis=0))
            noise.append(acc.std(axis=0))
        base = self.feedback_ohm(gains[0])
        return hdr_mod.HdrTable(gain_sw=gains,
                                gain=np.array([self.feedback_ohm(sw) / base for sw in gains]),
                                zero=np.asarray(zeros), noise=np.asarray(noise),
                                clip=self.params["HDR_CLIP"]["value"])

    def _hdr_frame(self, frame):
        """In HDR, convert a frame with the gain each taxel was read at. Without HDR the
        frame is returned as it came; None drops a frame still scanned in HDR after HDR
        was turned off."""
        table = self._hdr_table
        if table is None:
            return frame
        if not self._hdr_on:
            tagged = (np.nan_to_num(frame.taxels, nan=0.0) > hdr_mod.RAW_MASK).any()
            return None if tagged else frame
        frame.hdr_raw = frame.taxels.copy()     # as sent: the pass in bits 12-15, NaN unread
        frame.taxels, frame.gain = table.decode(frame.hdr_raw, board_zero=frame.mode != 0)
        frame.hdr = table
        frame.units = "counts above zero at the highest gain"
        return frame

    # --- firmware ----------------------------------------------------------------

    def dfu(self):
        """Reboot into the STM32 bootloader for reflashing (the `B` command)."""
        self.link.command("B", self.node, match=lambda ln: ln == "R,OK,B")


class Bus:
    """Every sensor behind a bus controller, each a Sensor.

        with Bus() as bus:
            for s in bus:                  # one per online node
                print(s.node, s.manifest)
            for node, frame in bus.frames():
                ...
    """

    def __init__(self, port=None, *, timeout=5.0, link=None, **session):
        self.link = BusLink(port, timeout, link=link)
        try:
            caps = self.link.command("?", node=None, until="R,END")
            mcaps = [c for c in caps if c.startswith("R,MCAPS,")]
            if not mcaps:
                raise DeviceError("this port did not answer as an OpenTac bus controller")
            self.version = int(mcaps[0].split(",")[2])
            self.nodes = self.link.list_nodes()
            self.sensors = [Sensor(node=n["node"], _link=self.link, **session)
                            for n in self.nodes if n["online"]]
        except Exception:
            self.link.close()
            raise

    def __getitem__(self, node) -> Sensor:
        for s in self.sensors:
            if s.node == node:
                return s
        raise DeviceError(f"no online node {node}; online: {[s.node for s in self.sensors]}")

    def __iter__(self):
        return iter(self.sensors)

    def __len__(self):
        return len(self.sensors)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        for s in self.sensors:
            s._leave_hdr()
        self.link.close()

    def frames(self, limit=None):
        """Yield (node, frame) from every node holding a skin, in arrival order.

        Each frame's t_us is the controller's clock, shared by all nodes.
        """
        active = {s.node: s.stored["active"] for s in self.sensors if s.stored}
        if not active:
            raise DeviceError("no node on this bus holds a skin; run `opentac setup` for one first")
        by_node = {s.node: s for s in self.sensors}
        seen = 0
        while limit is None or seen < limit:
            got = self.link.readline_any()
            if got is None or got[0] not in active:
                continue
            node, t_us, line = got
            try:
                frame = parse_frame(line, active[node])
            except ValueError as exc:
                warnings.warn(f"node {node}: skipping a malformed frame: {exc}", stacklevel=2)
                continue
            if frame is not None:
                frame = by_node[node]._hdr_frame(frame)
            if frame is None:
                continue
            frame.node, frame.t_us = node, t_us
            seen += 1
            yield node, frame

    def pick(self, colour=(255, 0, 0), keys=None) -> Sensor | None:
        """Choose a node by its LED: one lights up, Space moves on, Enter picks it, q cancels."""
        if not self.sensors:
            raise DeviceError("no online node to pick from")
        saved = {s.node: s.get("LED_MODE") for s in self.sensors}
        keys = iter(keys) if keys is not None else _terminal_keys()
        try:
            for s in self.sensors:
                s.set("LED_MODE", "manual")
                s._command("c 0 0 0")
            k = 0
            while True:
                for j, s in enumerate(self.sensors):
                    if j == k:
                        s._command("c {} {} {}".format(*colour))
                    elif j == (k - 1) % len(self.sensors):
                        s._command("c 0 0 0")
                s = self.sensors[k]
                skin = f"skin {s.stored['config_id']:08X}" if s.stored else "no skin"
                print(f"\r\033[K  lit: node {s.node} ({s.board_name}, {skin})   "
                      f"Space next, Enter choose, q cancel", end="", flush=True)
                key = next(keys, "q")
                if key in ("\r", "\n", ""):
                    print()
                    return s
                if key in ("q", "\x1b", "\x03"):
                    print()
                    return None
                k = (k + 1) % len(self.sensors)
        finally:
            for s in self.sensors:
                s._command("c 0 0 0")
                s.set("LED_MODE", saved[s.node])


def _terminal_keys():
    """Single keypresses from the terminal; lines if stdin is not a terminal."""
    import sys
    if not sys.stdin.isatty():
        for line in sys.stdin:
            yield line.rstrip("\n")[:1]
        return
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            yield sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
