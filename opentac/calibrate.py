"""Counts -> newtons, from a calibration file.

    from opentac.calibrate import Calibration

    cal = Calibration("force_calibration.json")
    force = cal.skin(sensor.manifest, velostat="v2")
    newtons = force.taxels(frame, zero)        # N per taxel, NaN where not read
    total = force.total(frame, zero)

One curve per design (taxel pitch x electrode gap) and velostat. Every taxel of
a design shares it, and a frame's force is the sum of its taxels:

    c   = max(counts - zero - deadband, 0)
    F_k = a * c**b                       N on taxel k   (b = 1: linear)

The calibration file is a JSON object with a "curves" table:
curves[design][velostat] = {"model": "power"|"linear", "power": {"a_N", "b"},
"linear": {"a_N", "b"}, "deadband_counts", "sat_counts", "counts_fitted_max",
"mux_settle_us", "status"}. The design key is "p<pitch mm>_gap<gap mm>", e.g.
"p2.0_gap0.2", read from the manifest. A curve holds only for the stack it was
measured on (velostat, foam, readout board gain, MUX_SETTLE_US).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from opentac.device.config import derive_pitch_um
from opentac.frame import Frame
from opentac.manifest import load


class CalibrationError(ValueError):
    pass


def design_key(manifest) -> tuple[str, float]:
    """('p2.0_gap0.2', pitch mm) from a manifest: pitch from `pixels`, gap from `footprint`."""
    raw = load(manifest).raw
    px_um, py_um = derive_pitch_um(raw)
    if px_um != py_um:
        raise CalibrationError(f"pitch differs in x and y ({px_um} / {py_um} um); "
                               f"only square taxels are calibrated")
    gap = raw.get("footprint", {}).get("gap")
    if gap is None:
        gap = raw.get("design", {}).get("params", {}).get("gap_mm")
    if gap is None:
        raise CalibrationError("manifest has no footprint.gap or design.params.gap_mm")
    return f"p{px_um / 1000:.1f}_gap{gap:.1f}", px_um / 1000.0


class Curve:
    """One design x velostat: counts above the zero -> N on one taxel."""

    def __init__(self, design, velostat, entry, model=None):
        if entry.get("status") == "insufficient":
            raise CalibrationError(f"{design} x {velostat}: {entry.get('reason', 'no curve')}")
        self.design, self.velostat, self.entry = design, velostat, entry
        self.model = model or entry["model"]
        self.a = float(entry[self.model]["a_N"])
        self.b = float(entry[self.model]["b"])
        self.deadband = float(entry.get("deadband_counts", 0.0))
        self.sat_counts = float(entry.get("sat_counts", 4095))
        self.counts_fitted_max = float(entry.get("counts_fitted_max", np.inf))
        self.mux_settle_us = entry.get("mux_settle_us")

    def force(self, counts):
        """Counts above the zero (any shape) -> N. NaN stays NaN."""
        c = np.clip(np.asarray(counts, float) - self.deadband, 0, None)
        return self.a * (c if self.b == 1 else c ** self.b)


class Calibration:
    """A calibration file, passed in by the user."""

    def __init__(self, path):
        path = Path(path)
        if not path.exists():
            raise CalibrationError(f"no calibration file at {path}")
        self.path = path
        self.data = json.loads(path.read_text(encoding="utf-8"))
        if "curves" not in self.data:
            raise CalibrationError(f"{path.name} has no 'curves' table")

    def curve(self, design, velostat, model=None) -> Curve:
        curves = self.data["curves"]
        if design not in curves:
            raise CalibrationError(f"design {design} is not in {self.path.name}; it has "
                                   f"{', '.join(sorted(curves))}")
        if velostat not in curves[design]:
            raise CalibrationError(f"{design} has no curve for velostat {velostat!r}; it has "
                                   f"{', '.join(sorted(curves[design]))}")
        return Curve(design, velostat, curves[design][velostat], model)

    def skin(self, manifest, velostat, model=None) -> "SkinForce":
        return SkinForce(manifest, self.curve(design_key(manifest)[0], velostat, model))


class SkinForce:
    """One skin with one curve: frames of counts in `pixels` order -> N per taxel."""

    def __init__(self, manifest, curve: Curve):
        self.curve = curve
        _, self.pitch_mm = design_key(manifest)
        self.n = load(manifest).n

    @staticmethod
    def _values(frame):
        if isinstance(frame, Frame):
            return np.asarray(frame.taxels, float)
        return np.asarray(frame, float)

    def counts_above_zero(self, frame, zero=None):
        """Counts minus the zero. Xampling frames are already above the board's zero, and HDR
        frames above the zero at every gain: pass none."""
        v = self._values(frame)
        return v - (0.0 if zero is None else np.asarray(zero, float))

    def taxels(self, frame, zero=None):
        """N per taxel; NaN where a taxel was not read. A saturated taxel gives a lower bound."""
        return self.curve.force(self.counts_above_zero(frame, zero))

    def total(self, frame, zero=None):
        return np.nansum(self.taxels(frame, zero), axis=-1)

    def saturated(self, frame):
        """Taxels at or above the curve's saturation count (raw counts, raster frames). In
        HDR, only taxels there at the lowest gain, which no further pass could re-read."""
        hdr, sent = getattr(frame, "hdr", None), getattr(frame, "hdr_raw", None)
        if hdr is not None and sent is not None:
            return hdr.saturated(sent, at=self.curve.sat_counts)
        return self._values(frame) >= self.curve.sat_counts

    def extrapolated(self, frame, zero=None):
        """Above the highest count the fit saw, but not saturated."""
        c = self.counts_above_zero(frame, zero) - self.curve.deadband
        return (c > self.curve.counts_fitted_max) & ~self.saturated(frame)
