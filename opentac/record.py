"""Recording takes: every frame between start and stop, with what is needed to read them later.

    from opentac.device import Sensor
    from opentac.record import record, load_take

    path = record(Sensor(), "recordings", seconds=5, description="tape roll, edge-on")
    take = load_take(path)          # {"manifest", "info", plus every array}
    frames = list(iter_frames(path))  # same Frame objects a live board yields
    # opentac replay recordings/<take>   plays it in the heatmap viewer

A take is a folder named <date>_<time>_<slug> holding three files (format
"opentac_view_take/1"):

    tactile.npz     raw (frames x taxels, int16, manifest `pixels` order; -1 = not read),
                    t_host, frame_id, generation, dt_us, dropped, accel, gyro, quat, euler,
                    the geometry (xy mm, grid_rc, taxel_wh, pixel_rot_deg, outline),
                    `sampled` (frames x taxels bool) when xampling, `baseline` if one was set
    manifest.json   the skin's manifest, byte for byte as the board was set up with
    take.json       description, board and skin, session settings, frame statistics, notes
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

import numpy as np

from opentac import manifest as manifest_mod

TAKE_FORMAT = "opentac_view_take/1"


def slugify(text: str, limit: int = 40) -> str:
    """The first words of a description, as a folder-name suffix."""
    slug = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return slug[:limit].rstrip("_")


def geometry(manifest) -> dict:
    """The drawn layout as plain arrays to store beside the counts."""
    view = manifest_mod.drawn(manifest)
    pixels = view["pixels"]
    return dict(
        xy=np.array([[p[0], p[1]] for p in pixels], dtype=float),
        grid_rc=np.array([[p[2], p[3]] for p in pixels], dtype=np.int32),
        taxel_wh=np.array([view.get("w", 0.0), view.get("h", 0.0)], dtype=float),
        pixel_rot_deg=float(view.get("pixel_rot_deg", 0.0) or 0.0),
        outline=np.array(view.get("outline") or [], dtype=float).reshape(-1, 2))


def session(sensor) -> dict:
    """The static half of take.json: board, skin and settings. Ask before streaming."""
    m, stored = sensor.manifest, sensor.stored or {}
    mode = {0: "raster", 1: "xamp_fixed", 2: "xamp_grow"}.get(sensor.get("TAC_MODE"), "?")
    settings = dict(mode=mode, mux_settle_us=sensor.get("MUX_SETTLE_US"),
                    imu=bool(sensor.params.get("VIB_ENABLE", {}).get("value")))
    if mode != "raster":
        settings["xamp"] = dict(m=sensor.get("XAMP_M"), thresh=sensor.get("XAMP_THRESH"),
                                search_min=sensor.get("XAMP_SEARCH_MIN"),
                                bump_fill=bool(sensor.get("XAMP_BUMP_FILL")))
    if "GAIN_SW" in sensor.params:
        settings["gain_sw"] = sensor.get("GAIN_SW")
        settings["feedback_ohm"] = round(sensor.feedback_ohm(), 1)
    return dict(
        manifest=dict(file=m.name if m else None, id=f"{m.config_id:08X}" if m else None,
                      n_taxels=m.n if m else stored.get("active"),
                      drawn_from="pixels_true" if m and m.get("pixels_true") else "pixels"),
        board=dict(name=sensor.board_name, node=sensor.node,
                   port=getattr(sensor.link, "port", None),
                   config_id=f"{stored['config_id']:08X}" if stored else None,
                   rows=stored.get("rows"), cols=stored.get("cols"), active=stored.get("active")),
        settings=settings)


class Take:
    """Frames between start and stop. add() is cheap enough to call from a reader thread."""

    def __init__(self, outdir, manifest, description="", context=None):
        self.outdir = Path(outdir)
        self.manifest = manifest_mod.load(manifest)
        self.description = description or ""
        self.context = context or {}
        self.t_start = time.time()
        self.t, self.raw, self.meta, self.imu, self.sampled = [], [], [], [], []
        self.lock = threading.Lock()

    def add(self, frame):
        values = np.where(frame.mask, np.nan_to_num(frame.taxels, nan=-1), -1).astype(np.int16)
        imu = frame.imu or {}
        with self.lock:
            self.t.append(frame.t)
            self.raw.append(values)
            self.meta.append((getattr(frame, "index", len(self.t) - 1),
                              getattr(frame, "generation", 0), getattr(frame, "dt_us", 0),
                              getattr(frame, "dropped", 0)))
            self.imu.append(imu.get("accel_mg", [0] * 3) + imu.get("gyro_mdps", [0] * 3)
                            + imu.get("quat_wxyz", [0.0] * 4) + imu.get("euler_deg", [0.0] * 3))
            self.sampled.append(None if frame.mask.all() else frame.mask.copy())

    def __len__(self):
        return len(self.t)

    def _dir(self) -> Path:
        stem = time.strftime("%Y%m%d_%H%M%S", time.localtime(self.t_start))
        slug = slugify(self.description)
        name = f"{stem}_{slug}" if slug else stem
        path, n = self.outdir / name, 2
        while path.exists():                 # two takes inside one second
            path, n = self.outdir / f"{name}_{n}", n + 1
        return path

    def save(self, baseline=None, t_stop=None) -> Path | None:
        """Write the take. Returns its folder, or None if no frame arrived."""
        with self.lock:
            if not self.t:
                return None
            t_stop = time.time() if t_stop is None else t_stop
            raw = np.stack(self.raw)
            t = np.asarray(self.t, dtype=float)
            meta = np.asarray(self.meta, dtype=np.int64)
            imu = np.asarray(self.imu, dtype=float)
            sampled = list(self.sampled)
        path = self._dir()
        path.mkdir(parents=True)
        arrays = dict(raw=raw, t_host=t, frame_id=meta[:, 0],
                      generation=meta[:, 1].astype(np.int32), dt_us=meta[:, 2].astype(np.uint32),
                      dropped=meta[:, 3].astype(np.int32), accel=imu[:, 0:3], gyro=imu[:, 3:6],
                      quat=imu[:, 6:10], euler=imu[:, 10:13], **geometry(self.manifest))
        if any(s is not None for s in sampled):
            mask = np.ones(raw.shape, dtype=bool)
            for k, s in enumerate(sampled):
                if s is not None:
                    mask[k] = s
            arrays["sampled"] = mask
        if baseline is not None:
            arrays["baseline"] = np.asarray(baseline, dtype=np.float32)
        np.savez_compressed(path / "tactile.npz", **arrays)
        self.manifest.save(path / "manifest.json")

        span = float(t[-1] - t[0])
        info = dict(self.context)
        info.update(
            format=TAKE_FORMAT, take=path.name, description=self.description,
            t_start=self.t_start, t_stop=t_stop, duration_s=t_stop - self.t_start,
            n_frames=len(t), n_taxels=int(raw.shape[1]),
            fps=(len(t) - 1) / span if span > 0 else None,
            missing_frames=int(meta[-1, 0] - meta[0, 0] + 1 - len(t)),
            dropped_by_board=int(meta[:, 3].max() - meta[:, 3].min()),
            baseline=("subtracted on screen only; stored in the npz as `baseline`"
                      if baseline is not None else
                      "none: raw counts, or the board's own zero for xampling"),
            files=["tactile.npz", "manifest.json", "take.json"],
            notes=dict(
                clock="t_host is the computer's unix time (s) when the frame arrived; the "
                      "scan happened over the dt_us before it",
                units="raw is ADC counts as the board sent them. In raster that is an absolute "
                      "12-bit reading; with xampling it is counts above the board's own zero, "
                      "and `sampled` says which taxels a frame read (raw is -1 elsewhere). "
                      "Newtons are not applied: use opentac.calibrate",
                ordering="raw[k, j] is taxel j of frame k, in the manifest's `pixels` order. "
                         "Taxel j sits at xy[j] mm and is taxel_wh mm across",
                geometry="xy/outline are the drawn layout (pixels_true for a simplified skin, "
                         "else pixels), in the manifest's mm frame"))
        (path / "take.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        return path


def record(sensor, outdir, seconds=None, description="", log=print) -> Path | None:
    """Record from one sensor until `seconds` pass or Ctrl-C. Returns the take's folder."""
    if sensor.manifest is None:
        raise ValueError("this computer has no manifest for the board's skin: run "
                         "`opentac setup <manifest>` first")
    take = Take(outdir, sensor.manifest, description, session(sensor))
    log(f"recording{f' for {seconds} s' if seconds else ''}... Ctrl-C to stop")
    try:
        for frame in sensor.frames():
            take.add(frame)
            if seconds is not None and frame.t - take.t_start >= seconds:
                break
    except KeyboardInterrupt:
        pass
    path = take.save()
    if path is None:
        log("no frames arrived; nothing written")
    else:
        log(f"saved {path}: {len(take)} frames")
    return path


def load_take(path) -> dict:
    """A take folder as {"manifest": Manifest, "info": take.json dict, <npz arrays>...}."""
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f"{path} is not a take folder")
    info_path = path / "take.json"
    if not info_path.exists():
        raise ValueError(f"{path} is not a {TAKE_FORMAT} take (no take.json)")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("format") != TAKE_FORMAT:
        raise ValueError(f"{path} is not a {TAKE_FORMAT} take (format {info.get('format')!r})")
    with np.load(path / "tactile.npz") as npz:
        out = {k: npz[k] for k in npz.files}
    out["manifest"] = manifest_mod.load(path / "manifest.json")
    out["info"] = info
    return out


_MODE_FROM_NAME = {"raster": 0, "xamp_fixed": 1, "xamp_grow": 2}


def iter_frames(path):
    """Yield one `opentac.frame.Frame` per stored sample, in recording order.

    Unread taxels (xampling, or a stored -1) come back as NaN with `mask` False,
    the same way a live board reports them. Board bookkeeping (`index`,
    `generation`, `mode`, `dt_us`, `dropped`) is restored when the npz has it.
    """
    from opentac.device.protocol import BoardFrame

    take = load_take(path)
    raw = take["raw"]
    t = np.asarray(take["t_host"], dtype=float)
    n = len(t)
    sampled = take.get("sampled")
    frame_id = take.get("frame_id", np.arange(n))
    generation = take.get("generation", np.zeros(n, dtype=int))
    dt_us = take.get("dt_us", np.zeros(n, dtype=int))
    dropped = take.get("dropped", np.zeros(n, dtype=int))
    accel, gyro = take.get("accel"), take.get("gyro")
    quat, euler = take.get("quat"), take.get("euler")
    mode_name = (take["info"].get("settings") or {}).get("mode", "raster")
    mode = _MODE_FROM_NAME.get(mode_name, 0)
    units = "counts above zero" if mode else "counts"
    for i in range(n):
        values = np.asarray(raw[i], dtype=float)
        mask = np.asarray(sampled[i], dtype=bool) if sampled is not None else (values >= 0)
        taxels = np.where(mask, values, np.nan)
        imu = {}
        if accel is not None:
            imu = dict(accel_mg=accel[i].tolist(), gyro_mdps=gyro[i].tolist(),
                       quat_wxyz=quat[i].tolist(), euler_deg=euler[i].tolist())
        yield BoardFrame(
            taxels=taxels, t=float(t[i]), mask=mask, imu=imu, units=units,
            index=int(frame_id[i]), generation=int(generation[i]),
            mode=int(mode), dt_us=int(dt_us[i]), dropped=int(dropped[i]))


def peak_frame_index(path) -> int:
    """Index of the frame whose strongest taxel is the take's strongest taxel."""
    take = load_take(path)
    raw = np.asarray(take["raw"], dtype=float)
    sampled = take.get("sampled")
    if sampled is not None:
        shown = np.where(sampled, raw, np.nan)
    else:
        shown = np.where(raw >= 0, raw, np.nan)
    peaks = np.nanmax(shown, axis=1)
    peaks = np.where(np.isfinite(peaks), peaks, -np.inf)
    if not np.isfinite(peaks).any():
        return 0
    return int(np.nanargmax(peaks))
