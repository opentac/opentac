"""Live heatmap of one or more sensors, each taxel drawn where it physically is.

    opentac view                 the board on USB, or every node of a bus controller
    opentac view --shear         with one shear arrow per bump
    opentac view --save takes/   Enter starts and stops a take
    opentac replay TAKE          play a recorded take in the same heatmap

Keys on a live view (click the window first):
    b      zero: subtract the mean of the last frames (nothing touching). With
           xampling, the board measures a new zero instead.
    c      clear the zero
    a      autoscale on/off
    s      shear arrows and bump boxes on/off (skins with shear bumps)
    Enter  start / stop a take (with --save)
    q      quit

Keys on a replay:
    space  pause / continue
    left / right   step one frame
    home / end     first / last frame
    a, s, q        as on a live view

Frames arrive in the manifest's `pixels` order, so value k is drawn at pixels[k]
with no reshaping. Xampling frames carry only the taxels read this frame; the
others are drawn in blue, below the colour scale.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from opentac import manifest as manifest_mod
from opentac.device.config import derive_pitch_um
from opentac.record import Take, iter_frames, load_take, peak_frame_index, session

BASELINE_FRAMES = 30
UNSAMPLED_COLOR = "#3a6ea5"
SHEAR_ARROW_COLOR = "#39ff6a"
BUMP_OUTLINE_COLOR = "cyan"
PALETTE = [(255, 64, 64), (64, 255, 64), (64, 128, 255), (255, 200, 0), (255, 0, 255),
           (0, 255, 255), (255, 128, 0), (160, 64, 255), (0, 255, 150), (255, 90, 160)]


class ShearField:
    """One shear vector per bump, as the manifest defines it.

    Each taxel under a bump pulls the bump's vector along its unit direction from
    the bump centre, weighted by its reading: on a 2x2 group, (right - left,
    top - bottom). A component is zero on an axis where all of a group's taxels
    sit on one side (a bump the outline cut short). This is a difference of
    counts, not a calibrated force, and it only means shear with the bump sheet on.
    """

    def __init__(self, view: dict):
        shear = view.get("shear") or {}
        if not shear.get("bumps") or "b" not in shear:
            raise ValueError("this skin's manifest has no shear bumps")
        pix = np.array([p[:2] for p in view["pixels"]], dtype=float)
        group = np.asarray(shear["b"])
        self.centres = np.asarray(shear["bumps"], dtype=float)
        self.member = np.nonzero(group >= 0)[0]
        self.group = group[self.member]
        self.member_xy = pix[self.member]
        d = pix[self.member] - self.centres[self.group]
        norm = np.linalg.norm(d, axis=1)
        norm[norm == 0] = 1.0
        self.ux, self.uy = d[:, 0] / norm, d[:, 1] / norm
        n = len(self.centres)
        self.valid = np.zeros((n, 2), dtype=bool)
        for g in range(n):
            dg = d[self.group == g]
            if len(dg):
                self.valid[g] = [(dg[:, 0] > 1e-6).any() and (dg[:, 0] < -1e-6).any(),
                                 (dg[:, 1] > 1e-6).any() and (dg[:, 1] < -1e-6).any()]
        if n > 1:
            dist = np.linalg.norm(self.centres[:, None] - self.centres[None], axis=2)
            np.fill_diagonal(dist, np.inf)
            self.spacing = float(np.median(dist.min(axis=1)))
        else:
            self.spacing = 5.0

    def vectors(self, values, min_total=300.0):
        """(sx, sy, total) per bump; bumps with less than `min_total` counts get no arrow."""
        n = len(self.centres)
        v = np.nan_to_num(np.asarray(values, dtype=float)[self.member], nan=0.0).clip(0, None)
        sx = np.bincount(self.group, weights=self.ux * v, minlength=n) * self.valid[:, 0]
        sy = np.bincount(self.group, weights=self.uy * v, minlength=n) * self.valid[:, 1]
        total = np.bincount(self.group, weights=v, minlength=n)
        quiet = total < min_total
        sx[quiet] = 0.0
        sy[quiet] = 0.0
        return sx, sy, total


class Panel:
    """One skin drawn into one matplotlib axes."""

    def __init__(self, ax, manifest, title, vmax=4095.0, shear=False, colour=None):
        import matplotlib.pyplot as plt
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Polygon

        self.ax, self.title, self.vmax = ax, title, vmax
        self.colour = "#%02x%02x%02x" % colour if colour else "black"
        view = manifest_mod.drawn(manifest)
        pixels = view["pixels"]
        w = view.get("w") or derive_pitch_um(view)[0] / 1000.0 * 0.9
        h = view.get("h") or w
        tilt = np.radians(float(view.get("pixel_rot_deg", 0.0) or 0.0))
        c, s = np.cos(tilt), np.sin(tilt)

        def box(cx, cy, x0, y0, x1, y1):
            return Polygon([(cx + u * c - v * s, cy + u * s + v * c)
                            for u, v in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))], closed=True)

        cmap = plt.get_cmap("inferno").with_extremes(under=UNSAMPLED_COLOR)
        self.taxels = PatchCollection([box(p[0], p[1], -w / 2, -h / 2, w / 2, h / 2) for p in pixels],
                                      cmap=cmap, edgecolor="none")
        self.taxels.set_array(np.zeros(len(pixels)))
        self.taxels.set_clim(0, vmax)
        ax.add_collection(self.taxels)
        outline = view.get("outline")
        if outline:
            xs, ys = zip(*(list(outline) + [outline[0]]))
            ax.plot(xs, ys, color="0.6", lw=1)
        xs = [p[0] for p in pixels] + ([p[0] for p in outline] if outline else [])
        ys = [p[1] for p in pixels] + ([p[1] for p in outline] if outline else [])
        pad = max(w, h)
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.figure.colorbar(self.taxels, ax=ax, label="counts")

        self.field = self.arrows = self.boxes = None
        try:
            self.field = ShearField(view)
        except ValueError:
            pass
        if self.field is not None:
            from matplotlib.patches import Rectangle
            outlines = []
            for g in range(len(self.field.centres)):
                xy = self.field.member_xy[self.field.group == g]
                if len(xy):
                    x0, y0 = xy.min(axis=0) - (w / 2, h / 2)
                    x1, y1 = xy.max(axis=0) + (w / 2, h / 2)
                    outlines.append(Rectangle((x0, y0), x1 - x0, y1 - y0))
            self.boxes = PatchCollection(outlines, facecolor="none", edgecolor=BUMP_OUTLINE_COLOR,
                                         linewidth=0.6, alpha=0.7, zorder=2)
            ax.add_collection(self.boxes)
            n = len(self.field.centres)
            self.arrows = ax.quiver(self.field.centres[:, 0], self.field.centres[:, 1],
                                    np.zeros(n), np.zeros(n), color=SHEAR_ARROW_COLOR,
                                    angles="xy", scale_units="xy",
                                    scale=vmax / self.field.spacing, width=0.006, zorder=3,
                                    edgecolor="black", linewidth=0.5, minlength=0)
            self.show_shear(shear)

    def show_shear(self, on):
        if self.arrows is not None:
            self.arrows.set_visible(on)
            self.boxes.set_visible(on)

    @property
    def shear_on(self):
        return self.arrows is not None and self.arrows.get_visible()

    def draw(self, values, autoscale=False, note=""):
        """values: one per taxel, NaN where not read."""
        shown = np.where(np.isnan(values), -1.0, values)
        self.taxels.set_array(shown)
        if autoscale:
            self.taxels.set_clim(0, max(float(np.nanmax(values, initial=0)), 1.0))
        else:
            self.taxels.set_clim(0, self.vmax)
        if self.shear_on:
            sx, sy, _ = self.field.vectors(values)
            self.arrows.set_UVC(sx, sy)
        self.ax.set_title(f"{self.title}\n{note}", color=self.colour, fontsize=9)


class Source:
    """One sensor's newest frame, recent history and zero, filled by a reader thread."""

    def __init__(self, sensor, colour=None):
        self.sensor = sensor
        self.colour = colour
        self.lock = threading.Lock()
        self.latest = None
        self.recent = []
        self.baseline = None
        self.take = None
        self.fps, self._count, self._t0 = 0.0, 0, time.time()

    def push(self, frame):
        with self.lock:
            self.latest = frame
            self.recent.append(np.asarray(frame.taxels, float))
            del self.recent[:-BASELINE_FRAMES]
            if self.take is not None:
                self.take.add(frame)
            self._count += 1
            now = time.time()
            if now - self._t0 >= 1.0:
                self.fps, self._count, self._t0 = self._count / (now - self._t0), 0, now


class Reader(threading.Thread):
    """Reads frames as fast as they come; the window only ever draws the newest."""

    def __init__(self, sources, bus=None):
        super().__init__(daemon=True)
        self.sources = sources            # node (or None) -> Source
        self.bus = bus
        self.error = None
        self._halt = threading.Event()

    def run(self):
        try:
            if self.bus is not None:
                for node, frame in self.bus.frames():
                    if self._halt.is_set():
                        return
                    if node in self.sources:
                        self.sources[node].push(frame)
            else:
                src = next(iter(self.sources.values()))
                for frame in src.sensor.frames():
                    if self._halt.is_set():
                        return
                    src.push(frame)
        except Exception as exc:          # shown in the window
            self.error = exc

    def stop(self):
        self._halt.set()
        self.join(timeout=2.0)


def _use_matplotlib(snapshot):
    """Agg for a PNG snapshot so a machine with no display can still draw."""
    import matplotlib
    if snapshot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name, key in (("keymap.save", "s"), ("keymap.back", "c")):
        matplotlib.rcParams[name] = [k for k in matplotlib.rcParams[name] if k != key]
    return matplotlib, plt


def view(sensors, bus=None, shear=False, vmax=4095.0, save=None, description="",
         snapshot=None, log=print) -> int:
    """Show live heatmaps of `sensors` (Sensor objects holding skins with cached manifests).

    Pass `bus` when the sensors are nodes of one Bus. `save` is a folder for
    takes (Enter starts and stops one). `snapshot` saves one frame to a PNG and returns.
    """
    _, plt = _use_matplotlib(snapshot)

    sensors = [s for s in sensors if s.manifest is not None]
    if not sensors:
        raise ValueError("no sensor here has a manifest on this computer: run "
                         "`opentac setup <manifest>` first")
    n = len(sensors)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6.5 * rows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)

    sources, panels = {}, {}
    for k, (s, ax) in enumerate(zip(sensors, axes.flat)):
        colour = PALETTE[(s.node or 0) % len(PALETTE)] if bus is not None else None
        if bus is not None:                # the node's LED takes its panel's colour
            s.set("LED_MODE", "manual")
            s._command("c {} {} {}".format(*colour))
        label = (f"node {s.node} · " if s.node is not None else "") + (s.manifest.name or "skin")
        sources[s.node] = Source(s, colour)
        panels[s.node] = Panel(ax, s.manifest, label, vmax, shear, colour)
    contexts = {s.node: session(s) for s in sensors} if save else {}

    if snapshot:
        for s in sensors:
            frame = s.read()
            panels[s.node].draw(frame.taxels, note=f"peak {np.nanmax(frame.taxels):.0f}")
        fig.savefig(snapshot, dpi=120, bbox_inches="tight")
        log(f"saved {snapshot}")
        return 0

    state = {"autoscale": False, "reader": Reader(sources, bus)}
    state["reader"].start()

    def stop_takes():
        for src in sources.values():
            with src.lock:
                take, src.take = src.take, None
            if take is not None:
                path = take.save(src.baseline)
                log(f"saved {path}: {len(take)} frames" if path else "take caught no frames")

    def toggle_take():
        if save is None:
            log("no --save folder: run again with --save DIR to record takes")
        elif any(src.take is not None for src in sources.values()):
            stop_takes()
        else:
            for node, src in sources.items():
                desc = description if bus is None else f"{description} node {node}".strip()
                take = Take(save, src.sensor.manifest, desc, contexts[node])
                with src.lock:
                    src.take = take
            log(f"recording to {save}... Enter to stop")

    def rezero_boards():
        # Commands and the reader share one serial stream: stop it first.
        state["reader"].stop()
        for src in sources.values():
            src.sensor.link.reset_input_buffer()
            try:
                z = src.sensor.zero()
                log(f"zeroed node {src.sensor.node}: noise median <= {z['noise_median']}, "
                    f"max {z['noise_max']}")
            except Exception as exc:
                log(f"zero failed: {exc}")
        state["reader"] = Reader(sources, bus)
        state["reader"].start()

    def on_key(event):
        key = event.key
        if key in ("enter", "return"):
            toggle_take()
        elif key == "b":
            if any(src.latest is not None and src.latest.mode != 0 for src in sources.values()):
                rezero_boards()
            else:
                for src in sources.values():
                    with src.lock:
                        if src.recent:
                            src.baseline = np.mean(src.recent, axis=0)
        elif key == "c":
            for src in sources.values():
                src.baseline = None
        elif key == "a":
            state["autoscale"] = not state["autoscale"]
        elif key == "s":
            for p in panels.values():
                p.show_shear(not p.shear_on)

    def on_close(_event=None):
        stop_takes()                        # never lose an open take to a closed window
        state["reader"].stop()

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", on_close)

    def update(_):
        reader = state["reader"]
        for node, src in sources.items():
            with src.lock:
                frame, baseline, take = src.latest, src.baseline, src.take
            if reader.error is not None:
                panels[node].ax.set_title(f"stream stopped: {reader.error}", color="red")
                continue
            if frame is None:
                continue
            values = np.asarray(frame.taxels, float)
            if baseline is not None and frame.mode == 0:
                values = np.clip(values - baseline, 0, None)
            note = f"{src.fps:4.0f} fps  peak {np.nanmax(values, initial=0):4.0f}"
            if frame.mode != 0:
                note += f"  read {int(frame.mask.sum())}/{len(values)}"
            elif baseline is not None:
                note += "  (zeroed)"
            if src.sensor.rebooted:
                note += "  BOARD RESET: session settings lost"
            if take is not None:
                note += f"  REC {time.time() - take.t_start:4.1f} s"
            panels[node].draw(values, state["autoscale"], note)
        return []

    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, update, interval=33, cache_frame_data=False)  # noqa: F841
    log("keys: b zero, c clear zero, a autoscale, s shear arrows, q quit"
        + (f", Enter start/stop a take -> {save}" if save else ""))
    try:
        plt.show()
    finally:
        on_close()
    return 0


def _replay_interval_ms(frames, speed):
    """Animation step in ms, from the take's own timestamps, scaled by `speed`."""
    if len(frames) < 2 or speed <= 0:
        return 33
    span = float(frames[-1].t - frames[0].t)
    if span <= 0:
        return 33
    dt = 1000.0 * span / (len(frames) - 1) / speed
    return max(1, min(200, int(round(dt))))


def replay(path, shear=False, vmax=4095.0, snapshot=None, frame=None, speed=1.0,
           log=print) -> int:
    """Play a recorded take in the same heatmap as `view`.

    `path` is a take folder (`tactile.npz`, `manifest.json`, `take.json`).
    `snapshot` draws one frame to a PNG and returns: `frame` selects it
    (0-based); the default is the frame with the strongest taxel. `speed` is
    playback against the recorded timestamps (1 is realtime).
    """
    if speed <= 0:
        raise ValueError(f"speed must be positive, got {speed}")
    take = load_take(path)
    frames = list(iter_frames(path))
    if not frames:
        raise ValueError(f"{path} holds no frames")
    n = len(frames)
    info = take["info"]
    name = info.get("take") or Path(path).name
    fps = info.get("fps")
    log(f"{name}: {n} frames, {take['manifest'].n} taxels"
        + (f", {fps:.0f} fps" if fps else ""))

    _, plt = _use_matplotlib(snapshot)
    fig, ax = plt.subplots(figsize=(6, 6.5))
    panel = Panel(ax, take["manifest"], name, vmax, shear)

    def note_for(i):
        fr = frames[i]
        peak = float(np.nanmax(fr.taxels, initial=0))
        line = f"{i + 1}/{n}  t={fr.t - frames[0].t:6.3f} s  peak {peak:4.0f}"
        if not fr.mask.all():
            line += f"  read {int(fr.mask.sum())}/{len(fr.taxels)}"
        return line

    if snapshot:
        i = peak_frame_index(path) if frame is None else frame
        if i < 0 or i >= n:
            raise ValueError(f"frame {i} is outside 0..{n - 1}")
        panel.draw(frames[i].taxels, note=note_for(i))
        fig.savefig(snapshot, dpi=120, bbox_inches="tight")
        log(f"saved {snapshot} (frame {i})")
        plt.close(fig)
        return 0

    state = {"i": 0, "paused": False, "autoscale": False}
    panel.draw(frames[0].taxels, note=note_for(0))

    def on_key(event):
        key = event.key
        if key == " ":
            state["paused"] = not state["paused"]
        elif key in ("left", "right", "home", "end"):
            state["paused"] = True
            if key == "left":
                state["i"] = max(0, state["i"] - 1)
            elif key == "right":
                state["i"] = min(n - 1, state["i"] + 1)
            elif key == "home":
                state["i"] = 0
            else:
                state["i"] = n - 1
            panel.draw(frames[state["i"]].taxels, state["autoscale"], note_for(state["i"]))
            fig.canvas.draw_idle()
        elif key == "a":
            state["autoscale"] = not state["autoscale"]
            panel.draw(frames[state["i"]].taxels, state["autoscale"], note_for(state["i"]))
            fig.canvas.draw_idle()
        elif key == "s":
            panel.show_shear(not panel.shear_on)
            panel.draw(frames[state["i"]].taxels, state["autoscale"], note_for(state["i"]))
            fig.canvas.draw_idle()

    def update(_):
        if not state["paused"] and state["i"] < n - 1:
            state["i"] += 1
        elif not state["paused"] and state["i"] >= n - 1:
            state["paused"] = True
        i = state["i"]
        extra = "  PAUSED" if state["paused"] else ""
        panel.draw(frames[i].taxels, state["autoscale"], note_for(i) + extra)
        return []

    fig.canvas.mpl_connect("key_press_event", on_key)
    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, update, interval=_replay_interval_ms(frames, speed),
                         cache_frame_data=False)  # noqa: F841
    log("keys: space pause, left/right step, home/end, a autoscale, s shear, q quit")
    plt.show()
    return 0
