"""Live heatmap of one or more sensors, each taxel drawn where it physically is.

    opentac view                 the board on USB, or every node of a bus controller
    opentac view --shear         with one shear arrow per bump
    opentac view --save takes/   Enter starts and stops a take
    opentac view --hdr           high dynamic range (60-pin board): zeroes at every gain
                                 first (hands off), then a gain map beside each heatmap
    opentac view --floor 30      draw taxels below 30 counts as 0 (after b: above the zero)

Keys (click the window first):
    b      zero: subtract the mean of the last frames (nothing touching). With
           xampling, the board measures a new zero instead.
    c      clear the zero
    a      autoscale on/off
    s      shear arrows and bump boxes on/off (skins with shear bumps)
    h      with --hdr: HDR on / off (one fixed gain), on the same scene
    Enter  start / stop a take (with --save)
    q      quit

With --hdr the colour scale is logarithmic, in counts above the zero at the highest
gain, so a light touch and a heavy press show on one scale; b zeroes at every gain.

Frames arrive in the manifest's `pixels` order, so value k is drawn at pixels[k]
with no reshaping. Xampling frames carry only the taxels read this frame; the
others are drawn in blue, below the colour scale.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from opentac import manifest as manifest_mod
from opentac.device.config import derive_pitch_um
from opentac.record import Take, session

BASELINE_FRAMES = 30
UNSAMPLED_COLOR = "#3a6ea5"
SHEAR_ARROW_COLOR = "#39ff6a"
BUMP_OUTLINE_COLOR = "cyan"
HDR_FLOOR = 10.0        # counts: the bottom of the HDR colour scale
GAIN_COLORS = ("#463480", "#2f6c8e", "#1e9c89", "#5ec962", "#dfe318")   # gain 1 ... the lowest
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


def _skin_cells(ax, manifest):
    """(drawn view, one Polygon per taxel, taxel w, h) for a skin, with its outline drawn
    into `ax` and the axes framed around it."""
    from matplotlib.patches import Polygon

    view = manifest_mod.drawn(manifest)
    pixels = view["pixels"]
    w = view.get("w") or derive_pitch_um(view)[0] / 1000.0 * 0.9
    h = view.get("h") or w
    tilt = np.radians(float(view.get("pixel_rot_deg", 0.0) or 0.0))
    c, s = np.cos(tilt), np.sin(tilt)

    def box(cx, cy, x0, y0, x1, y1):
        return Polygon([(cx + u * c - v * s, cy + u * s + v * c)
                        for u, v in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))], closed=True)

    cells = [box(p[0], p[1], -w / 2, -h / 2, w / 2, h / 2) for p in pixels]
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
    return view, cells, w, h


class Panel:
    """One skin drawn into one matplotlib axes.

    `hdr_top` (counts) switches to the HDR colour scale: logarithmic, from HDR_FLOOR
    to `hdr_top`, in counts above the zero at the highest gain.
    """

    def __init__(self, ax, manifest, title, vmax=4095.0, shear=False, colour=None, hdr_top=None):
        import matplotlib.pyplot as plt
        from matplotlib.collections import PatchCollection

        self.ax, self.title, self.vmax, self.hdr_top = ax, title, vmax, hdr_top
        self.colour = "#%02x%02x%02x" % colour if colour else "black"
        view, cells, w, h = _skin_cells(ax, manifest)
        if hdr_top is None:
            cmap = plt.get_cmap("inferno").with_extremes(under=UNSAMPLED_COLOR)
            self.taxels = PatchCollection(cells, cmap=cmap, edgecolor="none")
            self.taxels.set_array(np.zeros(len(cells)))
            self.taxels.set_clim(0, vmax)
            label = "counts"
        else:
            from matplotlib.colors import LogNorm
            base = plt.get_cmap("inferno")
            cmap = base.with_extremes(under=base(0.0), bad=UNSAMPLED_COLOR)   # bad: not read
            self.taxels = PatchCollection(cells, cmap=cmap, norm=LogNorm(HDR_FLOOR, hdr_top),
                                          edgecolor="none")
            self.taxels.set_array(np.full(len(cells), HDR_FLOOR))
            label = "counts above zero, at the highest gain (log)"
        ax.add_collection(self.taxels)
        ax.figure.colorbar(self.taxels, ax=ax, label=label)

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
        if self.hdr_top is not None:
            # read taxels below the scale take its bottom colour; NaN (not read) the blue
            self.taxels.set_array(np.where(np.isnan(values), np.nan,
                                           np.maximum(np.nan_to_num(values), HDR_FLOOR / 2)))
            top = self.hdr_top
            if autoscale:
                top = max(float(np.nanmax(values, initial=0)), HDR_FLOOR * 10)
            self.taxels.set_clim(HDR_FLOOR, top)
        else:
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


class GainPanel:
    """The gain each taxel was read at this frame (HDR), drawn like the skin beside it."""

    def __init__(self, ax, manifest, labels):
        from matplotlib.collections import PatchCollection
        from matplotlib.colors import BoundaryNorm, ListedColormap

        self.ax = ax
        _, cells, _, _ = _skin_cells(ax, manifest)
        n = len(labels)
        self.cells = PatchCollection(cells, cmap=ListedColormap(GAIN_COLORS[:n]).with_extremes(
                                         bad=UNSAMPLED_COLOR),
                                     norm=BoundaryNorm(np.arange(n + 1) - 0.5, n), edgecolor="none")
        self.cells.set_array(np.zeros(len(cells)))
        ax.add_collection(self.cells)
        bar = ax.figure.colorbar(self.cells, ax=ax, ticks=range(n), label="gain")
        bar.ax.set_yticklabels(labels)

    def draw(self, passes, note=""):
        self.cells.set_array(np.asarray(passes, dtype=float))
        self.ax.set_title(f"gain each taxel was read at\n{note}", fontsize=9)


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


def floored(values, floor):
    """Values below `floor` as 0 (NaN, a taxel not read, stays NaN); unchanged if floor <= 0."""
    values = np.asarray(values, float)
    return np.where(values < floor, 0.0, values) if floor > 0 else values


def view(sensors, bus=None, shear=False, vmax=4095.0, save=None, description="",
         snapshot=None, log=print, hdr=False, floor=0.0) -> int:
    """Show live heatmaps of `sensors` (Sensor objects holding skins with cached manifests).

    Pass `bus` when the sensors are nodes of one Bus. `save` is a folder for
    takes (Enter starts and stops one). `snapshot` saves one frame to a PNG and returns.
    `hdr` reads in HDR (Sensor.hdr(), which zeroes at every gain first unless it is
    already on), with a gain map beside each heatmap and h to switch HDR off and on.
    `floor` (counts, as drawn) draws taxels below it as 0, on screen only: takes keep
    every value.
    """
    import matplotlib
    if snapshot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 's' and 'c' are the viewer's shear and clear keys; matplotlib also binds
    # them by default to save and back, which would open a save dialog or step
    # the view back. Release them before the figure is made.
    for name, key in (("keymap.save", "s"), ("keymap.back", "c")) + ((("keymap.home", "h"),) if hdr else ()):
        matplotlib.rcParams[name] = [k for k in matplotlib.rcParams[name] if k != key]

    sensors = [s for s in sensors if s.manifest is not None]
    if not sensors:
        raise ValueError("no sensor here has a manifest on this computer: run "
                         "`opentac setup <manifest>` first")
    n = len(sensors)
    if hdr:                                # each sensor: its heatmap, then its gain map
        for s in sensors:
            if not s.hdr_on:
                s.hdr()
        fig, grid = plt.subplots(n, 2, figsize=(12, 6.5 * n), squeeze=False)
        axes, gain_axes = grid[:, 0], grid[:, 1]
    else:
        cols = min(n, 3)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6.5 * rows), squeeze=False)
        for ax in axes.flat[n:]:
            ax.set_visible(False)
        axes, gain_axes = axes.flat, [None] * n

    sources, panels, gain_panels = {}, {}, {}
    for k, (s, ax, gax) in enumerate(zip(sensors, axes, gain_axes)):
        colour = PALETTE[(s.node or 0) % len(PALETTE)] if bus is not None else None
        if bus is not None:                # the node's LED takes its panel's colour
            s.set("LED_MODE", "manual")
            s._command("c {} {} {}".format(*colour))
        label = (f"node {s.node} · " if s.node is not None else "") + (s.manifest.name or "skin")
        sources[s.node] = Source(s, colour)
        table = s._hdr_table if hdr else None
        panels[s.node] = Panel(ax, s.manifest, label, vmax, shear, colour,
                               hdr_top=None if table is None else 4095.0 / table.gain[-1])
        if table is not None:
            gain_panels[s.node] = GainPanel(gax, s.manifest, table.labels())
    contexts = {s.node: session(s) for s in sensors} if save else {}

    def hdr_note(src, frame):
        """(gain map's pass per taxel, note) for a frame in the HDR view."""
        table = src.sensor._hdr_table
        if frame.gain is not None:
            passes, _ = table.passes(frame.hdr_raw)
            saturated = int(table.saturated(frame.hdr_raw).sum())
            note = f"HDR: {int((passes > 0).sum())} re-read at lower gains"
        else:
            passes = np.zeros(len(frame.taxels))
            raw = np.nan_to_num(frame.taxels)
            if frame.mode != 0:                 # Xampling: above the board's zero
                raw = raw + table.zero[0]
            saturated = int((raw >= table.clip).sum())
            note = "HDR off: one fixed gain"
        passes = np.where(frame.mask, passes, np.nan)
        return passes, note + (f", {saturated} saturated" if saturated else "")

    if snapshot:
        for s in sensors:
            frame = s.read()
            values = floored(np.clip(frame.taxels, 0, None) if frame.gain is not None
                             else frame.taxels, floor)
            panels[s.node].draw(values, note=f"peak {np.nanmax(values):.0f}")
            if s.node in gain_panels:
                passes, note = hdr_note(sources[s.node], frame)
                gain_panels[s.node].draw(passes, note)
        fig.savefig(snapshot, dpi=120, bbox_inches="tight")
        log(f"saved {snapshot}")
        return 0

    state = {"autoscale": False, "reader": Reader(sources, bus), "hdr": bool(hdr)}
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

    def set_hdr(on, zero=False):
        # Commands and the reader share one serial stream: stop it first.
        state["reader"].stop()
        for src in sources.values():
            try:
                table = src.sensor.hdr(True, zero=zero)
                if not on:
                    src.sensor.hdr(False)
                # HDR off: counts above the zero at the highest gain, as in HDR, when the
                # fixed gain is that gain
                same = src.sensor.params["GAIN_SW"]["value"] == table.gain_sw[0]
                src.baseline = table.zero[0] if not on and same else None
            except Exception as exc:
                log(f"HDR change failed: {exc}")
        state["hdr"] = on
        state["reader"] = Reader(sources, bus)
        state["reader"].start()

    def on_key(event):
        key = event.key
        if key in ("enter", "return"):
            toggle_take()
        elif key == "h" and gain_panels:
            set_hdr(not state["hdr"])
            log("HDR on" if state["hdr"] else "HDR off: one fixed gain")
        elif key == "b" and gain_panels:
            log("zeroing at every gain: nothing may touch the skin...")
            set_hdr(state["hdr"], zero=True)
            log("zeroed")
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
            if frame.gain is not None:          # HDR: already above the zero at every gain
                values = np.clip(values, 0, None)
            elif baseline is not None and frame.mode == 0:
                values = np.clip(values - baseline, 0, None)
            values = floored(values, floor)
            note = f"{src.fps:4.0f} fps  peak {np.nanmax(values, initial=0):4.0f}"
            if node in gain_panels:
                passes, hdr_text = hdr_note(src, frame)
                gain_panels[node].draw(passes, hdr_text)
            if frame.mode != 0:
                note += f"  read {int(frame.mask.sum())}/{len(values)}"
            elif baseline is not None and frame.gain is None and node not in gain_panels:
                note += "  (zeroed)"
            if src.sensor.rebooted:
                note += "  BOARD RESET: session settings lost"
            if take is not None:
                note += f"  REC {time.time() - take.t_start:4.1f} s"
            if floor > 0:
                note += f"  floor {floor:g}"
            panels[node].draw(values, state["autoscale"], note)
        return []

    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, update, interval=33, cache_frame_data=False)  # noqa: F841
    log("keys: b zero, c clear zero, a autoscale, s shear arrows, q quit"
        + (", h HDR on/off (b zeroes at every gain: nothing touching)" if gain_panels else "")
        + (f", Enter start/stop a take -> {save}" if save else ""))
    try:
        plt.show()
    finally:
        on_close()
    return 0
