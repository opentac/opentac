"""Manifest -> DeviceConfig: the bytes a readout board stores to know its skin.

The board needs only what it takes to scan: array size, which connector line
each row and column reaches, which taxels exist, the pitch, and which taxels
share a shear bump. Everything else in the manifest stays on the computer.
The byte layout is documented in firmware/opentac_readout/ot_devcfg.h; this is
the only other place that knows it.

Which line a manifest pad reaches belongs to the readout board, not the skin,
so it lives in one data file per board (boards/*.json) and every skin goes
through it.
"""
from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

from opentac.manifest import ManifestError

MAGIC = b"OTC1"
VERSION = 2             # ot_devcfg.h OTCFG_VERSION: row/col orders are line codes
HEADER_BYTES = 28
MAX_ROWS = 32
MAX_COLS = 32
NO_SHEAR = 0xFFFF
NO_POSITION = 0xFF
MUX = 32                # line codes 0..31 are shift-register steps, 32..63 mux channels

BOARDS_DIR = Path(__file__).with_name("boards")


@dataclass
class Board:
    """One readout board: its name as the firmware reports it, and its connector."""

    name: str
    title: str
    pad_line: dict[int, int]          # manifest pad -> line code
    hardware_checked: bool
    checked: str
    gain: dict | None = None
    raw: dict = field(default_factory=dict, repr=False)

    def line(self, pad: int) -> int:
        try:
            return self.pad_line[int(pad)]
        except KeyError:
            pads = sorted(self.pad_line)
            raise ManifestError(
                f"pad {pad} is not on the {self.name} connector (it carries pads "
                f"{pads[0]}..{pads[-1]}): this skin was made for a different readout board"
            ) from None


def boards() -> dict[str, Board]:
    """Every board this package knows, by the name the firmware reports in `?`."""
    out = {}
    for path in sorted(BOARDS_DIR.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        out[raw["board"]] = Board(
            name=raw["board"], title=raw["title"],
            pad_line={int(p[0]): int(p[3]) for p in raw["pads"]},
            hardware_checked=bool(raw.get("hardware_checked")),
            checked=raw.get("checked", ""), gain=raw.get("gain"), raw=raw)
    return out


#: What people call each board -> the name its firmware reports. The docs say
#: "60-pin board" and "30-pin pair"; the firmware and the CLI name them
#: READOUT60 / readout60 and NODE16 / node16. READOUT32 is the earlier 70-pin board.
ALIASES = {
    "readout60": "READOUT60", "60pin": "READOUT60", "60-pin": "READOUT60",
    "node16": "NODE16", "30pin": "NODE16", "30-pin": "NODE16",
    "readout32": "READOUT32", "70pin": "READOUT32", "70-pin": "READOUT32",
}

#: How the documentation names each board (https://opentac.github.io/hardware/).
LABELS = {"READOUT60": "60-pin board", "NODE16": "30-pin pair",
          "READOUT32": "earlier 70-pin board"}

#: The boards new sensors are made for; READOUT32 is still read, but not offered.
CURRENT = ("READOUT60", "NODE16")


def canonical(name: str) -> str:
    """The firmware's name for a board: '60pin', '60-pin', 'readout60' -> 'READOUT60'.

    Raises ManifestError for a name no board goes by.
    """
    key = str(name).strip().lower()
    if key in ALIASES:
        return ALIASES[key]
    for known in boards():            # a board added as boards/*.json, by its own name
        if known.lower() == key:
            return known
    raise ManifestError(f"no board called {name!r}: use readout60 or 60pin (the 60-pin "
                        f"board), node16 or 30pin (the 30-pin pair), or readout32 or 70pin "
                        f"(the earlier 70-pin board)")


def label(name: str) -> str:
    """'READOUT60' -> '60-pin board (READOUT60)', the way messages name a board."""
    return f"{LABELS[name]} ({name})" if name in LABELS else name


def board(name: str) -> Board:
    """A board by any of its names, case-insensitive: 'READOUT60', 'node16', '60pin', '30-pin'."""
    known = boards()
    key = canonical(name)
    if key not in known:
        raise ManifestError(f"no wiring table for board {key!r}; known: {', '.join(known)}")
    return known[key]


def pitch_mm(manifest: dict) -> tuple[float, float]:
    """Taxel pitch in mm, x and y, unrounded: the smallest gap between distinct
    coordinates in `pixels` (see derive_pitch_um)."""
    pixels = manifest["pixels"]
    xs = sorted({round(p[0], 6) for p in pixels})
    ys = sorted({round(p[1], 6) for p in pixels})

    def smallest_step(values, axis):
        if len(values) < 2:
            raise ManifestError(f"cannot derive the {axis} pitch: only one distinct {axis} in pixels")
        return min(b - a for a, b in zip(values, values[1:]))

    return smallest_step(xs, "x"), smallest_step(ys, "y")


def derive_pitch_um(manifest: dict, warnings: list | None = None) -> tuple[int, int]:
    """Taxel pitch in micrometres, x and y, from `pixels`.

    Row index is pad order, not geometry, so ox/oy/px/py do not place taxels.
    What the permutation does not change is the spacing, so the pitch is the
    smallest gap between distinct coordinates; |px|/|py| and
    design.params.pitch_*_mm only check that answer.
    """
    pitch_x_mm, pitch_y_mm = pitch_mm(manifest)
    notes = warnings if warnings is not None else []
    for axis, derived, declared in (("x", pitch_x_mm, abs(manifest.get("px", 0.0))),
                                    ("y", pitch_y_mm, abs(manifest.get("py", 0.0)))):
        if declared and abs(derived - declared) > 1e-4:
            notes.append(f"pitch_{axis} from pixels is {derived:.4f} mm but "
                         f"|{'px' if axis == 'x' else 'py'}| is {declared:.4f} mm; using the pixels value")
    params = manifest.get("design", {}).get("params", {})
    for axis, derived, key in (("x", pitch_x_mm, "pitch_x_mm"), ("y", pitch_y_mm, "pitch_y_mm")):
        declared = params.get(key)
        if declared and abs(derived - declared) > 1e-4:
            notes.append(f"pitch_{axis} from pixels is {derived:.4f} mm but design.params.{key} "
                         f"is {declared:.4f} mm; using the pixels value")
    return round(pitch_x_mm * 1000.0), round(pitch_y_mm * 1000.0)


def _fit_line(pairs):
    """Least-squares (origin, pitch) through (grid index, coordinate) pairs; None for one index."""
    n = len(pairs)
    sk = sum(k for k, _ in pairs)
    sv = sum(v for _, v in pairs)
    skk = sum(k * k for k, _ in pairs)
    skv = sum(k * v for k, v in pairs)
    den = n * skk - sk * sk
    if den == 0:
        return None
    pitch = (n * skv - sk * sv) / den
    return (sv - pitch * sk) / n, pitch


def derive_geometry(manifest, rows, cols, pitch_x_mm, pitch_y_mm):
    """Logical row/col -> grid position (top to bottom, left to right).

    From the lattice, not from rank: an empty logical row leaves a real gap, and
    ranking what is left would close it. Checked by rebuilding every pixel.

    The pitch is the exact one from pitch_mm(), not the whole micrometres the
    board stores. Coordinates rounded to 0.1 um can also make the smallest step
    a little short of the true pitch (2.6936 against 2.69368 mm on the 60-pin
    insole, 2.3 um short by row 29), so the rebuild is also checked against a
    least-squares fit of the grid, and a grid that fits either within 1 um is
    regular.
    """
    ys, xs = {}, {}
    for x, y, r, c in (p[:4] for p in manifest["pixels"]):
        ys.setdefault(r, round(y, 6))
        xs.setdefault(c, round(x, 6))
    origin_x, origin_y = min(xs.values()), max(ys.values())

    def grid_index(value, origin, pitch, axis, index):
        steps = (value - origin) / pitch
        nearest = round(steps)
        if abs(steps - nearest) > 0.25:
            raise ManifestError(
                f"{axis} {index} sits {steps:.4f} pitches from the grid origin, not a whole "
                f"number: the taxels are not on one regular grid, so the board cannot place them")
        return nearest

    row_geom = [grid_index(ys[r], origin_y, -pitch_y_mm, "row", r) if r in ys else NO_POSITION
                for r in range(rows)]
    col_geom = [grid_index(xs[c], origin_x, pitch_x_mm, "col", c) if c in xs else NO_POSITION
                for c in range(cols)]
    for axis, geom, extent in (("row", row_geom, rows), ("col", col_geom, cols)):
        placed = [g for g in geom if g != NO_POSITION]
        if len(set(placed)) != len(placed):
            raise ManifestError(f"two logical {axis}s share one grid position")
        if placed and max(placed) >= extent:
            raise ManifestError(f"{axis} positions reach {max(placed)} but the grid is only "
                                f"{extent} {axis}s: pixels span more grid than r/c declares")

    # Each axis must rebuild every pixel within 1 um, from the smallest step or from the fit.
    pix = [p[:4] for p in manifest["pixels"]]
    for axis, geom, coord, origin, pitch in (("x", col_geom, 0, origin_x, pitch_x_mm),
                                             ("y", row_geom, 1, origin_y, -pitch_y_mm)):
        index = 3 if axis == "x" else 2
        pairs = [(geom[p[index]], p[coord]) for p in pix]
        lines = [(origin, pitch)]
        fit = _fit_line(pairs)
        if fit is not None:
            lines.append(fit)
        worst = [max(abs(o + k * s - v) for k, v in pairs) for o, s in lines]
        if min(worst) > 1e-3:
            k = max(range(len(pairs)), key=lambda i: abs(origin + pairs[i][0] * pitch - pairs[i][1]))
            raise ManifestError(f"grid index and pitch do not reproduce pixels[{k}]: {axis} is "
                                f"off the grid by {min(worst) * 1000:.1f} um")
    return row_geom, col_geom


def extract(manifest: dict, on: Board) -> dict:
    """The acquisition fields of a manifest, laid onto one board's connector.

    Raises ManifestError when the skin cannot be read on that board. Soft
    disagreements go in the result's "warnings".
    """
    warnings: list[str] = []
    try:
        rows, cols = manifest["r"], manifest["c"]
        mask_rows, pixels = manifest["m"], manifest["pixels"]
        rp, cp = manifest["rp"], manifest["cp"]
    except KeyError as exc:
        raise ManifestError(f"manifest has no {exc.args[0]!r}: it cannot be scanned") from None
    if not 1 <= rows <= MAX_ROWS:
        raise ManifestError(f"r = {rows}, outside 1..{MAX_ROWS}")
    if not 1 <= cols <= MAX_COLS:
        raise ManifestError(f"c = {cols}, outside 1..{MAX_COLS}")
    if len(mask_rows) != rows:
        raise ManifestError(f"m has {len(mask_rows)} rows, r says {rows}")
    for i, row in enumerate(mask_rows):
        if len(row) != cols:
            raise ManifestError(f"m[{i}] is {len(row)} wide, c says {cols}")
        if set(row) - {"0", "1"}:
            raise ManifestError(f"m[{i}] has characters other than 0/1")
    active = [(r, c) for r in range(rows) for c in range(cols) if mask_rows[r][c] == "1"]
    if len(active) != len(pixels):
        raise ManifestError(f"m has {len(active)} active taxels but pixels has {len(pixels)}")
    # The firmware emits taxels row-major over the mask; that must be pixels order.
    for k, (px, (r, c)) in enumerate(zip(pixels, active)):
        if (px[2], px[3]) != (r, c):
            raise ManifestError(
                f"pixels[{k}] is at (row {px[2]}, col {px[3]}) but the row-major walk of m "
                f"expects (row {r}, col {c}): pixels is not in row-major order")
    if len(rp) != rows:
        raise ManifestError(f"rp has {len(rp)} entries, r says {rows}")
    if len(cp) != cols:
        raise ManifestError(f"cp has {len(cp)} entries, c says {cols}")
    row_order = [on.line(p) for p in rp]
    col_order = [on.line(p) for p in cp]
    lines = row_order + col_order
    if len(set(lines)) != len(lines):
        raise ManifestError("two rows/columns land on the same connector line")
    # One line of each taxel must be driven and the other sensed.
    unreadable = [(r, c) for r, c in active if (row_order[r] >= MUX) == (col_order[c] >= MUX)]
    if unreadable:
        r, c = unreadable[0]
        side = "mux" if row_order[r] >= MUX else "shift register"
        raise ManifestError(
            f"{len(unreadable)} of {len(active)} taxels cannot be read on {on.name}: both of "
            f"their lines land on the {side} (e.g. row {r} on pad {rp[r]} and column {c} on "
            f"pad {cp[c]}). This skin was laid out for a different readout board")

    pitch_x_um, pitch_y_um = derive_pitch_um(manifest, warnings)
    row_geom, col_geom = derive_geometry(manifest, rows, cols, *pitch_mm(manifest))

    shear = manifest.get("shear")
    shear_groups = None
    if shear is not None and "b" in shear:
        b = shear["b"]
        if len(b) != len(pixels):
            raise ManifestError(f"shear.b has {len(b)} entries, pixels has {len(pixels)}")
        bump_count = len(shear.get("bumps", []))
        shear_groups = []
        for k, group in enumerate(b):
            if group == -1:
                shear_groups.append(NO_SHEAR)
                continue
            if bump_count and not 0 <= group < bump_count:
                raise ManifestError(f"shear.b[{k}] = {group}, outside 0..{bump_count - 1}")
            if group >= NO_SHEAR:
                raise ManifestError(f"shear.b[{k}] = {group}, too large to encode")
            shear_groups.append(group)

    return {"board": on.name, "rows": rows, "cols": cols, "row_order": row_order,
            "col_order": col_order, "row_geom": row_geom, "col_geom": col_geom,
            "active": active, "pitch_x_um": pitch_x_um, "pitch_y_um": pitch_y_um,
            "shear_groups": shear_groups, "warnings": warnings}


def build_config(spec: dict, config_id: int) -> bytes:
    """The DeviceConfig bytes for an extract()ed spec, CRC included."""
    rows, cols = spec["rows"], spec["cols"]
    mask = bytearray((rows * cols + 7) // 8)
    for r, c in spec["active"]:
        bit = r * cols + c
        mask[bit >> 3] |= 1 << (bit & 7)
    shear = spec["shear_groups"]
    payload = (bytes(spec["row_order"]) + bytes(spec["col_order"]) +
               bytes(spec["row_geom"]) + bytes(spec["col_geom"]) + bytes(mask))
    if shear:
        payload += struct.pack(f"<{len(shear)}H", *shear)
    header = struct.pack("<4sHBBIIHHII", MAGIC, VERSION, rows, cols,
                         spec["pitch_x_um"], spec["pitch_y_um"], len(spec["active"]),
                         len(shear) if shear else 0, config_id, len(payload))
    assert len(header) == HEADER_BYTES, len(header)
    body = header + payload
    return body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)


def config_id(manifest_bytes: bytes) -> int:
    """The id a board stores for a skin: the CRC-32 of the manifest file's bytes."""
    return zlib.crc32(manifest_bytes) & 0xFFFFFFFF


def device_config(manifest, on: Board | str) -> tuple[bytes, dict]:
    """(config bytes, spec) for a manifest (anything opentac.manifest.load takes) on a board."""
    from opentac.manifest import load
    m = load(manifest)
    on = board(on) if isinstance(on, str) else on
    spec = extract(m.raw, on)
    return build_config(spec, config_id(m.source_bytes)), spec


def fits(manifest) -> dict[str, str | None]:
    """Board name -> None if the skin can be read on it, else the reason it cannot."""
    from opentac.manifest import load
    m = load(manifest)
    out = {}
    for name, b in boards().items():
        try:
            extract(m.raw, b)
            out[name] = None
        except ManifestError as exc:
            out[name] = str(exc)
    return out


def infer_board(manifest) -> str | None:
    """The one board a skin is made for, from fits(); None if that is not one board.

    The earlier 70-pin board counts only when no current board fits.
    """
    ok = [name for name, reason in fits(manifest).items() if reason is None]
    current = [name for name in ok if name in CURRENT]
    if len(current) == 1:
        return current[0]
    if not current and len(ok) == 1:
        return ok[0]
    return None


def describe_lines(codes) -> str:
    """[31, 30, ..., 18] -> 'shift-register steps 31..18'."""
    sr = [c for c in codes if c < MUX]
    mux = [c - MUX for c in codes if c >= MUX]
    parts = []
    if sr:
        parts.append(f"shift-register steps {sr[0]}..{sr[-1]}")
    if mux:
        parts.append(f"mux channels {mux[0]}..{mux[-1]}")
    return " + ".join(parts)
