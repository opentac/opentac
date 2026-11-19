"""What the back silkscreen says: a taxel-core QR, and a registration grid.

`silk_label` puts the design name on the front, inboard of the connector, so a
fabricated board says which design it is. This is the other half of the same
problem: a board that has been separated from its manifest entirely.

Two marks, and they answer different questions.

The QR carries the taxel core (see `export/taxel_core.py`): grid dimensions,
both pitches, the grid origin, taxel size, the interdigitated footprint
parameters, the connector pad number for every row and column net, the
active-taxel mask, and the pad-order permutation. That is typically under 100
bytes, which is a QR small enough to print, and it is everything needed to
read a sensor -- the scan order, where each taxel physically sits, and which
pad drives it.

It prints as a link, HTTPS://OPENTAC.GITHUB.IO/Q/<payload> (taxel_core.qr_link),
so a phone camera opens the website's decoder page. The link is 28 characters
longer than the payload, which usually costs one QR version, 1.4 mm a side
per version. A board too small for it gets the bare payload,
which a phone shows as text for the decoder page or `opentac manifest
from-qr`; a board too small for that gets no QR. The code is decoded and
checked against the manifest before it is printed. A board the regular grid
cannot describe (unevenly spaced columns, columns numbered out of order) gets
its columns laid out by order, approximately spaced, with a warning; a board
that cannot be described even so gets no QR and a warning.

The grid carries the board's shape, by making the photograph measurable. The
outline is deliberately not in the QR (it is 80-95% of the payload, and the
physical board already has it), so it comes from a picture instead. A flex
sensor is the hard case: it is almost never flat when photographed, so a
single scale factor -- or four corner fiducials -- recovers the outline wrong
exactly where the board curves. A grid samples the whole surface, so the photo
can be unwarped locally.

The grid sits on the manifest coordinate frame with its origin at (0, 0), not
on the board's bounding box, so an intersection read off a photo is a taxel
coordinate. That is most of the value and it costs nothing.

Mirroring. Back-layer artwork is stored in board (top-view) coordinates in
both KiCad and Gerber; the fab mirrors it at production. So anything here
appears mirrored when you look at the physical back. The two marks want
opposite things about that:

  * The QR is chiral. A mirrored QR is rejected by most decoders, so it is
    pre-mirrored about its own centre and therefore reads correctly off the
    physical back. In a top-view editor it looks backwards, as intended.

  * The grid must not be mirrored. A grid line at x = 10 mm is physically at
    x = 10 mm; seeing it flipped from the back is a fact about which side you
    are standing on, not an error to correct. Mirroring it would make the
    printed coordinates disagree with the board's real geometry.

The origin L-marker discloses the handedness, so a pipeline reading a back
photo can detect the flip and account for it.

Sizing is pinned to JLCPCB's flex silkscreen rules, which are the binding
constraint (they are the same as their rigid limits; flex silkscreen is not a
special order):

    line width      >= 0.15 mm hard, 0.20 mm recommended
    hollow gap      >  0.25 mm   -- and a knocked-out QR is a hollow feature
    auto-widening   lines under the minimum are silently widened, which would
                    destroy the pre-compensation below without any warning

At MODULE_MM = 0.35 with PRECOMP_MM = 0.05 the thinnest ink feature is 0.25 mm
and the smallest knockout is 0.45 mm, which clears every recommended limit.
0.30 mm sits exactly on the recommended ink minimum; 0.25 mm sits on the hard
one, where auto-widening lives.

Polarity. Silkscreen is light ink on dark coverlay, so drawing ink where the QR
is dark produces an inverted code, and decoder support for those is
inconsistent. The ink is therefore the light field and the dark modules are
knocked out as bare coverlay, giving an ordinary dark-on-light code. KiCad
unions overlapping silk and cannot subtract, so that knockout is built as
horizontal runs of ink rather than as "rectangle minus modules".

Pre-compensation. Screen printing spreads ink by a roughly fixed amount, which
eats the knockouts. Shrinking the ink by that amount at design time cancels it
-- ordinary screen-print practice, and worth about 2x in tolerable spread.
PRECOMP_MM, the amount removed from every ink edge, is an estimate of that
spread rather than a value calibrated on fabricated boards.
"""

import json
import math

import numpy as np
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import polylabel

from opentac.cad.export.taxel_core import (
    TaxelCoreError, b41encode, check as check_core, check_ordered, encode as encode_core, qr_link)

# Geometry, all mm. See the sizing note above before changing MODULE_MM.
MODULE_MM = 0.35
PRECOMP_MM = 0.05
QUIET_MODULES = 4          # QR spec minimum
QR_CLEAR_MM = 0.8          # board-level breathing room around the quiet zone

GRID_MINOR_MM = 5.0
GRID_MAJOR_EVERY = 2       # -> a heavier line every 10 mm
GRID_W_MINOR_MM = 0.15     # JLCPCB hard minimum; thinner gets auto-widened
GRID_W_MAJOR_MM = 0.25
GRID_MIN_SEG_MM = 1.0      # shorter stubs are noise, not registration
ORIGIN_ARM_MM = 2.0
EDGE_CLEAR_MM = 0.2        # silk this close to the cut line is a print defect

# The grid scale, spelled out. A photograph of a grid does not say whether the
# fine spacing is 5 mm or 10 mm, and guessing wrong scales the whole recovered
# outline by 2x -- so the board states it.
#
# Just the text, sitting inside one cell. The grid lines are the dimension; the
# label only has to say which number goes with them, and a cell it sits inside
# is unambiguous about which spacing it means. Anything more -- a bar, end ticks
# -- duplicates what the lines already do and lines up with nothing.
DIM_TEXT_H_MM = 1.2        # JLCPCB flex minimum is 1.0; 1.2 leaves margin
DIM_STROKE_W_MM = 0.2      # their recommended silkscreen width, not the 0.15 floor
DIM_CELL_MARGIN_MM = 0.6   # text to cell wall

SUBDIV = 16                # sub-cells per module when pre-compensating

# What the QR says, in order of preference: the link, then the bare payload for
# a board too small for the link. See the module docstring.
QR_FORMS = ("link", "bare")


# --------------------------------------------------------------------------
# QR artwork
# --------------------------------------------------------------------------

def _qr_matrix(payload):
    """Dark-module matrix including the quiet zone. Returns None if segno is
    missing, so a board still exports -- without a back mark -- rather than
    failing the request over an optional feature."""
    try:
        import segno
    except ImportError:
        return None, None
    qr = segno.make(payload, error="M", mode="alphanumeric", micro=False)
    rows = [list(r) for r in qr.matrix]
    n = len(rows)
    size = n + 2 * QUIET_MODULES
    out = np.zeros((size, size), dtype=bool)
    for y in range(n):
        for x in range(n):
            out[y + QUIET_MODULES, x + QUIET_MODULES] = bool(rows[y][x])
    return out, qr.version


def _erode(mask, d):
    """Erode a boolean region by d cells (square structuring element).

    Padded rather than rolled, so the outer edge of the ink field erodes
    inward like every other boundary instead of wrapping around.
    """
    padded = np.zeros((mask.shape[0] + 2 * d, mask.shape[1] + 2 * d), dtype=bool)
    padded[d:-d, d:-d] = mask
    out = np.ones_like(mask)
    for dy in range(2 * d + 1):
        for dx in range(2 * d + 1):
            out &= padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
    return out


def _ink_runs(dark, precomp_modules):
    """Ink spans as (y0, y1, x0, x1) in module units.

    With pre-compensation the run edges do not land on module boundaries, so
    the field is rasterised at SUBDIV per module, eroded, cut into runs, and
    vertically identical neighbours merged back -- which collapses the sub-rows
    to the two or three bands per module row that actually differ.
    """
    if precomp_modules <= 0:
        grid, s = ~dark, 1
    else:
        s = SUBDIV
        big = np.kron(~dark, np.ones((s, s), dtype=bool))
        grid = _erode(big, max(1, int(round(precomp_modules * s))))

    per_row = []
    h, w = grid.shape
    for y in range(h):
        spans, x = [], 0
        while x < w:
            if not grid[y, x]:
                x += 1
                continue
            x0 = x
            while x < w and grid[y, x]:
                x += 1
            spans.append((x0, x))
        per_row.append(spans)

    out, y = [], 0
    while y < h:
        y2 = y + 1
        while y2 < h and per_row[y2] == per_row[y]:
            y2 += 1
        for x0, x1 in per_row[y]:
            out.append((y / s, y2 / s, x0 / s, x1 / s))
        y = y2
    return out


def _qr_text(manifest, form, ordered=False):
    """(blob, payload, text): the taxel core, its base41 payload, and what a
    QR of this form prints. ordered: columns laid out by order (see
    taxel_core._by_order)."""
    blob = encode_core(manifest, by_order=ordered)
    payload = b41encode(blob)
    return blob, payload, (qr_link(payload) if form == "link" else payload)


def qr_rects(manifest, center, form="link", ordered=False):
    """QR ink rectangles in board coords (y-up), pre-mirrored in X.

    `form` is "link" or "bare" (QR_FORMS). Returns (rects, info) where info
    carries the printed text, version, module count and size so the caller can
    report what it placed.
    """
    try:
        blob, payload, text = _qr_text(manifest, form, ordered)
        dark, version = _qr_matrix(text)
    except Exception:
        return [], None
    if dark is None:
        return [], None
    n = dark.shape[0]
    side = n * MODULE_MM
    cx, cy = center
    x0, y1 = cx - side / 2.0, cy + side / 2.0

    rects = []
    for a, b, c, d in _ink_runs(dark, PRECOMP_MM / MODULE_MM):
        # X mirrored about the QR centre: back-layer artwork is stored in
        # top-view coordinates, so this is what makes it read off the
        # physical back. See the module docstring.
        rx0 = x0 + (n - d) * MODULE_MM
        rx1 = x0 + (n - c) * MODULE_MM
        ry1 = y1 - a * MODULE_MM
        ry0 = y1 - b * MODULE_MM
        rects.append((rx0, ry0, rx1, ry1))
    return rects, {
        "form": form, "text": text, "payload": payload, "bytes": len(blob),
        "version": version, "modules": n, "size_mm": side,
        "module_mm": MODULE_MM, "precomp_mm": PRECOMP_MM,
        "center": (cx, cy),
    }


# --------------------------------------------------------------------------
# registration grid
# --------------------------------------------------------------------------

def _inset(edge_cuts, half_width):
    """The board, shrunk by half a line width plus edge clearance.

    Clipping the line's centreline to the raw outline and then widening it is
    wrong, and wrong in a way that only shows up at the boundary: the rect's
    corners overhang the cut line wherever the edge is sloped or curved.
    Insetting first means a span that survives is one where the full-width
    line actually fits.

    Returns None when nothing survives -- a board narrower than a grid line.
    """
    poly = Polygon([(float(x), float(y)) for x, y in edge_cuts])
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return None
    inner = poly.buffer(-(half_width + EDGE_CLEAR_MM))
    return None if inner.is_empty else inner


def _spans_along(inner, coord, axis, lo, hi):
    """Spans of the line {axis == coord} that lie inside `inner`.

    Shapely does the clipping, so concave outlines and the curved grown edge
    both work: a line that crosses a gap between two parts of the board comes
    out as separate spans, which a bounding-box clip would not manage.
    """
    line = (LineString([(coord, lo), (coord, hi)]) if axis == 0
            else LineString([(lo, coord), (hi, coord)]))
    hit = line.intersection(inner)
    if hit.is_empty:
        return []
    parts = getattr(hit, "geoms", [hit])
    out = []
    for part in parts:
        if part.geom_type != "LineString" or part.is_empty:
            continue
        xs, ys = part.xy
        vals = ys if axis == 0 else xs
        out.append((min(vals), max(vals)))
    return sorted(out)


def _subtract(span, holes):
    out = [span]
    for h0, h1 in holes:
        nxt = []
        for s0, s1 in out:
            if h1 <= s0 or h0 >= s1:
                nxt.append((s0, s1))
                continue
            if s0 < h0:
                nxt.append((s0, h0))
            if h1 < s1:
                nxt.append((h1, s1))
        out = nxt
    return out


def grid_rects(edge_cuts, keepouts=()):
    """Registration-grid rectangles in board coords (y-up). Not mirrored."""
    if len(edge_cuts) < 3:
        return []
    xs = [float(p[0]) for p in edge_cuts]
    ys = [float(p[1]) for p in edge_cuts]
    lo_x, hi_x, lo_y, hi_y = min(xs), max(xs), min(ys), max(ys)

    # One inset per line width, not per line: buffer() is the expensive call.
    inner = {}
    for w in (GRID_W_MINOR_MM, GRID_W_MAJOR_MM):
        inner[w] = _inset(edge_cuts, w / 2.0)

    rects = []
    for axis, lo, hi, span_lo, span_hi in (
            (0, lo_x, hi_x, lo_y, hi_y), (1, lo_y, hi_y, lo_x, hi_x)):
        for k in range(int(math.floor(lo / GRID_MINOR_MM)),
                       int(math.ceil(hi / GRID_MINOR_MM)) + 1):
            coord = k * GRID_MINOR_MM
            if not (lo <= coord <= hi):
                continue
            w = (GRID_W_MAJOR_MM if k % GRID_MAJOR_EVERY == 0 else GRID_W_MINOR_MM)
            if inner[w] is None:
                continue
            half = w / 2.0
            holes = []
            for bx0, by0, bx1, by1 in keepouts:
                if axis == 0 and bx0 - half <= coord <= bx1 + half:
                    holes.append((by0, by1))
                elif axis == 1 and by0 - half <= coord <= by1 + half:
                    holes.append((bx0, bx1))
            holes.sort()
            for span in _spans_along(inner[w], coord, axis, span_lo, span_hi):
                for s0, s1 in _subtract(span, holes):
                    if s1 - s0 < GRID_MIN_SEG_MM:
                        continue
                    if axis == 0:
                        rects.append((coord - half, s0, coord + half, s1))
                    else:
                        rects.append((s0, coord - half, s1, coord + half))
    return rects


def dim_label(edge_cuts, qr_center, qr_size_mm):
    """The pitch, as text centred in one grid cell. Returns (polylines, bbox).

    The cell is chosen, not computed from the QR: the label has to sit inside
    the lines it describes, so it is placed on the same lattice they are. The
    nearest fully-enclosed cell to the QR wins, which keeps it findable without
    needing a keep-out -- text inside a cell never crosses a grid line.

    Mirrored in X like the QR, and for the same reason: text is chiral and this
    is read off the physical back. The grid itself is not mirrored.
    """
    from opentac.cad.export.stroke_font import text_strokes, text_width

    pitch = GRID_MINOR_MM
    if abs(pitch - round(pitch)) > 1e-9:
        return [], None          # the stroke font has no decimal point
    text = "%dMM" % round(pitch)

    tw = text_width(text, DIM_TEXT_H_MM)
    if tw > pitch - 2 * DIM_CELL_MARGIN_MM:
        return [], None          # would touch the cell walls

    inner = _inset(edge_cuts, DIM_STROKE_W_MM / 2.0)
    if inner is None:
        return [], None

    xs = [float(p[0]) for p in edge_cuts]
    ys = [float(p[1]) for p in edge_cuts]
    anchor = qr_center or ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)
    qr_half = (qr_size_mm / 2.0 + QR_CLEAR_MM) if qr_size_mm else 0.0

    cells = []
    for i in range(int(math.floor(min(xs) / pitch)), int(math.ceil(max(xs) / pitch))):
        for j in range(int(math.floor(min(ys) / pitch)), int(math.ceil(max(ys) / pitch))):
            cx, cy = (i + 0.5) * pitch, (j + 0.5) * pitch
            if (abs(cx - anchor[0]) < qr_half + pitch / 2.0
                    and abs(cy - anchor[1]) < qr_half + pitch / 2.0):
                continue         # under the QR or its quiet zone
            if not inner.contains(box(i * pitch, j * pitch,
                                      (i + 1) * pitch, (j + 1) * pitch)):
                continue         # cell is clipped by the board edge
            cells.append(((cx - anchor[0]) ** 2 + (cy - anchor[1]) ** 2, cx, cy))
    if not cells:
        return [], None

    _, cx, cy = min(cells)
    x0 = cx - tw / 2.0
    y0 = cy - DIM_TEXT_H_MM / 2.0
    placed = [[(x0 + (tw - x), y0 + y) for x, y in poly]      # mirrored in X
              for poly in text_strokes(text, DIM_TEXT_H_MM)]

    pad = DIM_STROKE_W_MM / 2.0
    pts = [p for poly in placed for p in poly]
    bbox = (min(p[0] for p in pts) - pad, min(p[1] for p in pts) - pad,
            max(p[0] for p in pts) + pad, max(p[1] for p in pts) + pad)
    return placed, bbox


def origin_rects(edge_cuts):
    """An L at manifest (0, 0): the origin, both axis directions, and -- because
    it is chiral -- which side of the board you are looking at.

    Dropped entirely when it does not fit inside the board. The manifest origin
    is a coordinate, not a guaranteed location: nothing says the outline the
    user drew contains it.
    """
    h = GRID_W_MAJOR_MM / 2.0
    a = GRID_W_MAJOR_MM * 1.6
    rects = [(-h, -h, ORIGIN_ARM_MM, h),
             (-h, -h, h, ORIGIN_ARM_MM),
             (-a, -a, a, a)]
    inner = _inset(edge_cuts, 0.0)
    if inner is None:
        return []
    return [r for r in rects if inner.contains(box(*r))]


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def _qr_center(edge_cuts, tcx, tcy, size):
    """Where the QR goes.

    First choice is the centre of the taxel field: every board has one, it is
    where a reader looks, and it keeps placement predictable across a family of
    designs. But it is not guaranteed to be on the board -- the U-notch test
    case puts that point squarely inside the notch -- so it is checked, not
    assumed.

    The fallback is the pole of inaccessibility: the interior point furthest
    from any edge, which is the best place a square of unknown size can go on a
    concave outline. `polylabel` finds it far more reliably than the centroid,
    which on a horseshoe or a U sits outside the board entirely.

    `size` is the printed side of the QR, quiet zone included.
    """
    inner = _inset(edge_cuts, 0.0)
    if inner is None or size is None:
        return (tcx, tcy)

    half = size / 2.0
    if inner.contains(box(tcx - half, tcy - half, tcx + half, tcy + half)):
        return (tcx, tcy)

    try:
        pole = polylabel(inner if inner.geom_type == "Polygon"
                         else max(inner.geoms, key=lambda g: g.area), tolerance=0.1)
    except Exception:
        return (tcx, tcy)
    return (pole.x, pole.y)


def _qr_size_mm(manifest, form="link", ordered=False):
    """Printed side of the QR, quiet zone included, without building artwork."""
    try:
        dark, _ = _qr_matrix(_qr_text(manifest, form, ordered)[2])
    except Exception:
        return None
    return None if dark is None else dark.shape[0] * MODULE_MM


def backmark(manifest_json, enable=True):
    """Never raises. See _backmark below for the real work."""
    try:
        return _backmark(manifest_json, enable=enable)
    except Exception as exc:
        # A legend is not worth a board. The PDF path in main.py takes the same
        # stance for the same reason: an optional artefact must not be able to
        # cost the user their Gerbers.
        return {"rects": [], "strokes": [], "stroke_w": DIM_STROKE_W_MM,
                "qr": None,
                "warnings": ["back silkscreen mark skipped: %s: %s"
                             % (type(exc).__name__, exc)]}


def _backmark(manifest_json, enable=True):
    """Every back-silkscreen rectangle for one board.

    Takes the manifest JSON text that `export_manifest` already produced, so
    the mark is guaranteed to describe the board that was actually exported
    rather than a re-derivation of it.

    Returns {"rects": [(x0, y0, x1, y1), ...], "qr": info|None} in board
    coordinates, y-up. Callers negate Y for KiCad exactly as they do for every
    other primitive.
    """
    if not enable or not manifest_json:
        return {"rects": [], "strokes": [], "stroke_w": DIM_STROKE_W_MM,
                "qr": None, "warnings": []}
    try:
        manifest = json.loads(manifest_json)
    except (TypeError, ValueError):
        return {"rects": [], "strokes": [], "stroke_w": DIM_STROKE_W_MM,
                "qr": None, "warnings": []}
    manifest.pop("_readme", None)
    if not manifest.get("m") or not manifest.get("pixels"):
        return {"rects": [], "strokes": [], "stroke_w": DIM_STROKE_W_MM,
                "qr": None, "warnings": []}

    edge_cuts = manifest.get("edge_cuts") or manifest.get("outline") or []

    tx = [p[0] for p in manifest["pixels"]]
    ty = [p[1] for p in manifest["pixels"]]
    tcx, tcy = (min(tx) + max(tx)) / 2.0, (min(ty) + max(ty)) / 2.0

    # The link if it fits, else the bare payload, else nothing. A QR that
    # overhangs the cut line is worse than no QR: the fab clips it and the
    # remains do not decode. So each form is placed and checked, and what was
    # printed instead is said rather than left to be found on the board.
    inner = _inset(edge_cuts, 0.0)
    qr, info, center, missed = [], None, None, []
    # A QR that decodes to the wrong sensor is worse than none, so the code is
    # decoded and checked against the manifest before anything is printed. A
    # board the regular grid cannot describe (a wrap's seams, columns numbered
    # out of order) gets its columns laid out by order: their order is exact,
    # their positions approximate, and the mark says by how much.
    ordered, notes = False, []
    try:
        check_core(manifest)
        forms = QR_FORMS
    except TaxelCoreError as exc:
        try:
            _, off_rows, off_cols = check_ordered(manifest)
            forms, ordered = QR_FORMS, True
            moved = [axis for axis, off in (("rows", off_rows), ("columns", off_cols)) if off > 0.011]
            notes.append("the taxel-core QR spaces the %s evenly, in their true order: decoded "
                         "taxels are up to %.2f mm from their positions in the manifest"
                         % (" and ".join(moved) or "columns", max(off_rows, off_cols)))
        except TaxelCoreError:
            forms = ()
            notes.append("taxel-core QR omitted: %s; registration grid kept" % exc)
    for form in forms:
        size = _qr_size_mm(manifest, form, ordered)
        if size is None:
            break                # no segno, or a grid the format cannot hold
        at = _qr_center(edge_cuts, tcx, tcy, size)
        half = size / 2.0
        if inner is not None and inner.contains(
                box(at[0] - half, at[1] - half, at[0] + half, at[1] + half)):
            qr, info = qr_rects(manifest, at, form, ordered)
            center = at if info else None
            break
        missed.append((form, size))

    keep = []
    warnings = list(notes)
    strokes, dim_bb = [], None
    if info:
        half = info["size_mm"] / 2.0 + QR_CLEAR_MM
        keep.append((center[0] - half, center[1] - half,
                     center[0] + half, center[1] + half))
        if missed:
            warnings.append(
                "the QR link (%.2f mm) does not fit inside the board outline; "
                "the back carries the bare taxel-core code (%.2f mm) instead, which "
                "a phone shows as text: paste it at opentac.github.io/qr/ or run "
                "opentac manifest from-qr" % (missed[0][1], info["size_mm"]))
    elif missed:
        warnings.append(
            "taxel-core QR (%s) does not fit anywhere inside the board outline; "
            "back QR omitted, registration grid kept"
            % ", ".join("%.2f mm as %s" % (size, "a link" if form == "link" else "the bare code")
                        for form, size in missed))

    # No keep-out: the label sits inside a cell, so it cannot cross a grid line.
    strokes, dim_bb = dim_label(edge_cuts, center if info else None,
                                info["size_mm"] if info else None)
    if not strokes:
        warnings.append("no whole grid cell free for the pitch label; "
                        "grid pitch is unlabelled")

    rects = qr + grid_rects(edge_cuts, keepouts=keep) + origin_rects(edge_cuts)
    return {"rects": rects, "strokes": strokes, "stroke_w": DIM_STROKE_W_MM,
            "qr": info, "warnings": warnings}
