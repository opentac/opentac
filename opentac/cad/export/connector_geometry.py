"""Connector footprint geometry helpers for direct manufacturing exports."""

import os
import re
import numpy as np
from shapely.geometry import LineString, Point as ShapelyPoint, Polygon as ShapelyPolygon
from shapely.ops import polygonize, unary_union

from opentac.cad.export.connector_contacts import contact_pads, stiffener_note
from opentac.cad.pipeline.tail_extension import tail_extension_from_layout
from opentac.cad.pipeline.geometry import tail_geometry


SUPPORTED_CONNECTOR_EXPORT_POS = {20, 30, 32, 40, 50, 60, 70, 80, 90, 100}


def local_to_world(lx, ly, t_hat, n_out, pad1_world):
    """Transform raw connector-local coordinates to app-world coordinates.

    Local +X maps to +t_hat, local +Y to -n_out; the KiCad and Gerber
    exporters both use this convention for the connector body, Edge.Cuts and
    stiffener silkscreen. Flipping the sign would reorient the whole connector
    body, not only its pads. The pad drill offset is handled separately in
    _resolve_pads().
    """
    wx = pad1_world[0] + lx * t_hat[0] - ly * n_out[0]
    wy = pad1_world[1] + lx * t_hat[1] - ly * n_out[1]
    return np.array([wx, wy], dtype=float)


def connector_export_dir(layout, connector_dir: str) -> str:
    """Return connector_dir only when physical connector export is enabled.

    The requirements are a supported size and a footprint file on disk. A
    custom tail extension is not required: connector_pose and the Edge.Cuts
    stitcher both handle the no-extension case (the base tail_geometry pose),
    so a board exported without a drawn tail still carries its connector
    footprint.
    """
    if not connector_dir:
        return ""
    if int(layout.connector_pos) not in SUPPORTED_CONNECTOR_EXPORT_POS:
        return ""
    mod_file = os.path.join(connector_dir, f"{layout.connector_pos}_pos_0.5mm_FPC.kicad_mod")
    return connector_dir if os.path.exists(mod_file) else ""


def connector_pose(layout, outline_xy, params):
    """Return (t_hat, n_out, pad1_world) matching KiCad connector placement."""
    tail_ext = tail_extension_from_layout(layout)
    if tail_ext:
        return (
            np.array(tail_ext["t_hat"], dtype=float),
            np.array(tail_ext["n_out"], dtype=float),
            np.array(tail_ext["pad1_world"], dtype=float),
        )

    tail = tail_geometry(layout.outline_poly, outline_xy[0], outline_xy[1], layout.connector_pos, params,
                         fit_width=(layout.tail or {}).get("fit_width_mm"),
                         fit_offset=(layout.tail or {}).get("fit_offset_mm"))
    return tail["t_hat"], tail["n_out"], np.array(tail["pad1_world"], dtype=float)


def connector_pad_geometries(layout, outline_xy, params, connector_dir):
    """Parse connector pads and return transformed pad rectangles/drill centers."""
    model = connector_geometries(layout, outline_xy, params, connector_dir)
    return model["pads"] if model else []


def connector_edge_geometries(layout, outline_xy, params, connector_dir):
    """Parse connector footprint Edge.Cuts using the same transform as KiCad export."""
    model = connector_geometries(layout, outline_xy, params, connector_dir)
    if not model:
        return {"lines": [], "arcs": []}
    return {"lines": model["edge_lines"], "arcs": model["edge_arcs"]}


def connector_user_geometries(layout, outline_xy, params, connector_dir):
    """Parse connector documentation geometry using the same transform as pads."""
    model = connector_geometries(layout, outline_xy, params, connector_dir)
    if not model:
        return {"lines": [], "arcs": [], "texts": []}
    return {
        "lines": model["user_lines"],
        "arcs": model["user_arcs"],
        "texts": model["user_texts"],
    }


def connector_open_edge_endpoints(layout, outline_xy, params, connector_dir):
    """Return the two open endpoints of the connector Edge.Cuts graph."""
    edges = connector_edge_geometries(layout, outline_xy, params, connector_dir)
    segments = list(edges["lines"])
    for start, _mid, end in edges["arcs"]:
        segments.append((start, end))
    if not segments:
        return None

    nodes = []
    degree = []

    def node_index(pt):
        p = np.asarray(pt, dtype=float)
        for i, existing in enumerate(nodes):
            if np.linalg.norm(p - existing) < 0.01:
                return i
        nodes.append(p)
        degree.append(0)
        return len(nodes) - 1

    for a, b in segments:
        ia = node_index(a)
        ib = node_index(b)
        degree[ia] += 1
        degree[ib] += 1

    open_nodes = [nodes[i] for i, d in enumerate(degree) if d == 1]
    if len(open_nodes) < 2:
        return None
    if len(open_nodes) == 2:
        return open_nodes

    # If duplicated graphic fragments create more than two degree-one nodes,
    # choose the pair closest to the tail extension endpoint.
    tail_ext = tail_extension_from_layout(layout)
    target = np.array(tail_ext["pad_row_center"], dtype=float) if tail_ext else np.mean(open_nodes, axis=0)
    ranked = sorted(open_nodes, key=lambda p: np.linalg.norm(p - target))
    return ranked[:2]


def connector_opening(layout, outline_xy, params, connector_dir, shift_mm=0.0):
    """The connector's open edge (A, B), the unit direction from it into the tab,
    and how deep the tab runs past it; None without a connector footprint.
    `shift_mm` moves the connector that far along that direction (the export's
    synthetic tail extension moves it main.AUTO_TAIL_EXTENSION_MM out)."""
    model = connector_geometries(layout, outline_xy, params, connector_dir)
    open_pts = connector_open_edge_endpoints(layout, outline_xy, params, connector_dir) if model else None
    if not open_pts:
        return None
    n = np.asarray(model["n_out"], dtype=float)
    A, B = (np.asarray(p, dtype=float) for p in open_pts)
    mid = 0.5 * (A + B)
    pts = [p for a, b in model["edge_lines"] for p in (a, b)] + [p for arc in model["edge_arcs"] for p in arc]
    depth = max(float((np.asarray(p, dtype=float) - mid) @ n) for p in pts)
    return A + shift_mm * n, B + shift_mm * n, n, depth


# Board narrower than this beside the connector's tab (mean width) is left to
# stitched_outline_segments, which cuts the connector body 0.02 mm wide.
SLIVER_MM = 0.01


def clip_beside_tab(board, A, B, n_in, depth):
    """`board` without the board that stands beside the connector's tab past its
    opening A-B.

    Past the opening the footprint's own Edge.Cuts draw the tab, so board there
    wider than the tab is left standing next to the tab's side lines, which a
    fab mills as slits: a typed tail wider than the tab, the routing canvas a
    board keeps when follow padding is off, a shoulder padded past an enlarged
    connector. Removed here, the board steps in to the tab along the opening
    line and its outline ends at A and B. Only board connected to the tab past
    the opening is touched: a separate part of the board beside the tail is
    not the tail. Copper past the opening runs on the pad lanes, inside the
    tab, so no wire loses its board. Returns `board` itself when there is
    nothing to remove."""
    if board is None or board.is_empty:
        return board
    A, B, n = (np.asarray(v, dtype=float) for v in (A, B, n_in))
    width = float(np.linalg.norm(B - A))
    if width < 1e-9:
        return board
    t = (B - A) / width
    far = float(depth) + 1.0
    minx, miny, maxx, maxy = board.bounds
    span = float(np.hypot(maxx - minx, maxy - miny)) + width
    tab = ShapelyPolygon([A, B, B + far * n, A + far * n])
    sides = unary_union([ShapelyPolygon([A, A - span * t, A - span * t + far * n, A + far * n]),
                         ShapelyPolygon([B, B + span * t, B + span * t + far * n, B + far * n])])
    past = board.intersection(unary_union([tab, sides]))
    cut = []
    for part in _polygon_parts(past):
        if part.intersection(tab).area <= 1e-6:
            continue                       # not the tail: another part of the board
        # what stands beside the tab, but not hair-thin slivers along its sides:
        # overlay leaves them where the board's edge runs on a side exactly, and
        # a tail unioned with a 1 um overlap leaves a 1 um strip. The export
        # takes anything within 0.02 mm of the connector body away anyway
        # (stitched_outline_segments), so they never reach Edge.Cuts.
        cut += [q for q in _polygon_parts(part.intersection(sides))
                if 2.0 * q.area / q.length > SLIVER_MM]
    if not cut:
        return board
    clipped = board.difference(unary_union(cut))
    if not clipped.is_valid:
        clipped = clipped.buffer(0)
    if len(_polygon_parts(clipped)) > len(_polygon_parts(board)):
        return board                       # never split the board (the Edge.Cuts check reports it)
    return clipped


def _polygon_parts(geom):
    """The polygons of non-negligible area in a shapely result."""
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom] if geom.area > 1e-6 else []
    return [p for g in getattr(geom, "geoms", []) for p in _polygon_parts(g)]


def drop_tiny_edges(ring, keep=(), tol=0.002):
    """A closed ring without its near-zero-length edges.

    Polygon unions and buffers leave the odd edge well under a micron long;
    KiCad rejects those as a malformed outline. Merges each such vertex into
    its neighbour, never moving a vertex listed in `keep` (the connector's
    open endpoints the outline must meet exactly)."""
    ring = np.asarray(ring, dtype=float)
    keep = [np.asarray(k, dtype=float) for k in keep]

    def kept(p):
        return any(np.linalg.norm(p - k) < 1e-9 for k in keep)

    out = []
    for p in ring:
        if out and np.linalg.norm(p - out[-1]) < tol:
            if kept(p):
                out[-1] = p
            continue
        out.append(p)
    while len(out) > 3 and np.linalg.norm(out[0] - out[-1]) < tol:
        if kept(out[-1]):
            out.pop(0)
        else:
            out.pop()
    return np.asarray(out, dtype=float) if len(out) >= 3 else ring


def stitched_outline_segments(layout, outline_xy, params, connector_dir):
    """Return board Edge.Cuts segments with the connector landing zone cut away.

    The connector footprint Edge.Cuts are intentionally open on the side facing
    the tail; the board outline must end exactly at those two open endpoints.
    Dragging the endpoints of one outline edge (the "cap") onto the open
    endpoints does not work: the board carries material deeper than the open
    edge (the base tail's connector-body allowance, and the landing of the
    export's tail extension, which runs on to the pad drills), and dragging the
    deepest cap folds segments back inward across the corridor. Instead, this
    builds the connector body region (its Edge.Cuts path closed along the open
    edge), subtracts it from the board polygon so the outline is cut along the
    open edge, drops the opening segment itself, and snaps the two cut vertices
    exactly onto the connector's open endpoints.

    That only meets the footprint cleanly while the board is no wider than the
    connector's tab past the open edge: board beside the tab is left standing
    next to the footprint's tab-side lines, which a fab then mills as slits
    (follow_padding.tail_wire_hold keeps the tail's wires from growing it;
    clip_beside_tab, run where the board is finished, removes any other;
    export/edge_cuts_check.py reports any board where it is not).
    """
    final_xy = getattr(layout, "final_outline_xy", None)
    if final_xy is None:
        return None
    pts = np.asarray(final_xy, dtype=float)
    if len(pts) < 4:
        return None

    pts = drop_tiny_edges(pts)
    plain = [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
    open_pts = connector_open_edge_endpoints(layout, outline_xy, params, connector_dir)
    if not open_pts:
        return plain
    A = np.asarray(open_pts[0], dtype=float)
    B = np.asarray(open_pts[1], dtype=float)

    edges = connector_edge_geometries(layout, outline_xy, params, connector_dir)
    lines = [LineString([tuple(a), tuple(b)]) for a, b in edges["lines"]]
    lines += [LineString([tuple(a), tuple(m), tuple(b)]) for a, m, b in edges["arcs"]]
    lines.append(LineString([tuple(A), tuple(B)]))
    try:
        bodies = [g for g in polygonize(unary_union(lines)) if g.area > 1.0]
        body = max(bodies, key=lambda g: g.area)
        board = ShapelyPolygon(pts)
        if not board.is_valid:
            board = board.buffer(0)
        cut = board.difference(body.buffer(0.02, join_style=2))
        if hasattr(cut, "geoms"):
            cut = max((g for g in cut.geoms if not g.is_empty), key=lambda g: g.area)
        if cut.is_empty or cut.area < 1.0:
            return plain
        ring = np.asarray(cut.exterior.coords[:-1], dtype=float)
    except Exception:
        return plain

    # Snap the cut vertices onto the exact open endpoints, then emit segments,
    # skipping any that lie along the opening (the connector side continues
    # from A and B with its own Edge.Cuts).
    snap = 0.2
    for i in range(len(ring)):
        if np.linalg.norm(ring[i] - A) < snap:
            ring[i] = A
        elif np.linalg.norm(ring[i] - B) < snap:
            ring[i] = B
    ring = drop_tiny_edges(ring, keep=(A, B))
    opening = LineString([tuple(A), tuple(B)])
    segments = []
    for i in range(len(ring)):
        p1 = ring[i]
        p2 = ring[(i + 1) % len(ring)]
        if np.linalg.norm(p2 - p1) < 1e-6:
            continue
        if (opening.distance(ShapelyPoint(p1)) < 0.1
                and opening.distance(ShapelyPoint(p2)) < 0.1):
            continue                      # the opening itself -- stays open
        segments.append((p1, p2))
    return segments

def connector_geometries(layout, outline_xy, params, connector_dir):
    """Resolve raw connector footprint geometry in board world coordinates."""
    connector_dir = connector_export_dir(layout, connector_dir)
    if not connector_dir:
        return None
    mod_file = os.path.join(connector_dir, f"{layout.connector_pos}_pos_0.5mm_FPC.kicad_mod")
    if not os.path.exists(mod_file):
        return None

    text = PathReader(mod_file).read()
    t_hat, n_out, pad1_route_world = connector_pose(layout, outline_xy, params)
    pad_anchor_y = _tail_side_pad_edge_y(text)
    pad1_world = pad1_route_world + pad_anchor_y * n_out
    # The footprints draw each finger as a drilled through-hole pad on every
    # copper layer; the export makes them plain pads on the contact face only,
    # and words the stiffener note for that face (export/connector_contacts.py).
    return {
        "pads": contact_pads(_resolve_pads(text, t_hat, n_out, pad1_world), params),
        "edge_lines": _resolve_edge_lines(text, t_hat, n_out, pad1_world),
        "edge_arcs": _resolve_edge_arcs(text, t_hat, n_out, pad1_world),
        "user_lines": _resolve_user_lines(text, t_hat, n_out, pad1_world),
        "user_arcs": _resolve_user_arcs(text, t_hat, n_out, pad1_world),
        "user_texts": [{**item, "text": stiffener_note(item["text"], params)}
                       for item in _resolve_user_texts(text, t_hat, n_out, pad1_world)],
        "t_hat": np.array(t_hat, dtype=float),
        "n_out": np.array(n_out, dtype=float),
        "pad1_route_world": pad1_route_world,
        "pad1_world": pad1_world,
        "pad_anchor_y": pad_anchor_y,
    }


def _resolve_pads(text, t_hat, n_out, pad1_world):
    """Parse each pad and re-center its rectangle on the drill position.

    This connector family authors each pad's rectangle centered on its own
    "at" point (e.g. spanning local y=[-1.75, 1.75]) with the drill offset
    toward one end ("(offset 0 -1.5)") — taken literally, the pad's far
    edge stops 1.5mm short of the connector body's mechanical tip edge (the
    same file's Edge.Cuts tip line at local y=-3.25). Each pad rectangle is
    therefore re-centred on its drill (local y=-1.5 for this family), so the
    pad's far edge lands exactly on the tip line. The drill then sits at the
    pad's own centre, so "drill_center" and "center" coincide.
    """
    pads = _parse_pad_blocks(text)
    out = []
    for pad in pads:
        raw_cx, raw_cy = pad["at"]
        sx, sy = pad["size"]
        drill_offset = pad.get("drill_offset", (0.0, 0.0))
        cx = raw_cx + drill_offset[0]
        cy = raw_cy + drill_offset[1]
        local_corners = [
            (cx - sx / 2, cy - sy / 2),
            (cx + sx / 2, cy - sy / 2),
            (cx + sx / 2, cy + sy / 2),
            (cx - sx / 2, cy + sy / 2),
        ]
        dcx, dcy = cx, cy
        out.append({
            "pad_num": pad["pad_num"],
            "kind": pad["kind"],
            "shape": pad["shape"],
            "size": pad["size"],
            "layers": pad["layers"],
            "drill": pad.get("drill"),
            "drill_offset": drill_offset,
            "local_at": pad["at"],
            "center": local_to_world(cx, cy, t_hat, n_out, pad1_world),
            "drill_center": local_to_world(dcx, dcy, t_hat, n_out, pad1_world),
            "corners": np.array([local_to_world(x, y, t_hat, n_out, pad1_world) for x, y in local_corners]),
        })
    return out


def _resolve_edge_lines(text, t_hat, n_out, pad1_world):
    lines = []
    for m in re.finditer(
        r'fp_line\s*\(\s*start\s+([-\d.]+)\s+([-\d.]+)\s*\)\s*'
        r'\(\s*end\s+([-\d.]+)\s+([-\d.]+)\s*\).*?Edge\.Cuts',
        text, re.DOTALL):
        lx1, ly1 = float(m.group(1)), float(m.group(2))
        lx2, ly2 = float(m.group(3)), float(m.group(4))
        lines.append((
            local_to_world(lx1, ly1, t_hat, n_out, pad1_world),
            local_to_world(lx2, ly2, t_hat, n_out, pad1_world),
        ))
    return lines


def _resolve_edge_arcs(text, t_hat, n_out, pad1_world):
    arcs = []
    for m in re.finditer(
        r'fp_arc\s*\(\s*start\s+([-\d.]+)\s+([-\d.]+)\s*\)\s*'
        r'\(\s*mid\s+([-\d.]+)\s+([-\d.]+)\s*\)\s*'
        r'\(\s*end\s+([-\d.]+)\s+([-\d.]+)\s*\).*?Edge\.Cuts',
        text, re.DOTALL):
        ls = (float(m.group(1)), float(m.group(2)))
        lm = (float(m.group(3)), float(m.group(4)))
        le = (float(m.group(5)), float(m.group(6)))
        arcs.append((
            local_to_world(*ls, t_hat, n_out, pad1_world),
            local_to_world(*lm, t_hat, n_out, pad1_world),
            local_to_world(*le, t_hat, n_out, pad1_world),
        ))
    return arcs


def _resolve_user_lines(text, t_hat, n_out, pad1_world):
    lines = []
    for m in re.finditer(
        r'fp_line\s*\(\s*start\s+([-\d.]+)\s+([-\d.]+)\s*\)\s*'
        r'\(\s*end\s+([-\d.]+)\s+([-\d.]+)\s*\).*?'
        r'\(\s*stroke\s*\(\s*width\s+([-\d.]+)\s*\).*?'
        r'\(\s*layer\s+"Dwgs\.User"\s*\)',
        text, re.DOTALL):
        lines.append({
            "start": local_to_world(float(m.group(1)), float(m.group(2)), t_hat, n_out, pad1_world),
            "end": local_to_world(float(m.group(3)), float(m.group(4)), t_hat, n_out, pad1_world),
            "width": float(m.group(5)),
            "layer": "Dwgs.User",
        })

    for m in re.finditer(
        r'fp_rect\s*\(\s*start\s+([-\d.]+)\s+([-\d.]+)\s*\)\s*'
        r'\(\s*end\s+([-\d.]+)\s+([-\d.]+)\s*\).*?'
        r'\(\s*stroke\s*\(\s*width\s+([-\d.]+)\s*\).*?'
        r'\(\s*layer\s+"Dwgs\.User"\s*\)',
        text, re.DOTALL):
        x1, y1 = float(m.group(1)), float(m.group(2))
        x2, y2 = float(m.group(3)), float(m.group(4))
        width = float(m.group(5))
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        for i in range(4):
            lx1, ly1 = corners[i]
            lx2, ly2 = corners[(i + 1) % 4]
            lines.append({
                "start": local_to_world(lx1, ly1, t_hat, n_out, pad1_world),
                "end": local_to_world(lx2, ly2, t_hat, n_out, pad1_world),
                "width": width,
                "layer": "Dwgs.User",
            })
    return lines


def _resolve_user_arcs(text, t_hat, n_out, pad1_world):
    arcs = []
    for m in re.finditer(
        r'fp_arc\s*\(\s*start\s+([-\d.]+)\s+([-\d.]+)\s*\)\s*'
        r'\(\s*mid\s+([-\d.]+)\s+([-\d.]+)\s*\)\s*'
        r'\(\s*end\s+([-\d.]+)\s+([-\d.]+)\s*\).*?'
        r'\(\s*stroke\s*\(\s*width\s+([-\d.]+)\s*\).*?'
        r'\(\s*layer\s+"Dwgs\.User"\s*\)',
        text, re.DOTALL):
        arcs.append({
            "start": local_to_world(float(m.group(1)), float(m.group(2)), t_hat, n_out, pad1_world),
            "mid": local_to_world(float(m.group(3)), float(m.group(4)), t_hat, n_out, pad1_world),
            "end": local_to_world(float(m.group(5)), float(m.group(6)), t_hat, n_out, pad1_world),
            "width": float(m.group(7)),
            "layer": "Dwgs.User",
        })
    return arcs


def _resolve_user_texts(text, t_hat, n_out, pad1_world):
    texts = []
    base_rot = float(np.degrees(np.arctan2(-t_hat[1], t_hat[0])))
    for block in _extract_blocks(text, '(fp_text user'):
        if 'layer "Cmts.User"' not in block:
            continue
        head = re.search(r'\(fp_text\s+user\s+"((?:\\.|[^"])*)"', block)
        at = re.search(r'\(at\s+([-\d.]+)\s+([-\d.]+)(?:\s+([-\d.]+))?', block)
        size = re.search(r'\(size\s+([-\d.]+)\s+([-\d.]+)\)', block)
        thickness = re.search(r'\(thickness\s+([-\d.]+)\)', block)
        justify = re.search(r'\(justify\s+([^\)]*)\)', block)
        if not (head and at):
            continue
        local_rot = float(at.group(3)) if at.group(3) else 0.0
        texts.append({
            "text": head.group(1),
            "at": local_to_world(float(at.group(1)), float(at.group(2)), t_hat, n_out, pad1_world),
            "rot": base_rot + local_rot,
            "layer": "Cmts.User",
            "size": (
                float(size.group(1)) if size else 1.0,
                float(size.group(2)) if size else 1.0,
            ),
            "thickness": float(thickness.group(1)) if thickness else 0.15,
            "justify": justify.group(1).strip() if justify else "",
        })
    return texts


def _tail_side_pad_edge_y(text):
    """Return the raw pad edge nearest the connector's open tail-side outline."""
    pads = _parse_pad_blocks(text)
    if not pads:
        return 0.0
    open_y = _open_edge_mid_y(text)
    candidates = []
    for pad in pads:
        _cx, cy = pad["at"]
        _sx, sy = pad["size"]
        candidates.extend([cy - sy / 2, cy + sy / 2])
    if open_y is None:
        offset_ys = [pad.get("drill_offset", (0.0, 0.0))[1] for pad in pads]
        avg_offset_y = float(np.mean(offset_ys)) if offset_ys else 0.0
        if abs(avg_offset_y) > 1e-9:
            return max(candidates) if avg_offset_y < 0 else min(candidates)
        return max(candidates)
    return min(candidates, key=lambda y: abs(y - open_y))


def _open_edge_mid_y(text):
    segments = []
    for m in re.finditer(
        r'fp_line\s*\(\s*start\s+([-\d.]+)\s+([-\d.]+)\s*\)\s*'
        r'\(\s*end\s+([-\d.]+)\s+([-\d.]+)\s*\).*?Edge\.Cuts',
        text, re.DOTALL):
        segments.append(((float(m.group(1)), float(m.group(2))),
                         (float(m.group(3)), float(m.group(4)))))
    for m in re.finditer(
        r'fp_arc\s*\(\s*start\s+([-\d.]+)\s+([-\d.]+)\s*\)\s*'
        r'\(\s*mid\s+([-\d.]+)\s+([-\d.]+)\s*\)\s*'
        r'\(\s*end\s+([-\d.]+)\s+([-\d.]+)\s*\).*?Edge\.Cuts',
        text, re.DOTALL):
        segments.append(((float(m.group(1)), float(m.group(2))),
                         (float(m.group(5)), float(m.group(6)))))
    if not segments:
        return None

    nodes = []
    degree = []

    def node_index(pt):
        for i, existing in enumerate(nodes):
            if np.linalg.norm(np.array(pt) - np.array(existing)) < 0.01:
                return i
        nodes.append(pt)
        degree.append(0)
        return len(nodes) - 1

    for a, b in segments:
        ia = node_index(a)
        ib = node_index(b)
        degree[ia] += 1
        degree[ib] += 1

    open_nodes = [nodes[i] for i, d in enumerate(degree) if d == 1]
    if len(open_nodes) < 2:
        return None
    return float(np.mean([p[1] for p in open_nodes]))


class PathReader:
    def __init__(self, path):
        self.path = path

    def read(self):
        with open(self.path, "r") as f:
            return f.read()


def _extract_blocks(text, prefix):
    lines = text.splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.lstrip().startswith(prefix):
            i += 1
            continue
        block = [line]
        depth = line.count('(') - line.count(')')
        i += 1
        while i < len(lines) and depth > 0:
            block.append(lines[i])
            depth += lines[i].count('(') - lines[i].count(')')
            i += 1
        blocks.append('\n'.join(block))
    return blocks


def _parse_pad_blocks(text):
    blocks = []
    for block in _extract_blocks(text, '(pad "'):
        pad = _parse_pad_block(block)
        if pad:
            blocks.append(pad)
    return blocks


def _parse_pad_block(block):
    first = re.search(r'\(pad\s+"(\d+)"\s+(\S+)\s+(\S+)', block)
    at = re.search(r'\(at\s+([-\d.]+)\s+([-\d.]+)', block)
    size = re.search(r'\(size\s+([-\d.]+)\s+([-\d.]+)', block)
    layers = re.search(r'\(layers\s+([^\)]*)\)', block)
    if not (first and at and size and layers):
        return None
    drill = re.search(r'\(drill\s+([-\d.]+)', block)
    offset = re.search(r'\(offset\s+([-\d.]+)\s+([-\d.]+)\)', block)
    return {
        "pad_num": int(first.group(1)),
        "kind": first.group(2),
        "shape": first.group(3),
        "at": (float(at.group(1)), float(at.group(2))),
        "size": (float(size.group(1)), float(size.group(2))),
        "layers": layers.group(1).replace('"', '').split(),
        "drill": float(drill.group(1)) if drill else None,
        "drill_offset": (float(offset.group(1)), float(offset.group(2))) if offset else (0.0, 0.0),
    }
