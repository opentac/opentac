"""Design rule checks for routed traces."""

from .types import Route, DrcViolation
from .geometry import seg_seg_dist
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import linemerge, nearest_points, unary_union


def drc_check(
    col_routes: list[Route],
    row_routes: list[Route],
    trace_w_mm: float,
    clearance_mm: float,
    outline_xy: np.ndarray | None = None,
    board_edge_clear_mm: float = 0.0,
    board_poly=None,
    taxels=None,
    vias=None,
) -> list[DrcViolation]:
    """Check same-layer route clearance and optional board-edge clearance.

    Each copper layer is checked on its own: a row route with .layer set (the
    inner layers of a 4-layer board) is never compared with B.Cu copper it
    merely overlaps in plan view.

    taxels (optional, a TaxelCopper) also checks the F.Cu wires against the
    taxels' own copper — see _check_taxels.

    vias (optional, the connector's transition vias, export/connector_contacts.py)
    are through vias: each is checked against every other net's wires on every
    layer, against the other vias, and against the board edge — see _check_vias."""
    G = trace_w_mm + clearance_mm
    violations = []

    if col_routes:
        violations.extend(_check_layer(col_routes, G, "F.Cu"))
        if taxels is not None:
            violations.extend(_check_taxels(col_routes, taxels, trace_w_mm / 2 + clearance_mm))
    for layer, routes in _by_layer(row_routes, "B.Cu"):
        violations.extend(_check_layer(routes, G, layer))
    if board_edge_clear_mm > 0 and (outline_xy is not None or board_poly is not None):
        edge_min = trace_w_mm / 2 + board_edge_clear_mm
        violations.extend(_check_board_edge(col_routes, outline_xy, edge_min, "F.Cu", board_poly))
        for layer, routes in _by_layer(row_routes, "B.Cu"):
            violations.extend(_check_board_edge(routes, outline_xy, edge_min, layer, board_poly))
    if vias:
        violations.extend(_check_vias(
            vias, [*col_routes, *row_routes], trace_w_mm, clearance_mm,
            outline_xy, board_edge_clear_mm, board_poly))

    return violations


def _check_vias(vias, routes, trace_w_mm, clearance_mm, outline_xy=None,
                board_edge_clear_mm=0.0, board_poly=None) -> list[DrcViolation]:
    """Through vias (anything with .xy, .dia, .net_name) against the wires of
    other nets on any layer, against each other, and against the board edge."""
    out = []
    # Every wire segment once, as arrays, so each via is one vectorised
    # point-to-segment distance over the whole board.
    seg_a, seg_b, seg_net, seg_layer = [], [], [], []
    for route in routes:
        if route.polyline_xy is None:
            continue
        P = np.asarray(route.polyline_xy, dtype=float)
        if len(P) < 2:
            continue
        seg_a.append(P[:-1]); seg_b.append(P[1:])
        seg_net += [route.net_name] * (len(P) - 1)
        seg_layer += [getattr(route, "layer", None) or "via"] * (len(P) - 1)
    A = np.vstack(seg_a) if seg_a else np.zeros((0, 2))
    B = np.vstack(seg_b) if seg_b else np.zeros((0, 2))
    nets = np.array(seg_net, dtype=object)
    for i, v in enumerate(vias):
        c = np.asarray(v.xy, dtype=float)
        r = float(v.dia) / 2
        need = r + trace_w_mm / 2 + clearance_mm
        here = np.array([c, c])
        if len(A):
            d_ab = B - A
            L2 = np.einsum("ij,ij->i", d_ab, d_ab)
            t = np.clip(np.einsum("ij,ij->i", c - A, d_ab) / np.where(L2 > 0, L2, 1.0), 0.0, 1.0)
            dist = np.linalg.norm(A + t[:, None] * d_ab - c, axis=1)
            for k in np.flatnonzero((dist < need - 1e-6) & (nets != v.net_name)):
                out.append(DrcViolation(
                    layer=seg_layer[k], net1=v.net_name, net2=nets[k], type="via clearance",
                    seg1=here, seg2=np.array([A[k], B[k]]), distance=float(dist[k]), min_required=need))
        for w in vias[i + 1:]:
            if w.net_name == v.net_name:
                continue
            d = float(np.linalg.norm(np.asarray(w.xy, dtype=float) - c))
            need_vv = (float(v.dia) + float(w.dia)) / 2 + clearance_mm
            if d < need_vv - 1e-6:
                out.append(DrcViolation(
                    layer="via", net1=v.net_name, net2=w.net_name, type="via clearance",
                    seg1=here, seg2=np.array([w.xy, w.xy]), distance=d, min_required=need_vv))
    if board_edge_clear_mm > 0 and (outline_xy is not None or board_poly is not None):
        poly = board_poly if board_poly is not None else Polygon(outline_xy)
        if not poly.is_empty:
            for v in vias:
                p = Point(float(v.xy[0]), float(v.xy[1]))
                need = float(v.dia) / 2 + board_edge_clear_mm
                d = poly.boundary.distance(p) if poly.contains(p) else -poly.boundary.distance(p)
                if d < need - 1e-6:
                    c = np.asarray(v.xy, dtype=float)
                    out.append(DrcViolation(
                        layer="via", net1=v.net_name, net2="board edge", type="via edge",
                        seg1=np.array([c, c]), seg2=np.array([c, c]), distance=d, min_required=need))
    return out


class TaxelCopper:
    """The taxels' F.Cu copper for _check_taxels: each active taxel's square
    inset by edge_clear_mm (the comb's outer frame), with its column net."""

    def __init__(self, placements_xy, col_net_per_pixel, removed, pixel_w_mm, pixel_h_mm, edge_clear_mm):
        from shapely.geometry import box
        hw, hh = pixel_w_mm / 2 - edge_clear_mm, pixel_h_mm / 2 - edge_clear_mm
        self.items = [(col_net_per_pixel[i], box(x - hw, y - hh, x + hw, y + hh))
                      for i, (x, y) in enumerate(np.asarray(placements_xy, dtype=float))
                      if i not in removed]

    @classmethod
    def of(cls, layout, removed, params):
        return cls(layout.placements_xy, layout.col_net_per_pixel, set(removed or ()),
                   params.pixel_w_mm, params.pixel_h_mm, params.edge_clear_mm)


def _check_taxels(col_routes: list[Route], taxels: TaxelCopper, need: float) -> list[DrcViolation]:
    """F.Cu wires against the taxels' combs: a column wire passing a taxel
    must clear its copper like any other copper (on F.Cu both electrodes are
    combs, so grazing one shorts a column to a row). A wire's own column's
    taxels are skipped — it lands on them by design. Without this check, a
    smoothed column wire can cut a taxel's corner closer than the clearance,
    which the wire-to-wire check does not see."""
    from shapely.strtree import STRtree
    if not taxels.items or not col_routes:
        return []
    boxes = [b for _, b in taxels.items]
    tree = STRtree(boxes)
    out = []
    for r in col_routes:
        P = np.asarray(r.polyline_xy, dtype=float)
        for s in range(len(P) - 1):
            seg = LineString([P[s], P[s + 1]])
            if seg.length < 1e-9:
                continue
            for k in tree.query(seg.buffer(need)):
                net, b = taxels.items[int(k)]
                if net == r.net_name:
                    continue
                d = seg.distance(b)
                if d < need - 1e-6:
                    out.append(DrcViolation(
                        layer="F.Cu", net1=r.net_name, net2=f"TAXEL:{net}", type="taxel_clearance",
                        seg1=np.array([P[s], P[s + 1]]), seg2=np.array(b.exterior.coords[:2]),
                        distance=float(d), min_required=float(need)))
    return out


def _by_layer(routes: list[Route], default: str) -> list[tuple[str, list[Route]]]:
    """routes grouped by copper layer (Route.layer, or `default`), in first-seen order"""
    groups: dict[str, list[Route]] = {}
    for r in routes or ():
        groups.setdefault(getattr(r, "layer", None) or default, []).append(r)
    return list(groups.items())


def _check_layer(routes: list[Route], G: float, layer_name: str) -> list[DrcViolation]:
    """Check all route pairs on one layer.

    Exact-result bounding-box prefilters at the track and segment level: a
    pair whose boxes are separated by >= G cannot come within G, so skipping
    it changes nothing. Without them this loop is O(tracks^2 * segments^2)
    in pure Python and dominates fixed-mode generates, which call it to score
    every routing attempt.
    """
    violations = []
    N = len(routes)

    prepared = []
    for r in routes:
        P = r.polyline_xy
        if P is None or len(P) < 2:
            prepared.append(None)
            continue
        P = np.asarray(P, dtype=float)
        lo = np.minimum(P[:-1], P[1:])
        hi = np.maximum(P[:-1], P[1:])
        prepared.append((P, lo, hi, P.min(axis=0), P.max(axis=0)))

    for i in range(N):
        if prepared[i] is None:
            continue
        Pi, ilo, ihi, imin, imax = prepared[i]
        for j in range(i + 1, N):
            if routes[j].net_name == routes[i].net_name:
                continue   # same net may touch (hug router emits several tracks per net)
            if prepared[j] is None:
                continue
            Pj, jlo, jhi, jmin, jmax = prepared[j]
            if (imin[0] > jmax[0] + G or jmin[0] > imax[0] + G
                    or imin[1] > jmax[1] + G or jmin[1] > imax[1] + G):
                continue
            for si in range(len(Pi) - 1):
                cand = np.flatnonzero(
                    (jlo[:, 0] <= ihi[si, 0] + G) & (jhi[:, 0] >= ilo[si, 0] - G)
                    & (jlo[:, 1] <= ihi[si, 1] + G) & (jhi[:, 1] >= ilo[si, 1] - G))
                for sj in cand:
                    d = seg_seg_dist(Pi[si], Pi[si + 1], Pj[sj], Pj[sj + 1])
                    if d < G - 1e-9:
                        typ = "crossing" if d < 1e-9 else "clearance"
                        violations.append(DrcViolation(
                            layer=layer_name,
                            net1=routes[i].net_name,
                            net2=routes[j].net_name,
                            type=typ,
                            seg1=np.array([Pi[si], Pi[si + 1]]),
                            seg2=np.array([Pj[sj], Pj[sj + 1]]),
                            distance=d,
                            min_required=G,
                        ))

    return violations


def _check_board_edge(
    routes: list[Route],
    outline_xy: np.ndarray,
    min_required: float,
    layer_name: str,
    board_poly=None,
) -> list[DrcViolation]:
    """Check trace centerlines against the final board boundary.

    Every stretch of a segment that lies outside the board, or inside it but
    closer than min_required to the edge, is its own violation, reported at
    that stretch's worst point. Distance is the centerline's distance to the
    edge, negative when the centerline is outside the board.

    Measuring whole stretches, rather than sampling points and keeping one
    result per segment, means copper outside the board is always measured,
    and a harmless near-miss at the connector end of a long segment (which
    main._drop_connector_zone_edge_flags discards) cannot stand in for a real
    fault further along it.
    """
    violations = []
    outline_poly = board_poly if board_poly is not None else Polygon(outline_xy)
    if outline_poly.is_empty:
        return violations
    boundary = outline_poly.boundary
    band = boundary.buffer(min_required)

    for route in routes:
        P = route.polyline_xy
        if P is None or len(P) < 2:
            continue
        P = np.asarray(P, dtype=float)
        for si in range(len(P) - 1):
            seg = LineString([P[si], P[si + 1]])
            if seg.length <= 0:
                continue
            bad = seg.intersection(band).union(seg.difference(outline_poly))
            if not bad.is_empty and bad.geom_type != "LineString":
                bad = unary_union(_line_pieces(bad))
                if bad.geom_type == "MultiLineString":
                    bad = linemerge(bad)   # one fault, one flag
            for piece in _line_pieces(bad):
                worst_pt, dist = _worst_point(piece, outline_poly, boundary)
                if dist >= min_required - 1e-9:
                    continue
                violations.append(DrcViolation(
                    layer=layer_name,
                    net1=route.net_name,
                    net2="BOARD_EDGE",
                    type="edge_clearance",
                    seg1=np.array([P[si], P[si + 1]]),
                    seg2=np.array([worst_pt, worst_pt]),
                    distance=dist,
                    min_required=min_required,
                ))

    return violations


def _line_pieces(geom):
    """The non-degenerate LineStrings in a (multi)line intersection result."""
    if geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [geom] if geom.length > 1e-9 else []
    return [g for g in getattr(geom, "geoms", []) for g in _line_pieces(g)]


def _worst_point(piece, outline_poly, boundary, step=0.05):
    """(point, signed distance) where `piece` is worst: deepest outside the
    board if any of it is outside, else closest to the edge."""
    outside = _line_pieces(piece.difference(outline_poly))
    if outside:
        best = None
        for part in outside:
            n = max(2, int(np.ceil(part.length / step)) + 1)
            for t in np.linspace(0.0, 1.0, n):
                pt = part.interpolate(t, normalized=True)
                d = boundary.distance(pt)
                if best is None or d > best[1]:
                    best = (np.array([pt.x, pt.y]), d)
        return best[0], -float(best[1])
    p, _ = nearest_points(piece, boundary)
    return np.array([p.x, p.y]), float(piece.distance(boundary))
