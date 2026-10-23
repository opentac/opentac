"""The connector's contact face, and the vias that bring every wire to it.

The tail ends in gold fingers that a ZIF connector presses on from one face.
Which face that has to be depends on the connector and on how the sensor is
mounted relative to it, so it is a setting: Params.contact_side "bottom" (the
default) or "top" (the taxel side). A finger is a plain pad on that face's copper only, with no drill, and the
coverlay is opened across the finger row on that face only. The PI stiffener
goes on the other face, and the exported drawing and note say so. A drilled
finger would read to a fabricator's CAM as a solder pad, and the ZIF contact
would press on a hole.

A wire routed on another copper layer (the columns on F.Cu, and on a 4-layer
board the rows on In1/In2) changes to the contact face through a via in the
tail, before the stiffener, so no hole is under the stiffener or in the area
the connector's contacts press on. The via goes on the wire, where it crosses a
line VIA_ROW_SETBACK_MM in front of the stiffener's inner edge, and the rest of
the wire keeps its routed path, on the contact face. Adjacent fingers are
0.5 mm apart, closer than one via plus its clearance (0.35 + 0.2 mm), so the
vias stand in two rows, odd pads in one and even pads in the other; where the
wires run straight into the fingers a via clears its neighbour's wire by
0.5 - 0.175 - 0.1 = 0.225 mm. Wires already on the contact face need no via.

Every via is placed only where it, and the stretch of wire it moves to the
contact face, clear every other net's copper on every layer, the other vias,
the taxel field and the board's edge; a wire whose first row does not allow
that tries the other row and then lines further from the stiffener. On a very
short cable the wires of the two faces may still be crossing over each other
right in front of the stiffener, and a wire can have no such place. Its finger
then keeps the footprint's plated hole (FALLBACK), as every finger had before
this choice existed: the hole joins the layers inside the tab, as it always
did, so the board is never one the earlier export would not have made.

The transitions are made at export time on copies of the routes
(main._build_outputs), so the taxels, the routes in edit_data and the manifest
are the same for either face, and both board writers and the pipeline DRC see
the same copper.
"""
from dataclasses import dataclass, field, replace

import numpy as np

CONTACT_SIDES = ("bottom", "top")

# Distance of the two via rows from the stiffener's inner edge, toward the
# board. The nearer row keeps a via's copper 0.825 mm clear of the stiffener's
# edge, where the flex bends; the rows are 0.6 mm apart, so a via and the
# nearest via of the other row (one pad over) are 0.78 mm apart centre to
# centre.
VIA_ROW_SETBACK_MM = (1.0, 1.6)
# Further lines tried, in order, when a via cannot go on its own row.
VIA_SETBACK_FALLBACKS_MM = (2.2, 2.8, 3.4)

_TOL = 1e-3


def contact_side(params) -> str:
    side = getattr(params, "contact_side", None) or "bottom"
    if side not in CONTACT_SIDES:
        raise ValueError(f'contact_side must be "bottom" or "top", got {side!r}.')
    return side


def contact_copper(params) -> str:
    """The copper layer the connector's fingers are on."""
    return "B.Cu" if contact_side(params) == "bottom" else "F.Cu"


def contact_mask(params) -> str:
    """The mask (coverlay) layer opened across the fingers."""
    return "B.Mask" if contact_side(params) == "bottom" else "F.Mask"


def contact_pads(pads, params):
    """The footprint's pads as gold fingers on the contact face: plain pads on
    that face's copper and mask only, no drill. Position and size are the
    footprint's own; "drill_center" stays as the pad's centre, where the
    wires end."""
    layers = [contact_copper(params), contact_mask(params)]
    return [{**pad, "kind": "smd", "drill": None, "layers": list(layers),
             "footprint_kind": pad.get("kind"), "footprint_drill": pad.get("drill")}
            for pad in pads]


def with_drilled(pads, drilled_pads, params):
    """`pads` (from contact_pads) with the fingers in `drilled_pads` given back
    the footprint's plated through hole on every copper layer: the fallback for
    a wire that has no place for a transition via (contact_transitions)."""
    drilled = {int(n) for n in drilled_pads or ()}
    if not drilled:
        return pads
    mask = contact_mask(params)
    return [{**pad, "kind": pad.get("footprint_kind") or "thru_hole", "drill": pad.get("footprint_drill"),
             "layers": ["*.Cu", mask]} if int(pad["pad_num"]) in drilled and pad.get("footprint_drill") else pad
            for pad in pads]


def stiffener_note(text: str, params) -> str:
    """The footprint's stiffener note with the faces of the chosen contact
    side. The footprints are drawn for bottom contacts ("... PI stiffener on TOP
    (taxel side) ... contacts on BOTTOM")."""
    if contact_side(params) == "bottom" or "stiffener" not in text:
        return text
    return (text.replace("stiffener on TOP (taxel side)", "stiffener on BOTTOM")
                .replace("contacts on BOTTOM", "contacts on TOP (taxel side)"))


@dataclass
class TransitionVia:
    """A through via that brings one net's wire to the contact face."""
    xy: np.ndarray
    net_name: str
    pad_num: int
    dia: float
    drill: float


@dataclass
class ContactTransitions:
    col_routes: list
    row_routes: list
    vias: list = field(default_factory=list)
    # Fingers that keep the footprint's plated hole (with_drilled): their wire
    # had no place for a transition via and reaches the finger on its own layer.
    drilled_pads: set = field(default_factory=set)
    # Kept for callers; the fallback above leaves nothing to report here.
    problems: list = field(default_factory=list)


def _route_layer(route, is_col: bool) -> str:
    if is_col:
        return "F.Cu"
    return getattr(route, "layer", None) or "B.Cu"


def _stiffener_inner_ly(model, to_local) -> float:
    """Local y (toward the board) of the stiffener's inner edge: the Dwgs.User
    outline's edge nearest the board. Falls back to the footprint's open edge
    (the landing), where the stiffener starts on every footprint drawn so far."""
    ys = [to_local(p)[1] for item in model.get("user_lines", [])
          if item.get("layer", "Dwgs.User") == "Dwgs.User"
          for p in (item["start"], item["end"])]
    if ys:
        return max(ys)
    edge_ys = [to_local(p)[1] for a, b in model.get("edge_lines", []) for p in (a, b)]
    return max(edge_ys) if edge_ys else 0.0


def contact_transitions(model, params, col_routes, row_routes, board_poly=None,
                        field_boxes=()) -> ContactTransitions:
    """Copies of the routes with every wire that reaches a finger on another
    layer cut at its transition via and continued on the contact face, and
    the vias; see the module docstring. `model` is connector_geometries()
    (None: no connector exported, the routes are left as they are),
    `board_poly` the board the vias must stay inside, `field_boxes` shapely
    polygons covering the taxel field, which no via or moved wire may touch."""
    from shapely import STRtree
    from shapely.geometry import LineString, Point

    out = ContactTransitions(list(col_routes), list(row_routes))
    if not model or not model.get("pads"):
        return out
    contact = contact_copper(params)
    t_hat = np.asarray(model["t_hat"], dtype=float)
    n_out = np.asarray(model["n_out"], dtype=float)
    origin = np.asarray(model["pad1_world"], dtype=float)

    def to_local(p):
        d = np.asarray(p, dtype=float) - origin
        return np.array([float(d @ t_hat), -float(d @ n_out)])

    def to_world(lx, ly):
        return origin + lx * t_hat - ly * n_out

    stiff_ly = _stiffener_inner_ly(model, to_local)
    pads = {int(p["pad_num"]): p for p in model["pads"]}
    via_dia, via_drill = float(params.via_dia_mm), float(params.via_drill_mm)
    tw, cl = float(params.trace_w_mm), float(params.clearance_mm)
    via_r = via_dia / 2.0
    edge = float(getattr(params, "board_edge_clear_mm", 0.0) or 0.0)

    # Every wire segment, labelled with its net: all layers (for the vias,
    # which go through every layer) and the contact face (for moved wires).
    tagged = [(r, True) for r in col_routes] + [(r, False) for r in row_routes]
    segs, seg_net, cseg, cseg_net = [], [], [], []
    for r, is_col in tagged:
        if r.polyline_xy is None or len(r.polyline_xy) < 2:
            continue
        P = np.asarray(r.polyline_xy, dtype=float)
        on_contact = _route_layer(r, is_col) == contact
        for a, b in zip(P[:-1], P[1:]):
            line = LineString([a, b])
            segs.append(line); seg_net.append(r.net_name)
            if on_contact:
                cseg.append(line); cseg_net.append(r.net_name)
    all_tree = STRtree(segs) if segs else None
    contact_tree = STRtree(cseg) if cseg else None
    field = list(field_boxes or ())
    field_tree = STRtree(field) if field else None
    placed_vias, placed_stubs = [], []          # (Point, net), (LineString, net)

    def clear(via_xy, stub_xy, net) -> bool:
        p = Point(float(via_xy[0]), float(via_xy[1]))
        stub = LineString(stub_xy)
        need_vw = via_r + tw / 2 + cl - 1e-6      # via to another net's wire
        need_ww = tw + cl - 1e-6                  # wire to wire, centre lines
        if all_tree is not None and any(seg_net[i] != net for i in all_tree.query(p, predicate="dwithin", distance=need_vw)):
            return False
        if contact_tree is not None and any(cseg_net[i] != net for i in contact_tree.query(stub, predicate="dwithin", distance=need_ww)):
            return False
        for q, other in placed_vias:
            if other != net and (q.distance(p) < via_dia + cl - 1e-6 or q.distance(stub) < need_vw):
                return False
        for line, other in placed_stubs:
            if other != net and line.distance(stub) < need_ww:
                return False
        if field_tree is not None and (len(field_tree.query(p, predicate="dwithin", distance=via_r + cl))
                                       or len(field_tree.query(stub, predicate="dwithin", distance=tw / 2 + cl))):
            return False
        if board_poly is not None and not board_poly.is_empty:
            if not board_poly.contains(p) or board_poly.boundary.distance(p) < via_r + edge - 1e-6:
                return False
        return True

    def ends_on_finger(P, pad):
        c = to_local(pad["center"])
        half_w, half_l = float(pad["size"][0]) / 2.0, float(pad["size"][1]) / 2.0

        def on(p):
            q = to_local(p)
            return abs(q[0] - c[0]) <= half_w + _TOL and abs(q[1] - c[1]) <= half_l + _TOL
        if on(P[-1]):
            return P, False
        if on(P[0]):
            return P[::-1], True
        return None, False

    def cut(P, via_ly):
        """(head, stub, via) of the wire P (finger last) cut where it crosses
        local y = via_ly, nearest the finger; None if it never gets there."""
        L = np.array([to_local(p) for p in P])
        j = len(L) - 1
        while j > 0 and L[j - 1][1] < via_ly - _TOL:
            j -= 1
        if j == 0 or L[j - 1][1] < via_ly - _TOL or L[j][1] >= via_ly:
            return None
        a, b = L[j - 1], L[j]
        k = (via_ly - a[1]) / (b[1] - a[1])
        via = to_world(*(a + k * (b - a)))
        # A via on a vertex of the wire replaces that vertex rather than
        # adding a zero-length step beside it.
        head = np.vstack([P[:j - 1], via]) if np.linalg.norm(P[j - 1] - via) < 1e-6 else np.vstack([P[:j], via])
        stub = np.vstack([via, P[j + 1:]]) if np.linalg.norm(P[j] - via) < 1e-6 else np.vstack([via, P[j:]])
        return head, stub, via

    # The wires that need a via, in pad order, so that neighbours alternate rows.
    todo = []
    for r, is_col in tagged:
        layer = _route_layer(r, is_col)
        pad = pads.get(int(getattr(r, "pad_num", -1) or -1))
        if layer == contact or pad is None or r.polyline_xy is None or len(r.polyline_xy) < 2:
            continue
        P, reverse = ends_on_finger(np.asarray(r.polyline_xy, dtype=float), pad)
        if P is not None:
            todo.append((int(pad["pad_num"]), r, P, reverse))
    todo.sort(key=lambda item: item[0])

    done = {}
    for pad_num, r, P, reverse in todo:
        own = VIA_ROW_SETBACK_MM[0 if pad_num % 2 else 1]
        other = VIA_ROW_SETBACK_MM[1 if pad_num % 2 else 0]
        for setback in (own, other, *VIA_SETBACK_FALLBACKS_MM):
            got = cut(P, stiff_ly + setback)
            if got is None:
                continue
            head, stub, via = got
            if clear(via, stub, r.net_name):
                if reverse:
                    head, stub = head[::-1], stub[::-1]
                done[id(r)] = (head, stub, via)
                placed_vias.append((Point(float(via[0]), float(via[1])), r.net_name))
                placed_stubs.append((LineString(stub), r.net_name))
                out.vias.append(TransitionVia(xy=via, net_name=r.net_name, pad_num=pad_num,
                                              dia=via_dia, drill=via_drill))
                break
        else:
            out.drilled_pads.add(pad_num)

    new_col, new_row, stubs = [], [], []
    for r, is_col in tagged:
        if id(r) not in done:
            (new_col if is_col else new_row).append(r)
            continue
        head, stub, _ = done[id(r)]
        (new_col if is_col else new_row).append(replace(r, polyline_xy=head))
        stubs.append(replace(r, polyline_xy=stub, layer=None if contact == "F.Cu" else "B.Cu"))
    if contact == "F.Cu":
        out.col_routes, out.row_routes = new_col + stubs, new_row
    else:
        out.col_routes, out.row_routes = new_col, new_row + stubs
    return out
