"""The connector's gold fingers: plain pads on the contact face, no drill.

contact_side "bottom" (the default) or "top": the fingers are on that face's
copper only, the coverlay is opened across them on that face only, the
stiffener note names the other face, and every wire that reaches the
connector on another copper layer changes to the contact face through a via
in the tail, outside the stiffener. The taxels, the routes and the connector
pads are the same for either face. See export/connector_contacts.py.
"""
import base64
import io
import json
import re
import shutil
import zipfile
from unittest import mock

import numpy as np
import pytest
from shapely.geometry import MultiPoint, Point, Polygon

from opentac.cad import main
from opentac.cad.export.connector_contacts import contact_transitions
from opentac.cad.pipeline.types import Params, Route

KICAD = shutil.which("kicad-cli")
SQUARE = [[-15, -15], [15, -15], [15, 15], [-15, 15]]
CONTACT = {"bottom": "B.Cu", "top": "F.Cu"}


def _generate(direct=False, **kw):
    real = shutil.which
    with mock.patch("opentac.cad.main.shutil.which",
                    (lambda n: None if n == "kicad-cli" else real(n)) if direct else real):
        resp = main.generate(main.GenerateRequest(outline=SQUARE, router="hug", board_mode="expand", **kw))
    return resp, zipfile.ZipFile(io.BytesIO(base64.b64decode(resp.zip_b64)))


def _j1(pcb):
    i = pcb.index('_pos_0.5mm_FPC_resolved')
    return pcb[i:pcb.index('\n  )', i)]


def _drill_holes(z):
    return [tuple(map(float, m)) for m in
            re.findall(r'^X(-?[\d.]+)Y(-?[\d.]+)', z.read("gerber/board.drl").decode(), re.M)]


def _wires_needing_vias(resp, side, layers):
    """Connector nets whose wire reaches the connector on a layer other than
    the contact face, read from the routes in edit_data."""
    nets = set()
    for r in resp.edit_data["routes"]:
        layer = r.get("layer") or ("F.Cu" if r["net_name"].startswith("COL") else "B.Cu")
        if layer != CONTACT[side]:
            nets.add(r["net_name"])
    return nets


@pytest.mark.parametrize("layers", [2, 4])
@pytest.mark.parametrize("side", ["bottom", "top"])
def test_fingers_vias_and_stiffener(side, layers):
    resp, z = _generate(contact_side=side, copper_layers=layers)
    assert resp.drc["violations"] == 0, resp.drc
    pcb = z.read("board.kicad_pcb").decode()
    j1 = _j1(pcb)
    pads = re.findall(r'\(pad "(\d+)" (\w+) custom[^\n]*', j1)
    assert len(pads) == resp.stats["connector_pos"]
    assert {kind for _, kind in pads} == {"smd"}
    lines = [l for l in j1.splitlines() if '(pad "' in l]
    assert not any("(drill" in l for l in lines)
    assert {re.search(r'\(layers ([^)]*)\)', l).group(1) for l in lines} == {f'"{CONTACT[side]}"'}
    masks = re.findall(r'\(fill solid\) \(layer "([FB])\.Mask"\)', j1)
    assert masks == ["B" if side == "bottom" else "F"]
    note = re.findall(r'fp_text user "([^"]*stiffener[^"]*)"', j1)
    assert len(note) == 1
    if side == "bottom":
        assert "stiffener on TOP" in note[0] and "contacts on BOTTOM" in note[0]
    else:
        assert "stiffener on BOTTOM" in note[0] and "contacts on TOP" in note[0]

    # One via per wire that reaches the connector on another layer, and every
    # hole on the board is a taxel via or one of these.
    vias = [(float(a), -float(b)) for a, b in re.findall(r'^  \(via \(at ([-\d.]+) ([-\d.]+)\)', pcb, re.M)]
    assert len(vias) == len(_wires_needing_vias(resp, side, layers))
    if layers == 2:
        assert len(vias) == (resp.stats["col_routes"] if side == "bottom" else len(
            {r["net_name"] for r in resp.edit_data["routes"] if r["net_name"].startswith("ROW")}))
    assert len(_drill_holes(z)) == resp.stats["active_pixels"] + len(vias)

    # No hole in a finger or under the stiffener (its Dwgs.User outline).
    stiff = re.findall(r'\(fp_line \(start ([-\d.]+) ([-\d.]+)\) \(end ([-\d.]+) ([-\d.]+)\)[^\n]*"Dwgs\.User"', j1)
    pts = [(float(a), -float(b)) for s in stiff for a, b in (s[:2], s[2:])]
    stiffener = MultiPoint(pts).convex_hull
    finger_rects = []
    for l in lines:
        ax, ay = map(float, re.search(r'\(at ([-\d.]+) ([-\d.]+)', l).groups())
        corners = [(ax + float(x), -(ay + float(y))) for x, y in re.findall(r'\(xy ([-\d.]+) ([-\d.]+)\)', l)]
        finger_rects.append(Polygon(corners))
    r = main.Params().via_dia_mm / 2
    for v in vias:
        disc = Point(v).buffer(r)
        assert not disc.intersects(stiffener), v
        assert not any(disc.intersects(f) for f in finger_rects), v


def test_contact_side_changes_no_taxel_route_or_pad():
    a, za = _generate(contact_side="bottom")
    b, zb = _generate(contact_side="top")
    ma, mb = json.loads(za.read("manifest.json")), json.loads(zb.read("manifest.json"))
    for key in ("pixels", "rp", "cp", "edge_cuts"):
        assert ma[key] == mb[key], key
    assert a.edit_data["routes"] == b.edit_data["routes"]
    # The default is not written, so a bottom-contact manifest is what it was
    # before the setting existed.
    assert "contact_side" not in ma["design"]["params"]
    assert mb["design"]["params"]["contact_side"] == "top"


@pytest.mark.parametrize("side", ["bottom", "top"])
def test_both_writers_drill_the_same_holes(side):
    _, zk = _generate(contact_side=side)
    _, zd = _generate(direct=True, contact_side=side)
    hk = sorted((round(x, 3), round(y, 3)) for x, y in _drill_holes(zk))
    hd = sorted((round(x, 3), round(y, 3)) for x, y in _drill_holes(zd))
    if KICAD is None:
        pytest.skip("kicad-cli not installed")
    assert len(hk) == len(hd)
    assert np.allclose(np.array(hk), np.array(hd), atol=1e-3)


def test_transition_via_sits_on_the_wire():
    """The via goes on the wire where it crosses its via row, wherever the
    wire runs, and the rest of the wire keeps its path on the contact face. A
    wire that does not reach the row gets no via and is reported. Connector
    frame here: the tip toward -y, the board toward +y, local y = world y."""
    t_hat, n_out = np.array([1.0, 0.0]), np.array([0.0, -1.0])
    pad = {"pad_num": 3, "center": np.array([0.0, -1.5]), "size": (0.3, 3.5)}
    model = {"pads": [pad], "t_hat": t_hat, "n_out": n_out, "pad1_world": np.array([0.0, 0.0]),
             # the stiffener outline: tip at y = -3.25, inner edge at y = 2.75
             "user_lines": [{"start": (-1.0, 2.75), "end": (1.0, 2.75), "layer": "Dwgs.User"},
                            {"start": (-1.0, -3.25), "end": (1.0, -3.25), "layer": "Dwgs.User"}],
             "edge_lines": []}
    # wires drawn from the finger to the taxels, as the router draws them
    straight = Route("COL_0", 3, np.zeros(2), np.zeros(2), np.array([[0.0, -1.5], [0.0, 10.0]]))
    angled = Route("COL_1", 3, np.zeros(2), np.zeros(2), np.array([[0.0, -1.5], [0.0, 3.0], [2.0, 10.0]]))
    short = Route("COL_2", 3, np.zeros(2), np.zeros(2), np.array([[0.0, -1.5], [0.0, 3.0]]))

    out = contact_transitions(model, Params(), [straight], [])
    assert len(out.vias) == 1 and not out.problems
    assert np.allclose(out.vias[0].xy, [0.0, 2.75 + 1.0])       # odd pad: the nearer row
    (head,) = out.col_routes
    (stub,) = out.row_routes
    assert np.allclose(head.polyline_xy, [[0.0, 3.75], [0.0, 10.0]]) and head.layer is None
    assert np.allclose(stub.polyline_xy, [[0.0, -1.5], [0.0, 3.75]]) and stub.layer == "B.Cu"

    out = contact_transitions(model, Params(contact_side="top"), [straight], [])
    assert not out.vias and out.col_routes == [straight]         # already on the contact face

    out = contact_transitions(model, Params(), [angled], [])
    assert len(out.vias) == 1 and not out.problems
    via = out.vias[0].xy
    assert via[1] == pytest.approx(3.75) and via[0] == pytest.approx(2.0 * 0.75 / 7.0)
    (head,) = out.col_routes
    (stub,) = out.row_routes
    assert np.allclose(head.polyline_xy, [via, [2.0, 10.0]])
    assert np.allclose(stub.polyline_xy, [[0.0, -1.5], [0.0, 3.0], via])

    # a wire that never reaches a via line keeps the drilled finger instead
    out = contact_transitions(model, Params(), [short], [])
    assert not out.vias and out.drilled_pads == {3} and out.col_routes == [short]
    assert not out.problems


def _model_for(pads):
    return {"pads": pads, "t_hat": np.array([1.0, 0.0]), "n_out": np.array([0.0, -1.0]),
            "pad1_world": np.array([0.0, 0.0]),
            "user_lines": [{"start": (-5.0, 2.75), "end": (5.0, 2.75), "layer": "Dwgs.User"}],
            "edge_lines": []}


def test_via_moves_off_a_conflict_or_falls_back_to_the_drilled_finger():
    """A via goes on the first line in front of the stiffener where it and the
    wire it moves clear everything else; with no such line the finger keeps
    its plated hole and the wire its layer (the earlier construction)."""
    pad = {"pad_num": 3, "center": np.array([0.0, -1.5]), "size": (0.3, 3.5)}
    col = Route("COL_0", 3, np.zeros(2), np.zeros(2), np.array([[0.0, -1.5], [0.0, 10.0]]))
    # a bottom-layer row wire running sideways 1.0 mm in front of the
    # stiffener blocks the first row (the via would sit on it) ...
    across = Route("ROW_0", 9, np.zeros(2), np.zeros(2), np.array([[-5.0, 3.75], [5.0, 3.75]]), "B.Cu")
    out = contact_transitions(_model_for([pad]), Params(), [col], [across])
    # ... and the moved part of the column would cross it from any line in
    # front of it, so the finger keeps its drill and nothing crosses
    assert out.drilled_pads == {3} and not out.vias
    assert out.col_routes == [col] and out.row_routes == [across]

    # the same wire beyond the via lines leaves the first row free
    beyond = Route("ROW_0", 9, np.zeros(2), np.zeros(2), np.array([[-5.0, 8.0], [5.0, 8.0]]), "B.Cu")
    out = contact_transitions(_model_for([pad]), Params(), [col], [beyond])
    assert len(out.vias) == 1 and out.vias[0].xy[1] == pytest.approx(3.75) and not out.drilled_pads

    # a neighbour's wire 0.45 mm away on any layer: too close for a via (0.475 mm)
    near = Route("COL_1", 4, np.zeros(2), np.zeros(2), np.array([[0.45, -1.5], [0.45, 10.0]]))
    pad4 = {"pad_num": 4, "center": np.array([0.45, -1.5]), "size": (0.3, 3.5)}
    out = contact_transitions(_model_for([pad, pad4]), Params(contact_side="top"), [col, near], [])
    assert not out.vias and not out.drilled_pads                 # both already on the top face
    out = contact_transitions(_model_for([pad, pad4]), Params(), [col, near], [])
    assert out.drilled_pads == {3, 4} and not out.vias


def test_drilled_fallback_pad_is_the_footprint_pad():
    from opentac.cad.export.connector_contacts import contact_pads, with_drilled
    raw = [{"pad_num": 1, "kind": "thru_hole", "drill": 0.2, "layers": ["*.Cu", "B.Mask"]},
           {"pad_num": 2, "kind": "thru_hole", "drill": 0.2, "layers": ["*.Cu", "B.Mask"]}]
    for side, mask, cu in (("bottom", "B.Mask", "B.Cu"), ("top", "F.Mask", "F.Cu")):
        pads = contact_pads(raw, Params(contact_side=side))
        assert [(p["kind"], p["drill"], p["layers"]) for p in pads] == [("smd", None, [cu, mask])] * 2
        back = with_drilled(pads, {2}, Params(contact_side=side))
        assert (back[0]["kind"], back[0]["drill"]) == ("smd", None)
        assert (back[1]["kind"], back[1]["drill"], back[1]["layers"]) == ("thru_hole", 0.2, ["*.Cu", mask])


def test_unknown_contact_side_is_refused():
    with pytest.raises(Exception):
        _generate(contact_side="left")


def test_drc_checks_transition_vias():
    """A via too close to another net's wire, to another via, or to the board
    edge is a violation; its own net's wire is not."""
    from types import SimpleNamespace as V
    from opentac.cad.pipeline.drc import drc_check
    board = [[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]]
    own = Route("COL_0", 1, np.zeros(2), np.zeros(2), np.array([[0.0, -3.0], [0.0, 3.0]]))
    near = Route("COL_1", 2, np.zeros(2), np.zeros(2), np.array([[0.45, -3.0], [0.45, 3.0]]))
    far = Route("COL_2", 3, np.zeros(2), np.zeros(2), np.array([[0.5, -3.0], [0.5, 3.0]]))
    via = V(xy=np.array([0.0, 0.0]), dia=0.35, net_name="COL_0")
    # 0.475 mm needed (0.175 + 0.1 + 0.2): 0.45 is too close, 0.5 is clear
    v = drc_check([own, near], [], 0.2, 0.2, np.array(board), 0.2, vias=[via])
    assert [(x.type, x.net2) for x in v] == [("via clearance", "COL_1")]
    assert not drc_check([own, far], [], 0.2, 0.2, np.array(board), 0.2, vias=[via])
    other = V(xy=np.array([0.5, 0.0]), dia=0.35, net_name="COL_2")
    assert [x.type for x in drc_check([], [], 0.2, 0.2, np.array(board), 0.2, vias=[via, other])] == ["via clearance"]
    edge = V(xy=np.array([4.7, 0.0]), dia=0.35, net_name="COL_0")
    assert [x.type for x in drc_check([], [], 0.2, 0.2, np.array(board), 0.2, vias=[edge])] == ["via edge"]
