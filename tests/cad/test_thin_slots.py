"""A slot or slit narrower than a taxel must not cut through taxels or their wires.

Found on the Tesollo boards (2026-10-23): in fixed-outline mode a taxel was kept
when its four corners were inside the outline, so a 0.5 mm slot passed between
the corners of a 2.5 mm taxel. A column of taxels and the row wires beside them
ran across the slot, the tool's DRC said 0, and KiCad's DRC reported 30 errors.
"""
import base64
import io
import json
import zipfile

import numpy as np
from shapely.geometry import LineString, Polygon, box

from opentac.cad.main import GenerateRequest, generate
from opentac.cad.pipeline.drc import TaxelCopper, drc_check


def _slot_board(mode, layers, xc=21.0):
    a, b = xc - 0.25, xc + 0.25
    outline = [[0, 0], [40, 0], [40, 40], [b, 40], [b, 15], [a, 15], [a, 40], [0, 40]]
    res = generate(GenerateRequest(outline=outline, router="hug", board_mode=mode, copper_layers=layers,
                                   pixel_w_mm=2.5, pixel_h_mm=2.5, pitch_x_mm=2.5, pitch_y_mm=2.5))
    z = zipfile.ZipFile(io.BytesIO(base64.b64decode(res.zip_b64)))
    return res, json.loads(z.read("manifest.json"))


def _taxels_across(manifest):
    board = Polygon(manifest["edge_cuts"]).buffer(1e-6)
    w = manifest["w"] / 2
    return [p for p in manifest["pixels"] if not board.contains(box(p[0] - w, p[1] - w, p[0] + w, p[1] + w))]


def test_no_taxel_sits_across_a_slot_in_a_fixed_outline():
    res, m = _slot_board("fixed_keepout", 4)
    assert _taxels_across(m) == []
    assert res.drc.get("violations") == 0
    assert len(m["pixels"]) > 190          # only the column the slot cuts is lost


def test_grow_mode_keeps_the_taxels_its_board_closes_the_slot_over():
    """Grow mode builds a larger board around the drawn outline, which closes a
    narrow slit; the taxels there are on the board and stay."""
    res, m = _slot_board("expand", 2)
    assert _taxels_across(m) == []
    assert len(m["pixels"]) == 225


def test_drc_reports_a_taxel_the_board_edge_cuts_through():
    board = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    taxels = TaxelCopper(np.array([[5.0, 5.0], [9.5, 5.0]]), ["C1", "C2"], set(), 2.0, 2.0, 0.1)
    found = drc_check([], [], 0.2, 0.2, board_poly=board, taxels=taxels)
    hits = [v for v in found if v.type == "taxel_outside_board"]
    assert len(hits) == 1 and hits[0].net1 == "TAXEL:C2"


def test_rows_a_slit_cuts_are_jumped_and_the_board_sets_up():
    """The rows a slit cuts in two must stay one row each, joined around the slit
    (on a 4-layer board, on In2), so that `opentac setup` can read the board.
    Split nets put two logical rows on one row of taxels, which the readout
    cannot place; the layout scoring now prefers the joined result."""
    from collections import defaultdict
    from opentac.device import device_config
    from opentac.device.config import infer_board

    res, m = _slot_board("fixed_keepout", 4)
    lines = defaultdict(set)
    for x, y, r, c in (p[:4] for p in m["pixels"]):
        lines[round(y, 3)].add(r)
    assert all(len(v) == 1 for v in lines.values())
    device_config(m, infer_board(m))
