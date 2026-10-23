"""Solder mask (coverlay) openings JLCPCB's flex process can make.

A flex board's mask is a polyimide coverlay: the film between two openings must
be at least 0.3 mm. The export opens one window per taxel on the top, sized
from the pitch (trimmed into the electrodes on a dense board, with a note),
and on the bottom only one window across the connector's gold fingers: the
row vias are covered there. Checked on the KiCad board file, on the Gerbers
kicad-cli plots from it, and on the pure-Python Gerber writer's output.
"""
import base64
import io
import re
import shutil
import zipfile
from unittest import mock

import numpy as np
import pytest
from shapely.geometry import Point, Polygon, box

from opentac.cad import main
from opentac.cad.export.coverlay import COVERLAY_WEB_MM, opening_web, taxel_opening

KICAD = shutil.which("kicad-cli")
SQUARE = [[-20, -20], [20, -20], [20, 20], [-20, 20]]
TRIM_NOTE = "Taxel openings trimmed 0.05 mm into the electrodes to keep JLCPCB's 0.3 mm coverlay spacing."
BOARDS = {
    # 4.0 mm taxels on a 4.2 mm pitch: 0.4 mm copper gap, 0.05 mm margin
    "default": {},
    # 2.0 mm taxels on a 2.0 mm pitch: 0.2 mm copper gap, trimmed 0.05 mm
    "dense": {"pixel_w_mm": 2.0, "pixel_h_mm": 2.0, "pitch_x_mm": 2.0, "pitch_y_mm": 2.0},
}


def _generate(kw, direct=False):
    real = shutil.which
    with mock.patch("opentac.cad.main.shutil.which",
                    (lambda n: None if n == "kicad-cli" else real(n)) if direct else real):
        resp = main.generate(main.GenerateRequest(outline=SQUARE, router="hug", board_mode="expand", **kw))
    return resp, zipfile.ZipFile(io.BytesIO(base64.b64decode(resp.zip_b64)))


def _min_web(rects):
    """Smallest distance between any two of the polygons."""
    best = np.inf
    for i, a in enumerate(rects):
        for b in rects[i + 1:]:
            if a.distance(b) < 1.0:
                best = min(best, a.distance(b))
    return best


def _footprints(pcb):
    """(reference, x, y, block text) for every footprint in a .kicad_pcb: the
    exporter writes each top-level item from a line of its own, indented two
    spaces, and everything inside a footprint deeper."""
    out = []
    for block in re.split(r'\n(?=  \()', pcb):
        if not block.startswith('  (footprint "'):
            continue
        at = re.search(r'\n    \(at ([-\d.]+) ([-\d.]+)', block)
        ref = re.search(r'(?:reference|"Reference") "([^"]+)"', block)
        out.append((ref.group(1), float(at.group(1)), float(at.group(2)), block))
    return out


def test_opening_margin_from_the_pitch():
    o = taxel_opening(4.0, 4.0, 0.1, 4.2, 4.2)
    assert (o.margin_x, o.margin_y, o.note) == (0.05, 0.05, "")
    o = taxel_opening(4.0, 4.0, 0.1, 4.3, 4.3)          # 0.5 mm gap: JLCPCB's 0.075 mm margin
    assert (o.margin_x, o.margin_y) == (0.075, 0.075)
    o = taxel_opening(2.0, 2.0, 0.1, 2.0, 2.0)
    assert (o.margin_x, o.margin_y, o.half_w) == (-0.05, -0.05, 0.85)
    assert o.note == TRIM_NOTE
    o = taxel_opening(2.0, 2.0, 0.1, 2.0, 2.1)          # only the columns are tight
    assert (o.margin_x, o.margin_y) == (-0.05, 0.0)
    assert o.note.startswith("Taxel openings trimmed into the electrodes, 0.05 mm at the left and right")


def test_off_grid_taxel_shrinks_every_opening():
    centers = [(0, 0), (4.2, 0), (8.4, 0), (4.2, 4.2), (8.0, 4.25)]  # the last is 0.4 mm off the grid
    o = taxel_opening(4.0, 4.0, 0.1, 4.2, 4.2, centers)
    c = np.asarray(centers, dtype=float)
    d = c[:, None, :] - c[None, :, :]
    webs = opening_web(d[..., 0], d[..., 1], o.half_w, o.half_h)[np.triu_indices(len(c), 1)]
    assert webs.min() >= COVERLAY_WEB_MM - 1e-6
    assert o.margin_x < 0.05 and o.margin_y < 0.05


@pytest.mark.parametrize("name", sorted(BOARDS))
def test_board_mask_openings(name):
    """One F.Mask rectangle per taxel, 0.3 mm apart, around (or trimmed into)
    the electrodes; no pad opens a mask of its own; on B.Mask one opening
    across the connector fingers and none at a via; no paste anywhere."""
    resp, z = _generate(BOARDS[name])
    pcb = z.read("board.kicad_pcb").decode()
    assert resp.stats["mask_note"] == (TRIM_NOTE if name == "dense" else "")
    assert not any("Paste" in n for n in z.namelist()) and "Paste" not in pcb
    for pad in re.findall(r'\(pad [^\n]*', pcb):
        assert ".Mask" not in pad, pad

    fps = _footprints(pcb)
    taxels = [f for f in fps if f[0].startswith("PX")]
    assert len(taxels) == resp.stats["active_pixels"]
    w = BOARDS[name].get("pixel_w_mm", 4.0)
    m = resp.stats["coverlay"]["margin_x_mm"]
    assert m == (-0.05 if name == "dense" else 0.05)
    openings = []
    for ref, x, y, block in taxels:
        rects = re.findall(r'\(fp_rect \(start ([-\d.]+) ([-\d.]+)\) \(end ([-\d.]+) ([-\d.]+)\)'
                           r'[^\n]*\(fill solid\) \(layer "F\.Mask"\)', block)
        assert len(rects) == 1, ref
        x1, y1, x2, y2 = map(float, rects[0])
        # the electrode copper spans the taxel less edge_clear (0.1 mm) on each side
        assert x2 == pytest.approx(w / 2 - 0.1 + m, abs=1e-9) and x1 == -x2 and y1 == -y2
        openings.append(box(x + x1, y + y1, x + x2, y + y2))
    assert _min_web(openings) >= COVERLAY_WEB_MM - 1e-6

    j1 = [f for f in fps if f[0] == "J1"][0][3]
    assert "F.Mask" not in j1
    polys = re.findall(r'\(fp_poly \(pts ((?:\(xy [-\d.]+ [-\d.]+\) ?)+)\)[^\n]*\(layer "B\.Mask"\)', j1)
    assert len(polys) == 1
    finger_opening = Polygon([tuple(map(float, p)) for p in re.findall(r'\(xy ([-\d.]+) ([-\d.]+)\)', polys[0])])
    # Plain pads on the bottom copper, no drill (export/connector_contacts.py)
    fingers = re.findall(r'\(pad "\d+" smd custom[^\n]*\(at ([-\d.]+) ([-\d.]+) 0\)', j1)
    assert len(fingers) == resp.stats["connector_pos"]
    assert not re.search(r'\(pad "\d+"[^\n]*\(drill', j1)
    assert all(finger_opening.buffer(1e-6).contains(Point(float(a), float(b))) for a, b in fingers)
    vias = [(x + float(a), y + float(b)) for _, x, y, block in taxels
            for a, b in re.findall(r'thru_hole circle[^\n]*\(at ([-\d.]+) ([-\d.]+)\)', block)]
    assert len(vias) == len(taxels)
    vias += [(float(a), float(b)) for a, b in re.findall(r'^  \(via \(at ([-\d.]+) ([-\d.]+)\)', pcb, re.M)]
    assert not any(finger_opening.buffer(0.3).contains(Point(v)) for v in vias)
    assert sum("B.Mask" in f[3] for f in fps) == 1          # only the connector opens the bottom


def _kicad_regions(text):
    """Every G36/G37 region of a KiCad-plotted Gerber, as polygons (mm)."""
    polys = []
    for body in re.findall(r'G36\*\n(.*?)G37\*', text, re.S):
        pts = [(int(x) / 1e6, int(y) / 1e6) for x, y in re.findall(r'X(-?\d+)Y(-?\d+)D0[12]', body)]
        polys.append(Polygon(pts))
    return polys


@pytest.mark.skipif(KICAD is None, reason="kicad-cli not installed")
@pytest.mark.parametrize("name", sorted(BOARDS))
def test_plotted_mask_gerbers(name):
    resp, z = _generate(BOARDS[name])
    assert resp.stats["gerber_exporter"] == "kicad-cli"
    top = z.read("gerber/board-F_Mask.gts").decode()
    bottom = z.read("gerber/board-B_Mask.gbs").decode()
    assert "D03*" not in top and "D03*" not in bottom       # no per-pad flashes left
    front = _kicad_regions(top)
    assert len(front) == resp.stats["active_pixels"]
    assert _min_web(front) >= COVERLAY_WEB_MM - 1e-6
    back = _kicad_regions(bottom)
    assert len(back) == 1
    # the finger row: 0.5 mm pitch, 0.3 mm fingers
    n = resp.stats["connector_pos"]
    assert max(back[0].bounds[2] - back[0].bounds[0], back[0].bounds[3] - back[0].bounds[1]) \
        == pytest.approx((n - 1) * 0.5 + 0.3, abs=1e-5)


@pytest.mark.parametrize("name", sorted(BOARDS))
def test_direct_gerber_writer_masks(name):
    """The no-KiCad fallback (the hosted service) plots the same openings."""
    resp, z = _generate(BOARDS[name], direct=True)
    assert resp.stats["gerber_exporter"] == "direct"
    top = z.read("gerber/board-F_Mask.gts").decode()
    (w, h), = [tuple(map(float, a)) for a in re.findall(r'%ADD\d+R,([\d.]+)X([\d.]+)\*%', top)]
    assert w == pytest.approx(resp.stats["coverlay"]["opening_mm"][0], abs=1e-6)
    centers = [(int(x) / 1e6, int(y) / 1e6) for x, y in re.findall(r'X(-?\d+)Y(-?\d+)D03', top)]
    assert len(centers) == resp.stats["active_pixels"]
    front = [box(x - w / 2, y - h / 2, x + w / 2, y + h / 2) for x, y in centers]
    assert _min_web(front) >= COVERLAY_WEB_MM - 1e-6
    bottom = z.read("gerber/board-B_Mask.gbs").decode()
    assert "D03*" not in bottom and bottom.count("G36*") == 1
