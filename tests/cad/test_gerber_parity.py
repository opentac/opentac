"""The pure-Python Gerber and drill writer draws what KiCad plots.

Every board ordered from the hosted service is made from export/gerber.py and
export/excellon.py: the service image has no KiCad. With kicad-cli installed
the same export is plotted by KiCad from the .kicad_pcb instead. The two must
agree, among other things, on the sign of Y (a built-in layer or drill file
mirrored with respect to the design would give mirrored boards from the hosted
files) and on the via rings of the top copper.

The parity tests generate each board once and keep both writers' output (the
KiCad plot of the board file, and the built-in writer run on the same layout),
then compare what the files draw, read back into geometry by
tests/cad/gerber_geometry.py, not their text:

  * copper, mask, Edge.Cuts, B.Silkscreen, User.Drawings: the symmetric
    difference of the two drawings is at most 0.5 % of the layer's area and no
    connected piece of it exceeds 0.05 mm^2. (On the 31 example designs x 2/4
    layers x grow/fixed the difference is at most 0.15 % and 0.0005 mm^2, all
    of it on F.Cu, where KiCad plots the round finger ends as polygons.)
  * drill: the same holes in the same sizes, every position within 0.01 mm.
  * F.Silkscreen: the design name in the same place, bounding-box centres
    within 0.5 mm. The glyphs differ by design: KiCad strokes the label with
    its own font, the built-in writer with export/stroke_font.py.
  * User.Comments: KiCad strokes the connector's stiffener note as text; the
    built-in writer carries it as G04 comment lines. Not compared.

They need kicad-cli and skip without it. test_builtin_writer_orientation runs
everywhere: it checks the built-in files against the manifest, which is in the
same Y-up frame, so a sign flip fails it with or without KiCad.
"""
import base64
import io
import json
import re
import shutil
import zipfile

import numpy as np
import pytest

from opentac.cad import main
from opentac.cad.export.gerber import _arc_to_segments
from opentac.cad.export.kicad_footprint import pixel_geometry
from tests.cad.gerber_geometry import drill_difference, excellon_hits, layer_difference

KICAD = shutil.which("kicad-cli")

SQUARE = [[-20, -20], [20, -20], [20, 20], [-20, 20]]
L_SHAPE = [[0, 0], [40, 0], [40, 20], [12, 20], [12, 50], [0, 50]]
SQUARE_126 = [[-63, 0], [63, 0], [63, 126], [-63, 126]]
BOARDS = {
    "square_2L": dict(outline=SQUARE),
    "lshape_2L": dict(outline=L_SHAPE),
    # every pin of the 60-pin connector in use: the tail is as wide as its tab
    "60pin_full_width": dict(outline=SQUARE_126, readout_board="60pin",
                             target_cols=30, target_rows=30),
    "square_4L": dict(outline=SQUARE, copper_layers=4),
    "lshape_fixed": dict(outline=L_SHAPE, board_mode="fixed_keepout"),
}

MAX_REL_XOR = 0.005          # of the layer's area
MAX_XOR_PIECE_MM2 = 0.05
MAX_DRILL_OFFSET_MM = 0.01
MAX_LABEL_SHIFT_MM = 0.5
# compared on what the label is, not how it is drawn (module docstring)
LABEL_LAYER = "board-F_Silkscreen.gto"
NOT_COMPARED = {"board-User_Comments.gbr"}


def _request(kw):
    return main.GenerateRequest(router="hug", design_name="parity_check", **kw)


def both_writers(kw, monkeypatch):
    """(response, KiCad's files, the built-in writer's files) for one board.

    One generate: KiCad's plot is kept and the export is then sent down the
    no-KiCad path, so both writers draw the very same layout and routes.
    """
    plotted = {}
    real = main._manufacturing_files_from_kicad

    def plot_then_fall_back(pcb_text, copper_layers=2):
        plotted["kicad"] = real(pcb_text, copper_layers)
        return None, None, "direct"

    monkeypatch.setattr(main, "_manufacturing_files_from_kicad", plot_then_fall_back)
    resp = main.generate(_request(kw))
    monkeypatch.undo()
    gerbers, drill, source = plotted["kicad"]
    assert source == "kicad-cli", "kicad-cli did not plot the board"
    kicad = dict(gerbers, **{"board.drl": drill})
    z = zipfile.ZipFile(io.BytesIO(base64.b64decode(resp.zip_b64)))
    builtin = {n[len("gerber/"):]: z.read(n).decode() for n in z.namelist() if n.startswith("gerber/")}
    assert resp.stats["gerber_exporter"] == "direct"
    return resp, kicad, builtin


def parity_failures(kicad, builtin):
    """Every way the built-in files differ from KiCad's beyond tolerance."""
    bad = []
    if set(kicad) != set(builtin):
        bad.append(f"file sets differ: KiCad only {sorted(set(kicad) - set(builtin))}, "
                   f"built-in only {sorted(set(builtin) - set(kicad))}")
    for name in sorted(set(kicad) & set(builtin)):
        if name in NOT_COMPARED:
            continue
        if name.endswith(".drl"):
            d = drill_difference(kicad[name], builtin[name])
            if d["count_a"] != d["count_b"] or not d["same_tools"] \
                    or d["max_offset"] > MAX_DRILL_OFFSET_MM:
                bad.append(f"{name}: {d}")
            continue
        d = layer_difference(kicad[name], builtin[name])
        if name == LABEL_LAYER:
            if d["bounds_a"] is None or d["bounds_b"] is None:
                bad.append(f"{name}: the label is drawn by only one writer")
                continue
            ca = np.reshape(d["bounds_a"], (2, 2)).mean(axis=0)
            cb = np.reshape(d["bounds_b"], (2, 2)).mean(axis=0)
            if np.hypot(*(ca - cb)) > MAX_LABEL_SHIFT_MM:
                bad.append(f"{name}: label centred at {cb.round(3)}, KiCad's at {ca.round(3)}")
            continue
        if d["rel"] > MAX_REL_XOR or d["max_piece"] > MAX_XOR_PIECE_MM2:
            bad.append(f"{name}: XOR {d['xor']:.4f} mm^2 ({100 * d['rel']:.3f} % of "
                       f"{max(d['area_a'], d['area_b']):.1f}), largest piece {d['max_piece']:.5f} mm^2")
    return bad


@pytest.mark.skipif(KICAD is None, reason="kicad-cli not installed: no KiCad plot to compare against")
@pytest.mark.parametrize("name", sorted(BOARDS))
def test_builtin_gerbers_match_kicad(name, monkeypatch):
    _resp, kicad, builtin = both_writers(BOARDS[name], monkeypatch)
    assert parity_failures(kicad, builtin) == []


# ---- without KiCad ------------------------------------------------------------

def _points(text, ops="123"):
    """Every coordinate a Gerber file writes with these D01/D02/D03 operations,
    (x, y) in mm (format 4.6)."""
    import re
    return np.array([(int(x) / 1e6, int(y) / 1e6)
                     for x, y in re.findall(r"X(-?\d+)Y(-?\d+)D0[" + ops + "]", text)],
                    dtype=float).reshape(-1, 2)


def _flashes(text, aperture_def):
    """Positions flashed with the aperture `aperture_def` (e.g. "C,0.350000")."""
    import re
    code = re.search(r"%ADD(\d+)" + re.escape(aperture_def) + r"\*%", text).group(1)
    out, current = [], None
    for line in text.splitlines():
        m = re.fullmatch(r"D(\d+)\*", line)
        if m:
            current = m.group(1)
            continue
        m = re.fullmatch(r"X(-?\d+)Y(-?\d+)D03\*", line)
        if m and current == code:
            out.append((int(m.group(1)) / 1e6, int(m.group(2)) / 1e6))
    return np.array(out, dtype=float)


def _connector_tab(pcb_text):
    """J1's Edge.Cuts points (lines, and arcs sampled finely) in the board frame."""
    import re
    j1 = re.search(r'\n  \(footprint "[^"]*FPC_resolved".*?\n  \)', pcb_text, re.S).group(0)
    num = r"(-?[\d.]+)"
    pts = []
    for m in re.finditer(r"\(fp_line \(start " + num + " " + num + r"\) \(end " + num + " " + num
                         + r"\)[^\n]*\(layer \"Edge\.Cuts\"\)", j1):
        x1, y1, x2, y2 = map(float, m.groups())
        pts += [(x1, -y1), (x2, -y2)]
    for m in re.finditer(r"\(fp_arc \(start " + num + " " + num + r"\) \(mid " + num + " " + num
                         + r"\) \(end " + num + " " + num + r"\)[^\n]*\(layer \"Edge\.Cuts\"\)", j1):
        x1, y1, xm, ym, x2, y2 = map(float, m.groups())
        pts += _arc_to_segments((x1, -y1), (xm, -ym), (x2, -y2), n_segs=12)
    return np.asarray(pts, dtype=float).reshape(-1, 2)


def _same_points(a, b, tol=1e-3):
    a, b = np.asarray(a, float).reshape(-1, 2), np.asarray(b, float).reshape(-1, 2)
    if len(a) != len(b):
        return False
    from scipy.spatial import cKDTree
    return bool(cKDTree(b).query(a)[0].max() <= tol)


@pytest.mark.parametrize("name", ["lshape_2L", "square_4L"])
def test_builtin_writer_orientation(name, monkeypatch):
    """The built-in files sit in the manifest's frame, the frame KiCad plots in.

    No KiCad needed. The manifest's edge_cuts (the cut line up to the
    connector) plus the connector's own tab outline is exactly the Edge.Cuts
    extent; the taxel vias are where the manifest's taxels put them, in the
    drill file and as rings on the top copper; the top mask opens over the
    taxel centres. A mirrored writer moves all of them.
    """
    monkeypatch.setattr(main.shutil, "which", lambda n: None)
    resp = main.generate(_request(BOARDS[name]))
    assert resp.stats["gerber_exporter"] == "direct"
    z = zipfile.ZipFile(io.BytesIO(base64.b64decode(resp.zip_b64)))
    man = json.loads(z.read("manifest.json"))

    # Edge.Cuts: manifest cut line + connector tab, same extent, same way up.
    # The tab is the connector footprint's own Edge.Cuts in the board file
    # (KiCad's Y-down space, so Y negated back).
    tab = _connector_tab(z.read("board.kicad_pcb").decode())
    assert len(tab), "no connector tab to measure"
    expected = np.vstack([np.asarray(man["edge_cuts"], float), tab])
    got = _points(z.read("gerber/board-Edge_Cuts.gm1").decode())
    np.testing.assert_allclose(got.min(axis=0), expected.min(axis=0), atol=1e-3)
    np.testing.assert_allclose(got.max(axis=0), expected.max(axis=0), atol=1e-3)

    # The vias: drill hits and top-copper rings at every manifest taxel's via
    fp = man["footprint"]
    g = pixel_geometry(man["w"], man["h"], fp["trace_w"], fp["gap"], fp["center_clear"],
                       fp["edge_clear"], fp["via_dia"])
    taxels = np.array([p[:2] for p in man["pixels"]], dtype=float)
    vias = taxels + [g.via_x, -g.via_y]          # footprint frame is Y-down
    # plus the connector's transition vias (export/connector_contacts.py),
    # through vias on every copper layer, as the board file places them
    pcb = z.read("board.kicad_pcb").decode()
    transition = [(float(x), -float(y)) for x, y in re.findall(r"^  \(via \(at (-?[\d.]+) (-?[\d.]+)\)", pcb, re.M)]
    assert transition, "no transition vias"
    vias = np.vstack([vias, transition])
    drill = excellon_hits(z.read("gerber/board.drl").decode())
    assert _same_points([h[1:] for h in drill if abs(h[0] - fp["via_drill"]) < 1e-6], vias)
    top = z.read("gerber/board-F_Cu.gtl").decode()
    assert _same_points(_flashes(top, f"C,{fp['via_dia']:.6f}"), vias)

    # The top mask: one opening centred on each taxel
    mask = z.read("gerber/board-F_Mask.gts").decode()
    assert _same_points(_points(mask, ops="3"), taxels)
