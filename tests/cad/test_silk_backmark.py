"""The back silkscreen describes the board that was exported.

Three failures are possible, and none is visible by looking at a board file:

  1. The QR stops describing the board. It is derived from manifest.json, so a
     manifest change that the codec does not follow produces a mark that
     decodes cleanly to the wrong sensor, which is worse than no mark at all.
     The round trip below rebuilds every taxel position from the payload and
     compares it with the manifest it came from.

  2. Silk lands over the board edge. Clipping the line centreline to the
     outline and then widening it puts the rect corners over the cut line
     wherever the edge is sloped or curved. The fab clips such silk, and the
     defect shows only on the delivered boards.

  3. The mark takes the whole export down with it. A problem with the legend,
     such as a pitch too large for the encoder, must not become an error on
     /generate that costs the user their Gerbers. main.py treats the print PDF
     the same way, for the same reason.
"""
import json
import math

import pytest
from shapely.geometry import Polygon, box


from opentac.cad.export.silk_backmark import (
    DIM_CELL_MARGIN_MM, DIM_STROKE_W_MM, DIM_TEXT_H_MM, EDGE_CLEAR_MM,
    GRID_MINOR_MM, MODULE_MM, PRECOMP_MM, backmark)
from opentac.cad.export.taxel_core import (
    ABSENT, B41, LINK_PREFIX, TaxelCoreError, b41decode, decode, manifest_from_scan,
    qr_link, taxels)
from opentac.cad.main import generate, GenerateRequest

SQUARE = [[-20, -20], [20, -20], [20, 20], [-20, 20]]
U_NOTCH = [[-34, 0], [-12, 0], [-12, 34], [12, 34], [12, 0], [34, 0], [34, 56], [-34, 56]]

CASES = [
    ("square/expand", SQUARE, "expand", 2.0, 2.2),
    ("square/fixed", SQUARE, "fixed_keepout", 2.0, 2.2),
    ("u-notch/expand", U_NOTCH, "expand", 4.0, 4.2),
    # 22.4 mm pitch: a coarse pitch, which the encoder's pitch field holds.
    ("coarse pitch", [[-30, -30], [30, -30], [30, 30], [-30, 30]], "expand", 22.0, 22.4),
]


def _manifest(outline, mode, pw, pitch):
    resp = generate(GenerateRequest(
        outline=outline, router="hug", board_mode=mode,
        pixel_w_mm=pw, pixel_h_mm=pw, pitch_x_mm=pitch, pitch_y_mm=pitch))
    import base64
    import io
    import zipfile
    z = zipfile.ZipFile(io.BytesIO(base64.b64decode(resp.zip_b64)))
    return z.read("manifest.json").decode(), z


@pytest.mark.parametrize("tag,outline,mode,pw,pitch", CASES)
def test_qr_rebuilds_the_manifest_it_came_from(tag, outline, mode, pw, pitch):
    manifest_json, _ = _manifest(outline, mode, pw, pitch)
    mark = backmark(manifest_json)
    assert not mark["warnings"], "%s: %s" % (tag, mark["warnings"])
    assert mark["qr"], "%s: no QR produced" % tag

    core = decode(b41decode(mark["qr"]["payload"]))
    man = json.loads(manifest_json)

    rebuilt = {(r, c): (x, y) for x, y, r, c, _, _ in taxels(core)}
    ref = {(r, c): (x, y) for x, y, r, c in man["pixels"]}
    assert set(rebuilt) == set(ref), "%s: taxel set differs from the manifest" % tag

    worst = max(max(abs(rebuilt[k][0] - ref[k][0]), abs(rebuilt[k][1] - ref[k][1]))
                for k in ref)
    assert worst <= 0.01, "%s: taxel positions off by %.4f mm" % (tag, worst)

    # the pin map is the other half of what a readout needs
    assert core["rp"] == [p for p in man["rp"]], "%s: row pads differ" % tag
    assert core["cp"] == [p for p in man["cp"]], "%s: col pads differ" % tag


@pytest.mark.parametrize("tag,outline,mode,pw,pitch", CASES)
def test_the_pitch_label_sits_inside_one_grid_cell(tag, outline, mode, pw, pitch):
    """The grid lines are the dimension; the label only names it.

    That works only if the text sits inside a cell: a number floating across a
    line does not say which spacing it belongs to, which matters on a board
    with two line weights. Sitting inside one cell also means the label cannot
    cross a grid line, so no keep-out is needed.
    """
    manifest_json, _ = _manifest(outline, mode, pw, pitch)
    mark = backmark(manifest_json)
    assert mark["strokes"], "%s: no pitch label" % tag

    pts = [p for poly in mark["strokes"] for p in poly]
    pad = DIM_STROKE_W_MM / 2.0
    x0, x1 = min(p[0] for p in pts) - pad, max(p[0] for p in pts) + pad
    y0, y1 = min(p[1] for p in pts) - pad, max(p[1] for p in pts) + pad

    # one cell, on the same lattice the grid is drawn on
    ci, cj = math.floor(x0 / GRID_MINOR_MM), math.floor(y0 / GRID_MINOR_MM)
    cell = (ci * GRID_MINOR_MM, cj * GRID_MINOR_MM,
            (ci + 1) * GRID_MINOR_MM, (cj + 1) * GRID_MINOR_MM)
    assert cell[0] <= x0 and x1 <= cell[2] and cell[1] <= y0 and y1 <= cell[3], \
        "%s: label spans x %.2f..%.2f y %.2f..%.2f, crossing cell %r" \
        % (tag, x0, x1, y0, y1, cell)

    # and clear of the walls, so print spread cannot merge it into a line
    clear = min(x0 - cell[0], cell[2] - x1, y0 - cell[1], cell[3] - y1)
    assert clear >= DIM_CELL_MARGIN_MM - pad - 1e-9, \
        "%s: label is only %.3f mm from a cell wall" % (tag, clear)


def test_the_pitch_label_is_readable_after_fabrication():
    """Text is chiral, so it is mirrored like the QR and unlike the grid.

    Drawn unmirrored it would look right in KiCad's top view and read backwards
    on the board, which is the only side anyone reads it from.
    """
    manifest_json, _ = _manifest(SQUARE, "expand", 2.0, 2.2)
    mark = backmark(manifest_json)
    got = [p for poly in mark["strokes"] for p in poly]
    assert got, "no label strokes"

    from opentac.cad.export.stroke_font import text_strokes
    ref = [p for poly in text_strokes("%dMM" % round(GRID_MINOR_MM), DIM_TEXT_H_MM)
           for p in poly]
    assert len(got) == len(ref)

    ref_w = max(p[0] for p in ref) - min(p[0] for p in ref)
    got_w = max(p[0] for p in got) - min(p[0] for p in got)
    assert abs(ref_w - got_w) < 1e-6, "glyph run width changed under mirroring"

    first_ref = ref[0][0] - min(p[0] for p in ref)
    first_got = got[0][0] - min(p[0] for p in got)
    assert abs(first_got - (ref_w - first_ref)) < 1e-6, \
        "pitch text is not mirrored; it would read backwards on the board"


@pytest.mark.parametrize("tag,outline,mode,pw,pitch", CASES)
def test_nothing_is_printed_over_the_board_edge(tag, outline, mode, pw, pitch):
    manifest_json, _ = _manifest(outline, mode, pw, pitch)
    mark = backmark(manifest_json)
    man = json.loads(manifest_json)

    poly = Polygon(man["edge_cuts"])
    if not poly.is_valid:
        poly = poly.buffer(0)
    inner = poly.buffer(-EDGE_CLEAR_MM)

    outside = [r for r in mark["rects"] if not inner.buffer(1e-9).contains(box(*r))]
    assert not outside, (
        "%s: %d of %d back-silk rects come within %.2f mm of the cut line, worst %r"
        % (tag, len(outside), len(mark["rects"]), EDGE_CLEAR_MM, outside[0]))

    # the stroked label too, allowing for its pen width
    pad = DIM_STROKE_W_MM / 2.0
    for poly in mark["strokes"]:
        for x, y in poly:
            assert inner.buffer(1e-9).contains(box(x - pad, y - pad, x + pad, y + pad)), \
                "%s: dimension label runs within %.2f mm of the cut line" % (tag, EDGE_CLEAR_MM)


@pytest.mark.parametrize("tag,outline,mode,pw,pitch", CASES)
def test_the_gerber_and_the_kicad_file_agree(tag, outline, mode, pw, pitch):
    """Both exporters read the same geometry, so both must carry the same count.

    kicad-cli plots B.Silkscreen from the board file when it is installed and
    the pure-Python path draws it directly when it is not. Those are different
    code paths to the same artwork, and a board that only carries the mark on
    one of them is the failure this catches.
    """
    manifest_json, z = _manifest(outline, mode, pw, pitch)
    mark = backmark(manifest_json)
    n_rects = len(mark["rects"])
    n_seg = sum(len(poly) - 1 for poly in mark["strokes"])
    assert n_rects > 0 and n_seg > 0

    pcb = z.read("board.kicad_pcb").decode()
    # every back-silk primitive: filled rects for the QR, grid and origin,
    # plus one stroked segment per polyline in the dimension label
    assert pcb.count('(layer "B.SilkS")') == n_rects + n_seg, \
        "%s: kicad_pcb carries %d back-silk primitives, artwork has %d rects + %d segments" \
        % (tag, pcb.count('(layer "B.SilkS")'), n_rects, n_seg)

    names = z.namelist()
    assert "gerber/board-B_Silkscreen.gbo" in names, \
        "%s: no back-silkscreen Gerber in the export" % tag
    gbo = z.read("gerber/board-B_Silkscreen.gbo").decode()
    assert gbo.count("G36*") == n_rects, \
        "%s: Gerber region count does not match the artwork" % tag

    # the Gerber carries the strokes as aperture draws, not regions
    assert gbo.count("D01*") >= n_seg, \
        "%s: Gerber is missing dimension-label segments" % tag


def test_a_broken_mark_never_costs_the_board():
    """backmark() degrades. It does not raise, whatever it is handed."""
    for junk in (None, "", "not json", "{}", '{"m": []}',
                 json.dumps({"m": ["11"], "pixels": [], "r": 1, "c": 2}),
                 json.dumps({"m": ["1"], "pixels": [[0, 0, 0, 0]], "r": 99, "c": 99,
                             "w": 1, "h": 1, "rp": [1], "cp": [2]})):
        out = backmark(junk)
        assert out["rects"] == [] or isinstance(out["rects"], list)
        assert "warnings" in out


def test_artwork_clears_the_fab_silkscreen_limits():
    """JLCPCB flex: line >= 0.15 mm, hollow internal gap > 0.25 mm.

    A knocked-out QR is a hollow feature, so the knockout is the binding rule.
    Lines under the minimum are widened by the fab without notice, which would
    undo the pre-compensation, so the ink clears the minimum by design.
    """
    thinnest_ink = MODULE_MM - 2 * PRECOMP_MM
    smallest_knockout = MODULE_MM + 2 * PRECOMP_MM
    assert thinnest_ink >= 0.15, \
        "ink feature %.3f mm is under the fab minimum and would be auto-widened" % thinnest_ink
    assert smallest_knockout > 0.25, \
        "knockout %.3f mm is under the hollow-font gap limit" % smallest_knockout


# --------------------------------------------------------------------------
# the other end: a scanned string back to a manifest
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tag,outline,mode,pw,pitch", CASES)
def test_a_scan_rebuilds_a_usable_manifest(tag, outline, mode, pw, pitch):
    """What someone gets after pointing a phone at the board.

    Manifest-shaped on purpose: it has to drop into whatever already reads
    manifest.json, not into a second format nobody consumes.
    """
    manifest_json, _ = _manifest(outline, mode, pw, pitch)
    orig = json.loads(manifest_json)
    rec = manifest_from_scan(backmark(manifest_json)["qr"]["payload"])

    for key in ("v", "units", "r", "c", "footprint", "rp", "cp", "m"):
        assert rec[key] == orig[key], "%s: %s differs after a scan" % (tag, key)
    for key in ("w", "h", "px", "py", "ox", "oy"):
        assert abs(rec[key] - orig[key]) <= 0.01, \
            "%s: %s off by %.4f" % (tag, key, abs(rec[key] - orig[key]))

    a = {(r, c): (x, y) for x, y, r, c in orig["pixels"]}
    b = {(r, c): (x, y) for x, y, r, c in rec["pixels"]}
    assert set(a) == set(b), "%s: taxel set differs after a scan" % tag
    worst = max(max(abs(b[k][0] - a[k][0]), abs(b[k][1] - a[k][1])) for k in a)
    assert worst <= 0.01, "%s: taxels off by %.4f mm after a scan" % (tag, worst)

    # what it cannot carry is named, not faked
    for key in ABSENT:
        assert key not in rec, "%s: %s claims to be recovered but cannot be" % (tag, key)
    assert rec["recovered"]["absent"] == list(ABSENT)


QR_ALNUM = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"

# The four QR-alphanumeric characters that are mangled silently: each turns a
# payload into a different valid payload rather than an error.
SILENT_HAZARDS = {
    " ": "trimmed by scanners and clipboards",
    "%": "eaten by URL percent-decoding",
    "+": "URL-decoded back into a space",
    "*": "expanded by a shell if the CLI argument is unquoted",
}


def test_the_alphabet_excludes_every_silently_mangled_character():
    """The alphabet is base41, not RFC 9285's base45, without the four
    characters that are mangled silently.

    Density is set by needing N**3 >= 65536 to pack two bytes into three
    characters, which holds down to 41. QR's alphanumeric set has 45, so there
    are exactly four characters of headroom, and exactly four that fail
    silently; dropping them costs no density.
    """
    assert len(B41) == 41 and len(set(B41)) == 41
    assert set(B41) <= set(QR_ALNUM), "alphabet leaves QR alphanumeric mode"

    for ch, why in SILENT_HAZARDS.items():
        assert ch not in B41, "%r is still in the alphabet (%s)" % (ch, why)

    # and the density is unchanged
    assert len(B41) ** 3 >= 65536 and len(B41) ** 2 >= 256


def test_payloads_survive_common_transports():
    """Whitespace trimming, URL round-tripping and shell globbing leave the payload unchanged."""
    import glob
    import urllib.parse

    for tag, outline, mode, pw, pitch in CASES:
        manifest_json, _ = _manifest(outline, mode, pw, pitch)
        payload = backmark(manifest_json)["qr"]["payload"]
        reference = manifest_from_scan(payload)

        assert not (set(payload) & set(SILENT_HAZARDS)), \
            "%s: payload contains a silently-mangled character" % tag

        # a scanner or clipboard that trims
        assert manifest_from_scan("  \n" + payload + "\t\r\n ") == reference, \
            "%s: trimming changed the result" % tag
        assert manifest_from_scan("".join(payload.split())) == reference, \
            "%s: internal whitespace strip changed the result" % tag

        # a URL query round-trip, both quoting styles
        assert urllib.parse.unquote_plus(payload) == payload, \
            "%s: payload is altered by URL decoding" % tag
        assert urllib.parse.parse_qs("p=" + payload)["p"][0] == payload, \
            "%s: payload does not survive a query string" % tag

        # an unquoted shell argument: no glob metacharacters at all
        assert glob.escape(payload) == payload, \
            "%s: payload contains shell glob metacharacters" % tag


def test_a_mangled_scan_fails_loudly():
    """Every single-character substitution is rejected.

    A payload that decodes to a plausible-but-wrong pin mapping is worse than
    one that does not decode: someone wires a sensor up from it.
    """
    import random

    manifest_json, _ = _manifest(SQUARE, "expand", 2.0, 2.2)
    payload = backmark(manifest_json)["qr"]["payload"]
    rng = random.Random(0)

    for _ in range(200):
        i = rng.randrange(len(payload))
        ch = rng.choice([c for c in B41 if c != payload[i]])
        with pytest.raises(TaxelCoreError):
            manifest_from_scan(payload[:i] + ch + payload[i + 1:])


# --------------------------------------------------------------------------
# the link: what a phone camera opens
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tag,outline,mode,pw,pitch", CASES)
def test_the_qr_is_a_link_when_it_fits(tag, outline, mode, pw, pitch):
    """A phone camera opens a link; it only shows a bare payload as text.

    The link has to stay inside QR alphanumeric mode -- one character outside
    it and segno would have to fall back to byte mode, a much bigger code -- and
    the payload has to survive a browser: no '/' left (so no '.' or '..' path
    segment a browser would rewrite), and every link form decodes to the same
    manifest as the bare payload.
    """
    import segno

    manifest_json, _ = _manifest(outline, mode, pw, pitch)
    q = backmark(manifest_json)["qr"]
    assert q["form"] == "link", "%s: printed %s" % (tag, q["form"])
    assert q["text"] == qr_link(q["payload"]) == LINK_PREFIX + q["payload"].replace("/", "*")
    assert set(q["text"]) <= set(QR_ALNUM), "%s: link leaves alphanumeric mode" % tag
    assert segno.make(q["text"], error="M", mode="alphanumeric").version == q["version"]

    segment = q["text"][len(LINK_PREFIX):]
    assert "/" not in segment and segment not in (".", ".."), \
        "%s: the payload is more than one path segment" % tag

    reference = manifest_from_scan(q["payload"])
    host_path = q["text"][len("HTTPS://"):]
    for text in (q["text"], q["text"].lower(), host_path, " " + host_path.lower() + "\n",
                 q["text"].replace("*", "/"), "https://opentac.github.io/qr/#" + q["payload"]):
        assert manifest_from_scan(text) == reference, "%s: %r decodes differently" % (tag, text)


def _resized(manifest_json, side):
    """The same manifest on a square board of the given side, centred on the
    taxel field -- the QR's first choice of position."""
    man = json.loads(manifest_json)
    xs = [p[0] for p in man["pixels"]]
    ys = [p[1] for p in man["pixels"]]
    cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    h = side / 2.0
    man["edge_cuts"] = [[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy + h], [cx - h, cy + h]]
    return json.dumps(man)


def test_a_board_too_small_for_the_link_gets_the_bare_code():
    """The link costs 28 characters, a QR version, ~1.4 mm a side. A board that
    has room for the bare code but not the link still gets a code, and says
    which; one without room for either gets none, and says so."""
    manifest_json, _ = _manifest(SQUARE, "expand", 2.0, 2.2)
    big = backmark(manifest_json)["qr"]
    assert big["form"] == "link"
    link_mm = big["size_mm"]
    bare_payload = big["payload"]

    from opentac.cad.export.silk_backmark import _qr_size_mm
    bare_mm = _qr_size_mm(json.loads(manifest_json), "bare")
    assert bare_mm < link_mm

    small = _resized(manifest_json, (bare_mm + link_mm) / 2.0)
    mark = backmark(small)
    q = mark["qr"]
    assert q and q["form"] == "bare" and q["text"] == q["payload"] == bare_payload
    assert abs(q["size_mm"] - bare_mm) < 1e-9
    assert any("QR link" in w and "bare" in w for w in mark["warnings"]), mark["warnings"]
    assert manifest_from_scan(q["text"]) == manifest_from_scan(qr_link(bare_payload))

    # and it is still printed inside the board, quiet zone included
    poly = Polygon(json.loads(small)["edge_cuts"])
    (cx, cy), h = q["center"], q["size_mm"] / 2.0
    assert poly.buffer(1e-9).contains(box(cx - h, cy - h, cx + h, cy + h))

    tiny = backmark(_resized(manifest_json, bare_mm - 0.5))
    assert tiny["qr"] is None
    assert any("omitted" in w and "as a link" in w and "bare code" in w
               for w in tiny["warnings"]), tiny["warnings"]


def test_a_small_exported_board_carries_the_bare_code_in_both_files():
    """The whole export path on a board the link does not fit: a 10 mm square.

    The board file and the Gerber must both carry the bare code, and the
    /generate stats must say which code the back carries.
    """
    resp = generate(GenerateRequest(
        outline=[[-5, -5], [5, -5], [5, 5], [-5, 5]], router="hug", board_mode="expand",
        pixel_w_mm=2.0, pixel_h_mm=2.0, pitch_x_mm=2.2, pitch_y_mm=2.2))
    import base64
    import io
    import zipfile
    z = zipfile.ZipFile(io.BytesIO(base64.b64decode(resp.zip_b64)))
    manifest_json = z.read("manifest.json").decode()
    mark = backmark(manifest_json)
    assert mark["qr"] and mark["qr"]["form"] == "bare"

    stats = resp.stats["back_qr"]
    assert stats["form"] == "bare" and stats["text"] == mark["qr"]["payload"]
    assert any("QR link" in w for w in stats["warnings"])

    pcb = z.read("board.kicad_pcb").decode()
    n_seg = sum(len(poly) - 1 for poly in mark["strokes"])
    assert pcb.count('(layer "B.SilkS")') == len(mark["rects"]) + n_seg
    gbo = z.read("gerber/board-B_Silkscreen.gbo").decode()
    assert gbo.count("G36*") == len(mark["rects"])


def _blank_column(manifest, ci):
    """`manifest` with column `ci` emptied, as removing taxels after routing leaves it."""
    m = json.loads(json.dumps(manifest))
    m["pixels"] = [p for p in m["pixels"] if p[3] != ci]
    m["m"] = [row[:ci] + "0" + row[ci + 1:] for row in m["m"]]
    return m


@pytest.mark.parametrize("where", ["first", "middle", "last"])
def test_qr_decodes_when_a_column_has_no_active_taxel(where):
    """The decoder looks columns up by manifest index, so the column table needs an
    entry for an empty column too. Without one, every later column read its
    neighbour's entry, and the site and `from-qr` failed with 'index out of range'
    (found on the CoRL piano and drum pad)."""
    manifest_json, _ = _manifest(SQUARE, "expand", 2.0, 2.2)
    m = json.loads(manifest_json)
    ci = {"first": 0, "middle": m["c"] // 2, "last": m["c"] - 1}[where]
    m = _blank_column(m, ci)
    mark = backmark(json.dumps(m))
    assert mark["qr"], mark["warnings"]
    assert not any("omitted" in w for w in mark["warnings"])

    core = decode(b41decode(mark["qr"]["payload"]))
    assert core["m"] == m["m"]
    rebuilt = {(r, c): (x, y) for x, y, r, c, _, _ in taxels(core)}
    ref = {(r, c): (x, y) for x, y, r, c in m["pixels"]}
    assert set(rebuilt) == set(ref)
    worst = max(max(abs(rebuilt[k][0] - ref[k][0]), abs(rebuilt[k][1] - ref[k][1])) for k in ref)
    assert worst <= 0.01


def _decoded(mark):
    core = decode(b41decode(mark["qr"]["payload"]))
    return core, {(r, c): (x, y) for x, y, r, c, _, _ in taxels(core)}


def test_wrap_boards_get_a_qr_with_their_columns_in_order():
    """Columns that are not evenly spaced (a wrap's seams) or are numbered right to
    left (a pad block that reverses them) do not fit the regular grid. The QR then
    spaces the columns evenly in their true order, with a column-order table when
    the numbering needs one, and the mark says how far columns move. Rows stay
    exact."""
    manifest_json, _ = _manifest(SQUARE, "expand", 2.0, 2.2)
    m = json.loads(manifest_json)
    c = m["c"]

    reversed_cols = json.loads(manifest_json)
    reversed_cols["pixels"] = [[x, y, r, c - 1 - ci] for x, y, r, ci in m["pixels"]]
    reversed_cols["m"] = [row[::-1] for row in m["m"]]

    uneven = json.loads(manifest_json)
    uneven["pixels"] = [[x + (0.7 if ci >= c // 2 else 0.0), y, r, ci] for x, y, r, ci in m["pixels"]]

    for tag, case, table in (("reversed columns", reversed_cols, True), ("uneven columns", uneven, False)):
        mark = backmark(json.dumps(case))
        assert mark["qr"], (tag, mark["warnings"])
        assert any("spaces the columns evenly" in w for w in mark["warnings"]), tag
        core, got = _decoded(mark)
        assert (core["colperm"] is not None) == table, tag
        want = {(r, ci): (x, y) for x, y, r, ci in case["pixels"]}
        assert set(got) == set(want), tag
        assert max(abs(got[k][1] - want[k][1]) for k in want) <= 0.01, tag      # rows exact
        for r in {k[0] for k in want}:                                          # column order exact
            ks = [k for k in want if k[0] == r]
            assert sorted(ks, key=lambda k: want[k][0]) == sorted(ks, key=lambda k: got[k][0]), tag
        assert max(abs(got[k][0] - want[k][0]) for k in want) < 1.1, tag      # under half a 2.2 mm gap


def test_qr_is_left_off_when_two_columns_share_a_position():
    """Ordering cannot separate two columns at one x; the mark leaves the QR off
    and says so rather than print a code that decodes to the wrong sensor."""
    manifest_json, _ = _manifest(SQUARE, "expand", 2.0, 2.2)
    m = json.loads(manifest_json)
    x1 = next(x for x, _, _, ci in m["pixels"] if ci == 1)
    m["pixels"] = [[x1 if ci == 2 else x, y, r, ci] for x, y, r, ci in m["pixels"]]
    mark = backmark(json.dumps(m))
    assert mark["qr"] is None
    assert any(w.startswith("taxel-core QR omitted") for w in mark["warnings"])
    assert mark["rects"], "the registration grid should still be drawn"


def test_an_early_boards_header_flag_is_not_read_as_a_column_order():
    """The column-order bit was the anchor-holes flag on early boards (never
    shipped). A code with the bit set but no table after the mask decodes as
    before."""
    from opentac.cad.export.taxel_core import crc8, encode
    manifest_json, _ = _manifest(SQUARE, "expand", 2.0, 2.2)
    m = json.loads(manifest_json)
    plain = encode(m)
    body = bytearray(plain[:-1])
    body[0] |= 0x02                     # the bit after version, square, footprint, sparse x, sparse y
    flagged = bytes(body) + bytes([crc8(bytes(body))])
    a, b = decode(plain), decode(flagged)
    assert b["colperm"] is None
    assert taxels(a) == taxels(b)


def test_a_board_wrapped_the_other_way_gets_its_rows_in_order():
    """Seams across the rows (the Tesollo proximal) leave the rows unevenly spaced.
    The QR spaces the rows evenly in their true order; the row table already
    carries their order, so no column-order table is needed, and the columns
    stay exact."""
    manifest_json, _ = _manifest(SQUARE, "expand", 2.0, 2.2)
    m = json.loads(manifest_json)
    ys = sorted({y for _, y, _, _ in m["pixels"]})
    cut = ys[len(ys) // 2]
    m["pixels"] = [[x, y + (0.7 if y >= cut else 0.0), r, ci] for x, y, r, ci in m["pixels"]]
    mark = backmark(json.dumps(m))
    assert mark["qr"], mark["warnings"]
    assert any("spaces the rows evenly" in w for w in mark["warnings"])
    core, got = _decoded(mark)
    assert core["colperm"] is None
    want = {(r, ci): (x, y) for x, y, r, ci in m["pixels"]}
    assert max(abs(got[k][0] - want[k][0]) for k in want) <= 0.01                 # columns exact
    for ci in {k[1] for k in want}:                                               # row order exact
        ks = [k for k in want if k[1] == ci]
        assert sorted(ks, key=lambda k: want[k][1]) == sorted(ks, key=lambda k: got[k][1])
    assert max(abs(got[k][1] - want[k][1]) for k in want) < 1.1
