"""Tests for SVG outline import — rect, path, circle, units, transforms."""
import math

from fastapi.testclient import TestClient

from opentac.cad.importers.svg import parse_svg_outline, SvgImportError
from opentac.cad.main import app

client = TestClient(app)


def _svg(body, *, width="40mm", height="40mm", viewBox="0 0 40 40", extra_root=""):
    vb = f' viewBox="{viewBox}"' if viewBox is not None else ""
    w = f' width="{width}"' if width is not None else ""
    h = f' height="{height}"' if height is not None else ""
    return (
        '<?xml version="1.0"?>'
        f'<svg xmlns="http://www.w3.org/2000/svg"{w}{h}{vb}{extra_root}>'
        f"{body}</svg>"
    ).encode("utf-8")


def test_rect_square():
    result = parse_svg_outline(_svg('<rect x="0" y="0" width="20" height="20"/>',
                                    width="40mm", viewBox="0 0 40 40"))
    assert len(result.loops) == 1
    assert abs(result.areas[0] - 400.0) < 1e-6


def test_polygon_square():
    result = parse_svg_outline(_svg(
        '<polygon points="0,0 20,0 20,20 0,20"/>', width="40mm", viewBox="0 0 40 40"))
    assert len(result.loops) == 1
    assert abs(result.areas[0] - 400.0) < 1e-6
    assert len(result.loops[0]) == 4


def test_closed_path_with_relative_lines():
    result = parse_svg_outline(_svg(
        '<path d="M 0 0 l 20 0 l 0 20 l -20 0 z"/>', width="40mm", viewBox="0 0 40 40"))
    assert abs(result.areas[0] - 400.0) < 1e-6


def test_path_with_cubic_is_closed():
    # Rounded-corner square: three lines + one cubic approximating a quarter circle.
    result = parse_svg_outline(_svg(
        '<path d="M 0 0 H 20 V 15 C 20 17.761 18.761 20 16 20 H 0 Z"/>',
        width="40mm", viewBox="0 0 40 40"))
    assert len(result.loops) == 1
    assert result.areas[0] > 300


def test_circle_area():
    result = parse_svg_outline(_svg(
        '<circle cx="20" cy="20" r="10"/>', width="40mm", viewBox="0 0 40 40"))
    assert abs(result.areas[0] - math.pi * 100) < 1.0


def test_arc_path_quarter_then_close():
    result = parse_svg_outline(_svg(
        '<path d="M 0 0 H 20 A 5 5 0 0 1 15 5 V 20 H 0 Z"/>',
        width="40mm", viewBox="0 0 40 40"))
    assert len(result.loops) == 1
    assert result.areas[0] > 200


def test_multiple_loops_picks_largest():
    result = parse_svg_outline(_svg(
        '<rect x="0" y="0" width="20" height="20"/>'
        '<rect x="5" y="5" width="3" height="3"/>',
        width="40mm", viewBox="0 0 40 40"))
    assert len(result.loops) == 2
    assert result.areas[result.chosen_index] == max(result.areas)
    assert any("2 closed shapes" in w for w in result.warnings)


def test_group_translate():
    result = parse_svg_outline(_svg(
        '<g transform="translate(10,5)">'
        '<rect x="0" y="0" width="20" height="10"/>'
        "</g>",
        width="40mm", viewBox="0 0 40 40"))
    xs = [p[0] for p in result.loops[0]]
    ys = [p[1] for p in result.loops[0]]
    assert min(xs) == 10 and max(xs) == 30
    assert min(ys) == 5 and max(ys) == 15


def test_unitless_warns_and_assumes_mm():
    result = parse_svg_outline(_svg(
        '<rect x="0" y="0" width="20" height="20"/>',
        width="40", height="40", viewBox="0 0 40 40"))
    assert any("assuming" in w.lower() for w in result.warnings)
    assert abs(result.areas[0] - 400.0) < 1e-6


def test_width_in_inches_scales_viewbox():
    # viewBox 2 x 2 user units, width 2 in -> 1 user unit = 25.4 mm; 1x1 square is 645.16 mm²
    result = parse_svg_outline(_svg(
        '<rect x="0" y="0" width="1" height="1"/>',
        width="2in", height="2in", viewBox="0 0 2 2"))
    assert abs(result.areas[0] - (25.4 ** 2)) < 0.5


def test_open_geometry_raises():
    raised = False
    try:
        parse_svg_outline(_svg('<line x1="0" y1="0" x2="20" y2="0"/>',
                               width="40mm", viewBox="0 0 40 40"))
    except SvgImportError:
        raised = True
    assert raised


def test_not_svg_raises():
    raised = False
    try:
        parse_svg_outline(b"<html></html>")
    except SvgImportError:
        raised = True
    assert raised


def test_import_svg_endpoint():
    svg = _svg('<rect x="0" y="0" width="20" height="10"/>',
               width="40mm", viewBox="0 0 40 40")
    r = client.post("/import-svg", files={"file": ("pad.svg", svg, "image/svg+xml")})
    assert r.status_code == 200, r.text
    data = r.json()
    assert abs(data["areas"][data["chosen_index"]] - 200.0) < 1e-6
    assert len(data["loops"][data["chosen_index"]]) == 4


def test_import_svg_endpoint_rejects_open_path():
    svg = _svg('<path d="M 0 0 L 20 0"/>', width="40mm", viewBox="0 0 40 40")
    r = client.post("/import-svg", files={"file": ("open.svg", svg, "image/svg+xml")})
    assert r.status_code == 400
    assert "closed outline" in r.json()["detail"].lower()
