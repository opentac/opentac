"""SVG outline import — flatten 2D SVG geometry into closed polygons in millimetres.

Supported drawing: path, polygon, polyline, rect, circle, ellipse, line.
Transforms on the element and its ancestors are applied. Curves are flattened
to the same 0.05 mm chord as DXF import. Winding and cable-edge choice are
left to the web app, as for DXF.
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET

from opentac.cad.importers.dxf import (
    DxfImportResult,
    FLATTEN_TOL_MM,
    _shoelace_area,
    _weld_segments_into_loops,
)

# SVG user-length -> millimetres. px is CSS (96/in); unitless lengths stay 1
# and the document scale is decided from width/height + viewBox.
_UNIT_TO_MM = {
    "mm": 1.0,
    "cm": 10.0,
    "in": 25.4,
    "pt": 25.4 / 72.0,
    "pc": 25.4 / 6.0,
    "px": 25.4 / 96.0,
    "q": 0.25,          # quarter-millimetre
}

_LENGTH_RE = re.compile(r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*([a-zA-Z%]*)\s*$")
_NUM_RE = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_TRANSFORM_RE = re.compile(r"(matrix|translate|scale|rotate|skewX|skewY)\s*\(([^)]*)\)", re.I)
_SKIP_TAGS = frozenset({
    "defs", "clippath", "mask", "pattern", "marker", "symbol", "style",
    "script", "title", "desc", "metadata", "filter", "lineargradient",
    "radialgradient", "text", "tspan", "textpath", "image", "use",
})


class SvgImportError(ValueError):
    """Raised when an SVG file can't be parsed into a usable outline."""


def parse_svg_outline(content: bytes) -> DxfImportResult:
    warnings: list[str] = []
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("latin-1")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise SvgImportError(f"Could not parse SVG file: {exc}") from None

    if _local(root.tag) != "svg":
        raise SvgImportError("Not an SVG file (root element is not <svg>).")

    scale, unit_warning = _document_scale(root)
    if unit_warning:
        warnings.append(unit_warning)

    identity = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    segments, skipped = [], set()
    _walk(root, identity, scale, segments, skipped, warnings)

    if skipped:
        warnings.append("Skipped unsupported elements: " + ", ".join(sorted(skipped)))
    if not segments:
        raise SvgImportError("No drawable geometry found in SVG file.")

    loops = _weld_segments_into_loops(segments, warnings)
    if not loops:
        raise SvgImportError(
            "No closed outline found — check the SVG contains a closed path, "
            "polygon or shape (no gaps or open ends)."
        )

    areas = [abs(_shoelace_area(loop)) for loop in loops]
    chosen_index = max(range(len(loops)), key=lambda i: areas[i])
    if len(loops) > 1:
        warnings.append(
            f"{len(loops)} closed shapes found in file — using the largest "
            f"({areas[chosen_index]:.1f} mm²)."
        )
    xs = [p[0] for p in loops[chosen_index]]
    ys = [p[1] for p in loops[chosen_index]]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    if span > 400 and "assuming" in " ".join(warnings).lower():
        warnings.append(
            f"Imported outline is {span:.0f} mm across. If the SVG was drawn in "
            "pixels, re-export it with millimetre units (or a viewBox in mm)."
        )
    return DxfImportResult(loops=loops, areas=areas, chosen_index=chosen_index,
                           warnings=warnings)


def _local(tag: str) -> str:
    return tag.split("}", 1)[-1].lower() if tag else ""


def _walk(el, xf, scale, segments, skipped, warnings):
    tag = _local(el.tag)
    if tag in _SKIP_TAGS:
        if tag == "use":
            skipped.add("use")
        return
    if _is_hidden(el):
        return
    xf = _compose(xf, _parse_transform(el.get("transform")))
    if tag != "svg":
        try:
            segments.extend(_element_segments(el, xf, scale))
        except SvgImportError:
            raise
        except Exception:
            skipped.add(tag)
    for child in list(el):
        child_tag = _local(child.tag)
        if child_tag == "svg":
            skipped.add("nested-svg")
            continue
        _walk(child, xf, scale, segments, skipped, warnings)


def _is_hidden(el) -> bool:
    display = (el.get("display") or "").strip().lower()
    if display == "none":
        return True
    style = el.get("style") or ""
    return re.search(r"display\s*:\s*none", style, re.I) is not None


def _element_segments(el, xf, scale) -> list:
    tag = _local(el.tag)
    if tag == "path":
        return _path_segments(el.get("d") or "", xf, scale)
    if tag == "polygon":
        pts = _points_attr(el.get("points") or "")
        return _closed_poly_segments(pts, xf, scale)
    if tag == "polyline":
        pts = _points_attr(el.get("points") or "")
        if len(pts) < 2:
            return []
        if len(pts) >= 3 and _dist(pts[0], pts[-1]) < 1e-6:
            return _closed_poly_segments(pts[:-1], xf, scale)
        return _poly_segments(pts, xf, scale, closed=False)
    if tag == "rect":
        return _rect_segments(el, xf, scale)
    if tag == "circle":
        cx, cy = _user(el.get("cx"), 0.0), _user(el.get("cy"), 0.0)
        r = _user(el.get("r"), 0.0)
        if r <= 0:
            return []
        return _ellipse_segments(cx, cy, r, r, xf, scale)
    if tag == "ellipse":
        cx, cy = _user(el.get("cx"), 0.0), _user(el.get("cy"), 0.0)
        rx, ry = _user(el.get("rx"), 0.0), _user(el.get("ry"), 0.0)
        if rx <= 0 or ry <= 0:
            return []
        return _ellipse_segments(cx, cy, rx, ry, xf, scale)
    if tag == "line":
        a = (_user(el.get("x1"), 0.0), _user(el.get("y1"), 0.0))
        b = (_user(el.get("x2"), 0.0), _user(el.get("y2"), 0.0))
        return [_seg(a, b, xf, scale)]
    return []


def _user(text, default=0.0) -> float:
    if text is None or str(text).strip() == "":
        return default
    value, _unit = _split_length(str(text))
    return value if value is not None else default


def _split_length(text: str):
    m = _LENGTH_RE.match(text or "")
    if not m:
        return None, ""
    return float(m.group(1)), m.group(2).lower()


def _document_scale(root) -> tuple[float, str | None]:
    """User-unit -> mm, from width/height when they carry units, else 1 (mm assumed)."""
    vb = _viewbox(root.get("viewBox"))
    width_s, height_s = root.get("width"), root.get("height")
    w_val, w_unit = _split_length(width_s) if width_s else (None, "")
    if w_unit in ("", "%"):
        if vb is None:
            return 1.0, (
                "SVG has no usable unit information — assuming user units are "
                "millimetres. Verify the imported outline's scale."
            )
        return 1.0, (
            "SVG width has no physical unit — assuming user units are millimetres. "
            "Verify the imported outline's scale."
        )
    if w_unit not in _UNIT_TO_MM:
        return 1.0, f"SVG width unit {w_unit!r} is unknown — assuming millimetres."
    width_mm = w_val * _UNIT_TO_MM[w_unit]
    if vb is not None and vb[2] > 0:
        return width_mm / vb[2], None
    return _UNIT_TO_MM[w_unit], None


def _viewbox(text):
    if not text:
        return None
    nums = [float(x) for x in _NUM_RE.findall(text)]
    if len(nums) != 4:
        return None
    return nums  # minx, miny, w, h


def _parse_transform(text) -> tuple:
    """SVG transform list as (a, b, c, d, e, f), identity if empty."""
    xf = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    if not text:
        return xf
    for name, args in _TRANSFORM_RE.findall(text):
        nums = [float(x) for x in _NUM_RE.findall(args)]
        name = name.lower()
        if name == "matrix" and len(nums) == 6:
            piece = tuple(nums)
        elif name == "translate":
            tx = nums[0] if nums else 0.0
            ty = nums[1] if len(nums) > 1 else 0.0
            piece = (1, 0, 0, 1, tx, ty)
        elif name == "scale":
            sx = nums[0] if nums else 1.0
            sy = nums[1] if len(nums) > 1 else sx
            piece = (sx, 0, 0, sy, 0, 0)
        elif name == "rotate":
            ang = math.radians(nums[0] if nums else 0.0)
            cos, sin = math.cos(ang), math.sin(ang)
            cx = nums[1] if len(nums) > 1 else 0.0
            cy = nums[2] if len(nums) > 2 else 0.0
            # translate(cx,cy) rotate translate(-cx,-cy)
            piece = _compose((1, 0, 0, 1, cx, cy),
                             _compose((cos, sin, -sin, cos, 0, 0), (1, 0, 0, 1, -cx, -cy)))
        elif name == "skewx":
            t = math.tan(math.radians(nums[0] if nums else 0.0))
            piece = (1, 0, t, 1, 0, 0)
        elif name == "skewy":
            t = math.tan(math.radians(nums[0] if nums else 0.0))
            piece = (1, t, 0, 1, 0, 0)
        else:
            continue
        xf = _compose(xf, piece)
    return xf


def _compose(p, q):
    """Apply q then p: x' = P @ Q @ x (SVG left-to-right)."""
    a1, b1, c1, d1, e1, f1 = p
    a2, b2, c2, d2, e2, f2 = q
    return (a1 * a2 + c1 * b2, b1 * a2 + d1 * b2,
            a1 * c2 + c1 * d2, b1 * c2 + d1 * d2,
            a1 * e2 + c1 * f2 + e1, b1 * e2 + d1 * f2 + f1)


def _apply(xf, x, y, scale):
    a, b, c, d, e, f = xf
    return (scale * (a * x + c * y + e), scale * (b * x + d * y + f))


def _seg(a, b, xf, scale):
    return (_apply(xf, a[0], a[1], scale), _apply(xf, b[0], b[1], scale))


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _points_attr(text: str) -> list[tuple[float, float]]:
    nums = [float(x) for x in _NUM_RE.findall(text)]
    return [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]


def _poly_segments(pts, xf, scale, closed):
    if len(pts) < 2:
        return []
    segs = [_seg(pts[i], pts[i + 1], xf, scale) for i in range(len(pts) - 1)]
    if closed:
        segs.append(_seg(pts[-1], pts[0], xf, scale))
    return segs


def _closed_poly_segments(pts, xf, scale):
    if len(pts) >= 2 and _dist(pts[0], pts[-1]) < 1e-9:
        pts = pts[:-1]
    if len(pts) < 3:
        return []
    return _poly_segments(pts, xf, scale, closed=True)


def _rect_segments(el, xf, scale):
    x, y = _user(el.get("x"), 0.0), _user(el.get("y"), 0.0)
    w, h = _user(el.get("width"), 0.0), _user(el.get("height"), 0.0)
    if w <= 0 or h <= 0:
        return []
    rx = _user(el.get("rx"), 0.0)
    ry = _user(el.get("ry"), rx)
    if el.get("rx") is None and el.get("ry") is not None:
        rx = ry
    rx = min(max(rx, 0.0), w / 2.0)
    ry = min(max(ry, 0.0), h / 2.0)
    if rx <= 0 or ry <= 0:
        pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        return _poly_segments(pts, xf, scale, closed=True)
    # Rounded rect as line + quarter-ellipse corners, clockwise from top-left.
    pts = []
    pts.extend(_arc_points(x + rx, y + ry, rx, ry, math.pi, 1.5 * math.pi))
    pts.extend(_arc_points(x + w - rx, y + ry, rx, ry, 1.5 * math.pi, 2.0 * math.pi)[1:])
    pts.extend(_arc_points(x + w - rx, y + h - ry, rx, ry, 0.0, 0.5 * math.pi)[1:])
    pts.extend(_arc_points(x + rx, y + h - ry, rx, ry, 0.5 * math.pi, math.pi)[1:])
    return _poly_segments(pts, xf, scale, closed=True)


def _flatten_tol(xf, scale) -> float:
    """0.05 mm expressed in the element's user units, for curve subdivision."""
    stretch = max(abs(xf[0]), abs(xf[3]), abs(xf[1]), abs(xf[2]), 1e-9)
    return FLATTEN_TOL_MM / max(abs(scale) * stretch, 1e-9)


def _ellipse_n(rx, ry, tol=FLATTEN_TOL_MM):
    r = max(abs(rx), abs(ry), 1e-9)
    inner = max(-1.0, min(1.0, 1.0 - min(1.0, tol / r)))
    return max(24, int(math.ceil(2 * math.pi / max(math.acos(inner), 1e-6))))


def _ellipse_segments(cx, cy, rx, ry, xf, scale):
    pts = _arc_points(cx, cy, rx, ry, 0.0, 2.0 * math.pi, _flatten_tol(xf, scale))[:-1]
    return _poly_segments(pts, xf, scale, closed=True)


def _arc_points(cx, cy, rx, ry, a0, a1, tol=FLATTEN_TOL_MM):
    n = max(4, int(round(_ellipse_n(rx, ry, tol) * abs(a1 - a0) / (2 * math.pi))))
    return [(cx + rx * math.cos(a0 + (a1 - a0) * i / n),
             cy + ry * math.sin(a0 + (a1 - a0) * i / n)) for i in range(n + 1)]


def _path_tokens(d: str):
    i, n = 0, len(d)
    while i < n:
        c = d[i]
        if c in "MmLlHhVvCcSsQqTtAaZz":
            yield c
            i += 1
            continue
        if c in ", \t\r\n":
            i += 1
            continue
        m = _NUM_RE.match(d, i)
        if m:
            yield float(m.group(0))
            i = m.end()
            continue
        i += 1


def _path_segments(d: str, xf, scale) -> list:
    segs = []
    cx = cy = sx = sy = 0.0
    last_c = None  # last cubic control, for S
    last_q = None  # last quad control, for T
    cmd = None

    def add_line(x, y):
        nonlocal cx, cy, last_c, last_q
        segs.append(_seg((cx, cy), (x, y), xf, scale))
        cx, cy = x, y
        last_c = last_q = None

    def take(n, args):
        if len(args) < n:
            return None
        out = args[:n]
        del args[:n]
        return out

    def close_path():
        nonlocal cx, cy, last_c, last_q
        if _dist((cx, cy), (sx, sy)) > 1e-9:
            segs.append(_seg((cx, cy), (sx, sy), xf, scale))
        cx, cy = sx, sy
        last_c = last_q = None

    tokens = list(_path_tokens(d))
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if isinstance(tok, str):
            cmd = tok
            i += 1
            if cmd.upper() == "Z":
                close_path()
            continue
        if cmd is None:
            i += 1
            continue
        args = []
        while i < len(tokens) and not isinstance(tokens[i], str):
            args.append(float(tokens[i]))
            i += 1
        relative = cmd.islower()
        op = cmd.upper()
        first_of_m = op == "M"
        while args:
            if first_of_m:
                pair = take(2, args)
                if pair is None:
                    break
                x, y = pair
                if relative:
                    x, y = cx + x, cy + y
                cx, cy = x, y
                sx, sy = x, y
                last_c = last_q = None
                first_of_m = False
                op = "L"
                continue
            if op == "L":
                pair = take(2, args)
                if pair is None:
                    break
                x, y = pair
                if relative:
                    x, y = cx + x, cy + y
                add_line(x, y)
            elif op == "H":
                one = take(1, args)
                if one is None:
                    break
                add_line(cx + one[0] if relative else one[0], cy)
            elif op == "V":
                one = take(1, args)
                if one is None:
                    break
                add_line(cx, cy + one[0] if relative else one[0])
            elif op == "C":
                six = take(6, args)
                if six is None:
                    break
                x1, y1, x2, y2, x, y = six
                if relative:
                    x1, y1, x2, y2, x, y = (cx + x1, cy + y1, cx + x2, cy + y2, cx + x, cy + y)
                _cubic_segs(segs, (cx, cy), (x1, y1), (x2, y2), (x, y), xf, scale,
                            _flatten_tol(xf, scale))
                last_c, last_q = (x2, y2), None
                cx, cy = x, y
            elif op == "S":
                four = take(4, args)
                if four is None:
                    break
                x2, y2, x, y = four
                if relative:
                    x2, y2, x, y = cx + x2, cy + y2, cx + x, cy + y
                if last_c is None:
                    x1, y1 = cx, cy
                else:
                    x1, y1 = 2 * cx - last_c[0], 2 * cy - last_c[1]
                _cubic_segs(segs, (cx, cy), (x1, y1), (x2, y2), (x, y), xf, scale,
                            _flatten_tol(xf, scale))
                last_c, last_q = (x2, y2), None
                cx, cy = x, y
            elif op == "Q":
                four = take(4, args)
                if four is None:
                    break
                x1, y1, x, y = four
                if relative:
                    x1, y1, x, y = cx + x1, cy + y1, cx + x, cy + y
                _quad_segs(segs, (cx, cy), (x1, y1), (x, y), xf, scale)
                last_q, last_c = (x1, y1), None
                cx, cy = x, y
            elif op == "T":
                pair = take(2, args)
                if pair is None:
                    break
                x, y = pair
                if relative:
                    x, y = cx + x, cy + y
                if last_q is None:
                    x1, y1 = cx, cy
                else:
                    x1, y1 = 2 * cx - last_q[0], 2 * cy - last_q[1]
                _quad_segs(segs, (cx, cy), (x1, y1), (x, y), xf, scale)
                last_q, last_c = (x1, y1), None
                cx, cy = x, y
            elif op == "A":
                seven = take(7, args)
                if seven is None:
                    break
                rx, ry, phi, fa, fs, x, y = seven
                if relative:
                    x, y = cx + x, cy + y
                _arc_segs(segs, (cx, cy), rx, ry, phi, fa, fs, (x, y), xf, scale)
                last_c = last_q = None
                cx, cy = x, y
            else:
                args.clear()
                break
    return segs


def _flat_enough(p0, p1, p2, p3, tol):
    """Control points of a cubic sit within `tol` of the chord p0-p3."""
    def d(p):
        vx, vy = p3[0] - p0[0], p3[1] - p0[1]
        L2 = vx * vx + vy * vy
        if L2 < 1e-18:
            return _dist(p, p0)
        t = ((p[0] - p0[0]) * vx + (p[1] - p0[1]) * vy) / L2
        t = max(0.0, min(1.0, t))
        return math.hypot(p[0] - (p0[0] + t * vx), p[1] - (p0[1] + t * vy))
    return d(p1) <= tol and d(p2) <= tol


def _cubic_segs(segs, p0, p1, p2, p3, xf, scale, tol=FLATTEN_TOL_MM):
    stack = [(p0, p1, p2, p3, 0)]
    while stack:
        a, b, c, d, depth = stack.pop()
        if depth > 12 or _flat_enough(a, b, c, d, tol):
            segs.append(_seg(a, d, xf, scale))
            continue
        # de Casteljau split
        ab = _mid(a, b); bc = _mid(b, c); cd = _mid(c, d)
        abbc = _mid(ab, bc); bccd = _mid(bc, cd)
        m = _mid(abbc, bccd)
        stack.append((m, bccd, cd, d, depth + 1))
        stack.append((a, ab, abbc, m, depth + 1))


def _mid(p, q):
    return ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0)


def _quad_segs(segs, p0, p1, p2, xf, scale):
    # Raise quadratic to cubic: c1 = p0 + 2/3 (p1-p0), c2 = p2 + 2/3 (p1-p2)
    c1 = (p0[0] + 2.0 / 3.0 * (p1[0] - p0[0]), p0[1] + 2.0 / 3.0 * (p1[1] - p0[1]))
    c2 = (p2[0] + 2.0 / 3.0 * (p1[0] - p2[0]), p2[1] + 2.0 / 3.0 * (p1[1] - p2[1]))
    _cubic_segs(segs, p0, c1, c2, p2, xf, scale, _flatten_tol(xf, scale))


def _arc_segs(segs, p0, rx, ry, phi_deg, fa, fs, p1, xf, scale):
    """SVG elliptical arc -> line segments. Implementation notes of the SVG spec."""
    rx, ry = abs(rx), abs(ry)
    if rx < 1e-12 or ry < 1e-12 or _dist(p0, p1) < 1e-12:
        segs.append(_seg(p0, p1, xf, scale))
        return
    phi = math.radians(phi_deg)
    cos_p, sin_p = math.cos(phi), math.sin(phi)
    x1, y1 = p0
    x2, y2 = p1
    dx, dy = (x1 - x2) / 2.0, (y1 - y2) / 2.0
    x1p = cos_p * dx + sin_p * dy
    y1p = -sin_p * dx + cos_p * dy
    lam = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry)
    if lam > 1:
        s = math.sqrt(lam)
        rx, ry = rx * s, ry * s
    num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    den = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    coef = math.sqrt(max(0.0, num / den))
    if bool(fa) == bool(fs):
        coef = -coef
    cxp = coef * rx * y1p / ry
    cyp = -coef * ry * x1p / rx
    cx = cos_p * cxp - sin_p * cyp + (x1 + x2) / 2.0
    cy = sin_p * cxp + cos_p * cyp + (y1 + y2) / 2.0

    def angle(ux, uy, vx, vy):
        sign = 1.0 if ux * vy - uy * vx >= 0 else -1.0
        dot = max(-1.0, min(1.0, (ux * vx + uy * vy) / (math.hypot(ux, uy) * math.hypot(vx, vy))))
        return sign * math.acos(dot)

    ux, uy = (x1p - cxp) / rx, (y1p - cyp) / ry
    vx, vy = (-x1p - cxp) / rx, (-y1p - cyp) / ry
    theta1 = angle(1, 0, ux, uy)
    dtheta = angle(ux, uy, vx, vy)
    if not fs and dtheta > 0:
        dtheta -= 2 * math.pi
    elif fs and dtheta < 0:
        dtheta += 2 * math.pi
    n = max(4, int(round(_ellipse_n(rx, ry, _flatten_tol(xf, scale)) * abs(dtheta) / (2 * math.pi))))
    pts = [p0]
    for i in range(1, n):
        t = theta1 + dtheta * i / n
        x = cx + rx * math.cos(t) * cos_p - ry * math.sin(t) * sin_p
        y = cy + rx * math.cos(t) * sin_p + ry * math.sin(t) * cos_p
        pts.append((x, y))
    pts.append(p1)
    for a, b in zip(pts, pts[1:]):
        segs.append(_seg(a, b, xf, scale))
