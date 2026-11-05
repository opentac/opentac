"""Solder mask openings for a flex sensor's coverlay.

On a flex board the solder mask is a polyimide coverlay film with openings cut
where copper is exposed. JLCPCB's coverlay process needs at least
COVERLAY_WEB_MM of film between two openings, and suggests openings
COVERLAY_MARGIN_MM larger than the copper they expose. Per-pad openings would
break the first rule in two places: a taxel's electrode fingers are the
electrode gap (0.2 mm) apart, and the connector's gold fingers are 0.3 mm wide
at 0.5 mm pitch, so the film between openings would be 0.2 mm.

The export therefore opens:

- on top, one rectangle per taxel over all its electrode copper (both
  electrodes and its row via). The rectangle is the copper's extent grown by a
  margin m, per axis,

      m = min(COVERLAY_MARGIN_MM, (copper gap to the neighbouring taxel - COVERLAY_WEB_MM) / 2)

  so neighbouring openings are at least COVERLAY_WEB_MM apart. At 4.0 mm taxels
  on a 4.2 mm pitch (edge clearance 0.1 mm) the copper gap is 0.4 mm and
  m = 0.05 mm. A dense board (2.0 mm taxels on a 2.0 mm pitch: a 0.2 mm copper
  gap) gives m < 0: the opening stops |m| inside the electrodes' outer edge,
  and the export says so (TaxelOpening.note). The wiring between taxels stays
  covered.
- on the connector's contact face (the bottom unless Params.contact_side is
  "top"), one opening across the connector's gold fingers, from the first
  finger's outer edge to the last's (export/connector_contacts.py). Nothing
  else on the bottom: the row vias and the transition vias are covered.

kicad_pcb.py / kicad_footprint.py draw these as filled shapes on F.Mask and
B.Mask (the pads themselves carry no mask layer), and gerber.py, the
no-KiCad fallback, plots the same shapes. Both footprints carry KiCad's
allow_soldermask_bridges attribute: an opening that exposes both of a taxel's
electrodes, or every finger, is the intent, and without it KiCad's DRC reports
each one as a solder mask bridge.
"""
from dataclasses import dataclass

import numpy as np
from shapely import STRtree
from shapely.geometry import MultiPoint, box

COVERLAY_WEB_MM = 0.3
COVERLAY_MARGIN_MM = 0.075

# Positions are written to 1 nm (6 decimals in mm); a web this much under
# COVERLAY_WEB_MM is rounding, not a closer pair.
_TOL = 1e-6


@dataclass(frozen=True)
class TaxelOpening:
    """One taxel's top mask opening, centred on the taxel.

    margin_x / margin_y: how far the opening reaches past the electrode copper
    on the left and right / top and bottom; negative when it is trimmed into
    the electrodes to keep the film between openings COVERLAY_WEB_MM wide.
    """
    margin_x: float
    margin_y: float
    half_w: float
    half_h: float

    @property
    def trim_x(self) -> float:
        return max(0.0, -self.margin_x)

    @property
    def trim_y(self) -> float:
        return max(0.0, -self.margin_y)

    def rect(self) -> tuple[float, float, float, float]:
        """(x1, y1, x2, y2) relative to the taxel centre."""
        return (-self.half_w, -self.half_h, self.half_w, self.half_h)

    @property
    def note(self) -> str:
        """The plain note the export shows when an opening is trimmed; "" when not."""
        tx, ty = round(self.trim_x, 3), round(self.trim_y, 3)
        why = f"to keep JLCPCB's {COVERLAY_WEB_MM:g} mm coverlay spacing"
        if not tx and not ty:
            return ""
        if tx == ty:
            return f"Taxel openings trimmed {tx:g} mm into the electrodes {why}."
        sides = []
        if tx:
            sides.append(f"{tx:g} mm at the left and right")
        if ty:
            sides.append(f"{ty:g} mm at the top and bottom")
        return f"Taxel openings trimmed into the electrodes, {' and '.join(sides)}, {why}."

    def stats(self) -> dict:
        return {"margin_x_mm": round(self.margin_x, 4), "margin_y_mm": round(self.margin_y, 4),
                "trim_mm": round(max(self.trim_x, self.trim_y), 4),
                "opening_mm": [round(2 * self.half_w, 4), round(2 * self.half_h, 4)]}


def copper_half_extent(pixel_w: float, pixel_h: float, edge_clear: float) -> tuple[float, float]:
    """Half width and height of a taxel's electrode copper, about its centre.

    Everything stays inside the pixel inset by edge_clear
    (kicad_footprint.pixel_geometry): the spines reach the inset at the left
    and right, their round ends (and on most sizes the outermost fingers) at
    the top and bottom.
    """
    return pixel_w / 2 - edge_clear, pixel_h / 2 - edge_clear


def taxel_opening(pixel_w: float, pixel_h: float, edge_clear: float,
                  pitch_x: float, pitch_y: float, centers=None) -> TaxelOpening:
    """The top mask opening every taxel of a board gets.

    The margin comes from the grid (pitch and copper extent), per axis. When
    the taxel centres are given, the openings are also checked against the
    real placements -- diagonal neighbours and any taxel off the grid -- and
    shrunk further, equally on both axes, until every pair is at least
    COVERLAY_WEB_MM apart.
    """
    hx, hy = copper_half_extent(pixel_w, pixel_h, edge_clear)
    mx = round(min(COVERLAY_MARGIN_MM, (abs(pitch_x) - 2 * hx - COVERLAY_WEB_MM) / 2), 6)
    my = round(min(COVERLAY_MARGIN_MM, (abs(pitch_y) - 2 * hy - COVERLAY_WEB_MM) / 2), 6)
    if centers is not None:
        extra = _extra_trim(np.asarray(centers, dtype=float).reshape(-1, 2), hx + mx, hy + my)
        mx, my = round(mx - extra, 6), round(my - extra, 6)
    return TaxelOpening(margin_x=mx, margin_y=my, half_w=round(hx + mx, 6), half_h=round(hy + my, 6))


def taxel_opening_for(params, centers=None) -> TaxelOpening:
    """taxel_opening() from a pipeline Params."""
    return taxel_opening(params.pixel_w_mm, params.pixel_h_mm, params.edge_clear_mm,
                         params.pitch_x_mm, params.pitch_y_mm, centers)


def opening_web(dx, dy, half_w, half_h):
    """Film width between two equal axis-aligned openings whose centres are
    (dx, dy) apart; negative when they overlap."""
    gx = np.abs(dx) - 2 * half_w
    gy = np.abs(dy) - 2 * half_h
    apart = np.hypot(np.maximum(gx, 0.0), np.maximum(gy, 0.0))
    return np.where((gx > 0) | (gy > 0), apart, np.maximum(gx, gy))


def _extra_trim(centers: np.ndarray, half_w: float, half_h: float) -> float:
    """How much more both half sizes must shrink for every pair of openings to
    be COVERLAY_WEB_MM apart; 0 on a regular grid."""
    if len(centers) < 2:
        return 0.0
    rects = [box(x - half_w, y - half_h, x + half_w, y + half_h) for x, y in centers]
    tree = STRtree(rects)
    i, j = tree.query(rects, predicate="dwithin", distance=COVERLAY_WEB_MM - _TOL)
    keep = i < j
    if not keep.any():
        return 0.0
    d = centers[j[keep]] - centers[i[keep]]
    lo, hi = 0.0, min(half_w, half_h)
    for _ in range(40):
        mid = (lo + hi) / 2
        if np.all(opening_web(d[:, 0], d[:, 1], half_w - mid, half_h - mid) >= COVERLAY_WEB_MM - _TOL):
            hi = mid
        else:
            lo = mid
    return float(np.ceil(hi * 1e6) / 1e6)


def connector_mask_opening(connector_pads, layer: str):
    """One opening across every connector pad that opens `layer` ("F.Mask" or
    "B.Mask"): the convex hull of their corners, which for a row of equal
    fingers is the rectangle from the first finger's outer edge to the last's.
    Returns the corners (world coordinates, as the pads' own) or None."""
    pts = [tuple(map(float, c)) for pad in connector_pads or ()
           if layer in pad["layers"] or "*.Mask" in pad["layers"]
           for c in pad["corners"]]
    if not pts:
        return None
    hull = MultiPoint(pts).convex_hull
    if hull.geom_type != "Polygon":
        return None
    return [tuple(p) for p in hull.exterior.coords[:-1]]


def copper_layers_only(layers) -> list[str]:
    """A pad's layer list without its mask layers: the openings are drawn as
    their own shapes."""
    return [name for name in layers if not name.endswith(".Mask")]
