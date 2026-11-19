"""Pixel packing — place rectangular pixels on a grid inside a polygon."""

import numpy as np
from .geometry import make_polygon, polygon_buffer, inside_or_on_polygon, poly_bounds


def pack_pixels_in_polygon(
    outline_xy: np.ndarray,
    pixel_w_mm: float,
    pixel_h_mm: float,
    pitch_x_mm: float,
    pitch_y_mm: float,
    edge_keepout_mm: float,
    anchor: str = "top_right",
    phase: tuple = (0.0, 0.0),
    whole_within_xy: np.ndarray = None,
) -> tuple[np.ndarray, object]:
    """Place rectangular pixels inside a polygon on a rectangular grid.

    anchor:
        "top_right"  — grid anchored to the top-right corner, leaving whatever
                       slack is left over at the bottom-left for routing. The
                       default, used by grow-board mode.
        "top_center" — same top anchoring, but the leftover horizontal slack is
                       split evenly left/right. Used by fixed-outline keep-out
                       mode, where the routing space is already reserved by an
                       explicit keep-out band and the array should sit centered
                       inside what remains.
    phase:
        (ox, oy) lattice offset in mm, applied after anchoring — shifts every
        candidate center left/down by that amount. The default (0, 0) is the
        unshifted lattice; fixed+hug searches phases because which grid rows
        survive a carved region depends strongly on the lattice phase.

    Returns:
        placements: Mx2 array of pixel centers
        inner_poly: the shrunk polygon used for inclusion test
    """
    outer_poly = make_polygon(outline_xy)
    if outer_poly.is_empty or outer_poly.area <= 0:
        raise ValueError("Invalid polygon (empty or zero area).")

    if edge_keepout_mm > 0:
        inner_poly = polygon_buffer(outer_poly, -edge_keepout_mm)
    else:
        inner_poly = outer_poly

    if inner_poly.is_empty or inner_poly.area <= 0:
        return np.zeros((0, 2)), inner_poly

    xmin, xmax, ymin, ymax = poly_bounds(inner_poly)

    # Anchor grid from top-right corner (leaves room at bottom-left for routing)
    dx = pixel_w_mm / 2
    dy = pixel_h_mm / 2

    def walk(hi, lo, half, pitch):
        """Lattice centres from `hi` down to `lo`, one `pitch` apart."""
        out = []
        v = hi - half
        while v >= lo + half - 1e-9:
            out.append(v)
            v -= pitch
        out.reverse()
        return out

    x_centers = walk(xmax, xmin, dx, pitch_x_mm)
    y_centers = walk(ymax, ymin, dy, pitch_y_mm)

    if anchor == "top_center" and x_centers:
        # Split the leftover horizontal slack evenly instead of letting it all
        # collect on the left. Y stays top-anchored (pixels pushed away from
        # the connector edge).
        slack = (x_centers[0] - dx) - xmin
        if slack > 0:
            x_centers = [x - slack / 2 for x in x_centers]
    elif anchor not in ("top_right", "top_center"):
        raise ValueError(f"Unknown packing anchor: {anchor!r}")

    ox, oy = float(phase[0]), float(phase[1])
    if abs(ox) > 1e-12:
        x_centers = [x - ox for x in x_centers]
    if abs(oy) > 1e-12:
        y_centers = [y - oy for y in y_centers]

    corner_offsets = np.array([
        [-dx, -dy],
        [ dx, -dy],
        [ dx,  dy],
        [-dx,  dy],
    ])

    # whole_within_xy (the fixed-outline modes pass the board outline): a taxel
    # also needs its whole square inside that outline, less the edge keep-out,
    # not only its corners. A slot or slit narrower than a taxel passes between
    # the corners and would cut through its copper and the grid wires beside
    # it. Grow mode builds a larger board around the drawn outline, which
    # closes a narrow slit, so it keeps the corner test alone.
    from shapely.geometry import box
    from shapely.prepared import prep
    whole = None
    if whole_within_xy is not None:
        board = make_polygon(whole_within_xy)
        if edge_keepout_mm > 0:
            board = polygon_buffer(board, -edge_keepout_mm)
        whole = prep(board.buffer(1e-9))

    placed = []
    for cy in y_centers:
        for cx in x_centers:
            corners = corner_offsets + np.array([cx, cy])
            ok = inside_or_on_polygon(inner_poly, corners[:, 0], corners[:, 1])
            if np.all(ok) and (whole is None or whole.covers(box(cx - dx, cy - dy, cx + dx, cy + dy))):
                placed.append([cx, cy])

    if placed:
        return np.array(placed), inner_poly
    else:
        return np.zeros((0, 2)), inner_poly
