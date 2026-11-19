"""Net assignment, connector selection, pad mapping, and cable geometry."""

import dataclasses
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from .types import Layout, Params
from .packing import pack_pixels_in_polygon
from .geometry import (
    make_polygon, make_tail_poly, tail_geometry, remove_collinear, remove_collinear_closed,
    polygon_buffer, poly_bounds,
)
from . import keepout as keepout_mod
from .keepout import fixed_keepout_floor as _fixed_keepout_floor
from .keepout import _TOO_SMALL_MSG

# solve_grid_pixel_size: how many shrink rounds the "requested count actually
# packs" retry gets, and how much each round shrinks by. Only non-rectangular
# outlines need these — for a rectangle the first, exact division already fits.
MAX_GRID_FIT_ITERS = 40
GRID_FIT_SHRINK = 0.98

# Fixed-outline keep-out mode: how many carve/re-pack rounds the routing-demand
# fixed point gets. The keep-out only ever grows (see layout_fixed_shape_keepout),
# so this loop always terminates; the cap only bounds worst-case work.
MAX_CARVE_ITERS = 8

# How many times the converged keep-out may be recomputed from its own result
# and re-solved from that (narrower) starting point. Each pass is a full carve,
# and in practice the second pass already stops improving, so this is a cheap
# bound rather than a tuned one. See layout_fixed_shape_keepout.
MAX_RELAX_PASSES = 4


def _largest_polygon(poly) -> Polygon:
    """Return a single exterior polygon for preview/export metadata."""
    if isinstance(poly, Polygon):
        return poly
    if hasattr(poly, "geoms"):
        polys = [g for g in poly.geoms if isinstance(g, Polygon) and not g.is_empty]
        if polys:
            return max(polys, key=lambda g: g.area)
    return Polygon()


def _routing_demand(params, n_row: int, n_col: int, include_cable_length: bool) -> dict:
    """Per-side routing demand in mm, from the row/column net counts.

    The only caller is _resolved_routing_padding (grow-board mode, which
    pushes the outline outward); it always passes include_cable_length=True.
    Fixed-outline mode sizes its inward keep-out separately
    (keepout.analytic_widths, or route_hug.hug_keepout_widths for the hug
    router).

    include_cable_length only applies to the tail-side (bottom) demand: the
    cable/tail run lies *outside* the board, so it is reserved when growing
    outward.
    """
    base = max(0.0, float(params.routing_margin_mm))
    trace_pitch = params.trace_w_mm + params.clearance_mm
    route_count = n_row + n_col
    safety = max(1.0, 2 * trace_pitch)
    row_left_count = (n_row + 1) // 2
    row_right_count = n_row // 2
    left_padding = max(base, row_left_count * trace_pitch + safety)
    right_padding = max(base, row_right_count * trace_pitch + safety, n_col * trace_pitch * 0.25)
    top_padding = base
    bottom_padding = max(base, n_col * trace_pitch * 0.35)
    if include_cable_length:
        bottom_padding = max(bottom_padding, params.cable_length_mm)
    return {
        "left_padding_mm": float(left_padding),
        "right_padding_mm": float(right_padding),
        "side_padding_mm": float(max(left_padding, right_padding)),
        "top_padding_mm": float(top_padding),
        "bottom_padding_mm": float(bottom_padding),
        "route_count": int(route_count),
        "row_route_count": int(n_row),
        "col_route_count": int(n_col),
        "row_route_left_count": int(row_left_count),
        "row_route_right_count": int(row_right_count),
    }


def _resolved_routing_padding(params, n_row: int, n_col: int) -> dict:
    """Resolve automatic route-demand padding for final board geometry."""
    return _routing_demand(params, n_row, n_col, include_cable_length=True)


def _region_exterior_xy(region) -> np.ndarray | None:
    """Largest-part exterior ring of a carved region, as Nx2 vertices."""
    poly = _largest_polygon(region)
    if poly.is_empty or poly.area <= 0:
        return None
    coords = np.array(poly.exterior.coords[:-1], dtype=float)
    return coords if len(coords) >= 3 else None


def _make_directional_expansion_poly(outline_xy, tail, padding):
    """Build asymmetric expansion bands around the sensor outline."""
    outline_poly = make_polygon(outline_xy)
    minx, miny, maxx, maxy = outline_poly.bounds
    left = padding["left_padding_mm"]
    right = padding["right_padding_mm"]
    top = padding["top_padding_mm"]
    bottom = padding["bottom_padding_mm"]
    n_out = tail["n_out"]

    # After rotate_for_cable, the cable edge is horizontal. The tail outward
    # normal points toward the high-demand bottom/tail side.
    bottom_y = miny - bottom if n_out[1] < 0 else maxy
    top_y = maxy + top if n_out[1] < 0 else miny - top
    left_x = minx - left
    right_x = maxx + right

    if n_out[1] < 0:
        coords = [[left_x, bottom_y], [right_x, bottom_y], [right_x, top_y], [left_x, top_y]]
    else:
        coords = [[left_x, miny - top], [right_x, miny - top], [right_x, maxy + bottom], [left_x, maxy + bottom]]
    return Polygon(coords)


def _make_tail_shoulder_poly(outline_xy, tail, params, bottom_padding: float):
    """Extra final-board area around the tail/sensor shoulder."""
    p1 = outline_xy[0]
    p2 = outline_xy[1]
    edge_len = float(np.linalg.norm(p2 - p1))
    margin = max(0.0, float(params.routing_margin_mm))
    if margin <= 0:
        return Polygon(), {}

    t_hat = tail["t_hat"]
    n_out = tail["n_out"]
    mid = tail["mid"]
    # With physical connector export disabled, keep the tail end as a clean
    # rectangular cap instead of narrowing into connector-style side notches.
    half_width = edge_len / 2 + margin
    depth = max(tail["total_length"], params.cable_length_mm + margin)

    A = mid - half_width * t_hat
    B = mid + half_width * t_hat
    C = B + depth * n_out
    D = A + depth * n_out
    return Polygon([A, B, C, D]), {
        "tail_shoulder_expansion": True,
        "tail_shoulder_width_mm": 2 * half_width,
        "tail_shoulder_depth_mm": depth,
    }


def shoulder_fill_depth(tail_width: float) -> float:
    """How far into the board side cable_shoulder_fill looks for wedges."""
    return max(2.0, float(tail_width) / 2)


def cable_shoulder_fill(outline_poly, mid, t_hat, n_out, tail_width: float, depth: float | None = None) -> list:
    """Board the outline leaves out beside a tail wider than its cable edge.

    The tail is connector-width, usually wider than the cable edge, so it
    sticks out past the outline's cable-edge corners. Where the outline's
    bottom edge slopes away from the attachment line (e.g. the fingertip wrap
    band), a wedge opens up between the tail's top edge and the outline —
    board that belongs to neither shape, right where the outermost connector
    fan wires cross (DRC then flags that copper as lying outside the board).
    Close it: take a strip of the tail's width extending from the attachment
    line into the board side, subtract the outline, and keep the pieces
    touching the attachment line — exactly the wedges. Depth
    tail_width/2 comfortably covers any slope a usable cable edge produces (a
    narrowed tail passes the depth of the connector-width fill its wires were
    routed in); the touch filter keeps drawn concavities elsewhere in the strip
    untouched.

    Part of every un-grown board (layout_with_connector), and rebuilt from
    edit_data by every later request (main._layout_from_edit_data): the
    fixed-outline routes run through it."""
    mid, t_hat, n_out = (np.asarray(v, dtype=float) for v in (mid, t_hat, n_out))
    half_w = float(tail_width) / 2
    depth = shoulder_fill_depth(tail_width) if depth is None else float(depth)
    strip = Polygon([
        mid - half_w * t_hat, mid + half_w * t_hat,
        mid + half_w * t_hat - depth * n_out, mid - half_w * t_hat - depth * n_out,
    ])
    attachment = LineString([mid - half_w * t_hat, mid + half_w * t_hat])
    gaps = strip.difference(outline_poly)
    return [g.buffer(0.001, join_style=2)
            for g in getattr(gaps, "geoms", [gaps])
            if not g.is_empty and g.distance(attachment) < 1e-6]


def _grid_counts(placements_xy, tol=1e-6):
    """Distinct column and row counts in a packed grid."""
    if len(placements_xy) == 0:
        return 0, 0
    def distinct(values):
        out = []
        for v in sorted(values):
            if not out or abs(v - out[-1]) > tol:
                out.append(v)
        return len(out)
    return distinct(placements_xy[:, 0]), distinct(placements_xy[:, 1])


def solve_grid_pixel_size(outline_xy, params: Params, fill_xy=None) -> dict:
    """Solve taxel size and pitch from a requested column/row count.

    The inverse of the normal flow: instead of the user giving a taxel size and
    finding out how many fit, they give the grid they want and the taxel size
    follows. Pitch is set equal to the taxel size, so taxels tile edge to edge.

    The first estimate divides the packable region's bounding box, which is
    exact for a rectangular outline. A non-rectangular one loses corners — the
    packer only keeps a taxel whose four corners are all inside — so the result
    is verified by really packing it, and shrunk in small steps until the
    requested counts appear. Returns the solved values plus what was actually
    achieved, so the caller can report the achieved counts rather than the request.
    """
    cols = int(params.target_cols)
    rows = int(params.target_rows)
    if cols < 1 or rows < 1:
        raise ValueError("Target rows and columns must both be at least 1.")

    pack_poly = make_polygon(outline_xy)
    if fill_xy is not None:
        clipped = pack_poly.intersection(make_polygon(fill_xy))
        if clipped.is_empty or clipped.area <= 0:
            raise ValueError("Sensor fill region does not overlap the board outline.")
        pack_poly = _largest_polygon(clipped)

    inner = polygon_buffer(pack_poly, -params.edge_keepout_mm) if params.edge_keepout_mm > 0 else pack_poly
    inner = _largest_polygon(inner)
    if inner.is_empty or inner.area <= 0:
        raise ValueError(_TOO_SMALL_MSG)

    xmin, xmax, ymin, ymax = poly_bounds(inner)
    pack_xy = np.array(pack_poly.exterior.coords[:-1])

    pixel_w = (xmax - xmin) / cols
    pixel_h = (ymax - ymin) / rows
    if pixel_w <= 0 or pixel_h <= 0:
        raise ValueError(_TOO_SMALL_MSG)

    for _ in range(MAX_GRID_FIT_ITERS):
        placements, _ = pack_pixels_in_polygon(
            pack_xy, pixel_w, pixel_h, pixel_w, pixel_h,
            params.edge_keepout_mm, anchor="top_center")
        got_cols, got_rows = _grid_counts(placements)
        if got_cols >= cols and got_rows >= rows:
            return {
                "pixel_w_mm": pixel_w,
                "pixel_h_mm": pixel_h,
                "pitch_x_mm": pixel_w,
                "pitch_y_mm": pixel_h,
                "achieved_cols": got_cols,
                "achieved_rows": got_rows,
            }
        if got_cols < cols:
            pixel_w *= GRID_FIT_SHRINK
        if got_rows < rows:
            pixel_h *= GRID_FIT_SHRINK

    raise ValueError(
        f"Could not fit a {cols}x{rows} taxel grid in this outline. "
        "Draw a larger or squarer outline, reduce the requested rows/columns, "
        "or reduce the edge keepout.")


def layout_with_connector(
    outline_xy: np.ndarray,
    params: Params,
    fill_xy: np.ndarray = None,
    pack_anchor: str = "top_right",
    join_split_nets: bool = False,
    placements_override: np.ndarray = None,
    whole_within_xy: np.ndarray = None,
) -> Layout:
    """Full layout: pack pixels, assign nets, select connector, build geometry.

    join_split_nets is the hug-router mode: a row/column cut in two by a
    concavity stays one net (one connector pad) and the router bridges the gap
    around the concavity, instead of each piece getting its own pad.
    placements_override skips packing and lays out exactly the given pixel
    centers — used by layout_fixed_hug_keepout to drop disconnected grid
    fragments without re-packing (which could shift the surviving lattice).
    """
    outline_poly = make_polygon(outline_xy)

    if placements_override is not None:
        placements_xy = np.asarray(placements_override, dtype=float)
    else:
        pack_xy = outline_xy
        if fill_xy is not None:
            fill_poly = make_polygon(fill_xy)
            clipped = outline_poly.intersection(fill_poly)
            if clipped.is_empty or clipped.area <= 0:
                raise ValueError("Sensor fill region does not overlap the board outline.")
            if clipped.geom_type == "MultiPolygon":
                clipped = max(clipped.geoms, key=lambda g: g.area)
            pack_xy = np.array(clipped.exterior.coords[:-1])

        placements_xy, _ = pack_pixels_in_polygon(
            pack_xy,
            params.pixel_w_mm, params.pixel_h_mm,
            params.pitch_x_mm, params.pitch_y_mm,
            params.edge_keepout_mm,
            anchor=pack_anchor,
            whole_within_xy=whole_within_xy,
        )

    if len(placements_xy) == 0:
        raise ValueError("No pixels packed. Check outline/pitch/keepout.")

    # Assign nets (segmented by default; joined in hug-router mode)
    assign = assign_rowcol_nets_joined if join_split_nets else assign_rowcol_nets_segmented
    row_net_per_pixel, col_net_per_pixel, row_nets, col_nets = \
        assign(placements_xy, params.pitch_x_mm, params.pitch_y_mm)

    n_row = len(row_nets)
    n_col = len(col_nets)

    # Select connector (or use an explicit manual override, e.g. to match an
    # existing/larger readout board rather than the smallest auto-fit size).
    if params.connector_pos_override is not None:
        conn_pos, conn_swap = resolve_connector_override(params.connector_pos_override, n_row, n_col)
    else:
        conn_pos, conn_swap = select_connector_for_nets(n_row, n_col, params.readout_board)

    # Order nets for pad assignment
    col_nets_ordered = sort_nets_by_mean_xy(col_nets, placements_xy, col_net_per_pixel, "COL")
    row_nets_ordered = split_row_pin_order(
        sort_nets_by_mean_xy(row_nets, placements_xy, row_net_per_pixel, "ROW"))

    pad_to_net = build_pad_to_net_map_centered(conn_pos, conn_swap, row_nets_ordered, col_nets_ordered)

    # Cable geometry off outline edge 1->2
    p1 = outline_xy[0]
    p2 = outline_xy[1]
    # Laid out and routed with the connector-width tail even when the board
    # gets a narrow one: narrow_tail.narrow_routed_tail cuts it down after routing, so
    # the taxels and wires never depend on it.
    tail = tail_geometry(outline_poly, p1, p2, conn_pos, params)
    cable_poly = make_tail_poly(
        tail["mid"], tail["t_hat"], tail["n_out"], tail["tail_width"], tail["connector_width"],
        tail["cable_length"], tail["total_length"])

    # Union robustness: the cable rect's attachment edge lies exactly on the
    # outline's cable edge, and unioning polygons that touch along a shared
    # segment (instead of overlapping) can leave a zero-width seam retraced
    # by the result's exterior ring. board_poly's exterior then doubles back
    # along the cable edge, a phantom boundary runs through the board there,
    # and the board-edge DRC flags interior row wires crossing into the tail
    # as edge violations (the exported Edge.Cuts is unaffected — the tail
    # shoulders cover that copper). A 1um inflation of the cable rect
    # turns edge-contact into area overlap, which unions cleanly; 1um is
    # far below every clearance and the export's own precision. cable_poly
    # itself stays exact — the export and relief logic use it directly.
    cable_union = cable_poly.buffer(0.001, join_style=2)

    shoulder_fill = cable_shoulder_fill(outline_poly, tail["mid"], tail["t_hat"], tail["n_out"], tail["tail_width"])

    base_board_poly = unary_union([outline_poly, cable_union, *shoulder_fill])
    board_poly = base_board_poly
    routing_margin_applied = 0.0
    if params.preserve_sensors and params.auto_expand_board and params.routing_margin_mm > 0:
        routing_padding = _resolved_routing_padding(params, n_row, n_col)
        expanded_outline = _make_directional_expansion_poly(outline_xy, tail, routing_padding)
        shoulder_poly, shoulder = _make_tail_shoulder_poly(
            outline_xy, tail, params, routing_padding["bottom_padding_mm"])
        board_poly = unary_union([expanded_outline, cable_union, shoulder_poly, *shoulder_fill])
        routing_margin_applied = routing_padding["bottom_padding_mm"]
        tail["resolved"].update({
            "routing_left_padding_mm": routing_padding["left_padding_mm"],
            "routing_right_padding_mm": routing_padding["right_padding_mm"],
            "routing_side_padding_mm": routing_padding["side_padding_mm"],
            "routing_top_padding_mm": routing_padding["top_padding_mm"],
            "routing_bottom_padding_mm": routing_padding["bottom_padding_mm"],
            "routing_route_count": routing_padding["route_count"],
        })
        tail["resolved"].update(shoulder)

    tail["resolved"]["connector_export_enabled"] = False

    final_poly = _largest_polygon(board_poly)
    final_outline_xy = remove_collinear_closed(np.array(final_poly.exterior.coords[:-1])) if not final_poly.is_empty else outline_xy

    tail["resolved"].update({
        "pad_row_center": np.asarray(tail["pad_row_center"], dtype=float).tolist(),
        "t_hat": np.asarray(tail["t_hat"], dtype=float).tolist(),
        "n_out": np.asarray(tail["n_out"], dtype=float).tolist(),
    })

    # Pad row position
    pad_y_mm = tail["pad_row_center"][1]

    return Layout(
        placements_xy=placements_xy,
        row_net_per_pixel=row_net_per_pixel,
        col_net_per_pixel=col_net_per_pixel,
        connector_pos=conn_pos,
        connector_swap=conn_swap,
        pad_to_net=pad_to_net,
        pad_y_mm=pad_y_mm,
        outline_poly=outline_poly,
        cable_poly=cable_poly,
        board_poly=board_poly,
        sensor_outline_xy=outline_xy,
        final_outline_xy=final_outline_xy,
        routing_margin_applied_mm=routing_margin_applied,
        tail=tail["resolved"],
    )


def max_pack_count(outline_xy: np.ndarray, params: Params, fill_xy: np.ndarray = None) -> int:
    """Pixels that would fit with no wiring keep-out at all — the baseline the
    fixed-outline mode's "dropped for wiring" figure is measured against."""
    pack_xy = outline_xy
    if fill_xy is not None:
        clipped = make_polygon(outline_xy).intersection(make_polygon(fill_xy))
        region_xy = _region_exterior_xy(clipped)
        if region_xy is None:
            return 0
        pack_xy = region_xy
    placements, _ = pack_pixels_in_polygon(
        pack_xy, params.pixel_w_mm, params.pixel_h_mm,
        params.pitch_x_mm, params.pitch_y_mm, params.edge_keepout_mm,
        anchor="top_center", whole_within_xy=outline_xy)
    return len(placements)


def layout_fixed_shape_keepout(
    outline_xy: np.ndarray,
    params: Params,
    fill_xy: np.ndarray = None,
    seed_profile=None,
    baseline_pixels: int | None = None,
) -> tuple[Layout, object, dict]:
    """Layout for fixed-outline mode: keep the drawn outline, shrink the grid.

    The wiring keep-out is a variable-width band that follows the outline's own
    shape (see pipeline/keepout.py) — widest where wires pile up on their way to
    the tail, hugging the real edge where few pass. Pixels reaching into that
    band are simply never packed, so nothing is deleted after the fact.

    The band's width depends on the row/column net counts, which depend on the
    packing, which depends on the band. That fixed point is resolved by
    iterating with an element-wise max: widths only ever grow, so the inner loop
    cannot oscillate and is guaranteed to terminate.

    That ratchet alone, though, finds the *greatest* fixed point reachable from
    wherever it started rather than the tightest one, and the result depends on
    the starting width in a way that is backwards: a narrower start packs a
    denser first iteration, whose row count inflates the modelled demand, and
    np.maximum can never give that width back, so a narrower start can converge
    to a wider band. The ratchet is therefore wrapped in a relaxation loop:
    once it converges, the demand is recomputed from the layout that actually
    resulted (which needs less room than the transient dense one that set the
    width), the band is allowed to drop to it, and the ratchet runs again. Every
    converged candidate is scored and the densest is returned, so this can only
    ever match or beat the single-pass result.

    Returns (layout, profile, info). (The hug router does not use this — see
    layout_fixed_hug_keepout: its exact plan-driven demand needs no fixed
    point, and with joined nets this ratchet can keep shrinking the field.)
    """
    outline_poly = make_polygon(outline_xy)
    user_fill_poly = make_polygon(fill_xy) if fill_xy is not None else None

    floor = _fixed_keepout_floor(params)
    # A seed profile comes from _grow_keepout_for_failures: it is width that a
    # real routing failure demanded, so relaxation must never dip below it.
    lower_bound = seed_profile.widths.copy() if seed_profile is not None else None

    if seed_profile is not None:
        profile = seed_profile.copy()
    else:
        profile = keepout_mod.make_profile(outline_xy, floor)

    info = {
        "iterations": 0,
        "converged": False,
        "relax_passes": 0,
        "max_pack_pixels": int(baseline_pixels) if baseline_pixels is not None
                           else max_pack_count(outline_xy, params, fill_xy),
    }

    def carve_to_fixed_point(prof):
        """Run the ratchet to convergence. Returns (layout, profile, converged)."""
        layout = None
        converged = False
        for it in range(MAX_CARVE_ITERS):
            region = keepout_mod.region(outline_poly, prof)
            if user_fill_poly is not None and not region.is_empty:
                region = region.intersection(user_fill_poly)
            region_xy = _region_exterior_xy(region)
            if region_xy is None:
                raise ValueError(_TOO_SMALL_MSG)

            try:
                layout = layout_with_connector(outline_xy, params, region_xy, pack_anchor="top_center",
                                               whole_within_xy=outline_xy)
            except ValueError as exc:
                # A region can survive the carve geometrically and still be too
                # small for a single pixel. Report that as the keep-out mode's own
                # failure; every other ValueError here (e.g. "no connector fits")
                # is more specific than anything this layer could say.
                if not str(exc).startswith("No pixels packed"):
                    raise
                raise ValueError(_TOO_SMALL_MSG) from exc
            info["iterations"] += 1

            need = keepout_mod.analytic_widths(prof, params, layout)
            if np.all(need <= prof.widths + 1e-9):
                converged = True
                break
            prof.widths = np.maximum(prof.widths, need)
            keepout_mod.smooth_widths(prof)
        return layout, prof, converged

    best = None  # (n_pixels, layout, profile, converged)

    # Because the ratchet is path-dependent, *where the search starts* changes
    # which fixed point it lands on, and neither start dominates: a floor of 0
    # lets the model taper freely and usually wins, but on some boards the dense
    # first iteration inflates the demand enough that starting from a flat
    # routing_margin_mm ring ends up denser. Carving is cheap next to routing —
    # it is the same pack/measure loop with no router involved — so both starts
    # are tried and the one that packs more is kept.
    if seed_profile is not None:
        starts = [profile]
    else:
        starts = [profile]
        legacy = max(0.0, float(params.routing_margin_mm))
        if legacy > floor + 1e-9:
            starts.append(keepout_mod.make_profile(outline_xy, legacy))

    for start in starts:
        profile = start
        for relax_pass in range(MAX_RELAX_PASSES):
            layout, profile, converged = carve_to_fixed_point(profile)
            n = len(layout.placements_xy)
            if best is None or n > best[0]:
                best = (n, layout, profile.copy(), converged)
            info["relax_passes"] += 1

            # Give back width the *resulting* layout does not need. The demand
            # that set the current band was computed from a denser transient
            # layout, so this is generally strictly smaller.
            relaxed_widths = keepout_mod.analytic_widths(profile, params, layout)
            relaxed_widths = np.maximum(relaxed_widths, floor)
            if lower_bound is not None:
                relaxed_widths = np.maximum(relaxed_widths, lower_bound)
            candidate = keepout_mod.KeepoutProfile(profile.samples.copy(), relaxed_widths)
            keepout_mod.smooth_widths(candidate)
            if not np.any(candidate.widths < profile.widths - 1e-9):
                break  # nothing left to give back
            profile = candidate

    n_placed, layout, profile, converged = best
    info["converged"] = converged
    info["max_pack_pixels"] = max(info["max_pack_pixels"], n_placed)
    summary = profile.summary()
    layout.tail.update({
        "board_mode": "fixed_keepout",
        "fixed_board_routing": True,
        "fixed_keepout_min_mm": summary["min_mm"],
        "fixed_keepout_max_mm": summary["max_mm"],
        "fixed_keepout_mean_mm": summary["mean_mm"],
        "fixed_keepout_iterations": int(info["iterations"]),
        "fixed_keepout_converged": bool(info["converged"]),
        "max_pack_pixels": int(info["max_pack_pixels"]),
        "dropped_by_keepout": int(info["max_pack_pixels"] - n_placed),
    })
    return layout, profile, info


# Lattice phases tried per axis when packing a carved region (hug fixed mode).
# Which grid rows survive the carve depends strongly on the lattice phase, and
# a phase shift adds no nets (region ⊂ outline, so counts never exceed what the
# band was sized for).
PHASE_STEPS = 5


def _largest_grid_component(placements, params):
    """Indices of the largest connected component of a packed grid.

    The carved region can be connected while the packed grid is not (a waist
    too narrow for a pixel). The hug plan is only sound over one connected
    field — terminals of stranded fragments project onto the main contour and
    its laminarity breaks, which produces plan warnings and row-wire crossings
    in the output.
    """
    from shapely.geometry import box as _box
    if len(placements) == 0:
        return np.zeros(0, dtype=bool)
    hx, hy = params.pitch_x_mm / 2, params.pitch_y_mm / 2
    # Grow each cell by a hair before the union. Neighbouring cells are meant
    # to share an edge exactly, but the lattice coordinates are accumulated in
    # floating point, so a shared edge can land ~1e-15mm apart. unary_union
    # does not merge across that, the field reports as two components, and
    # everything outside the larger one is silently discarded -- a whole arm
    # of the field can be lost to a zero-width seam. The epsilon is far below
    # any real gap: a genuine break is at least a pitch wide.
    eps = 1e-6
    merged = unary_union([
        _box(x - hx - eps, y - hy - eps, x + hx + eps, y + hy + eps)
        for x, y in placements])
    if merged.geom_type != "MultiPolygon":
        return np.ones(len(placements), dtype=bool)
    biggest = max(merged.geoms, key=lambda g: g.area)
    return np.array([biggest.covers(Point(x, y)) for x, y in placements])


def _best_phase_placements(region_xy, params, board_xy=None) -> np.ndarray | None:
    """Pack the carved region at several lattice phases, keep the densest
    connected grid. Returns pixel centers (largest component only), or None."""
    best = None
    for i in range(PHASE_STEPS):
        for j in range(PHASE_STEPS):
            phase = (params.pitch_x_mm * i / PHASE_STEPS,
                     params.pitch_y_mm * j / PHASE_STEPS)
            placements, _ = pack_pixels_in_polygon(
                region_xy, params.pixel_w_mm, params.pixel_h_mm,
                params.pitch_x_mm, params.pitch_y_mm,
                params.edge_keepout_mm, anchor="top_center", phase=phase,
                whole_within_xy=board_xy)
            if len(placements) == 0:
                continue
            keep = _largest_grid_component(placements, params)
            kept = placements[keep]
            if best is None or len(kept) > len(best):
                best = kept
    return best


def layout_fixed_hug_keepout(
    outline_xy: np.ndarray,
    params: Params,
    fill_xy: np.ndarray = None,
    seed_profile=None,
    baseline_pixels: int | None = None,
    join_split_nets: bool = True,
) -> tuple[Layout, object, dict]:
    """Fixed-outline layout for the hug router: one-shot exact carve.

    The ratchet used with the other routers (layout_fixed_shape_keepout)
    re-measures demand on each shrunken field and only ever grows — with the
    hug router's joined nets the net structure shifts under it each iteration
    (segments merge/split, link corridors move), and on concave shapes the
    ratchet keeps widening the band until most of the field is lost. The hug
    plan's demand is exact, so no fixed point is needed: compute the band once
    from the full max-pack layout's routing plan, carve once, pack once.
    Under-reservation from the carve changing the net structure is corrected
    by the caller's escape-driven retry (grow_near + re-carve via
    seed_profile), which is local instead of global.
    """
    from .route_hug import hug_keepout_widths

    outline_poly = make_polygon(outline_xy)
    user_fill_poly = make_polygon(fill_xy) if fill_xy is not None else None
    floor = _fixed_keepout_floor(params)
    info = {
        "iterations": 1,
        "converged": True,
        "relax_passes": 0,
        "max_pack_pixels": int(baseline_pixels) if baseline_pixels is not None
                           else max_pack_count(outline_xy, params, fill_xy),
    }

    if seed_profile is not None:
        profile = seed_profile.copy()
    else:
        # Provisional full pack, only to size the wiring band. It uses the same
        # connector as the final layout when it fits; when the full grid is too
        # big for that board (the carve below takes rows away), any connector
        # will do here, and the board limit applies to the carved layout.
        try:
            full_layout = layout_with_connector(
                outline_xy, params, fill_xy, pack_anchor="top_center",
                join_split_nets=join_split_nets)
        except ConnectorCapacityError:
            full_layout = layout_with_connector(
                outline_xy, dataclasses.replace(params, readout_board="any",
                                                connector_pos_override=None),
                fill_xy, pack_anchor="top_center", join_split_nets=join_split_nets)
        profile = keepout_mod.make_profile(outline_xy, floor)
        profile.widths = np.maximum(
            profile.widths, hug_keepout_widths(profile, params, full_layout))
    keepout_mod.smooth_widths(profile)

    region = keepout_mod.region(outline_poly, profile)
    if user_fill_poly is not None and not region.is_empty:
        region = region.intersection(user_fill_poly)
    region_xy = _region_exterior_xy(region)
    if region_xy is None:
        raise ValueError(_TOO_SMALL_MSG)

    placements = _best_phase_placements(region_xy, params, board_xy=outline_xy)
    if placements is None:
        raise ValueError(_TOO_SMALL_MSG)
    try:
        layout = layout_with_connector(outline_xy, params,
                                       pack_anchor="top_center",
                                       join_split_nets=join_split_nets,
                                       placements_override=placements)
    except ValueError as exc:
        if not str(exc).startswith("No pixels packed"):
            raise
        raise ValueError(_TOO_SMALL_MSG) from exc

    n_placed = len(layout.placements_xy)
    info["max_pack_pixels"] = max(info["max_pack_pixels"], n_placed)
    summary = profile.summary()
    layout.tail.update({
        "board_mode": "fixed_keepout",
        "fixed_board_routing": True,
        "fixed_keepout_min_mm": summary["min_mm"],
        "fixed_keepout_max_mm": summary["max_mm"],
        "fixed_keepout_mean_mm": summary["mean_mm"],
        "fixed_keepout_iterations": int(info["iterations"]),
        "fixed_keepout_converged": bool(info["converged"]),
        "max_pack_pixels": int(info["max_pack_pixels"]),
        "dropped_by_keepout": int(info["max_pack_pixels"] - n_placed),
    })
    return layout, profile, info


# =========================================================================
# Net assignment: segment rows/cols when gaps exist
# =========================================================================

def cluster_axis(vals: np.ndarray, tol: float) -> np.ndarray:
    """Cluster 1D values with rolling-mean tolerance."""
    order = np.argsort(vals)
    sv = vals[order]
    labels = np.zeros(len(vals), dtype=int)
    cur = 1
    mu = sv[0]
    labels[order[0]] = cur
    for i in range(1, len(vals)):
        if abs(sv[i] - mu) > tol:
            cur += 1
            mu = sv[i]
        else:
            mu = 0.5 * (mu + sv[i])
        labels[order[i]] = cur
    return labels


def remap_labels(labels: np.ndarray, new_order: np.ndarray) -> np.ndarray:
    """Remap cluster labels according to sorted order."""
    K = labels.max()
    inv_map = np.zeros(K + 1, dtype=int)
    for new_idx, old_idx in enumerate(new_order):
        inv_map[old_idx] = new_idx + 1
    return inv_map[labels]


def assign_rowcol_nets_segmented(P: np.ndarray, pitch_x: float, pitch_y: float):
    """Assign segmented ROW/COL net names to pixels."""
    col_id = cluster_axis(P[:, 0], pitch_x * 0.55)
    row_id = cluster_axis(P[:, 1], pitch_y * 0.55)

    # Remap to dense indices sorted by mean coordinate
    unique_cols = np.unique(col_id)
    col_centers = np.array([P[col_id == c, 0].mean() for c in unique_cols])
    ord_c = unique_cols[np.argsort(col_centers)]
    col_idx = remap_labels(col_id, ord_c)

    unique_rows = np.unique(row_id)
    row_centers = np.array([P[row_id == r, 1].mean() for r in unique_rows])
    ord_r = unique_rows[np.argsort(row_centers)]
    row_idx = remap_labels(row_id, ord_r)

    N = len(P)
    row_net_per_pixel = [""] * N
    col_net_per_pixel = [""] * N

    # Column segmentation by gaps along Y
    n_cols = col_idx.max()
    for c in range(1, n_cols + 1):
        idx = np.where(col_idx == c)[0]
        if len(idx) == 0:
            continue
        ys = P[idx, 1]
        sort_order = np.argsort(ys)
        idx = idx[sort_order]
        ys = ys[sort_order]

        seg = 0
        col_net_per_pixel[idx[0]] = f"COL_{c-1}_{seg}"
        for k in range(1, len(idx)):
            if (ys[k] - ys[k - 1]) > pitch_y * 1.25:
                seg += 1
            col_net_per_pixel[idx[k]] = f"COL_{c-1}_{seg}"

    # Row segmentation by gaps along X
    n_rows = row_idx.max()
    for r in range(1, n_rows + 1):
        idx = np.where(row_idx == r)[0]
        if len(idx) == 0:
            continue
        xs = P[idx, 0]
        sort_order = np.argsort(xs)
        idx = idx[sort_order]
        xs = xs[sort_order]

        seg = 0
        row_net_per_pixel[idx[0]] = f"ROW_{r-1}_{seg}"
        for k in range(1, len(idx)):
            if (xs[k] - xs[k - 1]) > pitch_x * 1.25:
                seg += 1
            row_net_per_pixel[idx[k]] = f"ROW_{r-1}_{seg}"

    row_nets = sorted(set(row_net_per_pixel) - {""})
    col_nets = sorted(set(col_net_per_pixel) - {""})

    return row_net_per_pixel, col_net_per_pixel, row_nets, col_nets


def assign_rowcol_nets_joined(P: np.ndarray, pitch_x: float, pitch_y: float):
    """Assign ROW/COL net names where a gap does not split the net (hug mode).

    Reuses the segmented assignment's clustering exactly, then strips the
    segment suffix: COL_{c}_{0..k} all become COL_{c}. The hug router later
    re-derives the segments itself to plan the link traces that bridge each
    gap around the concavity.
    """
    row_seg, col_seg, _, _ = assign_rowcol_nets_segmented(P, pitch_x, pitch_y)
    row_net_per_pixel = [n.rsplit("_", 1)[0] if n else "" for n in row_seg]
    col_net_per_pixel = [n.rsplit("_", 1)[0] if n else "" for n in col_seg]
    row_nets = sorted(set(row_net_per_pixel) - {""})
    col_nets = sorted(set(col_net_per_pixel) - {""})
    return row_net_per_pixel, col_net_per_pixel, row_nets, col_nets


# =========================================================================
# Connector selection
# =========================================================================


# Connectors auto-selected by select_connector_for_nets(): every increment of
# 10 from 20 to 70. Each uses the mechanically detailed footprint in
# connectors/ (latch ear, chamfered lead-in, stiffener zone). connectors/ also
# holds a 32-pos footprint, which is intentionally excluded from selection here.
CONNECTOR_AUTO_OPTIONS = [20, 30, 40, 50, 60, 70]

# Larger sizes in the same family, offered only for an explicit manual
# override (e.g. to match an existing/larger readout board) via
# GenerateRequest.connector_pos_override — never chosen automatically.
CONNECTOR_UPSIZE_OPTIONS = [80, 90, 100]

CONNECTOR_FAMILY = CONNECTOR_AUTO_OPTIONS + CONNECTOR_UPSIZE_OPTIONS

# The OpenTac readout boards and the FPC connector each one carries. A sensor
# made for one of them uses that board's connector size; on both boards one
# connector half is wired to the drive side and the other to the sense side,
# which matches connector_segment_pad_ranges' split exactly (30-pos: 1-15 /
# 16-30, 60-pos: 1-30 / 31-60), so rows and columns may take either half.
READOUT_BOARDS = {"30pin": 30, "60pin": 60}
READOUT_BOARD_LIMITS = {"30pin": "15 rows x 15 columns", "60pin": "30 rows x 30 columns"}
class ConnectorCapacityError(ValueError):
    """The sensor has more rows or columns than the connector (or readout
    board) allowed for it can carry."""


# "auto": the 30-pin board when the grid fits it, else the 60-pin board.
# "any": the whole auto-selectable family (no OpenTac readout board for most sizes).
READOUT_BOARD_CHOICES = ("auto", "30pin", "60pin", "any")


def connector_options_for(readout_board: str) -> list[int]:
    """Connector sizes the selector may use for a readout-board choice."""
    if readout_board == "auto":
        return [READOUT_BOARDS["30pin"], READOUT_BOARDS["60pin"]]
    if readout_board in READOUT_BOARDS:
        return [READOUT_BOARDS[readout_board]]
    if readout_board == "any":
        return list(CONNECTOR_AUTO_OPTIONS)
    raise ValueError(
        f"readout_board must be one of {READOUT_BOARD_CHOICES}, got {readout_board!r}.")


# Pads deliberately left unconnected so the row/column boundary can land on the
# connector's true mechanical centre. Only 70-pos needs this: its 70 pads are an
# even count but the routed nets are split into two equal banks of 32, so three
# pads at each end are burned to make the halves symmetric about pad 35.5.
CONNECTOR_FIXED_NC = {
    70: [1, 2, 3, 68, 69, 70],
}


def connector_segment_pad_ranges(conn_pos: int):
    """Return (left_pads, right_pads, fixed_nc) for connector position count.

    The boundary between the two segments is the connector's centre line:
    build_pad_to_net_map_centered packs the row nets up against it from one
    side and the column nets from the other, and apply_tail_extension centres
    the tail corridor and the connector body on the same split point. So the
    two segments must hold an equal number of assignable pads — an uneven split
    silently slides the whole used-pad block off the connector housing (and
    with it the wire bundle), by half the imbalance.

    Derived from the pad count rather than tabulated per size, so every size
    splits exactly about its centre.
    """
    if conn_pos not in CONNECTOR_FAMILY and conn_pos != 32:
        raise ValueError(f"No segment definition for connector {conn_pos}")
    fixed_nc = CONNECTOR_FIXED_NC.get(conn_pos, [])
    assignable = [p for p in range(1, conn_pos + 1) if p not in fixed_nc]
    if len(assignable) % 2:
        raise ValueError(
            f"Connector {conn_pos} has an odd number of assignable pads "
            f"({len(assignable)}); it cannot be split evenly about its centre. "
            "Add a balancing pair to CONNECTOR_FIXED_NC.")
    half = len(assignable) // 2
    return assignable[:half], assignable[half:], fixed_nc


def _connector_capacity(conn_pos: int) -> tuple[int, int]:
    """Return (L, R) — the number of assignable (non-NC) pads on each side."""
    left_pads, right_pads, fixed_nc = connector_segment_pad_ranges(conn_pos)
    left_avail = [p for p in left_pads if p not in fixed_nc]
    right_avail = [p for p in right_pads if p not in fixed_nc]
    return len(left_avail), len(right_avail)


def _best_fit_swap(conn_pos: int, n_row: int, n_col: int):
    """Return (fits, swap) for whether/how n_row/n_col nets fit conn_pos."""
    L, R = _connector_capacity(conn_pos)
    fit_normal = n_row <= L and n_col <= R
    fit_swap = n_col <= L and n_row <= R
    if not (fit_normal or fit_swap):
        return False, False
    if fit_normal and not fit_swap:
        return True, False
    if fit_swap and not fit_normal:
        return True, True
    # Both fit: choose better utilization.
    s_normal = max(n_row / max(1, L), n_col / max(1, R))
    s_swap = max(n_col / max(1, L), n_row / max(1, R))
    return True, s_swap < s_normal - 1e-9


def select_connector_for_nets(n_row: int, n_col: int,
                              readout_board: str = "any") -> tuple[int, bool]:
    """Select the smallest connector that fits the nets, among the sizes the
    readout-board choice allows (see connector_options_for)."""
    options = connector_options_for(readout_board)
    if readout_board != "any" and not any(_best_fit_swap(cp, n_row, n_col)[0] for cp in options):
        board = "30pin" if readout_board == "30pin" else "60pin"
        hint = ("Choose the 60-pin readout board, or use a larger taxel pitch or a smaller outline."
                if board == "30pin" else
                "Use a larger taxel pitch or a smaller outline.")
        raise ConnectorCapacityError(
            f"This sensor needs {n_row} rows and {n_col} columns, but the "
            f"{board[:2]}-pin readout board reads at most {READOUT_BOARD_LIMITS[board]}. {hint}")
    capacities = []
    for cp in options:
        L, R = _connector_capacity(cp)
        capacities.append((cp, L, R))
        fits, swap = _best_fit_swap(cp, n_row, n_col)
        if fits:
            return cp, swap

    largest_cp, largest_l, largest_r = capacities[-1]
    suggested = next(
        (cp for cp in CONNECTOR_UPSIZE_OPTIONS if _best_fit_swap(cp, n_row, n_col)[0]),
        None,
    )
    suggestion = (
        f"pass connector_pos_override={suggested} to use a larger connector"
        if suggested is not None
        else f"reduce row/column count or add connector support larger than {CONNECTOR_FAMILY[-1]}-pos"
    )
    raise ConnectorCapacityError(
        f"No connector fits n_row={n_row} n_col={n_col}. "
        f"Largest auto-selectable connector is {largest_cp}-pos with capacity "
        f"{largest_l} nets on one side and {largest_r} nets on the other. "
        f"{suggestion}."
    )


def resolve_connector_override(conn_pos: int, n_row: int, n_col: int) -> tuple[int, bool]:
    """Validate and resolve swap orientation for a manually-forced connector size."""
    if conn_pos not in CONNECTOR_FAMILY:
        raise ValueError(
            f"connector_pos_override must be one of {CONNECTOR_FAMILY}, got {conn_pos}."
        )
    fits, swap = _best_fit_swap(conn_pos, n_row, n_col)
    if not fits:
        L, R = _connector_capacity(conn_pos)
        raise ConnectorCapacityError(
            f"{conn_pos}-pos connector cannot fit n_row={n_row} n_col={n_col} "
            f"(capacity {L} nets on one side, {R} on the other). "
            f"Choose a larger connector_pos_override."
        )
    return conn_pos, swap


def resize_pad_to_net_map(old_conn_pos: int, new_conn_pos: int, swap: bool, pad_to_net: dict) -> dict:
    """Rebuild pad_to_net for a different connector size on an already-routed
    board — used by the lightweight "use a bigger connector" operation.

    Deliberately does not recompute `swap`: it must stay exactly as routed,
    since flipping it would swap which physical side row vs. column nets
    land on and invalidate every existing route. Net order within each side
    is reconstructed from the *existing* pad_to_net (sorted by old pad
    number) rather than re-derived from packing/geometry, so this is a pure
    pad-number remap — physical positions of already-used pads are
    unchanged by connector size: only pad *numbers* shift, and only
    unused/NC pads move.
    """
    if new_conn_pos not in CONNECTOR_FAMILY:
        raise ValueError(f"connector_pos_override must be one of {CONNECTOR_FAMILY}, got {new_conn_pos}.")

    old_left_pads, old_right_pads, old_nc = connector_segment_pad_ranges(old_conn_pos)
    old_left_set = set(old_left_pads) - set(old_nc)
    old_right_set = set(old_right_pads) - set(old_nc)

    left_nets = [pad_to_net[p] for p in sorted(pad_to_net) if p in old_left_set]
    right_nets = [pad_to_net[p] for p in sorted(pad_to_net) if p in old_right_set]

    row_nets_ordered, col_nets_ordered = (left_nets, right_nets) if not swap else (right_nets, left_nets)
    n_row, n_col = len(row_nets_ordered), len(col_nets_ordered)

    L, R = _connector_capacity(new_conn_pos)
    fits = (n_row <= L and n_col <= R) if not swap else (n_col <= L and n_row <= R)
    if not fits:
        raise ValueError(
            f"{new_conn_pos}-pos connector cannot fit the existing routed net arrangement "
            f"({n_row} row nets, {n_col} column nets; {new_conn_pos}-pos has capacity "
            f"{L} nets on one side, {R} on the other). Choose a larger connector size."
        )

    return build_pad_to_net_map_centered(new_conn_pos, swap, row_nets_ordered, col_nets_ordered)


def build_pad_to_net_map_centered(conn_pos, swap, row_nets_ordered, col_nets_ordered):
    """Build pad-to-net mapping, pushing nets toward center of connector."""
    left_pads, right_pads, fixed_nc = connector_segment_pad_ranges(conn_pos)
    left_avail = sorted([p for p in left_pads if p not in fixed_nc])
    right_avail = sorted([p for p in right_pads if p not in fixed_nc])

    if not swap:
        left_nets = row_nets_ordered
        right_nets = col_nets_ordered
    else:
        left_nets = col_nets_ordered
        right_nets = row_nets_ordered

    n_l = len(left_nets)
    n_r = len(right_nets)

    # Push toward segment boundary (center of connector)
    left_use = left_avail[-n_l:]    # rightmost n_l of left segment
    right_use = right_avail[:n_r]   # leftmost n_r of right segment

    pad_to_net = {}
    for i, net in enumerate(left_nets):
        pad_to_net[left_use[i]] = net
    for i, net in enumerate(right_nets):
        pad_to_net[right_use[i]] = net

    return pad_to_net


def split_row_pin_order(row_nets_ordered):
    """Order row pads as top half, then bottom half reversed for split-side routing."""
    top_count = (len(row_nets_ordered) + 1) // 2
    top_rows = row_nets_ordered[:top_count]
    bottom_rows = row_nets_ordered[top_count:]
    return [*top_rows, *reversed(bottom_rows)]


def sort_nets_by_mean_xy(net_list, P, net_per_pixel, which):
    """Sort nets by mean coordinate of their pixels."""
    means = []
    for net in net_list:
        idx = [i for i, n in enumerate(net_per_pixel) if n == net]
        pts = P[idx]
        means.append(pts.mean(axis=0))

    means = np.array(means)
    if which == "COL":
        order = np.argsort(means[:, 0])  # left -> right
    else:
        order = np.argsort(-means[:, 1])  # top -> bottom (descending Y)

    return [net_list[i] for i in order]
