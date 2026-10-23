"""Excellon drill file generation for vias.

Coordinates are the board frame (mm, Y up) written as is, the frame KiCad's
drill writer and every Gerber in the same export use (see export/gerber.py).
"""

import numpy as np
from opentac.cad.pipeline.types import Layout, Params
from opentac.cad.export.kicad_footprint import pixel_geometry
from opentac.cad.export.connector_contacts import with_drilled
from opentac.cad.export.connector_geometry import connector_pad_geometries


def generate_drill_file(
    layout: Layout,
    removed_pix: set[int],
    params: Params,
    outline_xy=None,
    connector_dir: str = "",
    transition_vias=(),
    drilled_pads=(),
) -> str:
    """Generate Excellon drill file for the taxel vias, the connector's
    transition vias (export/connector_contacts.py) and any connector drill."""
    active_pix = sorted(set(range(len(layout.placements_xy))) - removed_pix)
    Pxy = layout.placements_xy
    g = pixel_geometry(params.pixel_w_mm, params.pixel_h_mm, params.trace_w_mm,
                       params.gap_mm, params.center_clear_mm, params.edge_clear_mm,
                       params.via_dia_mm)

    holes: list[tuple[float, float, float]] = []
    for k in active_pix:
        # pixel_geometry is the footprint's Y-down frame: local y goes in negated
        holes.append((params.via_drill_mm, Pxy[k, 0] + g.via_x, Pxy[k, 1] - g.via_y))

    if outline_xy is not None and connector_dir:
        for pad in with_drilled(connector_pad_geometries(layout, np.asarray(outline_xy, dtype=float), params,
                                                         connector_dir), drilled_pads, params):
            if pad.get("drill"):
                drill_xy = pad.get("drill_center", pad["center"])
                holes.append((float(pad["drill"]), float(drill_xy[0]), float(drill_xy[1])))

    for via in transition_vias or ():
        holes.append((float(via.drill), float(via.xy[0]), float(via.xy[1])))

    tools: dict[float, list[tuple[float, float]]] = {}
    for dia, x, y in holes:
        tools.setdefault(round(float(dia), 6), []).append((x, y))

    lines = []
    lines.append("M48")
    lines.append("; Drill file for tactile PCB vias and connector holes")
    lines.append("; Format: metric, absolute")
    lines.append("METRIC")
    for idx, dia in enumerate(sorted(tools), start=1):
        lines.append(f"T{idx}C{dia:.3f}")
    lines.append("%")

    for idx, dia in enumerate(sorted(tools), start=1):
        lines.append(f"T{idx}")
        for x, y in tools[dia]:
            lines.append(f"X{x:.4f}Y{y:.4f}")

    lines.append("M30")
    return '\n'.join(lines)
