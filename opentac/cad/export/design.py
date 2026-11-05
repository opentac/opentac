"""The design record: what an export has to carry for OpenTac-CAD to re-open it.

manifest.json describes the board that was made — taxel positions, the taxel
footprint, the real cut boundary. That is deliberately a description of the
*result*, and it is not enough to keep working on a design: `/generate` takes
about thirty-five inputs and the board description covers only a few of
them, so a re-import from it alone would have to guess board_mode, the router
and every clearance. The two records here close that gap, and they answer
two different questions:

  `design_block()` -> the "design" key inside manifest.json. The recipe:
      every /generate input, next to the outline the manifest already stores.
      Re-running it rebuilds the board from scratch on whatever pipeline is
      current. Small, human-readable, survives a pipeline change — and loses
      anything done by hand after generation.

  `design_file()`  -> design.json, its own entry in the ZIP. The save file:
      the editor's `edit_data` plus the live routes, which is exactly the
      state /export-edited, /redo-padding, the bump sheet and the case
      builder already run on. Restores hand-dragged routes, a tail extension
      and a swapped connector — but it is a snapshot of one pipeline version,
      so it carries a version stamp and is refused rather than silently
      misread when that stamp does not match the current pipeline.

Note on `removed_pix`: it is an output, not a user edit. Nothing in the UI
lets a user delete a taxel; the set is what the router could not wire, and a
recipe replay with the same params re-derives it. It is recorded anyway
because design.json has to restore the exact board, and because it tells a
reader of the manifest why the taxel count is what it is.

This module owns `params_to_dict` (main.py delegates to it) so that
export/manifest.py can build the design block without importing main, which
imports manifest.
"""

import base64
import io
import json
import zipfile

import numpy as np

from opentac.cad.pipeline.types import Params

# Bump when a field's meaning changes, not when one is added: unknown keys are
# ignored by every consumer, but a re-used key that means something else
# would be read wrong. The importer refuses a schema it does not know.
DESIGN_SCHEMA = 1

# Identifies the pipeline that wrote a design.json. An exact restore replays a
# stored layout and stored route polylines through the export chain, so it is
# only trustworthy against the pipeline that produced them. Bump this whenever
# routing, packing or board shaping changes in a way that would make an old
# snapshot export a different board than the one it previewed.
PIPELINE_VERSION = "2026.09.04"

APP_NAME = "OpenTac-CAD"


def params_to_dict(params: Params) -> dict:
    """Params as plain JSON, for edit_data and for the manifest design block.

    Deliberately omits target_cols/target_rows and connector_pos_override:
    those are *requests* the pipeline has already resolved by the time this
    runs. The solved taxel size and pitch are in pixel_*_mm/pitch_*_mm, so
    replaying this dict reproduces the same grid without re-solving it, and
    the resolved connector is recorded separately (see design_block).
    """
    return {
        "pixel_w_mm": params.pixel_w_mm,
        "pixel_h_mm": params.pixel_h_mm,
        "pitch_x_mm": params.pitch_x_mm,
        "pitch_y_mm": params.pitch_y_mm,
        "edge_keepout_mm": params.edge_keepout_mm,
        "trace_w_mm": params.trace_w_mm,
        "gap_mm": params.gap_mm,
        "clearance_mm": params.clearance_mm,
        "center_clear_mm": params.center_clear_mm,
        "edge_clear_mm": params.edge_clear_mm,
        "via_drill_mm": params.via_drill_mm,
        "via_dia_mm": params.via_dia_mm,
        "pad_pitch_mm": params.pad_pitch_mm,
        "cable_length_mm": params.cable_length_mm,
        "cable_length_auto": bool(getattr(params, "cable_length_auto", False)),
        "tail_width_mm": params.tail_width_mm,
        "tail_side_padding_mm": params.tail_side_padding_mm,
        "board_edge_clear_mm": params.board_edge_clear_mm,
        "board_mode": params.board_mode,
        # 2 or 4: a re-opened or edited 4-layer board must export its inner
        # layers again (Route.layer alone does not add them to the stack).
        "copper_layers": int(getattr(params, "copper_layers", 2) or 2),
        "preserve_sensors": params.preserve_sensors,
        "auto_expand_board": params.auto_expand_board,
        "routing_margin_mm": params.routing_margin_mm,
        "wiring_keepout_floor_mm": params.wiring_keepout_floor_mm,
        "follow_main_padding": params.follow_main_padding,
        "follow_main_padding_mm": params.follow_main_padding_mm,
        "smooth_follow_padding": params.smooth_follow_padding,
        "connector_shoulder_fillet_mm": params.connector_shoulder_fillet_mm,
        # Round-trips so a re-opened design keeps its silkscreen label instead
        # of silently re-exporting as an unnamed board.
        "design_name": params.design_name,
        "readout_board": params.readout_board,
        "narrow_tail": params.narrow_tail,
        # Only when it is not the default, so a bottom-contact design's
        # manifest (and its configuration id) is what it was before the
        # setting existed, and older versions of the tool still open it.
        **({"contact_side": params.contact_side}
           if (getattr(params, "contact_side", None) or "bottom") != "bottom" else {}),
    }


def _points(xy):
    if xy is None:
        return None
    arr = np.asarray(xy, dtype=float)
    if arr.ndim != 2 or len(arr) < 3:
        return None
    return [[round(float(x), 4), round(float(y), 4)] for x, y in arr]


def design_block(layout, params: Params, removed, fill_region=None) -> dict:
    """The `design` key of manifest.json — every input /generate needs.

    Pairs with the manifest's own top-level `outline`, which is already in the
    canonical cable-edge-down frame and is idempotent under rotate_for_cable,
    so it can be handed straight back to /generate as the drawn outline.
    """
    return {
        "app": APP_NAME,
        "schema": DESIGN_SCHEMA,
        "pipeline": PIPELINE_VERSION,
        # Not a Params field — the router choice lives on the request, and is
        # recorded onto layout.tail by whichever branch actually ran (so a hug
        # request that fell back to legacy is stored as legacy, which is what
        # reproduces the board).
        "router": (layout.tail.get("router", "legacy") if layout is not None else "legacy"),
        # The connector the pipeline actually resolved to, replayed as
        # connector_pos_override so a hand-upsized connector survives a
        # re-import instead of dropping back to the smallest that fits.
        "connector_pos": (int(layout.connector_pos) if layout is not None else None),
        "params": params_to_dict(params),
        # The sensorize zone, in the same rotated frame as `outline`. The
        # manifest records it nowhere else: layout.sensor_outline_xy is set to
        # the outline, not to this, so without this key a replay would lose it.
        "fill_region": _points(fill_region),
        # See the module docstring: an output, not a user edit.
        "removed_pix": sorted(int(i) for i in (removed or ())),
    }


def design_file(edit_data: dict, routes=None) -> dict:
    """design.json — the editor state needed to restore a board exactly."""
    return {
        "app": APP_NAME,
        "kind": "design",
        "schema": DESIGN_SCHEMA,
        "pipeline": PIPELINE_VERSION,
        "edit_data": edit_data,
        # Held beside edit_data rather than merged into it because these are
        # the live routes: on an edited export they are the user's dragged
        # polylines, which is exactly what edit_data["routes"] is not.
        "routes": routes if routes is not None else (edit_data.get("routes") or []),
    }


def zip_with_design(zip_b64: str, design: dict) -> str:
    """Return the export ZIP with design.json added.

    Done by rewriting the archive rather than threading a payload into
    _build_outputs, because the ZIP is built there on a deep copy of the
    layout while edit_data is built afterwards from the original (see
    `_outputs_with_connector_tail` in main.py). Rewriting
    leaves the export chain — the part that actually gets manufactured —
    completely untouched.
    """
    src = zipfile.ZipFile(io.BytesIO(base64.b64decode(zip_b64)))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for info in src.infolist():
            if info.filename == "design.json":
                continue
            zf.writestr(info, src.read(info.filename))
        zf.writestr("design.json", json.dumps(design, indent=2))
    return base64.b64encode(out.getvalue()).decode()
