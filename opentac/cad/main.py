"""FastAPI backend for tactile PCB generation."""

import base64
import copy
import io
import json
import logging
import os
import shutil
import subprocess
import tempfile
import traceback
import zipfile
from dataclasses import replace

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import nearest_points, unary_union

from opentac.cad.pipeline.types import Params, drop_removed_params
from opentac.cad.pipeline.types import DrcViolation, Layout, Route
from opentac.cad.pipeline.geometry import (
    rotate_for_cable, remove_collinear_closed, tail_geometry, tail_offset_mm,
    connector_dimensions, CONNECTOR_BODY_BELOW_MM,
    round_convex_corners, concave_fillet_infill,
)
from opentac.cad.pipeline.layout import (
    cable_shoulder_fill,
    layout_fixed_hug_keepout, layout_fixed_shape_keepout, layout_with_connector,
    resize_pad_to_net_map,
    solve_grid_pixel_size,
)
from opentac.cad.pipeline.bump_sheet import BumpParams, build_bump_sheet
from opentac.cad.pipeline.case import NECK_STABLE_MM, CaseParams, build_case, case_cable_check
from opentac.cad.pipeline import keepout as keepout_mod
from opentac.cad.pipeline.keepout import _TOO_SMALL_MSG as _KEEPOUT_TOO_SMALL_MSG
from opentac.cad.pipeline.route_columns import route_columns_grid
from opentac.cad.pipeline.route_hug import route_hug_all
from opentac.cad.pipeline.route_4layer import INNER_LAYERS, copper_layers_of, is_feed, route_for, sync_pads_to_feeds
from opentac.cad.pipeline.route_rows import make_row_entries_horizontal, route_rows_grid
from opentac.cad.pipeline.tail_extension import apply_tail_extension, tail_extension_from_layout
from opentac.cad.pipeline.follow_padding import (
    apply_main_follow_padding, cable_without_landing, route_containment_mm, tail_wire_hold,
)
from opentac.cad.pipeline.narrow_tail import narrow_routed_tail
from opentac.cad.pipeline.tail_length import (
    AUTO_TAIL_MIN_MM, REASON_CASE, REASON_FIT, REASON_MIN, REASON_TYPED, fit_tail_length, round_up_length,
    stretch_routed_tail, tail_trim_room, trim_routed_tail,
)
from opentac.cad.pipeline.validate import validate_routes
from opentac.cad.pipeline.smooth import chamfer_layer, los_simplify_layer, taxel_copper_keepout_mm
from opentac.cad.pipeline.drc import TaxelCopper, drc_check
from opentac.cad.export.kicad_pcb import generate_kicad_pcb, kicad_project_files
from opentac.cad.export.edge_cuts_check import edge_cuts_violations
from opentac.cad.export.kicad_footprint import build_pixel_footprint_text, make_footprint_name
from opentac.cad.export.coverlay import taxel_opening_for
from opentac.cad.export.manifest import export_manifest
from opentac.cad.export.design import (
    params_to_dict, design_file, zip_with_design, DESIGN_SCHEMA, PIPELINE_VERSION,
)
from opentac.cad.export.legacy_manifest import (
    is_legacy_manifest, recover as recover_legacy, geometry_svg,
)
from opentac.cad.export.case_zip import build_case_files, zip_with_case
from opentac.cad.export.print_pdf import generate_print_pdf
from opentac.cad.export.silk_backmark import backmark
from opentac.cad.export.velostat import velostat_polygon, velostat_dxf, velostat_fold_pdf
from opentac.cad.export.gerber import generate_gerbers
from opentac.cad.export.excellon import generate_drill_file
from opentac.cad.export.connector_contacts import CONTACT_SIDES, contact_transitions
from opentac.cad.export.connector_geometry import (
    clip_beside_tab, connector_edge_geometries, connector_export_dir, connector_open_edge_endpoints,
    connector_geometries, connector_opening, connector_pad_geometries,
)
from opentac.cad.export.svg_preview import generate_svg
from opentac.cad.importers.dxf import parse_dxf_outline, DxfImportError
from opentac.cad.importers.svg import parse_svg_outline, SvgImportError

app = FastAPI(title="OpenTac-CAD API", version="0.1.0")
logger = logging.getLogger(__name__)

_UNROUTED_MSG = (
    "Some rows or columns cannot be wired to the connector inside this fixed outline, even on its "
    "most conservative layout, and OpenTac-CAD does not produce a board with unconnected taxels. "
    "Enlarge the outline, coarsen the pitch, or switch to grow-board mode.")

MAX_DXF_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_SVG_UPLOAD_BYTES = MAX_DXF_UPLOAD_BYTES
# Upload limit for the STL endpoints (/unroll-mesh, /grow-region,
# /unroll-mesh-chain).
MAX_STL_UPLOAD_BYTES = 50 * 1024 * 1024


class JsonErrors:
    """Unhandled exceptions as a JSON 500 that still carries the CORS headers.

    Starlette turns an exception no handler caught into a plain-text 500 in
    ServerErrorMiddleware, which sits outside every user middleware, CORS
    included. The hosted page is cross-origin (GitHub Pages -> the hosted API
    server), so that 500 would reach the browser without
    Access-Control-Allow-Origin and surface as "TypeError: Failed to fetch",
    indistinguishable from the server being down. Registered before
    CORSMiddleware so CORS wraps it.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = False

        async def send_tracking(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, receive, send_tracking)
        except Exception as exc:
            if started:      # half a response is already out; nothing clean to send
                raise
            traceback.print_exc()
            msg = f"Server error: {type(exc).__name__}: {exc}"
            await JSONResponse(status_code=500, content={"detail": msg, "error": msg})(
                scope, receive, send)


# Order matters: middleware added later wraps what was added before, so JsonErrors
# must come first for its 500s to pass back out through CORS.
app.add_middleware(JsonErrors)
app.add_middleware(
    CORSMiddleware,
    # The hosted demo's page lives on GitHub Pages; a local run serves the page
    # itself (same origin), so it needs no entry here.
    allow_origins=[
        "https://opentac.github.io",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def no_cache_for_local_development(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


CONNECTOR_DIR = os.path.join(os.path.dirname(__file__), "connectors")



class GenerateRequest(BaseModel):
    outline: list[list[float]] = Field(..., description="Nx2 polygon vertices in mm")
    fill_region: list[list[float]] | None = Field(
        None, description="Optional Nx2 polygon (mm) constraining where sensors are packed")
    pixel_w_mm: float = 4.0
    pixel_h_mm: float = 4.0
    pitch_x_mm: float = 4.2
    pitch_y_mm: float = 4.2
    edge_keepout_mm: float = 0.5
    trace_w_mm: float = 0.2
    gap_mm: float = 0.2
    clearance_mm: float = 0.2
    center_clear_mm: float = 0.2
    edge_clear_mm: float = 0.1
    via_drill_mm: float = 0.15
    via_dia_mm: float = 0.35
    pad_pitch_mm: float = 0.5
    cable_length_mm: float | None = Field(
        None, gt=0,
        description="Tail length (mm) from the cable edge to the connector. Omit for "
                    "automatic: the shortest tail of at least 12 mm, in 0.5 mm steps, that "
                    "lets the narrow tail reach the width of its wires and lets the printed "
                    "case clear the connector body (stats tail_length_mm / "
                    "tail_length_reason). A number is used as given.")
    tail_width_mm: float | None = None
    tail_side_padding_mm: float = 0.0
    board_edge_clear_mm: float = 0.2
    board_mode: str | None = Field(
        None,
        description='Board-generation mode: "expand" (grow the board to fit the wiring) '
                    'or "fixed_keepout" (keep the drawn outline, shrink the sensor grid). '
                    "Omit to derive it from the legacy auto_expand_board flag.")
    copper_layers: int = Field(
        2,
        description="Copper layers: 2 (columns on F.Cu, rows on B.Cu, every wire run "
                    "round the sensor field) or 4 (rows leave on the inner layers -- In1 "
                    "carries each row to the connector, In2 the links across notches -- so "
                    "the field needs no wiring band for rows). Works in both board modes; "
                    "4 needs the hug router.")
    router: str = Field(
        "hug",
        description='"hug" (default: route wires hugging the sensor field) or '
                    '"legacy" (the column/row routers). "legacy" is used when '
                    'requested explicitly, as the guarded fallback for topologies '
                    'the hug plan cannot order planar, and to replay a design.json '
                    'that records no router. '
                    'Under "hug", rows/columns cut apart by a concavity stay one '
                    'net, bridged by a link trace around it. '
                    "Works in both board modes; in fixed_keepout the wiring "
                    "keep-out band is sized exactly from the hug routing plan.")
    preserve_sensors: bool = True
    auto_expand_board: bool = True
    routing_margin_mm: float = 3.0
    wiring_keepout_floor_mm: float | None = Field(
        None, ge=0.0,
        description="Fixed-outline mode: minimum wiring keep-out width (mm). "
                    "Omit to let the shape-following model decide.")
    follow_main_padding: bool = True
    follow_main_padding_mm: float = 2.0
    smooth_follow_padding: bool = True
    connector_pos_override: int | None = Field(
        None, description="Force a specific connector size instead of auto-selecting the smallest fit")
    connector_shoulder_fillet_mm: float = Field(
        1.0, ge=0.0,
        description="Fillet radius (mm) for the Edge.Cuts step created when a manually "
                    "enlarged connector is wider than its cable/tail. 0 disables.")
    target_cols: int | None = Field(
        None, ge=1,
        description="Target sensor columns. With target_rows, solves taxel size and pitch "
                    "from the drawn outline instead of using pixel_*_mm/pitch_*_mm. "
                    'Grow-board mode ("expand") only.')
    target_rows: int | None = Field(
        None, ge=1, description="Target sensor rows — see target_cols.")
    narrow_tail: bool = Field(
        True,
        description="After routing, cut the cable tail down to the traces it carries, "
                    "stepping out to the connector's width just before the connector. The "
                    "taxels and traces are the same either way. False: as wide as the "
                    "connector. A tail_width_mm or tail side padding wins over both, and a "
                    "cable shorter than about 5.5 mm keeps the connector's width.")
    contact_side: str = Field(
        "bottom",
        description='Which face of the tail carries the connector\'s gold fingers: "bottom" '
                    '(default) or "top" (the taxel side). The face that has to meet the connector\'s '
                    "contacts depends on the connector and on how the sensor is mounted relative to "
                    "it. The PI stiffener goes on the other face. The fingers are plain pads "
                    "on that face's copper only, with no drill; wires on another copper layer "
                    "cross to it through vias in the tail, before the stiffener. Changes no taxel "
                    "or route.")
    readout_board: str = Field(
        "auto",
        description='Which OpenTac readout board the sensor plugs into. "auto" uses the '
                    '30-pin board (up to 15 x 15 taxels) when the grid fits, else the 60-pin '
                    'board (up to 30 x 30). "30pin" or "60pin" force one; "any" allows every '
                    "connector size from 20 to 70 positions, most of which have no OpenTac "
                    "readout board. connector_pos_override, when set, wins.")
    design_name: str = Field(
        "",
        description="Name printed on the front silkscreen just inboard of the connector, "
                    "so a fabricated board identifies itself. Sanitised to [A-Za-z0-9_-] "
                    "and shrunk or truncated to fit; too long or too little room and the "
                    "label is dropped rather than run off the board. Empty = no label.")


class GenerateResponse(BaseModel):
    svg: str
    stats: dict
    drc: dict
    zip_b64: str
    edit_data: dict | None = None


class ExportEditedRequest(BaseModel):
    edit_data: dict
    routes: list[dict]


class ExportEditedResponse(BaseModel):
    svg: str
    stats: dict
    drc: dict
    zip_b64: str


class BumpSheetRequest(BaseModel):
    edit_data: dict
    params: dict | None = Field(
        None, description="Overrides for BumpParams fields; unknown keys are ignored")


class BumpSheetResponse(BaseModel):
    stl_b64: str
    stats: dict
    warnings: list[str]


class CaseRequest(BaseModel):
    edit_data: dict
    params: dict | None = Field(
        None, description="Overrides for CaseParams fields; unknown keys are ignored")
    bump_params: dict | None = Field(
        None, description="Overrides for BumpParams, used only when params.bumps is true")


class CaseResponse(BaseModel):
    bottom_stl_b64: str
    top_stl_b64: str
    stats: dict
    warnings: list[str]


class ZipWithCaseRequest(BaseModel):
    zip_b64: str = Field(..., description="The board ZIP from /generate or /export-edited")
    edit_data: dict
    params: dict | None = Field(
        None, description="Overrides for CaseParams fields, as for /case. bumps and hatch are "
                          "ignored: every lid style is built")
    bump_params: dict | None = Field(None, description="Overrides for BumpParams, for the bumped lid")
    primary: str = Field("bumps", description="The lid case/README.txt names as the one to "
                                              "print: bumps, flat or hatch")
    name: str = Field("", description="File stem for the case files, as the page names its downloads")


class ZipWithCaseResponse(BaseModel):
    zip_b64: str
    case_files: list[str] = Field(..., description="STL paths added under case/; empty if none")
    case_error: str | None = Field(
        None, description="Why the case, or one of its lids, was left out; null if all went in")
    warnings: list[str] = []


class RedoPaddingRequest(BaseModel):
    edit_data: dict
    routes: list[dict]
    params: dict | None = None
    connector_pos_override: int | None = Field(
        None, description="Swap to a different connector size on an already-routed board, without re-routing")


class RedoPaddingResponse(BaseModel):
    svg: str
    stats: dict
    drc: dict
    edit_data: dict


class DxfImportResponse(BaseModel):
    loops: list[list[list[float]]]
    areas: list[float]
    chosen_index: int
    warnings: list[str]


class OpenDesignRequest(BaseModel):
    """A previously exported manifest.json or design.json, already parsed."""
    file: dict
    prefer: str = Field(
        "restore",
        description='"restore" replays the exact board (design.json only); '
                    '"recipe" always re-generates from the stored inputs.')


class OpenDesignResponse(BaseModel):
    # "restore" — hand edit_data/routes to /export-edited and show that board
    #             back, unchanged, hand-dragged routes and all.
    # "recipe"  — put outline/params into the form and press Generate: the
    #             board is rebuilt from its inputs on the current pipeline.
    mode: str
    label: str
    outline: list[list[float]]
    # A /generate body fragment: every field but `outline` and `fill_region`.
    # Empty in "geometry" mode -- there is nothing to re-generate from.
    params: dict
    fill_region: list[list[float]] | None = None
    edit_data: dict | None = None
    routes: list[dict] | None = None
    # "geometry" mode only: the board drawn from its stored geometry. Supplied
    # here because there is no layout to run the normal preview through, and
    # no round trip through /export-edited to produce one.
    svg: str | None = None
    warnings: list[str] = []



class UnrollMeshResponse(BaseModel):
    outline: list[list[float]]
    region_face_count: int
    region_face_indices: list[int]
    developability_disagreement_mm: float
    warnings: list[str]
    # Populated only when this response used a straight-strut connection
    # (the two clicked regions don't touch on the mesh) — each region's own
    # boundary loop as [vertex_index, x, y, z] points, in original mesh
    # vertex indices, so a client can offer a "connect corners
    # manually" flow without a second round trip to re-fetch this data.
    strut_loop1: list[list[float]] | None = None
    strut_loop2: list[list[float]] | None = None
    # outline[i]'s original mesh vertex index, parallel to outline — only
    # populated once a strut connection has happened (empty otherwise, e.g.
    # a pure mesh-union selection, which can always be re-derived from
    # region_face_indices alone). Once non-empty, the caller echoes both
    # this and outline back as prior_outline/prior_outline_vertices on the
    # *next* shift-click, so a 3rd (or later) region can be strut-connected
    # onto this same running chain — a strut-joined outline is not a real
    # mesh region, so it cannot be re-unfolded from a face selection the
    # way a pure mesh-union chain can.
    outline_vertex_indices: list[int] = []


class GrowRegionResponse(BaseModel):
    region_face_indices: list[int]


class ChainConnection(BaseModel):
    needs_strut: bool
    # Each is [vertex_index, x, y, z, group_id] per boundary point, or None
    # when this connection didn't need a strut (the two regions touch).
    # loop_a is the running chain's boundary *as it stood before this
    # connection* (may span multiple earlier regions if they were joined by
    # touching, or multiple earlier *groups* if joined by an earlier strut —
    # group_id tags which original click each point ultimately traces back
    # to, so the frontend can color a multi-region chain by origin instead
    # of rendering it as one undifferentiated blob). loop_b is the newly
    # clicked region's own boundary, always a single fresh group.
    loop_a: list[list[float]] | None = None
    loop_b: list[list[float]] | None = None


class UnrollMeshChainResponse(BaseModel):
    outline: list[list[float]]
    region_face_count: int
    region_face_indices: list[int]
    warnings: list[str]
    # Parallel to the *gaps* between click_face_indices (length = N-1 for N
    # clicks) — connections[k] describes how click k connects to click k+1.
    connections: list[ChainConnection]
    # Parallel to click_face_indices: each click's own region's
    # developability disagreement (mm), measured before any joining. Near
    # zero = that surface flattens exactly; large = doubly curved
    # there, so the UI can point at the specific surface to cut flatter.
    region_developability_mm: list[float] = []
    # corners_first mode only: the 0-based gap index of the first strut
    # connection still waiting for user-picked corners — its marker loops
    # are the last entry in `connections`, and `outline` is empty. Null
    # once every cut is resolved (or in the default automatic mode).
    pending_connection: int | None = None




@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/import-dxf", response_model=DxfImportResponse)
async def import_dxf(file: UploadFile = File(...)):
    content = await file.read()
    if len(content) > MAX_DXF_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="DXF file too large (10MB limit).")

    try:
        result = parse_dxf_outline(content)
    except DxfImportError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return DxfImportResponse(
        loops=result.loops,
        areas=result.areas,
        chosen_index=result.chosen_index,
        warnings=result.warnings,
    )


@app.post("/import-svg", response_model=DxfImportResponse)
async def import_svg(file: UploadFile = File(...)):
    content = await file.read()
    if len(content) > MAX_SVG_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="SVG file too large (10MB limit).")

    try:
        result = parse_svg_outline(content)
    except SvgImportError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return DxfImportResponse(
        loops=result.loops,
        areas=result.areas,
        chosen_index=result.chosen_index,
        warnings=result.warnings,
    )


def _params_to_dict(params: Params) -> dict:
    """Params as plain JSON. Lives in export/design.py so that the manifest's
    design block and edit_data cannot drift apart -- they are the same dict."""
    return params_to_dict(params)


def _merge_param_overrides(edit_data: dict, overrides: dict | None) -> dict:
    if not overrides:
        return edit_data
    # copper_layers is a routing decision: changing it means re-routing, so
    # it is not something a re-export of existing routes can override
    allowed = (set(_params_to_dict(Params()).keys()) | {"contact_side"}) - {"copper_layers", "cable_length_auto"}
    clean = {k: v for k, v in overrides.items() if k in allowed}
    # A blank Tail length field is "automatic": the stored edit_data already
    # carries the length chosen for this board, and its routes end at it.
    if clean.get("cable_length_mm") is None:
        clean.pop("cable_length_mm", None)
    if not clean:
        return edit_data
    merged = dict(edit_data)
    merged_params = dict(merged.get("params", {}))
    merged_params.update(clean)
    merged["params"] = merged_params
    return merged


BOARD_MODES = ("expand", "fixed_keepout")

# Fixed-outline mode: how many times the keep-out may be widened and the grid
# re-packed in response to routing failures before the board is returned as-is
# with warnings.
MAX_ROUTE_RETRIES = 4


def _resolve_board_mode(board_mode: str | None, auto_expand_board: bool) -> str:
    """Resolve the requested board mode, accepting requests without board_mode.

    Requests and stored edit_data without board_mode carry only the
    auto_expand_board / preserve_sensors booleans; map those onto the two modes.
    """
    if board_mode is None:
        return "expand" if auto_expand_board else "fixed_keepout"
    if board_mode not in BOARD_MODES:
        raise ValueError(f"board_mode must be one of {list(BOARD_MODES)}, got {board_mode!r}.")
    return board_mode


def _params_for_board_mode(params: Params, board_mode: str) -> Params:
    """Derive the internal preserve_sensors/auto_expand_board flags from the mode.

    Neither flag is user-facing: expand mode preserves sensors and grows the
    outline; fixed mode preserves sensors and never grows it (the sensor grid
    is instead shrunk up front by the keep-out carve). The routers read
    preserve_sensors, and expand mode's trial probe flips it off
    deliberately — see the /generate expand branch.
    """
    return replace(
        params,
        board_mode=board_mode,
        preserve_sensors=True,
        auto_expand_board=(board_mode == "expand"),
    )


def _apply_target_grid(params: Params, outline_xy: np.ndarray, fill_xy, board_mode: str) -> Params:
    """Replace pixel size/pitch with values solved from a requested row/col count.

    A no-op unless the caller asked for both counts. Grow-board mode only: in
    fixed-outline mode the wiring keep-out band's width is itself a function of
    the row/column net counts (layout_fixed_shape_keepout's fixed point), so the
    packable region a taxel size would be solved against is not known until
    after the packing it determines — the two would have to be solved together.
    """
    if params.target_cols is None or params.target_rows is None:
        return params
    if board_mode != "expand":
        raise ValueError(
            "Target rows/columns are only supported in grow-board mode. "
            "Switch the board mode to expand, or set the taxel size directly.")
    solved = solve_grid_pixel_size(outline_xy, params, fill_xy)
    return replace(
        params,
        pixel_w_mm=solved["pixel_w_mm"],
        pixel_h_mm=solved["pixel_h_mm"],
        pitch_x_mm=solved["pitch_x_mm"],
        pitch_y_mm=solved["pitch_y_mm"],
    )


def _route_escapes(layout: Layout, routes: list[Route], params: Params) -> list[tuple[np.ndarray, float]]:
    """Points where routed copper leaves the board or crowds its edge.

    Fixed-outline mode cannot grow the board around a stray trace the way
    _apply_post_route_containment does for grow mode, so this is the signal that
    the keep-out has to give the wire more room *inside* instead. Returns
    (point, shortfall_mm) pairs, thinned to the worst point per locality.
    """
    board = layout.board_poly
    boundary = board.boundary
    required = params.trace_w_mm / 2 + params.board_edge_clear_mm
    found: list[tuple[np.ndarray, float]] = []

    for route in routes:
        P = np.asarray(route.polyline_xy, dtype=float)
        if len(P) < 2:
            continue
        line = LineString(P)

        # Exact geometry, not point sampling: drc_check measures segment-to-edge
        # distance continuously, so a sampled detector would pass boards that
        # drc_check then flags (a near-miss can fall between samples).
        outside = line.difference(board)
        if not outside.is_empty:
            for piece in getattr(outside, "geoms", [outside]):
                if piece.is_empty:
                    continue
                coords = np.asarray(piece.coords, dtype=float)
                dists = [boundary.distance(Point(c)) for c in coords]
                worst_i = int(np.argmax(dists))
                found.append((coords[worst_i], required + float(dists[worst_i])))
            continue

        d = float(line.distance(boundary))
        if d < required - 1e-9:
            p_line, _ = nearest_points(line, boundary)
            found.append((np.array([p_line.x, p_line.y]), required - d))

    # Thin to the worst point per ~4mm locality so one bad stretch does not
    # push the same part of the boundary out several times over.
    found.sort(key=lambda item: -item[1])
    kept: list[tuple[np.ndarray, float]] = []
    for pt, shortfall in found:
        if all(float(np.hypot(*(pt - other))) > 4.0 for other, _ in kept):
            kept.append((pt, shortfall))
    return kept


MAX_EDGE_PRUNE_ROUNDS = 4


def _prune_edge_escapes(layout: Layout, outline_xy: np.ndarray, params: Params,
                        col_routes: list[Route], row_routes: list[Route], hug: bool,
                        problem_pix=frozenset()):
    """Fixed-outline mode's final step for copper that leaves the board.

    The keep-out retry can only reserve room and re-pack; it cannot move a
    wire whose path does not depend on the packing — e.g. a fan wire that
    rises straight from its connector pad through a waist narrower than the
    bundle, and so runs outside the waist on every attempt. Whatever still
    escapes after it is dropped: the escaping net's wire goes, and so do all
    of that net's pixels, then the smaller field is re-routed — the same
    contract as a refused net in grow mode. Repeats until clean or out of
    rounds.

    Hug mode drops the pixels of nets the router reported failed
    (`problem_pix`, then each re-route's) the same way: when no attempt is
    clean, the best one would otherwise output nets known not to reach the
    connector, which KiCad reports as unconnected items.

    Returns (layout, col_routes, row_routes, dropped_escaping, dropped_failed,
    escapes, routing_problem_pix, incomplete).
    """
    dropped = dropped_failed = 0
    escapes = []
    routing_problem_pix: set = set(problem_pix)
    incomplete: list = []
    for _ in range(MAX_EDGE_PRUNE_ROUNDS + 1):
        bad_nets = {r.net_name for r in [*col_routes, *row_routes]
                    if _route_escapes(layout, [r], params)}
        failed = routing_problem_pix if hug else set()
        if not bad_nets and not failed:
            escapes = []
            break
        drop = {i for i, (c, r) in enumerate(zip(layout.col_net_per_pixel,
                                                  layout.row_net_per_pixel))
                if c in bad_nets or r in bad_nets}
        dropped_failed += len(failed - drop)
        dropped += len(drop)
        drop |= failed
        active = [i for i in range(len(layout.placements_xy)) if i not in drop]
        if not active:
            raise ValueError(_KEEPOUT_TOO_SMALL_MSG)
        sub = copy.deepcopy(layout)
        sub.placements_xy = layout.placements_xy[active]
        sub.col_net_per_pixel = [layout.col_net_per_pixel[i] for i in active]
        sub.row_net_per_pixel = [layout.row_net_per_pixel[i] for i in active]
        layout = sub
        if hug:
            col_routes, row_routes, _, refused = route_for(params)(layout, outline_xy, params)
            problem = set(refused)
        else:
            col_routes, col_problem = route_columns_grid(layout, outline_xy, params)
            row_routes, row_problem = route_rows_grid(layout, outline_xy, params)
            problem = set(col_problem) | set(row_problem)
        bad_pix, incomplete = validate_routes(
            layout, col_routes, row_routes, set(),
            outline_xy, params.pad_pitch_mm, params.trace_w_mm, params.clearance_mm)
        routing_problem_pix = problem | set(bad_pix)
        escapes = _route_escapes(layout, [*col_routes, *row_routes], params)
    return (layout, col_routes, row_routes, dropped, dropped_failed, escapes,
            routing_problem_pix, incomplete)


def _fixed_keepout_attempt_params(params: Params) -> list[Params]:
    """Keep-out floors to try in fixed-outline mode, densest first.

    The shape-following model in pipeline/keepout.py is what should decide the
    band width, so the first attempt gives it no floor at all and lets it ask
    for exactly what the wiring needs. That is markedly denser than a flat
    routing_margin_mm ring, with copper-to-edge clearance still several times
    the DRC minimum.

    It is not universally safe, though: on a board whose aspect ratio leaves the
    row router very little height to fan out in, packing to the model's own
    number can produce traces that cross. Rather than reserve for that
    everywhere, the dense attempt is *verified*, and a board that comes back
    with real defects falls back to the conservative routing_margin_mm floor.
    So the density is taken only where it demonstrably routes, and no board
    ends up worse than the conservative floor alone would make it.

    An explicit wiring_keepout_floor_mm from the caller is honoured as-is and
    disables the search.
    """
    if params.wiring_keepout_floor_mm is not None:
        return [params]
    return [
        replace(params, wiring_keepout_floor_mm=0.0),
        replace(params, wiring_keepout_floor_mm=max(0.0, float(params.routing_margin_mm))),
    ]


def _drc_crowding(col_routes: list[Route], row_routes: list[Route], params: Params):
    """Trace-to-trace DRC failures, as (point, shortfall_mm) pairs.

    Unrouted pixels and copper leaving Edge.Cuts are not the only failures in
    fixed-outline mode: a board whose traces merely *cross each other* passes
    both of those checks. Packing to what the shape-following model asks for
    makes the corridors as tight as the model allows, which is exactly when the
    row router starts running out of lanes, so the fixed-mode attempts have to
    see this signal too.

    Not fed back into _grow_keepout_for_failures: widening the band in response
    to a crossing re-packs the array, which re-routes everything and tends to
    move the congestion rather than relieve it. It counts instead toward each
    attempt's defect score in generate()'s fixed-outline branch, which is what
    selects the conservative keep-out floor over the dense one.
    """
    violations = drc_check(
        col_routes, row_routes, params.trace_w_mm, params.clearance_mm)
    found: list[tuple[np.ndarray, float]] = []
    for v in violations:
        mid = 0.5 * (np.asarray(v.seg1, dtype=float).mean(axis=0)
                     + np.asarray(v.seg2, dtype=float).mean(axis=0))
        found.append((mid, max(0.0, float(v.min_required - v.distance))))
    found.sort(key=lambda item: -item[1])
    kept: list[tuple[np.ndarray, float]] = []
    for pt, shortfall in found:
        if all(float(np.hypot(*(pt - other))) > 4.0 for other, _ in kept):
            kept.append((pt, shortfall))
    return kept


# what layout_fixed_hug_keepout records about the carve, carried onto a
# layout rebuilt from a subset of its taxels
_FIXED_CARVE_KEYS = ("board_mode", "fixed_board_routing", "fixed_keepout_min_mm",
                     "fixed_keepout_max_mm", "fixed_keepout_mean_mm",
                     "fixed_keepout_iterations", "fixed_keepout_converged", "max_pack_pixels")


def _drop_near_escapes(layout: Layout, outline_xy, params: Params, escapes, join_split_nets,
                       max_drop: int = 3):
    """Before the keep-out regrows: drop the taxel(s) nearest the escapes and
    re-route on the same grid.

    Regrowing the band means re-packing, and the re-pack is free to choose a
    whole new lattice: a near-miss then costs far more than the taxel next to
    it, since the re-packed grid can lose several taxels where dropping the one
    beside the escape loses only that one. Tries 1..max_drop taxels, nearest
    first; returns the first clean (layout, col_routes, row_routes, plan,
    incomplete), or None.
    The caller still runs the regrow and keeps whichever clean result holds
    more taxels, so this can only add."""
    P = np.asarray(layout.placements_xy, dtype=float)
    if len(P) <= max_drop + 1:
        return None
    pts = np.asarray([pt for pt, _ in escapes], dtype=float)
    d = np.min(np.hypot(P[:, None, 0] - pts[None, :, 0], P[:, None, 1] - pts[None, :, 1]), axis=1)
    order = np.argsort(d, kind="stable")
    for k in range(1, max_drop + 1):
        keep = np.sort(order[k:])
        try:
            lay = layout_with_connector(outline_xy, params, pack_anchor="top_center",
                                        join_split_nets=join_split_nets, placements_override=P[keep])
        except ValueError:
            return None
        lay.tail.update({key: layout.tail[key] for key in _FIXED_CARVE_KEYS if key in layout.tail})
        lay.tail["dropped_by_keepout"] = int(lay.tail.get("max_pack_pixels", len(P)) - len(keep))
        cols, rows, plan, failed = route_for(params)(lay, outline_xy, params)
        bad, incomplete = validate_routes(lay, cols, rows, set(), outline_xy, params.pad_pitch_mm,
                                          params.trace_w_mm, params.clearance_mm)
        if failed or bad or _route_escapes(lay, [*cols, *rows], params):
            continue
        return lay, cols, rows, plan, incomplete
    return None


def _grow_keepout_for_failures(profile, layout: Layout, problem_pix: set,
                               escapes: list[tuple[np.ndarray, float]], params: Params):
    """Widen the keep-out only where routing actually failed.

    An escaping wire names its own trouble spot, so the boundary near it is
    widened by however far it overran plus a trace pitch of headroom. A pixel
    the router could not wire names a spot too — the boundary nearest it. Widths
    elsewhere are untouched, which is the purpose of a shape-following
    keep-out: the top of the board should not pay for congestion at the tail.
    """
    step = params.trace_w_mm + params.clearance_mm
    changed = False

    if escapes:
        pts = [pt for pt, _ in escapes]
        amounts = [shortfall + step for _, shortfall in escapes]
        changed |= keepout_mod.grow_near(profile, pts, amounts)

    if problem_pix:
        P = np.asarray(layout.placements_xy, dtype=float)
        pts = [P[i] for i in sorted(problem_pix) if i < len(P)]
        if pts:
            changed |= keepout_mod.grow_near(profile, pts, [step] * len(pts))

    if not changed:
        # Nothing localizable — fall back to a uniform nudge so the retry is
        # not a no-op repeat of the same layout.
        profile.widths += step
        changed = True

    keepout_mod.smooth_widths(profile)
    return profile


def _without_expansion_padding(params: Params) -> Params:
    """Make auto-expand off mean no generated route/follow padding.

    board_edge_clear_mm is deliberately not zeroed here. It is a manufacturing
    clearance between copper and the board cut, not board-growth padding: the
    routers use it to steer away from the edge (route_columns.py /
    route_rows.py) and drc_check only tests copper-to-edge clearance when it is
    non-zero. Zeroing it in fixed-outline mode would let traces sit on — and run
    past — the Edge.Cuts line with a clean DRC report.
    """
    if params.auto_expand_board:
        return params
    return replace(
        params,
        follow_main_padding=False,
        follow_main_padding_mm=0.0,
        smooth_follow_padding=False,
    )


def _active_pad_split_center(layout: Layout, outline_xy: np.ndarray, params: Params) -> np.ndarray:
    row_pads = sorted(int(p) for p, net in layout.pad_to_net.items() if str(net).startswith("ROW_"))
    col_pads = sorted(int(p) for p, net in layout.pad_to_net.items() if str(net).startswith("COL_"))
    if row_pads and col_pads:
        split_pad = 0.5 * (max(row_pads) + min(col_pads))
    else:
        split_pad = (layout.connector_pos + 1) / 2.0
    x_mid = float(np.mean(np.asarray(outline_xy, dtype=float)[:2, 0]))
    pad_x_left = x_mid - (layout.connector_pos - 1) * params.pad_pitch_mm / 2
    return np.array([pad_x_left + (split_pad - 1) * params.pad_pitch_mm, layout.pad_y_mm], dtype=float)


def _route_to_dict(route: Route, layer: str) -> dict:
    return {
        "id": f"{layer}:{route.net_name}:{route.pad_num}",
        "layer": layer,
        "net_name": route.net_name,
        "pad_num": route.pad_num,
        "anchor_xy": np.asarray(route.anchor_xy, dtype=float).tolist(),
        "pad_xy": np.asarray(route.pad_xy, dtype=float).tolist(),
        "polyline_xy": np.asarray(route.polyline_xy, dtype=float).tolist(),
    }


def _build_edit_data(
    layout: Layout,
    outline_xy: np.ndarray,
    col_routes: list[Route],
    row_routes: list[Route],
    removed: set[int],
    params: Params,
    fill_xy=None,
) -> dict:
    sync_pads_to_feeds(layout, row_routes)
    minx, miny, maxx, maxy = layout.board_poly.bounds
    margin = 2.0
    cable_coords = []
    if layout.cable_poly and not layout.cable_poly.is_empty:
        cable_coords = [[float(x), float(y)] for x, y in layout.cable_poly.exterior.coords[:-1]]
    active_pad_split_center = _active_pad_split_center(layout, outline_xy, params)
    active_pad_center = active_pad_split_center.tolist()
    tail_ext_resolved = layout.tail.get("tail_extension_resolved", {}) or {}
    tail_ext_path = tail_ext_resolved.get("path") or []
    if tail_ext_resolved.get("enabled") and tail_ext_path:
        pad_center = np.asarray(tail_ext_path[0], dtype=float).tolist()
    else:
        existing_pad_center = layout.tail.get("pad_split_center")
        pad_center = existing_pad_center if existing_pad_center else active_pad_center
    layout.tail["pad_split_center"] = pad_center
    layout.tail["current_pad_split_center"] = active_pad_center
    tail_extension = dict(layout.tail.get("tail_extension", {}) or {})
    tail_extension_default_width = max(0.0, float(layout.tail.get("width_mm", 0.0)) - 0.7)
    if not tail_extension:
        tail_extension = {
            "enabled": False,
            "path": [pad_center],
            "width_mm": tail_extension_default_width,
            "clearance_mm": float(params.trace_w_mm / 2 + params.board_edge_clear_mm),
        }

    return {
        # Bumped only when a key changes meaning; see export/design.py.
        "version": 1,
        "schema": DESIGN_SCHEMA,
        "pipeline": PIPELINE_VERSION,
        "outline": np.asarray(outline_xy, dtype=float).tolist(),
        # The sensorize zone, in the same rotated frame as "outline". Held
        # separately from "sensor_outline", which layout.py sets to the
        # outline rather than to the fill region -- so without this key a
        # fill region would be lost by every round trip through edit_data
        # (redo-padding, a connector swap, an edited export).
        "fill_region": (np.asarray(fill_xy, dtype=float).tolist()
                        if fill_xy is not None and len(np.asarray(fill_xy)) >= 3 else None),
        "sensor_outline": np.asarray(layout.sensor_outline_xy if layout.sensor_outline_xy is not None else outline_xy, dtype=float).tolist(),
        "final_outline": np.asarray(layout.final_outline_xy if layout.final_outline_xy is not None else outline_xy, dtype=float).tolist(),
        "params": _params_to_dict(params),
        "removed_pix": sorted(int(i) for i in removed),
        "preview": {
            "xmin": minx - margin,
            "ymin": miny - margin,
            "xmax": maxx + margin,
            "ymax": maxy + margin,
        },
        "layout": {
            "placements_xy": np.asarray(layout.placements_xy, dtype=float).tolist(),
            "row_net_per_pixel": list(layout.row_net_per_pixel),
            "col_net_per_pixel": list(layout.col_net_per_pixel),
            "connector_pos": layout.connector_pos,
            "connector_swap": layout.connector_swap,
            "pad_to_net": {str(k): v for k, v in layout.pad_to_net.items()},
            "pad_y_mm": layout.pad_y_mm,
            "cable_poly": cable_coords,
            "routing_margin_applied_mm": layout.routing_margin_applied_mm,
            "tail": layout.tail,
            "tail_extension": tail_extension,
            "routing": layout.routing,
        },
        "routes": [
            *[_route_to_dict(r, "F.Cu") for r in col_routes],
            *[_route_to_dict(r, r.layer or "B.Cu") for r in row_routes],
        ],
    }


def _layout_from_edit_data(edit_data: dict) -> tuple[Layout, np.ndarray, Params, set[int]]:
    outline_xy = np.array(edit_data["outline"], dtype=float)
    final_outline_xy = np.array(edit_data.get("final_outline", edit_data["outline"]), dtype=float)
    stored_params = drop_removed_params(dict(edit_data["params"]))
    # edit_data without board_mode carries only the booleans.
    stored_params["board_mode"] = _resolve_board_mode(
        stored_params.get("board_mode"), bool(stored_params.get("auto_expand_board", True)))
    params = Params(**stored_params)
    layout_data = edit_data["layout"]
    tail_data = dict(layout_data.get("tail", {}))

    outline_poly = Polygon(outline_xy)
    pre_follow_xy = tail_data.get("pre_follow_final_outline")
    if pre_follow_xy:
        final_outline_xy = np.array(pre_follow_xy, dtype=float)
    final_outline_poly = Polygon(final_outline_xy)
    cable_coords = layout_data.get("cable_poly") or []
    cable_poly = Polygon(cable_coords) if len(cable_coords) >= 3 else None
    routing_margin_applied = float(layout_data.get("routing_margin_applied_mm", 0.0))
    if routing_margin_applied > 0 and final_outline_poly.is_valid and not final_outline_poly.is_empty:
        board_poly = final_outline_poly
    else:
        # 1um inflation before the union — edge-contact unions can leave a
        # phantom zero-width seam in the exterior ring; see the comment at
        # layout.py's base_board_poly construction for the failure it causes.
        # A narrow tail whose connector moved to a drawn extension leaves its
        # landing behind (follow_padding.cable_without_landing).
        board_cable = cable_poly
        if cable_poly and (layout_data.get("tail_extension") or {}).get("enabled"):
            board_cable = cable_without_landing(cable_poly, outline_xy, tail_data)
        board_poly = (unary_union([outline_poly, board_cable.buffer(0.001, join_style=2)])
                      if cable_poly else outline_poly)
        # ...and the shoulder fill that construction adds beside a tail wider
        # than the cable edge. Without it, every edited export of a fixed board
        # with a wide connector would put the fan wires there outside the board.
        if cable_poly and tail_data.get("width_mm") and tail_data.get("t_hat") and tail_data.get("n_out"):
            strip_mid = (0.5 * (outline_xy[0] + outline_xy[1])
                         + tail_offset_mm(tail_data) * np.asarray(tail_data["t_hat"], dtype=float))
            board_poly = unary_union([board_poly, *cable_shoulder_fill(
                outline_poly, strip_mid,
                tail_data["t_hat"], tail_data["n_out"], float(tail_data["width_mm"]),
                depth=tail_data.get("shoulder_depth_mm"))])

    layout = Layout(
        placements_xy=np.array(layout_data["placements_xy"], dtype=float),
        row_net_per_pixel=list(layout_data["row_net_per_pixel"]),
        col_net_per_pixel=list(layout_data["col_net_per_pixel"]),
        connector_pos=int(layout_data["connector_pos"]),
        connector_swap=bool(layout_data.get("connector_swap", False)),
        pad_to_net={int(k): v for k, v in layout_data["pad_to_net"].items()},
        pad_y_mm=float(layout_data["pad_y_mm"]),
        outline_poly=outline_poly,
        cable_poly=cable_poly,
        board_poly=board_poly,
        sensor_outline_xy=np.array(edit_data.get("sensor_outline", edit_data["outline"]), dtype=float),
        final_outline_xy=final_outline_xy,
        routing_margin_applied_mm=routing_margin_applied,
        tail=tail_data,
        routing=dict(layout_data.get("routing", {})),
    )
    if layout_data.get("tail_extension"):
        layout.tail["tail_extension"] = dict(layout_data.get("tail_extension") or {})
    removed = set(int(i) for i in edit_data.get("removed_pix", []))
    return layout, outline_xy, params, removed


def _resize_connector_pads(layout: Layout, new_conn_pos: int, routes: list[Route]) -> bool:
    """Swap layout.connector_pos to a different size on an already-routed
    board, without re-packing or re-routing.

    Physical pad *positions* for already-used pads are unchanged by
    connector size (only pad *numbers* shift — see resize_pad_to_net_map's
    docstring). But each Route object carries its own pad_num, assigned at
    the routing pass, and apply_tail_extension (for boards with a
    custom tail path) uses pad_num to compute each route's lane offset
    relative to the connector's center. If pad_num is left stale after a
    resize, that offset is computed against the *new* center using the
    *old* pad number — not the same net-to-pad relationship as before — so
    every route's lane silently shifts. Reassigning each route's pad_num to
    match its net's new pad restores that relationship exactly, which is
    what keeps routes physically unmoved (mathematically the same offset
    as pre-resize). This is not a routing change but a bookkeeping
    correction, so that no *accidental* routing change happens.

    Returns True if a resize actually happened. Closing the board outline
    around the new, wider connector body is a separate, later step — see
    _close_connector_shoulders — since apply_main_follow_padding
    unconditionally rebuilds layout.board_poly from scratch afterward and
    would otherwise discard any shape edit made here.
    """
    if int(new_conn_pos) == int(layout.connector_pos):
        return False
    new_pad_to_net = resize_pad_to_net_map(
        layout.connector_pos, new_conn_pos, layout.connector_swap, layout.pad_to_net)
    net_to_new_pad = {net: pad for pad, net in new_pad_to_net.items()}
    for route in routes:
        new_pad_num = net_to_new_pad.get(route.net_name)
        if new_pad_num is not None:
            route.pad_num = new_pad_num
    layout.pad_to_net = new_pad_to_net
    layout.connector_pos = int(new_conn_pos)
    # Remember that this board's connector was manually enlarged. layout.tail is
    # serialized into edit_data, so every later request can tell that the board
    # still needs its connector shoulder even though the size does not
    # change in that request. See _apply_connector_shoulder_if_needed.
    layout.tail["connector_pos_override"] = int(new_conn_pos)

    # apply_main_follow_padding caches its "main shell + tail pieces" union
    # in layout.tail["tail_geometry_lock"] so repeated /redo-padding calls
    # with the same connector are idempotent — but that means it would
    # silently keep reusing the *old* (narrower) connector envelope after a
    # resize. Drop the lock so it gets rebuilt fresh.
    layout.tail.pop("tail_geometry_lock", None)
    return True


def _close_connector_shoulders(layout: Layout, outline_xy: np.ndarray, params: Params, new_conn_pos: int) -> None:
    """Close the board's Edge.Cuts around a physically wider connector body,
    after every other outline-shaping pass (tail extension, post-route
    containment, follow-padding shrink-wrap) has already produced the final
    layout.board_poly — those passes are connector-width-agnostic and would
    otherwise leave the wider connector's body sticking out past the cut
    outline. Deliberately does not widen the cable/tail rectangle itself
    (that would override a user-chosen, purposely slim cable/tail width) —
    instead unions in a single rectangle covering just the connector-body
    zone (not the cable/tail run leading up to it) at the new connector's
    full width, overlapping 1mm into the existing board so the union always
    connects cleanly. The two new side edges where this rectangle is wider
    than the existing cable/corridor are the only visible change: a clean
    step out to the new connector's shoulder, right at the connector.
    """
    dims = connector_dimensions(new_conn_pos, params.pad_pitch_mm)
    overlap = 1.0
    ext = tail_extension_from_layout(layout)

    # Use the connector's *real* open-edge width (read straight from its
    # .kicad_mod file) rather than the CONNECTOR_BODY_ABOVE/BELOW_MM formula
    # below, which doesn't exactly match every real footprint — a mismatch
    # there leaves stitched_outline_segments (export/kicad_pcb.py,
    # which connects the board outline to this same real open edge when
    # emitting the physical connector footprint) bridging a gap with a
    # stray diagonal instead of a clean step.
    open_pts = connector_open_edge_endpoints(layout, outline_xy, params, CONNECTOR_DIR) if ext else None
    if open_pts:
        p_a = np.asarray(open_pts[0], dtype=float)
        p_b = np.asarray(open_pts[1], dtype=float)
        n_out = np.asarray(ext["n_out"], dtype=float)
        t_hat = (p_b - p_a) / np.linalg.norm(p_b - p_a)
        half_w = float(np.linalg.norm(p_b - p_a)) / 2
        pad_row_center = np.asarray(ext["pad_row_center"], dtype=float)
        edge_mid = 0.5 * (p_a + p_b)
        # Depth must safely cover the *whole* connector-body zone at this
        # width: back past the drawn tail path's own end point (pad_row_center
        # — the corridor's flat cap sits exactly there) on the near side,
        # and past the connector's physical tip (CONNECTOR_BODY_BELOW_MM
        # beyond pad_row_center) on the far side. Stopping exactly at the
        # open edge on the far side leaves a sliver of narrow, un-widened
        # corridor between it and the tip (the corridor's cap point is
        # slightly *past* the open edge, not before it), producing a
        # zigzag; stopping at a fixed offset from pad_row_center alone on
        # the near side can coincidentally land exactly on the open edge
        # for some connector sizes, producing a zero-area degenerate
        # rectangle that union silently drops. Overshooting generously on
        # both ends avoids all of that — stitched_outline_segments's own
        # fuzzy matching (span length + rough position) finds and connects
        # to the resulting long, unambiguous straight edge regardless of
        # exactly where within it the true open edge falls.
        gap_to_pad_row = float(np.dot(pad_row_center - edge_mid, n_out))
        near_dist = gap_to_pad_row + overlap
        far_dist = max(0.0, -gap_to_pad_row) + CONNECTOR_BODY_BELOW_MM
        A = edge_mid - half_w * t_hat - near_dist * n_out
        B = edge_mid + half_w * t_hat - near_dist * n_out
        C = edge_mid + half_w * t_hat + far_dist * n_out
        D = edge_mid - half_w * t_hat + far_dist * n_out
        shoulder = Polygon([A, B, C, D])
    else:
        # Exactly the new tab's width, so the board's edge runs straight into
        # the footprint's open endpoints. Padded past it (e.g. by
        # trace_w/2 + board_edge_clear_mm), the shoulder would run on beside
        # the tab and its side lines would be cut as slits. The wires keep the
        # pads they were routed to, one pad pitch or more inside the narrower
        # tab they were routed for.
        half_w = dims["tab_tip_width_mm"] / 2
        if ext:
            center = np.asarray(ext["pad_row_center"], dtype=float)
            t_hat = np.asarray(ext["t_hat"], dtype=float)
            n_out = np.asarray(ext["n_out"], dtype=float)
            y_start, y_end = -overlap, CONNECTOR_BODY_BELOW_MM
        else:
            p1, p2 = outline_xy[0], outline_xy[1]
            tail = tail_geometry(layout.outline_poly, p1, p2, layout.connector_pos, params,
                                 fit_width=(layout.tail or {}).get("fit_width_mm"),
                                 fit_offset=(layout.tail or {}).get("fit_offset_mm"))
            center, t_hat, n_out = tail["mid"], tail["t_hat"], tail["n_out"]
            y_start = max(0.0, tail["cable_length"] - overlap)
            y_end = tail["total_length"]
        A = center - half_w * t_hat + y_start * n_out
        B = center + half_w * t_hat + y_start * n_out
        C = center + half_w * t_hat + y_end * n_out
        D = center - half_w * t_hat + y_end * n_out
        shoulder = Polygon([A, B, C, D])

    # Soften the step this rectangle cuts into Edge.Cuts. Two separate corner
    # families need two different operations: the shoulder's own outward tip
    # corners are convex (rounded by shrinking the piece before the union), and
    # the two inward notches where the narrow cable/corridor meets the wide
    # shoulder are concave (rounded by adding a small wedge of material after
    # it). Both are expressed as union pieces so tail_geometry_lock can replay
    # them on later /redo-padding and export round trips.
    fillet_r = max(0.0, float(getattr(params, "connector_shoulder_fillet_mm", 0.0) or 0.0))
    if fillet_r > 0:
        shoulder = round_convex_corners(shoulder, fillet_r)
    pieces = [shoulder]

    layout.board_poly = unary_union([layout.board_poly, shoulder])
    if not layout.board_poly.is_valid:
        layout.board_poly = layout.board_poly.buffer(0)

    if fillet_r > 0:
        # Limit the concave fillets to the step's own neighbourhood so unrelated
        # inside corners elsewhere on the board keep the shape the user drew.
        infill = concave_fillet_infill(
            layout.board_poly, fillet_r, zone=shoulder.buffer(3 * fillet_r))
        # Only its area: where a tail extension meets the shoulder at an
        # angle the infill can come back with a bare line in it, which belongs
        # neither in the board nor in the replayed lock below (the next
        # export would fail on that line).
        areas = [g for g in getattr(infill, "geoms", [infill])
                 if g.geom_type == "Polygon" and not g.is_empty]
        if areas:
            # A wedge shares its edge with the cable it rounds, and a union of
            # shapes that only touch along an edge can keep a zero-width seam
            # there: on fixed outlines a 1 mm slit between the cable and the
            # fillet. Closing by 10 nm merges it (as fillet_inside_corners does).
            eps = 1e-5
            layout.board_poly = unary_union([layout.board_poly, *areas]).buffer(
                eps, join_style=2).buffer(-eps, join_style=2)
            if not layout.board_poly.is_valid:
                layout.board_poly = layout.board_poly.buffer(0)
            pieces.extend(areas)

    final_poly = _largest_polygon(layout.board_poly)
    if not final_poly.is_empty:
        layout.final_outline_xy = remove_collinear_closed(np.array(final_poly.exterior.coords[:-1], dtype=float))
        # _capture_pre_follow_outline (called earlier this request, before
        # apply_main_follow_padding) already snapshotted the *pre-shoulder*
        # outline into layout.tail["pre_follow_final_outline"] as the stable
        # baseline every future /redo-padding or /export-edited call resumes
        # from (so follow-padding's shrink-wrap doesn't compound across
        # repeated calls). Left alone, that snapshot would silently discard
        # this shoulder fix on the very next request — including the ZIP/
        # KiCad export, which is a fresh round trip through edit_data. Re-
        # capture it now so the shoulder becomes part of the baseline too.
        layout.tail["pre_follow_final_outline"] = layout.final_outline_xy.tolist()

    # apply_main_follow_padding (called earlier this request, before this
    # shoulder fix runs) also caches the exact set of "tail" polygons it
    # unioned into the board shape, in layout.tail["tail_geometry_lock"] —
    # a *separate* idempotency mechanism from pre_follow_final_outline
    # above. Every later request (another /redo-padding, or the ZIP/KiCad
    # export via /export-edited) reuses that cached piece list as-is,
    # regardless of pre_follow_final_outline, so leaving it out of date
    # would silently discard this shoulder on the very next request. Append
    # the shoulder into it so it's included from then on too.
    lock = layout.tail.get("tail_geometry_lock")
    if isinstance(lock, dict) and isinstance(lock.get("polygons"), list):
        for piece in pieces:
            if piece.is_empty:
                continue
            lock["polygons"].append([[float(x), float(y)] for x, y in piece.exterior.coords[:-1]])

    layout.tail["connector_min_width_mm"] = dims["tab_tip_width_mm"]
    layout.tail.update(dims)


def _apply_connector_shoulder_if_needed(layout: Layout, outline_xy: np.ndarray,
                                        params: Params, override: int | None = None) -> None:
    """Re-close the board around a manually enlarged connector, every request.

    Runs whenever the board carries an enlarged connector, not only when the
    current request changes its size. A later request that keeps the size but
    changes the tail (e.g. adds a tail extension) still needs the shoulder:
    _resize_connector_pads returns False there, and apply_main_follow_padding
    reuses its cached tail geometry only when the tail-extension state matches
    what was cached (see _locked_tail_pieces), so without this call the
    shoulder is lost. The tail then stays cable-width all the way to the tip
    with no landing for the connector body, and the exporter bridges the
    narrow board edge to the connector's much wider open edge with two long
    diagonals.

    Must run after every other outline-shaping pass, for the reason given in
    _close_connector_shoulders' own docstring.
    """
    conn = override if override is not None else layout.tail.get("connector_pos_override")
    if conn is None:
        return
    _close_connector_shoulders(layout, outline_xy, params, int(conn))


def _capture_pre_follow_outline(layout: Layout) -> None:
    if layout.final_outline_xy is not None:
        layout.tail["pre_follow_final_outline"] = np.asarray(layout.final_outline_xy, dtype=float).tolist()


def _finish_board(layout: Layout, col_routes: list[Route], row_routes: list[Route], params: Params) -> None:
    """The routed board's final shape: the tail cut to its wires, the safety net
    round every wire, then the board shrink-wrapped to the copper, never wider
    than the connector's tab past its opening."""
    narrow_routed_tail(layout, [*col_routes, *row_routes], params)
    _apply_post_route_containment(layout, col_routes, row_routes, params)
    _capture_pre_follow_outline(layout)
    apply_main_follow_padding(layout, [*col_routes, *row_routes], params)
    _clip_board_to_connector_tab(layout, np.asarray(layout.sensor_outline_xy, dtype=float), params)


def _auto_tail_length(layout: Layout, outline_xy: np.ndarray, col_routes: list[Route],
                      row_routes: list[Route], params: Params):
    """Finish a board routed with the minimum tail at the tail length it needs.

    See pipeline/tail_length.py. The narrow tail's own need is known from the
    pads before anything is built; the printed case's is measured on the
    finished outline with the case builder's own 2D check, and when it asks for
    more the tail is stretched further from the routed board and finished again
    (narrowing and padding only, a fraction of a routing pass). Returns the
    finished layout, routes and params, cable_length_mm set to the chosen
    length."""
    base = AUTO_TAIL_MIN_MM
    fit = fit_tail_length(layout, params)
    length = max(base, round_up_length(fit))
    reason = REASON_FIT if length > base else REASON_MIN
    cable_y = float(np.min(np.asarray(outline_xy, dtype=float)[:2, 1]))
    routed = (layout, col_routes, row_routes)
    for attempt in range(4):
        lay, cr, rr = copy.deepcopy(routed)
        p = replace(params, cable_length_mm=length)
        stretch_routed_tail(lay, [*cr, *rr], p, length - base)
        _finish_board(lay, cr, rr, p)
        check = case_cable_check(lay.final_outline_xy, cable_y, CaseParams())
        # A tail that never holds one width for the case to grip gets one from a
        # longer straight run; one as wide as the board cannot be helped.
        more = check["short_mm"] or (NECK_STABLE_MM if check["no_tail"] == "unsteady" else 0.0)
        if more <= 0 or attempt == 3:
            break
        length = round_up_length(length + more)
        reason = REASON_CASE
    lay.tail["length_auto"] = True
    lay.tail["length_reason"] = reason
    lay.tail["length_case_warnings"] = check["warnings"]
    return lay, cr, rr, p


def _relength_routed_tail(layout: Layout, col_routes: list[Route], row_routes: list[Route],
                          params: Params, old_length: float) -> None:
    """Redo padding with a Tail length typed that is not the board's own: the
    finished board made again at that length from its routes, without routing
    them again, the way _auto_tail_length makes it.

    Longer: the connector moves out and every pad wire gets a straight run into
    its moved pad (stretch_routed_tail). Shorter: the reverse, as far as every
    pad wire's own straight run into its pad allows (tail_trim_room); further
    than that, and with a drawn tail extension (its path starts at the old pad
    row), the board has to be routed again, and a ValueError says so."""
    routes = [*col_routes, *row_routes]
    new = float(params.cable_length_mm)
    delta = new - float(old_length)
    tail = layout.tail
    if (tail.get("tail_extension") or {}).get("enabled"):
        raise ValueError(
            f"The tail length cannot change to {new:g} mm under a drawn tail extension. Remove the "
            "extension first, or press Generate to make the board with the new length.")
    if delta < 0:
        room = tail_trim_room(layout, routes)
        if -delta > room + 1e-9:
            shortest = round_up_length(float(old_length) - room)
            hint = (f"It can be shortened to {shortest:g} mm without that. "
                    if shortest < float(old_length) - 1e-9 else "")
            raise ValueError(
                f"A {new:g} mm tail needs the wires into the connector routed again: their straight "
                f"runs into the pads are too short to cut. {hint}Press Generate to make the board "
                f"with a {new:g} mm tail.")
    # The tail back as it was routed (connector width; _finish_board narrows
    # it again), its old finished pieces dropped.
    if tail.get("mode") == "fit":
        tail.update({"mode": "auto", "width_mm": float(tail.get("routed_width_mm")
                                                      or tail.get("connector_min_width_mm")),
                     "fit_width_mm": None, "fit_offset_mm": None})
        for key in ("routed_width_mm", "shoulder_depth_mm", "step_fillet_mm", "landing_cover"):
            tail.pop(key, None)
    for key in ("tail_geometry_lock", "pre_follow_final_outline", "pad_split_center"):
        tail.pop(key, None)
    apply_tail_extension(layout, routes, params, CONNECTOR_DIR)      # none: records that
    if delta > 0:
        stretch_routed_tail(layout, routes, params, delta)
    else:
        trim_routed_tail(layout, routes, params, -delta)
    _finish_board(layout, col_routes, row_routes, params)
    tail["length_auto"] = False
    tail["length_reason"] = REASON_TYPED


def _routes_from_payload(routes_data: list[dict]) -> tuple[list[Route], list[Route]]:
    col_routes = []
    row_routes = []
    for item in routes_data:
        route = Route(
            net_name=item["net_name"],
            pad_num=int(item["pad_num"]),
            anchor_xy=np.array(item["anchor_xy"], dtype=float),
            pad_xy=np.array(item["pad_xy"], dtype=float),
            polyline_xy=np.array(item["polyline_xy"], dtype=float),
        )
        if item["layer"] == "F.Cu":
            col_routes.append(route)
        elif item["layer"] == "B.Cu":
            row_routes.append(route)
        elif item["layer"] in INNER_LAYERS:
            route.layer = item["layer"]
            row_routes.append(route)
        else:
            raise ValueError(f"Unsupported route layer: {item['layer']}")
    return col_routes, row_routes


def _largest_polygon(poly) -> Polygon:
    if isinstance(poly, Polygon):
        return poly
    if hasattr(poly, "geoms"):
        polys = [g for g in poly.geoms if isinstance(g, Polygon) and not g.is_empty]
        if polys:
            return max(polys, key=lambda g: g.area)
    return Polygon()



def _apply_post_route_containment(
    layout: Layout,
    col_routes: list[Route],
    row_routes: list[Route],
    params: Params,
) -> None:
    """Expand final board geometry to contain routed trace centerlines safely."""
    if not params.preserve_sensors or not params.auto_expand_board:
        layout.tail["post_route_containment_expansion"] = False
        layout.tail["max_route_outside_distance_mm"] = 0.0
        layout.tail["route_clearance_shortfall_mm"] = 0.0
        return

    original_board = layout.board_poly
    containment = route_containment_mm(params)
    # Wire the tail carries is held by the tail at the tail's own clearance;
    # padding it here widened the tail past the connector's tab.
    hold = tail_wire_hold(layout)
    additions = []
    max_outside = 0.0
    max_shortfall = 0.0

    for route in [*col_routes, *row_routes]:
        P = np.asarray(route.polyline_xy, dtype=float)
        if len(P) < 2:
            continue
        line = LineString(P)
        if line.is_empty:
            continue

        free = line if hold is None else line.difference(hold)
        if free.is_empty:
            continue
        buffered = free.buffer(containment, cap_style=2, join_style=2)
        if not buffered.is_empty and buffered.difference(original_board).area > 1e-9:
            additions.append(buffered)

        for s in range(len(P) - 1):
            seg_len = float(np.linalg.norm(P[s + 1] - P[s]))
            n_samp = max(3, int(np.ceil(seg_len / 0.5)))
            for qi in range(n_samp + 1):
                t = qi / n_samp
                pt = P[s] + t * (P[s + 1] - P[s])
                pt_obj = Point(float(pt[0]), float(pt[1]))
                if original_board.contains(pt_obj):
                    d = original_board.boundary.distance(pt_obj)
                    if d < containment:
                        max_shortfall = max(max_shortfall, containment - d)
                else:
                    max_outside = max(max_outside, original_board.boundary.distance(pt_obj))

    if additions:
        expanded = unary_union([original_board, *additions])
        layout.board_poly = expanded
        final_poly = _largest_polygon(expanded)
        if not final_poly.is_empty:
            layout.final_outline_xy = remove_collinear_closed(np.array(final_poly.exterior.coords[:-1], dtype=float))
        layout.tail["post_route_containment_expansion"] = True
        layout.tail["post_route_containment_buffer_mm"] = containment
    else:
        layout.tail["post_route_containment_expansion"] = False

    layout.tail["max_route_outside_distance_mm"] = float(max_outside)
    layout.tail["route_clearance_shortfall_mm"] = float(max_shortfall)


def _manufacturing_files_from_kicad(pcb_text: str, copper_layers: int = 2) -> tuple[dict[str, str] | None, str | None, str]:
    """Plot Gerber/drill files from the generated KiCad board when available.

    This keeps manufacturing output aligned with the .kicad_pcb file, especially
    for connector footprints whose native pad and Edge.Cuts behavior is best
    handled by KiCad itself. Hosted deployments without KiCad still use the
    direct Python Gerber fallback.
    """
    kicad_cli = shutil.which("kicad-cli")
    if not kicad_cli:
        return None, None, "direct"

    expected = {
        "board-F_Cu.gtl",
        "board-B_Cu.gbl",
        "board-Edge_Cuts.gm1",
        "board-F_Mask.gts",
        "board-B_Mask.gbs",
    }
    optional = {
        "board-User_Drawings.gbr",
        "board-User_Comments.gbr",
        "board-F_Silkscreen.gto",
        "board-B_Silkscreen.gbo",
    }
    base_layers = "F.Cu,B.Cu,F.Mask,B.Mask,Edge.Cuts,Dwgs.User,Cmts.User"
    if copper_layers == 4:
        expected |= {"board-In1_Cu.g1", "board-In2_Cu.g2"}
        base_layers = "F.Cu,In1.Cu,In2.Cu,B.Cu,F.Mask,B.Mask,Edge.Cuts,Dwgs.User,Cmts.User"
    try:
        with tempfile.TemporaryDirectory(prefix="otc_kicad_plot_") as tmp:
            pcb_path = os.path.join(tmp, "board.kicad_pcb")
            out_dir = os.path.join(tmp, "gerber")
            with open(pcb_path, "w") as f:
                f.write(pcb_text)

            def plot(layers):
                subprocess.run(
                    [
                        kicad_cli, "pcb", "export", "gerbers",
                        "-o", out_dir,
                        "-l", layers,
                        pcb_path,
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            # F.Silkscreen carries the design-name label; B.Silkscreen
            # carries the taxel-core QR and the registration grid. Both are
            # asked for separately because the board file uses the
            # pre-KiCad-6 layer names (F.SilkS / B.SilkS) and an older
            # kicad-cli may not accept the current ones, and the whole Gerber
            # set must not be lost over a legend. A refused layer costs the
            # marks, not the board: the retry plots the base layers only.
            try:
                plot(base_layers + ",F.Silkscreen,B.Silkscreen")
            except subprocess.CalledProcessError:
                plot(base_layers)
            subprocess.run(
                [
                    kicad_cli, "pcb", "export", "drill",
                    "-o", out_dir,
                    "--format", "excellon",
                    "--excellon-units", "mm",
                    pcb_path,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            gerbers = {}
            for fname in expected:
                path = os.path.join(out_dir, fname)
                if not os.path.exists(path):
                    return None, None, "direct"
                with open(path, "r", errors="replace") as f:
                    gerbers[fname] = f.read()
            for fname in optional:
                path = os.path.join(out_dir, fname)
                if os.path.exists(path):
                    with open(path, "r", errors="replace") as f:
                        gerbers[fname] = f.read()

            drill_path = os.path.join(out_dir, "board.drl")
            if not os.path.exists(drill_path):
                return None, None, "direct"
            with open(drill_path, "r", errors="replace") as f:
                drill_text = f.read()
            return gerbers, drill_text, "kicad-cli"
    except Exception:
        return None, None, "direct"


def _drop_connector_zone_edge_flags(violations, layout, outline_xy, params, connector_dir):
    """Drop board-edge DRC flags inside the connector landing zone — they are
    not real defects: the copper there is running onto the connector pads, and
    the 'edge' it is measured against is the tail cap that Edge.Cuts stitching
    removes from the exported outline (the connector body occupies that area).
    The zone is the connector's own placed footprint — pads plus its Edge.Cuts
    outline — buffered 0.6mm: the flags land within the copper-to-edge check
    distance (0.3mm) of the removed cap, which can sit outside the pad hull
    alone (on a long custom tail). Trace-to-trace violations are never
    touched."""
    if not connector_dir:
        return violations
    if not any(v.net2 == "BOARD_EDGE" for v in violations):
        return violations
    pads = connector_pad_geometries(layout, outline_xy, params, connector_dir)
    if not pads:
        return violations
    geoms = [Polygon(p["corners"]) for p in pads]
    edges = connector_edge_geometries(layout, outline_xy, params, connector_dir)
    for a, b in edges["lines"]:
        geoms.append(LineString([tuple(a), tuple(b)]))
    for a, m, b in edges["arcs"]:
        geoms.append(LineString([tuple(a), tuple(m), tuple(b)]))
    zone = unary_union(geoms).convex_hull.buffer(0.6)
    kept = []
    for v in violations:
        if v.net2 == "BOARD_EDGE":
            mid = (np.asarray(v.seg2[0], dtype=float) + np.asarray(v.seg2[1], dtype=float)) / 2
            if zone.covers(Point(mid[0], mid[1])):
                continue
        kept.append(v)
    return kept


# Length of the synthetic tail extension used when the user has not drawn one.
# Kept short: the base tail already reserves the connector body space; this
# only exists so the export goes through the tail-extension machinery, which
# moves the pads to the corridor end, extends every pad-start track onto the
# pad drills, and stitches Edge.Cuts to the connector's open edge exactly.
AUTO_TAIL_EXTENSION_MM = 1.0


def _clip_board_to_connector_tab(layout: Layout, outline_xy: np.ndarray, params: Params) -> bool:
    """Take away board beside the connector's tab past its opening
    (connector_geometry.clip_beside_tab), placed where the connector is
    exported: at the end of a drawn tail extension, else AUTO_TAIL_EXTENSION_MM
    further out than the tail's own pose (_outputs_with_connector_tail). Run
    wherever a board is finished, so edit_data (the printed case's input), the
    preview and every export file share one outline. Returns whether the board
    changed; a board with nothing beside the tab is left exactly as it was."""
    if layout.board_poly is None or layout.board_poly.is_empty:
        return False
    shift = 0.0 if tail_extension_from_layout(layout) else AUTO_TAIL_EXTENSION_MM
    opening = connector_opening(layout, outline_xy, params, CONNECTOR_DIR, shift)
    if opening is None:
        return False
    clipped = clip_beside_tab(layout.board_poly, *opening)
    if clipped is layout.board_poly:
        return False
    layout.board_poly = clipped
    final_poly = _largest_polygon(clipped)
    if not final_poly.is_empty:
        layout.final_outline_xy = remove_collinear_closed(np.array(final_poly.exterior.coords[:-1], dtype=float))
    return True


CONTACT_VIOLATION_TYPES = ("transition via", "via clearance", "via edge")


def _routing_violations(drc: dict) -> int:
    """DRC violations that bear on the routing: all of them except those of the
    connector's transition vias, which depend only on the contact side and are
    made at export time. generate() compares routing attempts by this count, so
    either contact side gets the same taxels and routes."""
    return int(drc["violations"]) - int(drc.get("contact_violations", 0))


def _outputs_with_connector_tail(
    layout: Layout,
    outline_xy: np.ndarray,
    col_routes: list[Route],
    row_routes: list[Route],
    removed: set[int],
    params: Params,
    routing_problem_pix: set[int] | None = None,
    incomplete_nets: list[str] | None = None,
    fill_xy: np.ndarray | None = None,
) -> tuple[str, dict, dict, str]:
    """_build_outputs, but a board with no user tail extension is exported
    through a synthetic AUTO_TAIL_EXTENSION_MM one. Without it, traces stop at
    the pad row ~3.3mm short of the connector pad drills and the board cap is
    not stitched to the connector outline. Applied to deep copies only: the
    caller's layout/routes (and therefore edit_data, the route editor, and the
    draw-a-tail prompt) are untouched, so a user extension added later starts
    from the unmodified routes."""
    sync_pads_to_feeds(layout, row_routes)
    ext = layout.tail.get("tail_extension") if layout.tail else None
    if ext and ext.get("enabled"):
        _clip_board_to_connector_tab(layout, outline_xy, params)
        return _build_outputs(layout, outline_xy, col_routes, row_routes, removed,
                              params, routing_problem_pix, incomplete_nets, fill_xy)

    lay = copy.deepcopy(layout)
    def _copies(routes):
        return [replace(r, polyline_xy=np.asarray(r.polyline_xy, dtype=float).copy(),
                        pad_xy=np.asarray(r.pad_xy, dtype=float).copy())
                for r in routes]
    cr, rr = _copies(col_routes), _copies(row_routes)
    c = np.asarray(lay.tail.get("pad_split_center") or lay.tail["pad_row_center"], dtype=float)
    n_out = np.asarray(lay.tail.get("n_out", [0.0, -1.0]), dtype=float)
    lay.tail["tail_extension"] = {
        "enabled": True,
        "path": [c.tolist(), (c + AUTO_TAIL_EXTENSION_MM * n_out).tolist()],
        "width_mm": max(0.0, float(lay.tail.get("width_mm", 0.0)) - 0.7),
        "clearance_mm": float(params.trace_w_mm / 2 + params.board_edge_clear_mm),
    }
    apply_tail_extension(lay, [*cr, *rr], params, CONNECTOR_DIR)
    lay.tail.setdefault("tail_extension_resolved", {})["auto"] = True
    _apply_post_route_containment(lay, cr, rr, params)
    _clip_board_to_connector_tab(lay, outline_xy, params)
    return _build_outputs(lay, outline_xy, cr, rr, removed,
                          params, routing_problem_pix, incomplete_nets, fill_xy)


def _build_outputs(
    layout: Layout,
    outline_xy: np.ndarray,
    col_routes: list[Route],
    row_routes: list[Route],
    removed: set[int],
    params: Params,
    routing_problem_pix: set[int] | None = None,
    incomplete_nets: list[str] | None = None,
    fill_xy: np.ndarray | None = None,
) -> tuple[str, dict, dict, str]:
    n_active = len(layout.placements_xy) - len(removed)
    # None (not empty) means "the caller doesn't know" -- every path except
    # /generate rebuilds from edit_data and has to read these back off the
    # layout, or it under-reports DRC. An explicitly empty set is a real
    # answer and is left alone.
    if routing_problem_pix is None:
        routing_problem_pix = set(int(i) for i in (layout.routing.get("routing_problem_pix") or ()))
    if incomplete_nets is None:
        incomplete_nets = list(layout.routing.get("incomplete_nets") or ())

    raw_connector_dir = os.path.join(os.path.dirname(__file__), "connectors")
    active_connector_dir = connector_export_dir(layout, raw_connector_dir)
    connector_export_enabled = bool(active_connector_dir)

    # The connector's fingers are on one face of the tail (contact_side); a
    # wire on another copper layer crosses to that face through a via in the
    # tail. Made on copies, for the DRC and both board writers alike: the
    # routes the caller holds (edit_data, the route editor) keep their layers.
    active_xy = np.asarray(layout.placements_xy, dtype=float)[
        sorted(set(range(len(layout.placements_xy))) - set(removed))]
    contacts = contact_transitions(
        connector_geometries(layout, outline_xy, params, active_connector_dir)
        if active_connector_dir else None,
        params, col_routes, row_routes, board_poly=layout.board_poly,
        field_boxes=[box(x - params.pitch_x_mm / 2, y - params.pitch_y_mm / 2,
                         x + params.pitch_x_mm / 2, y + params.pitch_y_mm / 2) for x, y in active_xy])
    cu_col, cu_row = contacts.col_routes, contacts.row_routes

    violations = drc_check(
        cu_col, cu_row, params.trace_w_mm, params.clearance_mm,
        outline_xy, params.board_edge_clear_mm, layout.board_poly,
        taxels=TaxelCopper.of(layout, removed, params), vias=contacts.vias)
    for problem in contacts.problems:
        xy = np.asarray(problem["xy"], dtype=float)
        violations.append(DrcViolation(
            layer=problem["layer"], net1=problem["net"], net2=f"pad {problem['pad']}",
            type="transition via", seg1=np.array([xy, xy]), seg2=np.array([xy, xy]),
            distance=0.0, min_required=0.0))
    violations = _drop_connector_zone_edge_flags(
        violations, layout, outline_xy, params, active_connector_dir)
    # The board outline and the connector footprint's Edge.Cuts, as exported,
    # must close into one outline with every line on it once: a footprint line
    # inside the board is milled as a slit, and KiCad's own DRC does not see it.
    violations.extend(edge_cuts_violations(layout, outline_xy, params, active_connector_dir))

    # The manifest is built first because the back silkscreen carries it: the
    # taxel-core QR has to describe the board that is actually exported, not a
    # re-derivation of it. See export/silk_backmark.py.
    manifest_json = export_manifest(
        layout, removed, params.pixel_w_mm, params.pixel_h_mm, outline_xy,
        params, active_connector_dir, fill_xy)

    pcb_text = generate_kicad_pcb(
        layout, outline_xy, cu_col, cu_row, removed, params,
        active_connector_dir, manifest_json=manifest_json, transition_vias=contacts.vias,
        drilled_pads=contacts.drilled_pads)

    # Every taxel's one top mask opening, as the board and Gerbers draw it
    # (export/coverlay.py); a dense board's is trimmed into the electrodes.
    active_idx = sorted(set(range(len(layout.placements_xy))) - set(removed))
    opening = taxel_opening_for(
        params, layout.placements_xy[active_idx] if active_idx else None)

    fp_name = make_footprint_name(
        "tactile_pixel", params.pixel_w_mm, params.pixel_h_mm,
        params.trace_w_mm, params.gap_mm, params.center_clear_mm,
        params.edge_clear_mm, params.via_drill_mm, params.via_dia_mm)
    fp_text, _ = build_pixel_footprint_text(
        fp_name, params.pixel_w_mm, params.pixel_h_mm,
        params.trace_w_mm, params.gap_mm, params.center_clear_mm,
        params.edge_clear_mm, params.via_drill_mm, params.via_dia_mm,
        opening=opening)

    gerber_files, drill_text, gerber_source = _manufacturing_files_from_kicad(
        pcb_text, copper_layers_of(params, row_routes))
    if gerber_files is None or drill_text is None:
        gerber_files = generate_gerbers(
            layout, outline_xy, cu_col, cu_row, removed, params,
            active_connector_dir, manifest_json=manifest_json, transition_vias=contacts.vias,
            drilled_pads=contacts.drilled_pads)
        drill_text = generate_drill_file(layout, removed, params, outline_xy, active_connector_dir,
                                         transition_vias=contacts.vias, drilled_pads=contacts.drilled_pads)
        gerber_source = "direct"
    svg_text = generate_svg(layout, outline_xy, col_routes, row_routes, removed, params,
                            active_connector_dir)
    # A 1:1 cutting template. Never fatal: a broken PDF must not cost the user
    # their Gerbers, which are the part that actually gets manufactured.
    pdf_info: dict = {}
    try:
        pdf_bytes = generate_print_pdf(layout, outline_xy, params, removed,
                                       active_connector_dir, info=pdf_info)
    except Exception:
        pdf_bytes, pdf_info = None, {}
    # Velostat cut files, same rule: optional extras, never allowed to fail
    # the export. Too big to fold -> one un-mirrored copy (fold=None); too big
    # for even that -> no PDF, DXF only.
    velo_dxf = velo_pdf = None
    velo_info: dict = {}
    try:
        velo_poly = velostat_polygon(layout, outline_xy)
        velo_dxf = velostat_dxf(velo_poly)
        velo_pdf = velostat_fold_pdf(velo_poly, info=velo_info)
    except Exception:
        velo_dxf = velo_pdf = None

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("board.kicad_pcb", pcb_text)
        for name, text in kicad_project_files(params, drilled_fingers=bool(contacts.drilled_pads)).items():
            zf.writestr(name, text)
        zf.writestr(f"{fp_name}.kicad_mod", fp_text)
        zf.writestr("manifest.json", manifest_json)
        if pdf_bytes:
            zf.writestr("printable_1to1.pdf", pdf_bytes)
        if velo_dxf:
            zf.writestr("velostat/velostat_cut.dxf", velo_dxf)
        if velo_pdf:
            zf.writestr("velostat/velostat_fold_template.pdf", velo_pdf)
        for fname, content in gerber_files.items():
            zf.writestr(f"gerber/{fname}", content)
        zf.writestr("gerber/board.drl", drill_text)

    stats = {
        "total_pixels": len(layout.placements_xy),
        "active_pixels": n_active,
        "removed_pixels": len(removed),
        "col_routes": len(col_routes),
        "row_routes": len(row_routes),
        "connector_pos": layout.connector_pos,
        "connector_swap": layout.connector_swap,
        "connector_export_enabled": connector_export_enabled,
        "gerber_exporter": gerber_source,
        "connector_export_warning": (
            "" if connector_export_enabled or not layout.tail.get("tail_extension_resolved", {}).get("enabled", False)
            else "Physical connector export supports only 20/30/32/40/50/60/70/80/90/100-pin footprints."
        ),
        "tail_length_mm": round(layout.tail.get("length_mm", params.cable_length_mm), 3),
        "tail_length_auto": bool(layout.tail.get("length_auto", False)),
        "tail_length_reason": layout.tail.get("length_reason", REASON_TYPED),
        "tail_width_mm": round(layout.tail.get("width_mm", 0.0), 3),
        "tail_side_padding_mm": round(layout.tail.get("side_padding_mm", params.tail_side_padding_mm), 3),
        "tail_clamp_reason": layout.tail.get("clamp_reason"),
        "tail_connector_min_width_mm": round(layout.tail.get("connector_min_width_mm", 0.0), 3),
        "connector_pad_center_span_mm": round(layout.tail.get("pad_center_span_mm", 0.0), 3),
        "connector_copper_span_mm": round(layout.tail.get("copper_span_mm", 0.0), 3),
        "connector_tab_tip_width_mm": round(layout.tail.get("tab_tip_width_mm", 0.0), 3),
        "connector_ear_stiffener_width_mm": round(layout.tail.get("ear_stiffener_width_mm", 0.0), 3),
        # The synthetic export-time extension (auto=True) must not read as a
        # user tail extension: this flag hides the frontend's draw-a-tail
        # prompt and flips its stats rows.
        "tail_extension_enabled": bool(
            layout.tail.get("tail_extension_resolved", {}).get("enabled", False)
            and not layout.tail.get("tail_extension_resolved", {}).get("auto", False)),
        "tail_extension_length_mm": round(layout.tail.get("tail_extension_resolved", {}).get("length_mm", 0.0), 3),
        "tail_extension_width_mm": round(layout.tail.get("tail_extension_resolved", {}).get("width_mm", 0.0), 3),
        "board_mode": params.board_mode,
        "copper_layers": copper_layers_of(params, row_routes),
        "rows_on_inner_layers": {L: len({r.net_name for r in row_routes if is_feed(r) and r.layer == L})
                                 for L in INNER_LAYERS},
        "rows_linked_on_in2": len({r.net_name for r in row_routes
                                   if r.layer == "In2.Cu" and not is_feed(r)}),
        "preserve_sensors": params.preserve_sensors,
        "auto_expand_board": params.auto_expand_board,
        "fixed_keepout_min_mm": round(layout.tail.get("fixed_keepout_min_mm", 0.0), 3),
        "fixed_keepout_max_mm": round(layout.tail.get("fixed_keepout_max_mm", 0.0), 3),
        "fixed_keepout_mean_mm": round(layout.tail.get("fixed_keepout_mean_mm", 0.0), 3),
        "fixed_keepout_iterations": int(layout.tail.get("fixed_keepout_iterations", 0)),
        "fixed_keepout_converged": bool(layout.tail.get("fixed_keepout_converged", False)),
        "fixed_keepout_route_attempts": int(layout.tail.get("fixed_keepout_route_attempts", 0)),
        "fixed_keepout_routing_clean": bool(layout.tail.get("fixed_keepout_routing_clean", False)),
        "fixed_keepout_worst_escape_mm": round(layout.tail.get("fixed_keepout_worst_escape_mm", 0.0), 3),
        "max_pack_pixels": int(layout.tail.get("max_pack_pixels", len(layout.placements_xy))),
        "dropped_by_keepout": int(layout.tail.get("dropped_by_keepout", 0)),
        "fixed_board_routing": bool(layout.tail.get("fixed_board_routing", False)),
        "perimeter_routing_attempted": bool(layout.tail.get("perimeter_routing_attempted", False)),
        "removal_reason": layout.tail.get("removal_reason", ""),
        "removed_by_routing_failure": int(layout.tail.get("removed_by_routing_failure", len(removed))),
        "bad_nets_before_removal": int(layout.tail.get("bad_nets_before_removal", 0)),
        "routing_margin_mm": params.routing_margin_mm,
        "routing_margin_applied_mm": round(layout.routing_margin_applied_mm, 3),
        "routing_left_padding_mm": round(layout.tail.get("routing_left_padding_mm", 0.0), 3),
        "routing_right_padding_mm": round(layout.tail.get("routing_right_padding_mm", 0.0), 3),
        "routing_side_padding_mm": round(layout.tail.get("routing_side_padding_mm", 0.0), 3),
        "routing_top_padding_mm": round(layout.tail.get("routing_top_padding_mm", 0.0), 3),
        "routing_bottom_padding_mm": round(layout.tail.get("routing_bottom_padding_mm", 0.0), 3),
        "routing_route_count": int(layout.tail.get("routing_route_count", 0)),
        "tail_shoulder_expansion": bool(layout.tail.get("tail_shoulder_expansion", False)),
        "tail_shoulder_width_mm": round(layout.tail.get("tail_shoulder_width_mm", 0.0), 3),
        "tail_pocket_depth_mm": round(layout.tail.get("tail_shoulder_depth_mm", 0.0), 3),
        "post_route_containment_expansion": bool(layout.tail.get("post_route_containment_expansion", False)),
        "post_route_containment_buffer_mm": round(layout.tail.get("post_route_containment_buffer_mm", 0.0), 3),
        "max_route_outside_distance_mm": round(layout.tail.get("max_route_outside_distance_mm", 0.0), 3),
        "route_clearance_shortfall_mm": round(layout.tail.get("route_clearance_shortfall_mm", 0.0), 3),
        "follow_main_padding": bool(layout.tail.get("follow_main_padding", params.follow_main_padding)),
        "follow_main_padding_mm": round(layout.tail.get("follow_main_padding_mm", params.follow_main_padding_mm), 3),
        "smooth_follow_padding": bool(layout.tail.get("smooth_follow_padding", params.smooth_follow_padding)),
        "final_outline_mode": layout.tail.get("final_outline_mode", "rectangular"),
        "router": layout.tail.get("router", "legacy"),
        "dropped_by_edge_prune": int(layout.tail.get("dropped_by_edge_prune", 0)),
        "dropped_failed_nets": int(layout.tail.get("dropped_failed_nets", 0)),
        "hug_plan_warnings": list(layout.tail.get("hug_plan_warnings", [])),
        "routing_problem_pixels": len(routing_problem_pix),
        "routing_warning_nets": len(incomplete_nets),
        "row_routes_left": int(layout.routing.get("row_routes_left", 0)),
        "row_routes_right": int(layout.routing.get("row_routes_right", 0)),
        "back_qr": _back_qr_stats(manifest_json),
        # The coverlay (solder mask) openings: one per taxel, at least 0.3 mm
        # apart. mask_note is "" unless the openings had to be trimmed into
        # the electrodes; the result panel shows it.
        "coverlay": opening.stats(),
        "mask_note": opening.note,
    }
    drc = {
        "violations": len(violations) + len(incomplete_nets),
        # The connector's transition vias (export/connector_contacts.py): part
        # of "violations", and left out where generate() chooses between
        # routing attempts, so the contact side never changes taxels or routes.
        "contact_violations": sum(v.type in CONTACT_VIOLATION_TYPES for v in violations),
        "routing_problem_pixels": sorted(int(i) for i in routing_problem_pix),
        "routing_warning_nets": sorted(incomplete_nets),
        "details": [
            {
                "layer": v.layer,
                "net1": v.net1,
                "net2": v.net2,
                "type": v.type,
                "distance": round(v.distance, 4),
                "min_required": round(v.min_required, 4),
            }
            for v in violations[:20]
        ],
    }
    zip_b64 = base64.b64encode(zip_buffer.getvalue()).decode()
    if pdf_bytes:
        stats["print_pdf"] = {
            "b64": base64.b64encode(pdf_bytes).decode(),
            "pages": int(pdf_info.get("pages", 1)),
            "board_w_mm": float(pdf_info.get("board_w_mm", 0.0)),
            "board_h_mm": float(pdf_info.get("board_h_mm", 0.0)),
        }
    if velo_dxf:
        stats["velostat"] = {
            "dxf_b64": base64.b64encode(velo_dxf.encode()).decode(),
            "fold_pdf_b64": base64.b64encode(velo_pdf).decode() if velo_pdf else None,
            "piece_w_mm": float(velo_info.get("piece_w_mm", 0.0)),
            "piece_h_mm": float(velo_info.get("piece_h_mm", 0.0)),
            "sheet": velo_info.get("sheet"),
            "fold": velo_info.get("fold"),
        }
    return svg_text, stats, drc, zip_b64


def _back_qr_stats(manifest_json) -> dict:
    """Which QR the back silkscreen carries, so the export can say so: "link"
    (a phone opens the decoder page), "bare" (a board too small for the link;
    a phone shows text) or None. See export/silk_backmark.py."""
    mark = backmark(manifest_json)
    q = mark["qr"]
    return {"form": q["form"] if q else None,
            "size_mm": round(q["size_mm"], 2) if q else None,
            "text": q["text"] if q else None,
            "warnings": list(mark["warnings"])}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    try:
        outline_xy = np.array(req.outline)
        if outline_xy.shape[1] != 2 or len(outline_xy) < 3:
            raise ValueError("Outline must have at least 3 vertices (Nx2).")

        fill_xy = None
        if req.fill_region:
            fill_xy = np.array(req.fill_region)
            if fill_xy.shape[1] != 2 or len(fill_xy) < 3:
                raise ValueError("Fill region must have at least 3 vertices (Nx2).")

        # Rotate so cable edge (edge 1->2) is horizontal at the bottom
        if fill_xy is not None:
            outline_xy, fill_xy = rotate_for_cable(outline_xy, fill_xy)
        else:
            outline_xy = rotate_for_cable(outline_xy)

        # A cable edge that is an interior chord — the outline continues
        # below it, so the connector sits in a bay with board (and sensors)
        # on three sides — is a topology neither router orders cleanly:
        # the hug plan goes non-planar (its refusal guard then leaves a
        # drastically reduced board) and legacy has no corridors out of
        # the bay. Moving the cable edge to the outer boundary avoids this,
        # so say so up front.
        cable_bay_warning = None
        if float(np.min(outline_xy[:, 1])) < -0.5:
            cable_bay_warning = (
                "The cable edge is an interior edge — the outline extends "
                f"{-float(np.min(outline_xy[:, 1])):.1f}mm below it, putting the "
                "connector in a bay. This routes poorly or not at all; move the "
                "cable edge to an outermost edge of the shape for full routing.")

        if req.contact_side not in CONTACT_SIDES:
            raise ValueError(f'contact_side must be "bottom" or "top", got {req.contact_side!r}.')
        if req.copper_layers not in (2, 4):
            raise ValueError(f"copper_layers must be 2 or 4, got {req.copper_layers!r}.")
        if req.trace_w_mm + req.clearance_mm > req.pad_pitch_mm + 1e-9:
            # Wires land on neighbouring connector pins pad_pitch apart; closer
            # than trace + clearance, every other one is refused and most of
            # the board cannot be wired.
            raise ValueError(
                f"Trace width + clearance ({req.trace_w_mm + req.clearance_mm:.2f} mm) is more than "
                f"the connector's pin pitch ({req.pad_pitch_mm:.2f} mm), so wires to neighbouring pins "
                "cannot keep the clearance. Use a thinner trace or a smaller clearance.")
        params = Params(
            pixel_w_mm=req.pixel_w_mm,
            pixel_h_mm=req.pixel_h_mm,
            pitch_x_mm=req.pitch_x_mm,
            pitch_y_mm=req.pitch_y_mm,
            edge_keepout_mm=req.edge_keepout_mm,
            trace_w_mm=req.trace_w_mm,
            gap_mm=req.gap_mm,
            clearance_mm=req.clearance_mm,
            center_clear_mm=req.center_clear_mm,
            edge_clear_mm=req.edge_clear_mm,
            via_drill_mm=req.via_drill_mm,
            via_dia_mm=req.via_dia_mm,
            pad_pitch_mm=req.pad_pitch_mm,
            # Automatic: routed with the minimum, lengthened after routing.
            cable_length_mm=AUTO_TAIL_MIN_MM if req.cable_length_mm is None else req.cable_length_mm,
            cable_length_auto=req.cable_length_mm is None,
            tail_width_mm=req.tail_width_mm,
            tail_side_padding_mm=req.tail_side_padding_mm,
            board_edge_clear_mm=req.board_edge_clear_mm,
            routing_margin_mm=req.routing_margin_mm,
            wiring_keepout_floor_mm=req.wiring_keepout_floor_mm,
            follow_main_padding=req.follow_main_padding,
            follow_main_padding_mm=req.follow_main_padding_mm,
            smooth_follow_padding=req.smooth_follow_padding,
            connector_pos_override=req.connector_pos_override,
            connector_shoulder_fillet_mm=req.connector_shoulder_fillet_mm,
            target_cols=req.target_cols,
            target_rows=req.target_rows,
            design_name=req.design_name,
            readout_board=req.readout_board,
            narrow_tail=req.narrow_tail,
            contact_side=req.contact_side,
            copper_layers=req.copper_layers,
        )
        board_mode = _resolve_board_mode(req.board_mode, req.auto_expand_board)
        params = _params_for_board_mode(params, board_mode)
        params = _without_expansion_padding(params)
        params = _apply_target_grid(params, outline_xy, fill_xy, board_mode)

        routing_problem_pix = set()
        incomplete = []

        if req.router not in ("legacy", "hug"):
            raise ValueError(f'router must be "legacy" or "hug", got {req.router!r}.')

        if req.router == "hug" and board_mode == "expand":
            # Hug router: joined nets (a split row/column stays one net), then
            # every wire hugs the sensor field and the wires routed before it.
            # The board is expanded up front only as a canvas — the router never
            # consults it — and post-route containment plus follow-padding
            # redraw the final edge cuts around the copper that actually exists.
            layout = layout_with_connector(outline_xy, params, fill_xy, join_split_nets=True)
            col_routes, row_routes, hug_plan, failed_pix = route_for(params)(layout, outline_xy, params)
            removed = set()
            # Capability boundary: when the plan itself reports a non-planar
            # ordering ("forced crossing") or the emission refuses a large
            # share of the board, this topology is beyond the hug model's
            # ordering rules (e.g. an outline whose row bundles' contour
            # spans interleave). Neither emitting crossings nor returning an
            # almost empty board is acceptable, so route the same request
            # with the legacy router and say so. Making these topologies
            # planar would need plan-driven side and pad assignment per
            # bundle.
            not_planar = any("not planar" in w for w in hug_plan.warnings)
            mass_refusal = len(failed_pix) > 0.25 * max(1, len(layout.placements_xy))
            if failed_pix and (not_planar or mass_refusal):
                # Only take the fallback when it is a clear win (a clean
                # legacy board). A legacy result with its own violations is
                # not an improvement over hug's reduced-but-clean board —
                # crossings must never be output from either path.
                try:
                    fallback = generate(req.model_copy(update={"router": "legacy"}))
                except HTTPException:
                    fallback = None
                if fallback is not None and _routing_violations(fallback.drc) == 0:
                    fallback.stats["router"] = "legacy"
                    fallback.stats["hug_fallback"] = True
                    fallback.stats["hug_plan_warnings"] = list(hug_plan.warnings)[:8]
                    return fallback
                hug_plan.warnings.append(
                    f"hug ordering cannot solve this topology ({len(failed_pix)} of "
                    f"{len(layout.placements_xy)} pixels' nets refused"
                    + (f"; legacy alternative had {fallback.drc['violations']} violations"
                       if fallback is not None else "; legacy alternative failed")
                    + ") — the board keeps only the taxels that route cleanly. Consider placing the "
                    "cable edge on the outline's outermost edge and giving "
                    "sensors a little spacing (pitch > pixel).")
            plan_warnings = list(hug_plan.warnings)
            if failed_pix:
                # The emission refused nets it could not draw without a DRC
                # violation (route_hug_all docstring). Their pixels are dead:
                # drop them and re-route the smaller field — repeatedly if a
                # retry uncovers another refusal — so the returned board
                # never contains a refused wire.
                for _ in range(4):
                    removed |= failed_pix
                    active = [i for i in range(len(layout.placements_xy)) if i not in removed]
                    retry_layout = copy.deepcopy(layout)
                    retry_layout.placements_xy = layout.placements_xy[active]
                    retry_layout.col_net_per_pixel = [layout.col_net_per_pixel[i] for i in active]
                    retry_layout.row_net_per_pixel = [layout.row_net_per_pixel[i] for i in active]
                    col_routes, row_routes, hug_plan, retry_failed = route_for(params)(
                        retry_layout, outline_xy, params)
                    plan_warnings.extend(w for w in hug_plan.warnings if w not in plan_warnings)
                    if not retry_failed:
                        break
                    failed_pix = {active[i] for i in retry_failed}
                else:
                    # Retries exhausted with refusals still standing (a
                    # non-planar topology can keep uncovering new
                    # refusals each round). Their routes were already
                    # stripped by route_hug_all; fold their pixels into
                    # `removed` too, or validate_routes counts the routeless
                    # nets as incomplete and the board is returned with
                    # violations.
                    removed |= failed_pix
            bad_pix, incomplete = validate_routes(
                layout, col_routes, row_routes, removed,
                outline_xy, params.pad_pitch_mm,
                params.trace_w_mm, params.clearance_mm)
            routing_problem_pix = set(bad_pix)
            layout.tail["router"] = "hug"
            if plan_warnings:
                layout.tail["hug_plan_warnings"] = plan_warnings
        elif board_mode == "expand":
            # Grow-board mode. The trial pass deliberately flips preserve_sensors
            # off to *probe* routability on the un-grown outline; it is thrown
            # away and only its removal count decides whether growth is needed.
            trial_params = replace(params, preserve_sensors=False, auto_expand_board=False)
            trial_layout = layout_with_connector(outline_xy, trial_params, fill_xy)
            trial_col_routes, trial_col_removed = route_columns_grid(trial_layout, outline_xy, trial_params)
            trial_row_routes, trial_row_removed = route_rows_grid(trial_layout, outline_xy, trial_params, trial_col_removed)
            trial_removed = trial_col_removed | trial_row_removed
            if trial_row_removed - trial_col_removed:
                trial_col_routes, trial_col_removed2 = route_columns_grid(
                    trial_layout, outline_xy, trial_params, trial_removed)
                trial_removed |= trial_col_removed2
            for _ in range(10):
                trial_bad_pix, _ = validate_routes(
                    trial_layout, trial_col_routes, trial_row_routes, trial_removed,
                    outline_xy, trial_params.pad_pitch_mm,
                    trial_params.trace_w_mm, trial_params.clearance_mm)
                if not trial_bad_pix:
                    break
                trial_removed |= trial_bad_pix
                trial_col_routes, trial_cr = route_columns_grid(trial_layout, outline_xy, trial_params, trial_removed)
                trial_row_routes, trial_rr = route_rows_grid(trial_layout, outline_xy, trial_params, trial_removed)
                trial_removed |= trial_cr | trial_rr

            route_count = len(trial_col_routes) + len(trial_row_routes)
            high_tail_density = route_count >= 32 or max(len(trial_col_routes), len(trial_row_routes)) >= 24
            expand_needed = bool(trial_removed) or high_tail_density

            layout_params = params if expand_needed and params.auto_expand_board else replace(params, auto_expand_board=False)
            layout = layout_with_connector(outline_xy, layout_params, fill_xy)
            params = layout_params
            col_routes, col_problem = route_columns_grid(layout, outline_xy, params)
            row_routes, row_problem = route_rows_grid(layout, outline_xy, params)
            removed = set()
            bad_pix, incomplete = validate_routes(
                layout, col_routes, row_routes, removed,
                outline_xy, params.pad_pitch_mm,
                params.trace_w_mm, params.clearance_mm)
            routing_problem_pix = set(col_problem) | set(row_problem) | set(bad_pix)
        else:
            # Fixed-outline keep-out mode: the drawn outline *is* the board. A
            # wiring keep-out band is reserved inside it and the sensor grid is
            # packed into what is left, so pixels that would reach into the
            # wiring zone are never placed to begin with. If routing still
            # fails, the keep-out grows on the implicated side and the grid is
            # re-packed — individual pixels are never deleted, which is what
            # keeps the surviving array a clean rectangular grid.
            # The hug router reuses this whole loop: only the demand model (its
            # plan gives exact per-sample wiring widths instead of the legacy
            # heuristic), the net assignment (joined), and the routing call
            # differ. The escape check and grow-on-failure retry stay as the
            # safety net for both.
            hug_fixed = req.router == "hug"
            hug_plan = None

            attempt_params = _fixed_keepout_attempt_params(params)
            if hug_fixed:
                # Joined nets buy true row/column semantics and fewer pads,
                # but their link corridors around a concavity lie inside the
                # material in fixed mode, and on tight shapes (arms only a few
                # pitches wide) they do not fit. Try joined first, split nets
                # second — the hug walk routes either — and the attempt
                # scoring keeps the denser clean result.
                attempt_specs = [(p, True) for p in attempt_params] \
                              + [(p, False) for p in attempt_params]
            else:
                attempt_specs = [(p, None) for p in attempt_params]
            best_attempt = None
            last_layout_err = None
            for try_i, (params, join_nets) in enumerate(attempt_specs):
                def _fixed_layout(p, seed_profile=None, baseline_pixels=None,
                                  _join=join_nets):
                    if hug_fixed:
                        return layout_fixed_hug_keepout(
                            outline_xy, p, fill_xy,
                            seed_profile=seed_profile,
                            baseline_pixels=baseline_pixels,
                            join_split_nets=_join)
                    return layout_fixed_shape_keepout(
                        outline_xy, p, fill_xy,
                        seed_profile=seed_profile, baseline_pixels=baseline_pixels)

                try:
                    layout, keepout_profile, keepout_info = _fixed_layout(params)
                except ValueError as exc:
                    if not hug_fixed:
                        raise   # legacy router: the layout error propagates
                    # Hug variant cannot pack at all (e.g. joined-net link
                    # corridors don't fit this shape) — try the next one.
                    last_layout_err = exc
                    continue
                baseline_pixels = keepout_info["max_pack_pixels"]
                removed = set()
                worst_escape = 0.0
                near_miss = None    # the best clean _drop_near_escapes repair seen

                for attempt in range(MAX_ROUTE_RETRIES):
                    if hug_fixed:
                        col_routes, row_routes, hug_plan, hug_failed = route_for(params)(
                            layout, outline_xy, params)
                        # Same capability boundary as the expand branch: a
                        # topology the hug plan cannot order planar falls
                        # back to the legacy router instead of emitting
                        # crossings or shrinking the field away.
                        # Only while nothing clean is in hand: a later spec
                        # (split nets) failing must not throw away a clean
                        # earlier one (joined) for the legacy router's board.
                        have_clean = best_attempt is not None and best_attempt[0][0] == 0
                        if hug_failed and not have_clean and (
                                any("not planar" in w for w in hug_plan.warnings)
                                or len(hug_failed) > 0.25 * max(1, len(layout.placements_xy))):
                            try:
                                fallback = generate(req.model_copy(update={"router": "legacy"}))
                            except HTTPException:
                                fallback = None
                            # Clean by itself — not clean because escaping
                            # nets were pruned away (_prune_edge_escapes).
                            if (fallback is not None and _routing_violations(fallback.drc) == 0
                                    and not fallback.stats.get("dropped_by_edge_prune")):
                                fallback.stats["router"] = "legacy"
                                fallback.stats["hug_fallback"] = True
                                fallback.stats["hug_plan_warnings"] = list(hug_plan.warnings)[:8]
                                return fallback
                            hug_plan.warnings.append(
                                "hug ordering cannot solve this topology and the "
                                "legacy alternative was not clean either — the "
                                "keep-out band is widened for the refused nets and "
                                "routing is retried.")
                        # Refused nets drive the keep-out growth retry exactly
                        # like a legacy routing failure would.
                        col_problem, row_problem = set(hug_failed), set()
                    else:
                        col_routes, col_problem = route_columns_grid(layout, outline_xy, params)
                        row_routes, row_problem = route_rows_grid(layout, outline_xy, params)
                    bad_pix, incomplete = validate_routes(
                        layout, col_routes, row_routes, removed,
                        outline_xy, params.pad_pitch_mm,
                        params.trace_w_mm, params.clearance_mm)
                    routing_problem_pix = set(col_problem) | set(row_problem) | set(bad_pix)
                    # Copper leaving Edge.Cuts is a failure the routers do not
                    # necessarily report, and one this mode cannot fix by growing
                    # the board — so it drives the retry directly.
                    escapes = _route_escapes(layout, [*col_routes, *row_routes], params)
                    worst_escape = max((s for _, s in escapes), default=0.0)
                    if (not routing_problem_pix and not escapes) or attempt == MAX_ROUTE_RETRIES - 1:
                        break
                    if hug_fixed and escapes and not routing_problem_pix:
                        # Copper too near the edge and nothing else wrong:
                        # also try giving up the taxel(s) beside it, on this
                        # same grid (see _drop_near_escapes). The regrow
                        # below still runs; the denser clean result is kept.
                        repaired = _drop_near_escapes(layout, outline_xy, params, escapes, join_nets)
                        if repaired is not None and (
                                near_miss is None
                                or len(repaired[0].placements_xy) > len(near_miss[0][0].placements_xy)):
                            near_miss = (repaired, keepout_profile.copy(), keepout_info)
                    if hug_fixed and len(layout.placements_xy) < 0.15 * max(1, baseline_pixels):
                        # Already below the too-small reject threshold; growth
                        # retries only shrink the field further. Abandoning them
                        # makes an infeasible shape fail fast instead of
                        # running through every retry first.
                        break
                    keepout_profile = _grow_keepout_for_failures(
                        keepout_profile, layout, routing_problem_pix, escapes, params)
                    try:
                        layout, keepout_profile, keepout_info = _fixed_layout(
                            params, seed_profile=keepout_profile,
                            baseline_pixels=baseline_pixels)
                    except ValueError:
                        if not hug_fixed:
                            raise   # legacy router: the layout error propagates
                        break   # growth left no room — keep this attempt's last result

                if near_miss is not None and (
                        routing_problem_pix or escapes
                        or len(near_miss[0][0].placements_xy) > len(layout.placements_xy)):
                    (layout, col_routes, row_routes, hug_plan, incomplete), keepout_profile, keepout_info = near_miss
                    routing_problem_pix, escapes, worst_escape = set(), [], 0.0
                # Score this attempt. Fewest real defects wins; among equally
                # clean results, the denser one. See _fixed_keepout_attempt_params.
                defects = (len(routing_problem_pix) + len(escapes)
                           + len(_drc_crowding(col_routes, row_routes, params)))
                score = (defects, -len(layout.placements_xy))
                snapshot = (score, params, layout, keepout_profile, keepout_info,
                            col_routes, row_routes, removed, routing_problem_pix,
                            escapes, incomplete, worst_escape, attempt, join_nets,
                            hug_plan)
                if best_attempt is None or score < best_attempt[0]:
                    best_attempt = snapshot
                # The legacy router stops at the first clean attempt. Hug
                # runs every spec: a clean joined result can still be much
                # sparser than split (link corridors use up material), and the
                # score's taxel term should decide that, not attempt order.
                # Ties prefer the earlier (joined) spec.
                if defects == 0 and not hug_fixed:
                    break

            if best_attempt is None:
                raise last_layout_err if last_layout_err is not None else ValueError(
                    "Fixed-outline layout failed for every attempt.")
            (_score, params, layout, keepout_profile, keepout_info, col_routes,
             row_routes, removed, routing_problem_pix, escapes, incomplete,
             worst_escape, attempt, join_nets, hug_plan) = best_attempt
            edge_pruned = failed_pruned = 0
            if escapes or (hug_fixed and routing_problem_pix):
                (layout, col_routes, row_routes, edge_pruned, failed_pruned, escapes,
                 pruned_problem, pruned_incomplete) = _prune_edge_escapes(
                    layout, outline_xy, params, col_routes, row_routes, hug_fixed,
                    problem_pix=routing_problem_pix)
                if edge_pruned or failed_pruned:
                    routing_problem_pix, incomplete = pruned_problem, pruned_incomplete
                    worst_escape = max((s for _, s in escapes), default=0.0)
            layout.tail["dropped_by_edge_prune"] = int(edge_pruned)
            layout.tail["dropped_failed_nets"] = int(failed_pruned)
            if hug_fixed:
                # A defect-laden result on a nearly-empty board is worse than
                # the error the legacy router raises for the same shapes: the
                # outline cannot hold both sensors and wiring.
                n_kept = len(layout.placements_xy)
                if _score[0] > 0 and n_kept < 0.15 * max(1, keepout_info["max_pack_pixels"]):
                    raise ValueError(_KEEPOUT_TOO_SMALL_MSG)
                layout.tail["hug_fixed_nets"] = "joined" if join_nets else "split"
                if hug_plan is not None and hug_plan.warnings:
                    layout.tail["hug_plan_warnings"] = list(hug_plan.warnings)
            # Never return nets known not to reach the connector. By here every
            # attempt (each keep-out floor; joined and split nets on the hug
            # router) has been scored and the best one's failed taxels pruned
            # and re-routed; what is still unconnected has no layout left.
            if routing_problem_pix or incomplete:
                raise ValueError(_UNROUTED_MSG)
            layout.tail["fixed_keepout_attempts"] = len(attempt_params)
            layout.tail["fixed_keepout_floor_used_mm"] = (
                params.wiring_keepout_floor_mm
                if params.wiring_keepout_floor_mm is not None else 0.0)

            layout.tail["perimeter_routing_attempted"] = True
            layout.tail["removed_by_routing_failure"] = 0
            layout.tail["bad_nets_before_removal"] = len(incomplete)
            layout.tail["fixed_keepout_route_attempts"] = attempt + 1
            layout.tail["fixed_keepout_routing_clean"] = not routing_problem_pix and not escapes
            layout.tail["fixed_keepout_worst_escape_mm"] = float(worst_escape)
            layout.tail["removal_reason"] = (
                "fixed-outline keep-out clean"
                if not routing_problem_pix and not escapes and not incomplete
                else "fixed-outline keep-out warnings only"
            )
            if hug_fixed:
                layout.tail["router"] = "hug"

        # Smooth routing: collapse staircases into long diagonals, then chamfer
        # any leftover sharp corners (both guarded; never add violations).
        # Skipped for the hug router: its paths are already smooth contour
        # hugs, and in grow mode the board covers the concavity, so the
        # line-of-sight pass would legally shortcut straight across the void
        # — undoing exactly the hugging the router exists to produce.
        if req.router != "hug":
            row_sensor_gap = max(params.trace_w_mm + params.clearance_mm, 0.5)
            col_taxel_gap = taxel_copper_keepout_mm(params)
            col_routes = los_simplify_layer(
                col_routes, layout, removed, outline_xy, params,
                pixel_keepout_mm=col_taxel_gap,
                protect_outer_col_fanout=True)
            row_routes = los_simplify_layer(
                row_routes, layout, removed, outline_xy, params,
                pixel_keepout_mm=row_sensor_gap,
                protect_outer_row_fanout=True)
            row_routes = make_row_entries_horizontal(row_routes)
            col_routes = chamfer_layer(
                col_routes, layout, removed, outline_xy, params,
                pixel_keepout_mm=col_taxel_gap,
                max_miter_mm=1.4,
                passes=5,
                angle_dot_limit=0.7,
                fractions=(1.0, 0.65, 0.4, 0.22, 0.12, 0.06, 0.025))
            row_routes = chamfer_layer(
                row_routes, layout, removed, outline_xy, params, pixel_keepout_mm=row_sensor_gap)
        if params.cable_length_auto:
            layout, col_routes, row_routes, params = _auto_tail_length(
                layout, outline_xy, col_routes, row_routes, params)
        else:
            _finish_board(layout, col_routes, row_routes, params)
            layout.tail["length_reason"] = REASON_TYPED

        # Generate outputs.
        # Record what the router could not do, alongside the geometric DRC.
        # drc["violations"] is len(geometry) + len(incomplete nets), and only
        # /generate knows the second half: /export-edited and /redo-padding
        # rebuild from edit_data, so without this an edited export of a board
        # with unroutable nets would report DRC 0 for unchanged copper.
        # Storing it on layout.routing -- which round-trips -- makes every
        # path agree.
        layout.routing["routing_problem_pix"] = sorted(int(i) for i in routing_problem_pix)
        layout.routing["incomplete_nets"] = sorted(incomplete)
        if cable_bay_warning:
            layout.tail.setdefault("hug_plan_warnings", [])
            if cable_bay_warning not in layout.tail["hug_plan_warnings"]:
                layout.tail["hug_plan_warnings"].insert(0, cable_bay_warning)
        svg_text, stats, drc, zip_b64 = _outputs_with_connector_tail(
            layout, outline_xy, col_routes, row_routes, removed, params,
            routing_problem_pix, incomplete, fill_xy)

        # Final-artifact guard for fixed-outline mode. The per-attempt check in
        # the keep-out loop scores routing *before* smoothing, post-route
        # containment and follow-padding have run, so a board can pass there and
        # still come out of the smoothing and output steps with violations.
        # Rather than reserve for that on every board, re-run once with the
        # conservative routing_margin_mm floor and keep whichever result is
        # cleaner. Bounded: the retry pins wiring_keepout_floor_mm, so the
        # branch below cannot fire a second time.
        if (board_mode == "fixed_keepout"
                and req.wiring_keepout_floor_mm is None
                and _routing_violations(drc) > 0):
            safe_req = req.model_copy(update={
                "wiring_keepout_floor_mm": max(0.0, float(req.routing_margin_mm))})
            # For the hug router the fallback exists to improve on the primary
            # result; if the conservative floor cannot produce a board at all
            # (the hug too-small guard fires with the wider band), keep the
            # primary result instead of letting the fallback's error replace
            # it. For the legacy router a fallback error propagates.
            if req.router == "hug":
                try:
                    fallback = generate(safe_req)
                except HTTPException:
                    fallback = None
            else:
                fallback = generate(safe_req)
            if fallback is not None and _routing_violations(fallback.drc) < _routing_violations(drc):
                return fallback

        edit_data = _build_edit_data(
            layout, outline_xy, col_routes, row_routes, removed, params, fill_xy)
        # design.json goes into the ZIP so a downloaded design can be re-opened
        # exactly, hand-edited routes included. Added here rather than inside
        # _build_outputs: that runs on a deep copy of the layout and edit_data
        # is built here from the original, so this is the first point where
        # the real editor state exists. See zip_with_design.
        zip_b64 = zip_with_design(zip_b64, design_file(edit_data))

        return GenerateResponse(
            svg=svg_text,
            stats=stats,
            drc=drc,
            zip_b64=zip_b64,
            edit_data=edit_data,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@app.post("/export-edited", response_model=ExportEditedResponse)
def export_edited(req: ExportEditedRequest):
    try:
        layout, outline_xy, params, removed = _layout_from_edit_data(req.edit_data)
        params = _without_expansion_padding(params)
        col_routes, row_routes = _routes_from_payload(req.routes)
        apply_tail_extension(layout, [*col_routes, *row_routes], params, CONNECTOR_DIR)
        _apply_post_route_containment(layout, col_routes, row_routes, params)
        _capture_pre_follow_outline(layout)
        apply_main_follow_padding(layout, [*col_routes, *row_routes], params)
        _apply_connector_shoulder_if_needed(layout, outline_xy, params)
        _clip_board_to_connector_tab(layout, outline_xy, params)
        svg_text, stats, drc, zip_b64 = _outputs_with_connector_tail(
            layout, outline_xy, col_routes, row_routes, removed, params,
            fill_xy=req.edit_data.get("fill_region"))
        # The caller's own state, not a rebuild of it: req.routes are the
        # dragged polylines, which is exactly what edit_data["routes"] is not.
        # Re-importing this ZIP therefore restores the edited board, not the
        # board it was generated from.
        zip_b64 = zip_with_design(
            zip_b64, design_file(req.edit_data, req.routes))
        return ExportEditedResponse(svg=svg_text, stats=stats, drc=drc, zip_b64=zip_b64)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Edited export error: {str(e)}")


@app.post("/bump-sheet", response_model=BumpSheetResponse)
def bump_sheet(req: BumpSheetRequest):
    """Build the printable shear-sensing overlay for an already-generated board.

    Takes the same `edit_data` the board editor holds, so the sheet always
    wraps the final outline — the grown, follow-padded board that actually gets
    manufactured — rather than the shape the user drew. Returns an STL rather
    than adding to the ZIP: it is a separate printed part, on a different
    machine, and users iterate on bump geometry without re-exporting Gerbers.
    """
    try:
        edit_data = req.edit_data
        missing = [k for k in ("params", "layout", "outline") if k not in edit_data]
        if missing:
            raise ValueError(
                f"edit_data is missing {', '.join(missing)} — generate a board first.")
        params = edit_data["params"]
        placements = np.asarray(edit_data["layout"]["placements_xy"], dtype=float)
        removed = {int(i) for i in (edit_data.get("removed_pix") or [])}
        active = [i for i in range(len(placements)) if i not in removed]
        # Outline edge 0->1 is the cable edge, laid horizontal by
        # rotate_for_cable, so its y is where the sheet stops: the tail carries
        # no taxels and must stay clear for the connector.
        outline = np.asarray(edit_data["outline"], dtype=float)
        cable_y = float(outline[:2, 1].min())

        overrides = {k: v for k, v in (req.params or {}).items()
                     if k in BumpParams.__dataclass_fields__ and v is not None}
        bp = BumpParams(**overrides)
        result = build_bump_sheet(
            edit_data.get("final_outline", edit_data["outline"]), cable_y,
            placements, active,
            float(params["pitch_x_mm"]), float(params["pitch_y_mm"]), bp)

        mesh = result.mesh
        stl_bytes = mesh.export(file_type="stl")
        minx, miny, minz = mesh.bounds[0]
        maxx, maxy, maxz = mesh.bounds[1]
        base_x, base_y = result.bump_base_dia_mm
        stats = {
            "bumps": int(len(result.bump_centers)),
            # Every taxel sits under a bump; these are the bumps whose 2x2
            # block the outline cut short, so part of them overhangs.
            "partial_bumps": int(result.partial_groups),
            "taxels_total": len(active),
            "bump_base_x_mm": round(base_x, 2),
            "bump_base_y_mm": round(base_y, 2),
            "bump_top_x_mm": round(result.bump_top_dia_mm[0], 2),
            "bump_top_y_mm": round(result.bump_top_dia_mm[1], 2),
            "bump_height_mm": round(bp.bump_height_mm, 2),
            "total_height_mm": round(result.total_height_mm, 2),
            "sheet_w_mm": round(float(maxx - minx), 1),
            "sheet_h_mm": round(float(maxy - miny), 1),
            "recess_depth_mm": round(bp.recess_depth_mm, 2),
            "recess_tolerance_mm": round(bp.recess_tolerance_mm, 2),
            "sheet_thickness_mm": round(bp.sheet_thickness_mm, 2),
            "skirt_width_mm": round(bp.skirt_width_mm, 2),
            "tab_width_mm": round(bp.tab_width_mm, 2),
            "tabs_per_edge": int(bp.tabs_per_edge),
            "bump_base_pct": round(100.0 * bp.bump_base_frac, 1),
            "bump_top_pct": round(100.0 * bp.bump_top_ratio, 1),
            "triangles": int(len(mesh.faces)),
            "volume_mm3": round(float(mesh.volume), 1),
            "watertight": bool(mesh.is_watertight),
            "bodies": int(mesh.body_count),
        }
        return BumpSheetResponse(
            stl_b64=base64.b64encode(stl_bytes).decode("ascii"),
            stats=stats, warnings=list(result.warnings))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bump sheet error: {str(e)}")


def _case_inputs(edit_data: dict, params: dict | None, bump_params: dict | None):
    """What build_case needs from a board's edit_data, plus the CaseParams.

    Shared by /case and /zip-with-case so the case in the ZIP is built from
    exactly the inputs the case panel uses. Raises ValueError for edit_data
    without a board.
    """
    missing = [k for k in ("params", "layout", "outline") if k not in edit_data]
    if missing:
        raise ValueError(
            f"edit_data is missing {', '.join(missing)} — generate a board first.")
    stored = edit_data["params"]
    placements = np.asarray(edit_data["layout"]["placements_xy"], dtype=float)
    removed = {int(i) for i in (edit_data.get("removed_pix") or [])}
    active = [i for i in range(len(placements)) if i not in removed]
    outline = np.asarray(edit_data["outline"], dtype=float)
    cable_y = float(outline[:2, 1].min())

    overrides = {k: v for k, v in (params or {}).items()
                 if k in CaseParams.__dataclass_fields__ and v is not None}
    overrides.pop("bump_params", None)
    cp = CaseParams(**overrides)
    if bump_params:
        bumps = {k: v for k, v in bump_params.items()
                 if k in BumpParams.__dataclass_fields__ and v is not None}
        cp.bump_params = BumpParams(**bumps)

    board = {
        "args": (edit_data.get("final_outline", edit_data["outline"]), cable_y,
                 placements, active,
                 float(stored["pitch_x_mm"]), float(stored["pitch_y_mm"])),
        "active": active,
        "tail_extension": (edit_data.get("layout") or {}).get("tail_extension"),
    }
    return board, cp


@app.post("/case", response_model=CaseResponse)
def case(req: CaseRequest):
    """Two printable case halves for an already-generated board.

    Separate from /bump-sheet rather than folded into it: that endpoint returns
    a single tray-shaped part and is wired to its own UI, and changing it to add
    a second STL would risk a working feature for no gain. The two share their
    geometry through pipeline/case.py importing bump_sheet's helpers, so the
    pocket, the skirt and the cable-edge opening cannot drift apart.

    The shear bumps are a flag on the lid, not a separate part -- a case whose
    top happens to carry bumps, rather than a bump sheet that happens to have a
    case around it.
    """
    try:
        board, cp = _case_inputs(req.edit_data, req.params, req.bump_params)
        result = build_case(*board["args"], cp, tail_extension=board["tail_extension"])

        stats = dict(result.stats)
        stats.update({
            "taxels": len(board["active"]),
            "recess_depth_mm": cp.recess_depth_mm,
            "plate_thickness_mm": cp.plate_thickness_mm,
            "lid_thickness_mm": cp.lid_thickness_mm,
            "skirt_width_mm": cp.skirt_width_mm,
            "bottom_watertight": bool(result.bottom.is_watertight),
            "top_watertight": bool(result.top.is_watertight),
        })
        warn = list(result.warnings)
        for mesh, name in ((result.bottom, "bottom"), (result.top, "top")):
            if not mesh.is_watertight:
                warn.append(f"The {name} piece is not a closed solid — it may not slice cleanly.")
        return CaseResponse(
            bottom_stl_b64=base64.b64encode(result.bottom.export(file_type="stl")).decode("ascii"),
            top_stl_b64=base64.b64encode(result.top.export(file_type="stl")).decode("ascii"),
            stats=stats, warnings=warn)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Case error: {str(e)}")


@app.post("/zip-with-case", response_model=ZipWithCaseResponse)
def zip_with_case_endpoint(req: ZipWithCaseRequest):
    """The board ZIP with the printed case added under case/.

    Called by the Download ZIP button, not by /generate: three lids are three
    boolean-heavy meshes, slow to build on the hosted server, and only a user
    who downloads needs them. See export/case_zip.py.

    Never fails because of the case. A board the case cannot be built for
    still gets its ZIP back, with case_error saying why; only a zip_b64 that
    is not a ZIP is an error (400), and the page then downloads the ZIP it
    already holds.
    """
    try:
        zipfile.ZipFile(io.BytesIO(base64.b64decode(req.zip_b64, validate=True))).close()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"zip_b64 is not a ZIP: {e}")

    warnings: list[str] = []
    try:
        board, cp = _case_inputs(req.edit_data, req.params, req.bump_params)
        built = build_case_files(*board["args"], cp, primary=req.primary, stem=req.name,
                                 tail_extension=board["tail_extension"])
        files, left_out, warnings = built.files, built.left_out, built.warnings
    except Exception as e:  # noqa: BLE001 - the board ZIP must still go out
        files = {}
        left_out = {"case": str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}"}

    def sentence(text):
        text = str(text).strip()
        return text if text.endswith((".", "!", "?")) else text + "."

    if not files:
        error = sentence(f"The printed case was left out: {left_out.get('case', 'it could not be built')}")
    elif left_out:
        error = " ".join(sentence(f"The {k} was left out: {v}") for k, v in left_out.items())
    else:
        error = None
    return ZipWithCaseResponse(
        zip_b64=zip_with_case(req.zip_b64, files),
        case_files=[p for p in files if p.endswith(".stl")],
        case_error=error, warnings=warnings)


@app.post("/redo-padding", response_model=RedoPaddingResponse)
def redo_padding(req: RedoPaddingRequest):
    try:
        edit_data = _merge_param_overrides(req.edit_data, req.params)
        layout, outline_xy, params, removed = _layout_from_edit_data(edit_data)
        params = _without_expansion_padding(params)
        col_routes, row_routes = _routes_from_payload(req.routes)
        resized = False
        if req.connector_pos_override is not None:
            resized = _resize_connector_pads(layout, req.connector_pos_override, [*col_routes, *row_routes])
        # A Tail length typed into the form: the board made again at it, or a
        # plain error saying to press Generate (_relength_routed_tail).
        old_length = float(layout.tail.get("length_mm") or req.edit_data["params"]["cable_length_mm"])
        if abs(float(params.cable_length_mm) - old_length) > 1e-6:
            params = replace(params, cable_length_auto=False)
            _relength_routed_tail(layout, col_routes, row_routes, params, old_length)
        else:
            apply_tail_extension(layout, [*col_routes, *row_routes], params, CONNECTOR_DIR)
            _apply_post_route_containment(layout, col_routes, row_routes, params)
            _capture_pre_follow_outline(layout)
            apply_main_follow_padding(layout, [*col_routes, *row_routes], params)
        # Fires whenever the board carries an enlarged connector, not only when
        # this particular request changed the size -- see
        # _apply_connector_shoulder_if_needed. Must run after every other
        # outline-shaping pass above, since apply_main_follow_padding
        # unconditionally rebuilds layout.board_poly from scratch and would
        # otherwise discard this connector-width shape edit.
        _apply_connector_shoulder_if_needed(
            layout, outline_xy, params, req.connector_pos_override)
        _clip_board_to_connector_tab(layout, outline_xy, params)
        fill_xy = edit_data.get("fill_region")
        svg_text, stats, drc, _ = _outputs_with_connector_tail(
            layout, outline_xy, col_routes, row_routes, removed, params, fill_xy=fill_xy)
        edit_data = _build_edit_data(
            layout, outline_xy, col_routes, row_routes, removed, params, fill_xy)
        return RedoPaddingResponse(svg=svg_text, stats=stats, drc=drc, edit_data=edit_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Redo padding error: {str(e)}")


# =========================================================================
# Re-opening a design that was exported earlier.
#
# manifest.json describes the board that was made, not the inputs that made
# it; the inputs are carried in its "design" block and in design.json (see
# export/design.py). Both records come back through here, which does
# validation and normalisation only -- it never routes. The caller then
# drives the same endpoints a hand-built design uses, so there is no second
# pipeline to keep in sync:
#
#   mode "restore" -> POST edit_data + routes to /export-edited
#   mode "recipe"  -> put outline + params in the form and POST /generate
# =========================================================================

def _recipe_params(params: dict, router: str, connector_pos: int | None) -> dict:
    """Stored params as a /generate body fragment.

    Filtered against GenerateRequest so a field dropped from the API in a
    later version is ignored instead of 422-ing the whole import, and so a
    hand-edited design file cannot inject unknown keys.
    """
    allowed = set(GenerateRequest.model_fields) - {"outline", "fill_region"}
    body = {k: v for k, v in (params or {}).items() if k in allowed}
    body["router"] = router if router in ("legacy", "hug") else "legacy"
    # A tail length the pipeline chose is chosen again, for whatever this
    # pipeline builds; a typed one is kept.
    if (params or {}).get("cable_length_auto"):
        body["cable_length_mm"] = None
    # Replayed as an override so a connector the user upsized by hand comes
    # back at that size rather than dropping to the smallest that fits.
    if connector_pos:
        body["connector_pos_override"] = int(connector_pos)
    # target_cols/target_rows are deliberately not replayed: the solved taxel
    # size and pitch are already in this dict, so re-solving them would be a
    # second answer to a question that has one. See params_to_dict.
    body.pop("target_cols", None)
    body.pop("target_rows", None)
    return body


@app.post("/open-design", response_model=OpenDesignResponse)
def open_design(req: OpenDesignRequest):
    try:
        data = req.file or {}
        if not isinstance(data, dict):
            raise ValueError("Not a JSON object.")

        schema = data.get("schema") if "edit_data" in data else (data.get("design") or {}).get("schema")
        if schema is not None and int(schema) > DESIGN_SCHEMA:
            raise ValueError(
                f"This design was written by a newer version of OpenTac-CAD "
                f"(design schema {schema}, this build reads {DESIGN_SCHEMA}). Update the app.")

        warnings: list[str] = []

        # --- a design recovered from a pre-design-block manifest ---
        # Geometry only: no nets, no pads, no routes, so /export-edited and the
        # route editor cannot run on it. Handled before the save-file branch
        # because it is a design.json shape and would otherwise be sent down a
        # restore path that needs a layout it does not have.
        if (data.get("edit_data") or {}).get("geometry_only") or is_legacy_manifest(data):
            manifest = data if is_legacy_manifest(data) else None
            if manifest is not None:
                edit_data, notes = recover_legacy(manifest)
                label = "recovered design"
            else:
                edit_data, notes = data["edit_data"], []
                label = data.get("label") or "recovered design"
            warnings.extend(notes)
            warnings.append(
                "This design was exported before OpenTac-CAD stored its settings, so only "
                "the board itself could be recovered: its outline, its cut edge and where every "
                "taxel sits, exactly as manufactured. You can build a case or a bump sheet from "
                "it. Route editing and a new Gerber/KiCad export are not available, because the "
                "wiring was never in the file -- re-routing it would give you copper that does "
                "not match the board you already had made.")
            return OpenDesignResponse(
                mode="geometry",
                label=label,
                outline=edit_data["outline"],
                params={},
                fill_region=None,
                edit_data=edit_data,
                routes=[],
                svg=geometry_svg(edit_data),
                warnings=warnings,
            )

        # --- design.json: the exact-restore save file ---
        if "edit_data" in data:
            edit_data = data.get("edit_data") or {}
            missing = [k for k in ("params", "layout", "outline") if k not in edit_data]
            if missing:
                raise ValueError(
                    f"design.json is missing {', '.join(missing)} — it is incomplete or not an "
                    "OpenTac-CAD design file.")
            routes = data.get("routes") or edit_data.get("routes") or []
            stored_pipeline = data.get("pipeline")
            router = (edit_data.get("layout", {}).get("tail") or {}).get("router", "legacy")
            recipe = _recipe_params(
                edit_data.get("params"), router,
                (edit_data.get("layout") or {}).get("connector_pos"))
            # A stored layout and stored route polylines are only meaningful
            # against the pipeline that produced them: replaying them on a
            # different one would export a board that never got previewed.
            # Fall back to the recipe, which is version-independent, instead
            # of restoring something unverifiable.
            stale = stored_pipeline != PIPELINE_VERSION
            if stale:
                warnings.append(
                    f"This design was saved by pipeline {stored_pipeline or 'unknown'} and this "
                    f"build is {PIPELINE_VERSION}, so the exact board (including any hand-edited "
                    "routes) cannot be restored. Re-generating it from its saved settings instead.")
            mode = "recipe" if (stale or req.prefer == "recipe") else "restore"
            return OpenDesignResponse(
                mode=mode,
                label=data.get("label") or "imported design",
                outline=edit_data["outline"],
                params=recipe,
                fill_region=edit_data.get("fill_region"),
                edit_data=edit_data if mode == "restore" else None,
                routes=routes if mode == "restore" else None,
                warnings=warnings,
            )

        # --- manifest.json: the recipe ---
        design = data.get("design")
        if not design:
            # is_legacy_manifest already claimed anything with taxels, above.
            raise ValueError("Not an OpenTac-CAD manifest.json or design.json.")

        outline = data.get("outline") or []
        if len(outline) < 3:
            raise ValueError("This manifest has no usable outline.")
        return OpenDesignResponse(
            mode="recipe",
            label=data.get("label") or "imported design",
            outline=outline,
            params=_recipe_params(design.get("params"), design.get("router", "legacy"),
                                  design.get("connector_pos")),
            fill_region=design.get("fill_region"),
            warnings=warnings,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Design import error: {str(e)}")





# =========================================================================
# Server-side STL/3D mesh unrolling, backed by pipeline/mesh_unroll.py.
#
# The web UI unfolds a connected selection in the browser (web/face-select.js)
# and calls /unroll-mesh-chain (from web/stl-viewer.js) to join disconnected
# selections: strut-connected multi-region chains and manual corner picking
# are computed here. /unroll-mesh and /grow-region are API-only.
# =========================================================================

@app.post("/unroll-mesh", response_model=UnrollMeshResponse)
async def unroll_mesh(
    file: UploadFile = File(...),
    click_face_index: int = Form(...),
    existing_faces: str = Form(""),  # comma-separated face indices from a prior
                                      # shift-click selection to extend, or "" for fresh
    angle_tol_deg: float = Form(2.0),
    # Optional manual strut connection points, from the "connect corners
    # yourself" flow: original mesh vertex indices, strut 1 connects
    # connect_v_a1 (on the prior/first-clicked region) to connect_v_b1 (on
    # the new/second-clicked region), strut 2 connects connect_v_a2 to
    # connect_v_b2. When all four are given, they replace the automatic
    # corner-selection heuristic entirely for this request. Only meaningful
    # when the two regions don't touch on the mesh (the strut-needed case).
    connect_v_a1: int | None = Form(None),
    connect_v_b1: int | None = Form(None),
    connect_v_a2: int | None = Form(None),
    connect_v_b2: int | None = Form(None),
    # Echoed back from a previous strut-connected response's `outline` /
    # `outline_vertex_indices` (see UnrollMeshResponse) to extend a running
    # chain with a 3rd (or later) region — prior_outline is a JSON [[x,y],...]
    # array, prior_outline_vertices is the parallel comma-separated original
    # mesh vertex indices. Present only once the selection already has at
    # least one strut connection in it.
    prior_outline: str = Form(""),
    prior_outline_vertices: str = Form(""),
):
    import json
    import trimesh
    from opentac.cad.pipeline.mesh_unroll import (
        grow_coplanar_region, unfold_face_region, check_developability,
        find_boundary_loops, extract_outline, _loop_signed_area,
        _build_face_adjacency, connect_regions_with_strut, connect_loop_to_region,
        _edge_key, region_boundary_loop,
    )

    content = await file.read()
    if len(content) > MAX_STL_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="STL file too large (50MB limit).")

    try:
        mesh = trimesh.load(io.BytesIO(content), file_type="stl")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse STL file: {e}")

    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    normals = np.array(mesh.face_normals)

    # click_face_index comes directly from the three.js raycast hit
    # (hit.faceIndex) on the client. This lines up 1:1 with trimesh's face
    # array: trimesh's default vertex-welding changes vertex indices/count
    # but preserves the raw STL triangle order and count exactly, so the two
    # loaders agree on "triangle N" even though STLLoader keeps unwelded
    # per-face vertices and trimesh doesn't. A nearest-centroid-to-click-point
    # lookup would be unreliable on complex/thin geometry (e.g. close
    # double-wall surfaces), where a nearby wrong triangle's centroid can be
    # closer to the click point than the clicked triangle's own centroid.
    if not (0 <= click_face_index < len(faces)):
        raise HTTPException(status_code=400, detail="click_face_index out of range for this mesh.")
    start_face = click_face_index

    new_region = grow_coplanar_region(faces, normals, start_face, angle_tol_deg, vertices=vertices)
    prior = set(int(x) for x in existing_faces.split(",") if x.strip())
    edge_faces = _build_face_adjacency(faces)

    def touches(a: set, b: set) -> bool:
        for fi in a:
            tri = faces[fi]
            for i in range(3):
                key = _edge_key(tri[i], tri[(i + 1) % 3])
                for nfi in edge_faces.get(key, []):
                    if nfi in b:
                        return True
        return False

    warnings = []
    manual_pairs = None
    if None not in (connect_v_a1, connect_v_b1, connect_v_a2, connect_v_b2):
        manual_pairs = ((connect_v_a1, connect_v_b1), (connect_v_a2, connect_v_b2))

    chain_verts = [int(x) for x in prior_outline_vertices.split(",") if x.strip()]
    chain_pts = json.loads(prior_outline) if prior_outline.strip() else None

    if chain_pts is not None and chain_verts:
        # Extending an already strut-connected chain onto a 3rd (or later)
        # region. Once a selection needs its first strut connection, the
        # resulting outline is not a real mesh region (some of its
        # boundary is a synthetic strut edge, not real mesh connectivity),
        # so it can't be re-derived from `existing_faces` the way a pure
        # mesh-union chain can — the caller echoes the previous response's
        # own outline back instead (see UnrollMeshResponse.outline_vertex_indices).
        # Always strut-connect here, even if this particular click happens
        # to also touch the mesh — simpler and consistent with already
        # being in "approximate" territory, rather than special-casing a
        # mix of real touches and approximated struts within one chain.
        loop1_orig = chain_verts
        loop1_uv = [np.array(p) for p in chain_pts]
        try:
            outline, outline_vidx = connect_loop_to_region(
                vertices, faces, loop1_orig, loop1_uv, new_region, start_face,
                connection_pairs=manual_pairs)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        warnings.append(
            "These two surfaces aren't directly connected on the mesh "
            "(e.g. a rounded or chamfered corner between them) — joined with "
            "a straight cut as an approximation. Check the preview before generating."
            if manual_pairs is None else
            "These two surfaces were joined with a straight cut using the "
            "corners you picked. Check the preview before generating."
        )
        disagreement = 0.0
        region_out = prior | new_region

        loop2_orig, _, p2_3d = region_boundary_loop(vertices, faces, new_region, start_face)
        strut_loop1 = [[float(v)] + vertices[v].tolist() for v in loop1_orig]
        strut_loop2 = [[float(v)] + p2_3d[k].tolist() for k, v in enumerate(loop2_orig)]

        return UnrollMeshResponse(
            outline=outline.tolist(),
            outline_vertex_indices=[int(v) for v in outline_vidx],
            region_face_count=len(region_out),
            region_face_indices=[int(f) for f in region_out],
            developability_disagreement_mm=disagreement,
            warnings=warnings,
            strut_loop1=strut_loop1,
            strut_loop2=strut_loop2,
        )

    elif not prior or touches(prior, new_region):
        # First click, or a shift-click landing on/adjacent to the existing
        # selection (a real sharp corner) — plain set union already
        # connects them; unfold_triangle_patch doesn't care about the angle
        # between facets at all, only that they're connected via a shared
        # edge, so a folded selection unfolds with the same zero-distortion
        # guarantee as a single flat face.
        region = prior | new_region
        uv, visited_local, orig_idx = unfold_face_region(vertices, faces, region, start_face)
        region_list = [start_face] + [f for f in region if f != start_face]
        if prior and not prior.issubset({region_list[i] for i in visited_local}):
            raise HTTPException(
                status_code=400,
                detail="This click isn't connected to the current selection on "
                       "this mesh — pick a point on the same connected surface.",
            )
        region_faces_orig = faces[region_list]
        _, inv = np.unique(region_faces_orig, return_inverse=True)
        sub_faces = inv.reshape(-1, 3)
        sub_vertices = vertices[orig_idx]
        disagreement = check_developability(sub_vertices, sub_faces, uv, visited_local)
        if disagreement > 1.0:
            warnings.append(
                f"Selected patch may not be fully flattenable — {disagreement:.2f}mm "
                f"disagreement detected. Check the preview before generating."
            )
        try:
            outline = extract_outline(sub_faces, uv)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        loops = find_boundary_loops(sub_faces)
        if len(loops) > 1:
            areas = sorted((abs(_loop_signed_area(loop, uv)) for loop in loops), reverse=True)
            if areas[0] > 0 and areas[1] / areas[0] > 0.5:
                warnings.append(
                    "Multiple similarly-sized boundary loops found — the outline "
                    "may be incomplete. Check the preview carefully before generating."
                )
            else:
                n_holes = len(loops) - 1
                warnings.append(
                    f"{n_holes} hole{'s' if n_holes != 1 else ''} detected in the "
                    f"selection and ignored (outline uses the outer perimeter only)."
                )
        region_out = region

    else:
        # The new click doesn't touch the existing selection — e.g. two
        # flat faces separated by a filleted/rounded corner, whose thin,
        # continuously-curving triangles are correctly excluded from both
        # by grow_coplanar_region's angle tolerance. Approximate the real
        # (possibly irregular, hard-to-select-reliably) connecting
        # surface with a straight cut between the two regions' nearest
        # corners instead of trying to select and flatten it: each region
        # unfolds cleanly on its own regardless of how complex the true
        # gap geometry is, which is what actually matters for a usable
        # board outline.
        #
        # `prior` may already span more than one connected component if
        # it was itself built via an earlier strut connection; only the
        # component nearest `new_region` gets connected this round; any
        # other, farther components stay in the selection for highlighting
        # but aren't reflected in this specific outline.
        visited = set()
        components = []
        for f in prior:
            if f in visited:
                continue
            comp = {f}
            queue = [f]
            while queue:
                fi = queue.pop()
                tri = faces[fi]
                for i in range(3):
                    key = _edge_key(tri[i], tri[(i + 1) % 3])
                    for nfi in edge_faces.get(key, []):
                        if nfi in prior and nfi not in comp:
                            comp.add(nfi)
                            queue.append(nfi)
            visited |= comp
            components.append(comp)

        centroids = vertices[faces].mean(axis=1)
        new_centroid = centroids[list(new_region)].mean(axis=0)
        prior_component = min(components, key=lambda c: np.linalg.norm(
            centroids[list(c)].mean(axis=0) - new_centroid))
        prior_start_face = next(iter(prior_component))

        try:
            outline, outline_vidx = connect_regions_with_strut(
                vertices, faces, prior_component, prior_start_face, new_region, start_face,
                connection_pairs=manual_pairs)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        warnings.append(
            "These two surfaces aren't directly connected on the mesh "
            "(e.g. a rounded or chamfered corner between them) — joined with "
            "a straight cut as an approximation. Check the preview before generating."
            if manual_pairs is None else
            "These two surfaces were joined with a straight cut using the "
            "corners you picked. Check the preview before generating."
        )
        disagreement = 0.0
        region_out = prior | new_region

        # Always include both loops' boundaries (not just when auto-picking
        # is used) so a client can offer "connect corners manually" as
        # a follow-up on *this* response without a second request just to
        # fetch the boundary data.
        loop1_orig, _, p1_3d = region_boundary_loop(vertices, faces, prior_component, prior_start_face)
        loop2_orig, _, p2_3d = region_boundary_loop(vertices, faces, new_region, start_face)
        strut_loop1 = [[float(v)] + p1_3d[k].tolist() for k, v in enumerate(loop1_orig)]
        strut_loop2 = [[float(v)] + p2_3d[k].tolist() for k, v in enumerate(loop2_orig)]

        return UnrollMeshResponse(
            outline=outline.tolist(),
            outline_vertex_indices=[int(v) for v in outline_vidx],
            region_face_count=len(region_out),
            region_face_indices=[int(f) for f in region_out],
            developability_disagreement_mm=disagreement,
            warnings=warnings,
            strut_loop1=strut_loop1,
            strut_loop2=strut_loop2,
        )

    return UnrollMeshResponse(
        outline=outline.tolist(),
        region_face_count=len(region_out),
        region_face_indices=[int(f) for f in region_out],
        developability_disagreement_mm=disagreement,
        warnings=warnings,
    )


@app.post("/grow-region", response_model=GrowRegionResponse)
async def grow_region(
    file: UploadFile = File(...),
    click_face_index: int = Form(...),
    angle_tol_deg: float = Form(2.0),
):
    """Lightweight, connection-free companion to /unroll-mesh: just grows and
    returns one clicked face's coplanar region, for live per-click
    highlighting while the user is still building up a multi-face
    selection (see /unroll-mesh-chain) — no strut/union math at all.
    """
    import trimesh
    from opentac.cad.pipeline.mesh_unroll import grow_coplanar_region

    content = await file.read()
    if len(content) > MAX_STL_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="STL file too large (50MB limit).")
    try:
        mesh = trimesh.load(io.BytesIO(content), file_type="stl")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse STL file: {e}")

    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    normals = np.array(mesh.face_normals)
    if not (0 <= click_face_index < len(faces)):
        raise HTTPException(status_code=400, detail="click_face_index out of range for this mesh.")

    region = grow_coplanar_region(faces, normals, click_face_index, angle_tol_deg, vertices=vertices)
    return GrowRegionResponse(region_face_indices=[int(f) for f in region])


@app.post("/unroll-mesh-chain", response_model=UnrollMeshChainResponse)
async def unroll_mesh_chain(
    file: UploadFile = File(...),
    click_face_indices: str = Form(...),  # comma-separated, in click order
    # JSON {"<connection index>": [[v_a1, v_b1], [v_a2, v_b2]], ...} — a
    # sparse map from a 0-based gap index (click k <-> click k+1 is index k)
    # to manually-picked corner pairs, for any connection the user fixed.
    # Omitted entries use the automatic same/crossed heuristic.
    connection_overrides: str = Form(""),
    # JSON list of per-click triangle-index lists, parallel to
    # click_face_indices. When given, click k's region is exactly
    # click_regions[k] (the caller's own selection — e.g. the web UI's
    # curved smooth-growth patches from web/face-select.js, which
    # grow_coplanar_region cannot reproduce: it only compares normals against
    # the clicked seed face, so a click can never select a patch that curves
    # away from its seed). Each list must contain its own click face index
    # and form one edge-connected patch. Empty string: each click's region
    # is grown with grow_coplanar_region.
    click_regions: str = Form(""),
    # When true, corner picking is mandatory: a strut-needed connection
    # without an entry in connection_overrides stops the walk and returns
    # that connection's marker loops with `pending_connection` set (outline
    # empty) instead of running the automatic corner heuristic. The caller
    # loops: pick corners, resend, until pending_connection is null.
    corners_first: bool = Form(False),
    angle_tol_deg: float = Form(2.0),
):
    """Computes an entire multi-face selection's outline in one request,
    rather than incrementally per click. Auto-connecting after every click
    would fix each earlier connection in place, so manual corner picking
    could only correct the *most recent* connection, and the running chain
    would have no per-origin identity for coloring: a multi-region chain's
    boundary would render in a single color in the corner-picker.
    This endpoint instead takes the *whole* ordered click list plus an
    optional sparse map of manual corner overrides per connection, and
    returns the final outline plus, for each connection that needed a strut,
    that pair's own two boundary loops (each point tagged with a `group_id`
    — see ChainConnection) so a "fix this connection" UI can color each
    original region distinctly, however deep into the chain it is. The
    caller only resends this same whole-chain request (with an updated
    overrides map) to apply a fix; there is no incremental replay state.
    """
    import json
    import trimesh
    from opentac.cad.pipeline.mesh_unroll import (
        grow_coplanar_region, unfold_face_region, check_developability,
        find_boundary_loops, extract_outline, _loop_signed_area,
        _build_face_adjacency, connect_loop_to_region, _edge_key,
        region_boundary_loop, simplify_outline_points, _reduce_boundary_loop,
        remove_needle_spikes,
    )

    content = await file.read()
    if len(content) > MAX_STL_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="STL file too large (50MB limit).")
    try:
        mesh = trimesh.load(io.BytesIO(content), file_type="stl")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse STL file: {e}")

    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    normals = np.array(mesh.face_normals)

    clicks = [int(x) for x in click_face_indices.split(",") if x.strip()]
    if not clicks:
        raise HTTPException(status_code=400, detail="No faces selected.")
    for c in clicks:
        if not (0 <= c < len(faces)):
            raise HTTPException(status_code=400, detail=f"click face index {c} out of range for this mesh.")

    overrides_raw = json.loads(connection_overrides) if connection_overrides.strip() else {}
    overrides = {int(k): v for k, v in overrides_raw.items()}

    edge_faces = _build_face_adjacency(faces)

    def touches(a: set, b: set) -> bool:
        for fi in a:
            tri = faces[fi]
            for i in range(3):
                key = _edge_key(tri[i], tri[(i + 1) % 3])
                for nfi in edge_faces.get(key, []):
                    if nfi in b:
                        return True
        return False

    if click_regions.strip():
        try:
            regions_raw = json.loads(click_regions)
        except ValueError:
            raise HTTPException(status_code=400, detail="click_regions is not valid JSON.")
        if not isinstance(regions_raw, list) or len(regions_raw) != len(clicks):
            raise HTTPException(
                status_code=400,
                detail="click_regions must be one face-index list per click face index.")
        regions = []
        for k, (c, raw) in enumerate(zip(clicks, regions_raw)):
            try:
                region = {int(f) for f in raw}
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail=f"click_regions[{k}] is not a list of face indices.")
            if not region:
                raise HTTPException(status_code=400, detail=f"click_regions[{k}] is empty.")
            for f in region:
                if not (0 <= f < len(faces)):
                    raise HTTPException(
                        status_code=400,
                        detail=f"click_regions[{k}] face index {f} out of range for this mesh.")
            if c not in region:
                raise HTTPException(
                    status_code=400,
                    detail=f"click face index {c} is not inside click_regions[{k}].")
            # The unfold walks edge-adjacency from the click face; a region
            # with detached pieces would silently lose them from the outline.
            reached = {c}
            stack = [c]
            while stack:
                fi = stack.pop()
                tri = faces[fi]
                for i in range(3):
                    key = _edge_key(tri[i], tri[(i + 1) % 3])
                    for nfi in edge_faces.get(key, []):
                        if nfi in region and nfi not in reached:
                            reached.add(nfi)
                            stack.append(nfi)
            if reached != region:
                raise HTTPException(
                    status_code=400,
                    detail=f"click_regions[{k}] is not a single connected surface "
                           f"({len(region) - len(reached)} of its faces don't connect to the click face).")
            regions.append(region)
    else:
        regions = [grow_coplanar_region(faces, normals, c, angle_tol_deg, vertices=vertices) for c in clicks]

    warnings = []
    # Per-surface developability check, computed on each click's own region
    # before any joining: near-zero means that surface flattens exactly; a
    # large value means real double curvature (e.g. a domed fingertip tip) and
    # the flat sensor will not lie on it without distortion.
    region_developability = []
    for k, (c, region) in enumerate(zip(clicks, regions)):
        uv_r, visited_r, orig_idx_r = unfold_face_region(vertices, faces, region, c)
        region_list_r = [c] + [f for f in region if f != c]
        _, inv_r = np.unique(faces[region_list_r], return_inverse=True)
        sub_faces_r = inv_r.reshape(-1, 3)
        d_r = float(check_developability(vertices[orig_idx_r], sub_faces_r, uv_r, visited_r))
        region_developability.append(d_r)
        if d_r > 1.0:
            warnings.append(
                f"Surface {k + 1} may not be fully flattenable — {d_r:.2f}mm "
                f"disagreement detected. Consider selecting a flatter piece of it."
            )
    current_faces = set(regions[0])
    current_start_face = clicks[0]
    # These three stay None as long as the chain is still a real,
    # re-unfoldable mesh region (only touching-merges so far). The first
    # strut connection flattens it into an explicit loop + per-vertex group
    # map, exactly like /unroll-mesh's chain-extension branch.
    current_loop_orig = None
    current_loop_uv = None
    current_group_of = {}
    current_group_id = 0
    connections = []

    for k in range(1, len(clicks)):
        new_region = regions[k]
        new_start = clicks[k]
        override = overrides.get(k - 1)
        # Once any strut has happened, always strut-connect from then on —
        # consistent with /unroll-mesh's chain branch (simpler than
        # special-casing a mix of real touches and approximated struts).
        do_touch = current_loop_orig is None and touches(current_faces, new_region) and override is None

        if do_touch:
            current_faces = current_faces | new_region
            connections.append(ChainConnection(needs_strut=False))
            continue

        if current_loop_orig is None:
            current_loop_orig, current_loop_uv, _ = region_boundary_loop(
                vertices, faces, current_faces, current_start_face)
            current_group_of = {v: current_group_id for v in current_loop_orig}

        new_group_id = current_group_id + 1
        loop2_orig, loop2_uv, _ = region_boundary_loop(vertices, faces, new_region, new_start)

        # Display-only reduced copies for the corner-picker's clickable
        # markers — current_loop_orig/loop2_orig (used below for the
        # actual connection math) stay full-resolution; reducing *those*
        # would break the automatic heuristic (see region_boundary_loop's
        # docstring), but the picker only needs a handful of well-
        # separated real corners, and vertices[v] works for any original
        # mesh vertex index regardless of which (full or reduced) loop it
        # came from, so no separate position bookkeeping is needed here.
        disp_a_orig, _ = _reduce_boundary_loop(current_loop_orig, current_loop_uv)
        disp_b_orig, _ = _reduce_boundary_loop(loop2_orig, loop2_uv)
        loop_a = [[float(v)] + vertices[v].tolist() + [float(current_group_of.get(v, current_group_id))]
                  for v in disp_a_orig]
        loop_b = [[float(v)] + vertices[v].tolist() + [float(new_group_id)] for v in disp_b_orig]

        connection_pairs = None
        if override:
            (a1, b1), (a2, b2) = override
            connection_pairs = ((int(a1), int(b1)), (int(a2), int(b2)))

        if corners_first and connection_pairs is None:
            # Corner picking is a required step in this mode: stop at the
            # first unresolved cut and hand back its two marker loops so the
            # user picks the joining corners — the automatic same/crossed
            # heuristic never runs. Later connections can't even be
            # described yet (their chain-side loop depends on this join),
            # hence one pending connection at a time.
            connections.append(ChainConnection(needs_strut=True, loop_a=loop_a, loop_b=loop_b))
            all_faces = set()
            for r in regions:
                all_faces |= r
            return UnrollMeshChainResponse(
                outline=[],
                region_face_count=len(all_faces),
                region_face_indices=[int(f) for f in all_faces],
                warnings=warnings,
                connections=connections,
                region_developability_mm=region_developability,
                pending_connection=k - 1,
            )

        try:
            outline, outline_vidx = connect_loop_to_region(
                vertices, faces, current_loop_orig, current_loop_uv, new_region, new_start,
                connection_pairs=connection_pairs)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Connection {k}↔{k + 1}: {e}")

        warnings.append(
            f"Connection {k}↔{k + 1}: these surfaces aren't directly connected on the "
            "mesh — joined with a straight cut as an approximation. Check the preview before generating."
            if connection_pairs is None else
            f"Connection {k}↔{k + 1}: joined with a straight cut using the corners you picked."
        )

        current_group_of = {v: current_group_of.get(v, new_group_id) for v in outline_vidx}
        current_faces = current_faces | new_region
        current_loop_orig = outline_vidx
        current_loop_uv = [np.array(p) for p in outline]
        current_group_id = new_group_id
        current_start_face = new_start

        connections.append(ChainConnection(needs_strut=True, loop_a=loop_a, loop_b=loop_b))

    if current_loop_orig is None:
        # Never needed a strut — a pure mesh-union chain, unfold + extract
        # fresh (identical to /unroll-mesh's touching branch).
        uv, visited_local, orig_idx = unfold_face_region(vertices, faces, current_faces, current_start_face)
        region_list = [current_start_face] + [f for f in current_faces if f != current_start_face]
        region_faces_orig = faces[region_list]
        _, inv = np.unique(region_faces_orig, return_inverse=True)
        sub_faces = inv.reshape(-1, 3)
        sub_vertices = vertices[orig_idx]
        disagreement = check_developability(sub_vertices, sub_faces, uv, visited_local)
        if disagreement > 1.0:
            warnings.append(
                f"Selected patch may not be fully flattenable — {disagreement:.2f}mm "
                f"disagreement detected. Check the preview before generating."
            )
        try:
            final_outline = extract_outline(sub_faces, uv)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        loops = find_boundary_loops(sub_faces)
        if len(loops) > 1:
            areas = sorted((abs(_loop_signed_area(loop, uv)) for loop in loops), reverse=True)
            if areas[0] > 0 and areas[1] / areas[0] > 0.5:
                warnings.append(
                    "Multiple similarly-sized boundary loops found — the outline "
                    "may be incomplete. Check the preview carefully before generating."
                )
            else:
                n_holes = len(loops) - 1
                warnings.append(
                    f"{n_holes} hole{'s' if n_holes != 1 else ''} detected in the "
                    f"selection and ignored (outline uses the outer perimeter only)."
                )
    else:
        # extract_outline() already simplifies the pure-touching case above;
        # a strut-assembled outline never goes through extract_outline (it's
        # built directly from two loops' "far" arcs plus straight struts),
        # so it never gets that cleanup — apply it here instead. Safe here
        # (unlike current_loop_orig/current_loop_uv mid-chain, which stay
        # full-resolution — see region_boundary_loop's docstring) because
        # nothing downstream needs per-point vertex identity once the whole
        # chain is assembled into its final form.
        final_outline = simplify_outline_points(np.array(current_loop_uv))

    # Needle spikes (out-and-back sub-tolerance-width protrusions from real
    # sliver triangles at a region boundary) can survive both branches'
    # simplification — Douglas-Peucker sees the spike length as deviation
    # worth keeping. Strip them from the final outline only; the pure-
    # touching branch needs a re-simplify afterwards to merge the
    # near-coincident neighbor pair each removed spike leaves behind.
    final_outline = simplify_outline_points(remove_needle_spikes(final_outline))

    return UnrollMeshChainResponse(
        outline=final_outline.tolist(),
        region_face_count=len(current_faces),
        region_face_indices=[int(f) for f in current_faces],
        warnings=warnings,
        connections=connections,
        region_developability_mm=region_developability,
    )


class DecodeBackmarkRequest(BaseModel):
    payload: str


class DecodeBackmarkResponse(BaseModel):
    manifest: dict
    absent: list


@app.post("/decode-backmark", response_model=DecodeBackmarkResponse)
def decode_backmark(req: DecodeBackmarkRequest):
    """Scanned back-silkscreen QR text -> a recovered manifest.json.

    The other end of export/silk_backmark.py: someone finds a sensor with no
    files, points a phone at the code on its back, and pastes what comes off it.
    What comes back is manifest-shaped so it drops into anything that already
    reads manifest.json, with `absent` naming the four keys the QR cannot carry.

    The payload is base41, whose alphabet deliberately excludes the four QR
    characters that get mangled silently -- space, '%', '+' and '*' -- so
    trimming, URL-decoding or an unquoted shell argument cannot turn it into a
    different valid payload. A CRC-8 catches whatever is left, because a
    plausible manifest with the wrong pin mapping is the worst thing this can
    return.
    """
    from opentac.cad.export.taxel_core import ABSENT, TaxelCoreError, manifest_from_scan

    try:
        manifest = manifest_from_scan(req.payload)
    except TaxelCoreError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="could not decode: %s" % exc)
    return DecodeBackmarkResponse(manifest=manifest, absent=list(ABSENT))


# Plain-English design chat. Registered before the frontend mount below,
# because that mount claims "/" and would otherwise swallow these routes.
#
# Guarded on purpose: chat is an optional extra, and a missing google-genai or
# an unset GEMINI_API_KEY must leave the rest of the app working exactly as it
# did. The frontend asks /api/chat/status and simply hides the Describe mode
# when it is not available.
try:
    from opentac.cad.chat import router as chat_router

    app.include_router(chat_router)
    logger.info("design assistant routes registered")
except Exception as _chat_exc:      # noqa: BLE001 - never break the site
    # Without a key the chat is off by design (a local run, a script); say
    # something only when a key was given but the chat still could not load.
    if os.environ.get("GEMINI_API_KEY"):
        print(f"[chat] disabled: {type(_chat_exc).__name__}: {_chat_exc} "
              "(install the chat's packages: uv run --extra demo opentac cad)")


# Serve the web frontend (index.html, JS, vendored Three.js) at the site root.
# Mounted after the API routes above so /health and /generate keep priority.
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "web")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
