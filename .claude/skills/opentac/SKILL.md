---
name: opentac
description: Design a tactile sensor PCB from a plain-English description — "a sensor shaped like a size 9 insole with the cable at the heel". Builds and validates the outline, runs the OpenTac-CAD pipeline locally, and returns KiCad files and Gerbers. Use for any tactile sensor, taxel array, pressure-sensing PCB, or sensor built from a described shape, a DXF outline, or a surface of an STL part.
---

# OpenTac-CAD

Turn a described sensor into a manufacturable PCB: a grid of taxels
packed into a shape, wired to an FPC connector, exported as KiCad + Gerbers.

## Setup

Everything runs locally, in process. There is no server to start.

**Run Python through the repo's environment with uv**, from the repository root:

```bash
uv run python my_sensor.py
```

The first `uv run` installs the pinned dependencies (a few minutes, once); it is
the same environment the app itself runs on, so anything that works here works in
the app. Do not use a bare `python3` or `py`: a system interpreter almost
certainly lacks these packages, and the error looks like a code fault rather than
a missing environment.

Put this skill's `helpers/` on `sys.path` and call the helpers. They call
`opentac.cad.main.generate()` directly — **never** start a server and never use
HTTP. The `/generate` route's handler holds the board-mode/router policy, the
fixed-outline retry loop and the post-route smoothing; calling
`opentac.cad.pipeline` modules directly would mean reimplementing all of it.

## The flow

1. **Get the outline.** Three sources:
   - *Described shape* — research the real dimensions (below), then build it
     with `helpers/outline.py`.
   - *DXF* — `importers.dxf.parse_dxf_outline()`. If they have real CAD, use
     it; it is more accurate than anything constructed from a description.
   - *STL surface* — `helpers/mesh.py` enumerates the part's flat surfaces,
     scores each for how well it flattens, and unrolls the chosen one. No
     clicking and no browser required.
2. **Place the cable.** Ask where the connector should exit if they did not
   say. This is a physical decision they care about, and it also decides the
   grid orientation.
3. **Define the taxels.** They may say a density, a count, a size, or nothing.
   Converse — see "Taxel sizing".
4. **Choose board_mode and copper layers, then preflight, then generate.**
   Grow mode unless the outline is a hard limit; 4 layers when packing or
   coverage is the priority. The router is always "hug", which is what
   `helpers/generate.py` sends in both modes — see "Board mode and router"
   and "Readout board and copper layers" below.
5. **Customize the tail** only if they want something beyond what the export
   already does. Every export already lands traces on the connector's pad
   drills automatically — see "Tail and connector" below.
6. **Report** what was actually built.

## Rules

**Look up real dimensions. Never recall them.** Web search gives authoritative
numbers and exposes the modelling question. For a size 9 insole it turns up a
foot length of ~254mm against an internal shoe length of 255-281mm, and a ball
width averaging 94.4mm across an 86-102mm range. Which one the sensor should
match is a real question — ask it. Guessing hides it.

**The cable edge must be a real edge.** Outline edge 0->1 is the cable edge and
`rotate_for_cable()` spins the whole board to lay it horizontal. Sample a
smooth shape naively and edge 0->1 is a short chord at an arbitrary angle: the
board rotates by that angle, the axis-aligned taxel grid skews across it, and
routing degrades severely. On a 270mm insole, a 20.4 degree accidental rotation
costs 40% of the taxels, widens the board from 100mm to 147mm, and produces 61
DRC violations. Always use `outline.with_flat_cable_edge()`, which
cuts an explicit flat first edge across wherever the connector exits.

**Wind the outline counter-clockwise.** `pipeline.geometry.ensure_ccw()` pins
vertex 0 and reverses the rest, so a clockwise polygon does not merely flip
winding — it silently swaps your cable edge for the *other* edge at vertex 0.
The board generates fine with the tail in the wrong place. `outline.orient_ccw()`
handles this; `outline.validate()` catches it.

**Smooth and resample — but only shapes that are actually smooth.** Linear
interpolation between hand-placed control points leaves visible faceting, and
`outline.resample_closed()` splines it away. Do *not* spline geometry that
already has real straight edges: a spline through the 8 vertices of a face
unrolled from an STL overshoots its corners and grows the board from 28x79mm to
51x83mm. `resample_closed()` guards against this by falling back to linear
whenever the curve leaves the source outline's envelope, so DXF and STL
outlines pass through unharmed — but pass `smooth=False` when you know the
shape is polygonal.

**Spend control points where the outline turns.** Splining is not a substitute
for describing the shape: `resample_closed()` interpolates what it is given, so
too few points produce a smooth curve that is still the wrong curve — a heel
that reads as a semicircle, an arch that barely dips. An 11-point insole comes
out 269mm long instead of 254mm and loses the arch; the worked example uses 23,
clustered where the curvature changes fastest. No single count suits every
shape: a rounded rectangle or a simple oval is fine with a handful, and
something with several distinct curves — an insole, a hand, a fingertip —
wants 20 or so. Judge it by whether the result still looks like the thing.

**Validate before generating.** `outline.validate()` and `preflight.preflight()`
cost milliseconds; the generate they guard takes about 1-6 s in grow mode and
10-45 s in fixed mode on a laptop.

## Board mode and router

**The router is not a choice. Always use `"hug"`.** It works in both board
modes — `helpers/generate.py::generate_board()` passes it by default, the
site always sends it, and `GenerateRequest.router` defaults to it too.

`"legacy"` is a simpler row/column router, available for comparison. Do not
design with it: on a concave C-shape under `fixed_keepout` it gives 6 taxels
and 25 DRC violations, 15 of them crossings, where hug on the same outline
gives 26 taxels and no crossing. Use it only to reproduce a board made with it,
or to measure what hug contributes on a given outline.

Board mode remains a real decision:

- **`board_mode="expand"` — grow mode, the default. Prefer it.** The board
  grows just enough to fit the wiring, then `follow_padding` (on by default)
  shrink-wraps it back toward the routed copper, so it follows the drawn shape
  rather than expanding into a shapeless outline. Every taxel the shape holds
  is kept, a row/column a concavity cuts apart stays one net with a link trace
  hugging the notch (see "Split nets"), and it is the fast mode. The board may
  extend a few mm past the drawn outline where wiring runs.
- **`board_mode="fixed_keepout"`** — the drawn outline *is* the board, exactly.
  A wiring keep-out band is reserved inside it, sized from the hug routing
  plan. Use it only when the user strictly needs that: the board must fit an
  enclosure, a slot or a part with no margin. It costs taxels (a concave
  C-shape keeps 26 against grow mode's 160) and time: it re-routes the board
  once per keep-out attempt.

**Which to pick**: grow, unless the user says the outline is a hard limit.
"In the shape of X" alone is not one; if fit might matter, build grow, say
how far the board extends past the outline, and offer fixed. Measured on a
size 9 insole (~29 x 28 grid, 3.3 x 8.6 mm pitch, laptop):

| mode, layers | taxels | board past the outline | time |
|---|---|---|---|
| grow, 2 | 574 | up to 8.3 mm | 5 s |
| grow, 4 | 574 | up to 3.9 mm | 5 s |
| fixed, 2 | 497 | none | 13 s |
| fixed, 4 | 540 | none | 43 s |

Report the actual board dimensions either way (see "Never compute the
board's size" below) so the user knows which one happened.

**Rows-and-columns targeting is grow-mode only.** `target_cols`/`target_rows`
raise in fixed mode, because the keep-out band's width depends on the net
counts that the packing determines. In fixed mode set the taxel size and let
the count follow.

**Never compute the board's size from the outline you sent.** The connector
union widens it. Read the real dimensions from `stats` and `manifest.json`.

## Readout board and copper layers

**Every sensor plugs into an OpenTac readout board**, and the board sets the
grid limit. Count rows and columns per direction, not their sum:

- **30-pin board** (`readout_board="30pin"`): up to 15 rows x 15 columns. Two
  stacked 16 x 22.8 mm boards; the tail ends in a 30-position connector.
- **60-pin board** (`readout_board="60pin"`): up to 30 rows x 30 columns. One
  37 x 34.3 mm board; 60-position connector.
- `readout_board="auto"` (the default) takes the 30-pin board when the grid
  fits it, else the 60-pin board. A grid over 30 in either direction fails
  with a message naming the limit: use larger taxels or a smaller area.

**Copper layers: `copper_layers=2` (default) or `4`.** On 2 layers every row
wire runs around the outside of the sensor field to the connector, so the
board needs a wiring band. On 4 layers rows run underneath on the inner layers
(In1 feeds, In2 links across notches), so the taxels reach closer to the edge.
**When precise packing or full coverage is the priority, use 4 layers, in grow
mode.** In grow mode 4 layers keep the board within ~2-4 mm of the drawn
outline where 2 layers spill ~8 mm (insole table above); in a fixed outline
they fit more taxels (insole 540 vs 497; D'Claw flat-top hexagon fingertip
497 vs 410). Use 4 for complex or tightly bounded parts such as a fingertip.
4-layer flex costs more to fabricate; say so.

## Taxel sizing

Width and height are independent (`pixel_w_mm`/`pixel_h_mm`,
`pitch_x_mm`/`pitch_y_mm`), and for an elongated shape they should differ. Rows
are the expensive direction: every row is a net that must reach the connector.

- **Neither dimension below 1.5mm.**
- Pitch slightly exceeds the taxel so they do not touch (the UI defaults are
  4.0mm taxels on a 4.2mm pitch).
- Rows and columns are each capped by the readout board: 15 on the 30-pin
  board, 30 on the 60-pin board (see above).

Ask the user what they are sensing before picking. Fine pressure distribution
wants small taxels; detecting presence over a large area does not.

## Tail and connector

Every export includes the connector footprint unconditionally, with
traces reaching all the way to the pad drills — the pipeline applies a
synthetic 1mm tail extension automatically at export time if the user never
drew one, so a plain export is already a complete, connectable board (0.000mm
track-to-drill gap). Drawing a custom tail is optional (a longer reach,
a bent path around an obstacle) — do it only if they ask for something the
automatic 1mm does not cover. To change a finished board without re-routing it
(a typed `cable_length_mm`, other padding settings), use the tool's Redo
padding with the held `edit_data`:
`opentac.cad.main.redo_padding(RedoPaddingRequest(edit_data=ed, routes=ed["routes"], params={...}))`.

The tail is narrow by default (`narrow_tail=True`): after routing it is cut down
to the traces it carries and steps out to the connector's width just before
the connector. Taxels and traces are identical to a connector-width tail, so do
not trade it off against taxel count. A `tail_width_mm` the user gives is used as
is. The length is automatic too: leave `cable_length_mm` out and the tail is made
as long as the narrow tail and the printed case need, at least 12 mm
(`stats["tail_length_mm"]`, `stats["tail_length_reason"]`). Pass a length only if
the user asks for one.

The tail's contacts are on the **bottom** copper by default; `contact_side: "top"`
puts them on the top (taxel) face. Which face has to meet the connector's contacts
depends on the connector and on how the user will mount the sensor, so do not
assume it: keep the default unless the user says how the sensor is mounted. The fingers are plain pads on
that face only, with no drill, and wires on another copper layer change to it
through vias in the tail just before the stiffener. The export opens solder
mask on that face only there, and its drawing layer says where the polyimide
stiffener goes: on the other face (the top, the taxel side, by default), 0.3 mm
total. Tell the user this when they order. On the top the mask has one opening per taxel, at least 0.3 mm
apart (JLCPCB's coverlay minimum); on a dense board the openings are trimmed
slightly into the electrodes and `stats["mask_note"]` says so.

## Split nets (concave shapes)

Where a shape is re-entrant along the grid axis — an insole's medial heel is
wider than its arch — a row or column enters the shape, leaves and returns.
The layout stage always splits it into two hops (`COL_5_0`, `COL_5_1`).

Hug joins them back into one net, bridged by a link trace that hugs around
the concavity (`assign_rowcol_nets_joined`) — no crossing is possible by
construction, in `fixed_keepout` as well as `expand`, including deep concave
shapes. A concave shape is not a reason to change anything, but `preflight()`
names the re-entrant band, which is worth repeating to the user because it
explains where the link trace will run.

The tradeoff is board area: the link runs into the notch, so a hug board
leaves the concavity less open than a hand-drawn fixed_keepout outline would.
Copper in the notch is expected on a correct join: the links run along its
walls.

Another DRC cause: **too many channels for the space** — the keep-out grows to
make room and takes sensing area. A `fixed_keepout_max_mm` above ~25mm on a
100mm board indicates it.

## What a good board looks like

Exactly two things: **no copper outside the board edge, and no traces
crossing.** `generate.summarize()` reports both. Everything else is detail.

**Two different things are counted in that number.** `drc["violations"]` is
geometry violations plus *incomplete nets*, and they mean opposite things
about the router. A `crossing` or `clearance` entry in `drc["details"]` is
copper that should not be where it is. An entry in
`drc["routing_warning_nets"]` is the reverse — hug **refused** to emit a net
rather than route it too close to its neighbour, so the board is safe but
some taxels are not connected (`drc["routing_problem_pixels"]` names them,
and `stats["hug_plan_warnings"]` says which net and why). For example, a
C-shape in fixed mode at 3.2mm pitch reports `ROW_0_0 pad0 would come within
DRC minimum of ROW_1_0 — net refused, not emitted` and leaves 2 taxels
unconnected. That is the "too many channels for the space" symptom above, not
a routing defect — the fix is a coarser pitch, a smaller grid, or grow mode,
never a retry.

**A nonzero DRC count is always a real problem, in every mode.** Copper
running onto the connector's own pads is not counted, so do not discount any
DRC line as "probably just the connector edge." If DRC is not clean, find out
why rather than retrying blindly — start with "Split nets" above.

## Reporting

Say what was built, not what was asked for. A 32x32 grid over a foot shape is
1024 grid cells but only ~535 taxels, because cells outside the outline are
dropped and wiring keep-out takes more. State the real number, before the run
where possible.

Point them at `preview.svg` first, and tell them `gerber/` is what a fab house
needs.

## Limits worth stating early

- At most 30 rows and 30 columns (60-pin board); 15 x 15 for the 30-pin board.
- One taxel grid per board — `fill_region` restricts *where* taxels go, but
  cannot give two densities on one board.
- **Doubly-curved surfaces cannot be sensorized.** A flex PCB bends in one
  direction at a time, so a sphere, dome or rounded fingertip pad cannot be
  covered without wrinkling or tearing. `check_developability()` reports the
  error in mm. Do not ship a warped board. Approximate the part with flat
  facets instead, and prefer a **hexagon**: a hexagonal prism with a flat top
  covers a round part well, and its top and six side faces unfold into one
  flat pattern ("leaves" around the top) that the pipeline can sensorize and
  that folds back onto the part. Worked example, the D'Claw 1-inch fingertip:
  a flat-top hexagon with 5 columns per leaf, 30 columns in total to fit the
  60-pin board, 4 copper layers, 546 taxels. Single-curvature surfaces
  (cylinders, cones) unroll directly with the STL tools.
