// PCB generation UI: outline editor + parameters + POST /generate + SVG/zip.
// The outline can come from hand-drawing, a selected STL face (stl-viewer.js),
// or a DXF / SVG import (dxf-import.js). A second, optional polygon (the
// "sensorize zone") constrains where sensors get packed inside the outline.
import { API_BASE } from "./config.js";
import { clearPrinted, designStem, downloadName, markPrinted, zipName } from "./design-name.js";
import { createRouteEditor } from "./route-editor.js";

const canvas = document.getElementById("outline-canvas");
const ctx = canvas.getContext("2d");
const outlineInfo = document.getElementById("outline-info");

// Vertices are stored in mm (y-up), in click order. First two = cable edge.
// `vertices` is the board outline; `fillVertices` is the optional sensorize
// zone. `editTarget` selects which array the drawing tools currently act on.
let vertices = [];
let fillVertices = [];
let editTarget = "outline"; // "outline" | "fill"
let foldLines = [];
// The cable edge (edge 0->1, where the connector tail exits) must be an
// explicit choice: an implicit "first edge drawn" default can place the
// connector on an interior chord the user never intended, leaving few or no
// sensors routable. Drawing or reshaping the outline clears this; pressing
// "Set cable edge" sets it. Presets and restored designs carry a designed
// cable edge and count as set; STL unfolds and DXF/SVG imports do not. Generate
// refuses to run while unset.
let cableEdgeConfirmed = false;
let selected = null;
let mode = "select";
let dragState = null;
let panState = null;
let previewPoint = null;
let arcState = null;
let angleReferenceEdge = null;
let spacePanActive = false;
let undoStack = [];
let redoStack = [];

// View transform: which mm point sits at the canvas centre, and px-per-mm.
const view = { scale: 4, cx: 0, cy: 0 };

function activeArray() { return editTarget === "fill" ? fillVertices : vertices; }
function setActiveArray(next) { if (editTarget === "fill") fillVertices = next; else vertices = next; }

const cadButtons = {
  select: document.getElementById("cad-select"),
  line: document.getElementById("cad-line"),
  rect: document.getElementById("cad-rect"),
  hline: document.getElementById("cad-hline"),
  vline: document.getElementById("cad-vline"),
  arc: document.getElementById("cad-arc"),
};
const cadUndoBtn = document.getElementById("cad-undo");
const cadRedoBtn = document.getElementById("cad-redo");
const cadDeleteBtn = document.getElementById("cad-delete");
const cadSetCableBtn = document.getElementById("cad-set-cable");
const cadMirrorHBtn = document.getElementById("cad-mirror-h");
const cadMirrorVBtn = document.getElementById("cad-mirror-v");
const cadSetAngleRefBtn = document.getElementById("cad-set-angle-ref");
const cadZoomOutBtn = document.getElementById("cad-zoom-out");
const cadZoomInBtn = document.getElementById("cad-zoom-in");
const cadFitBtn = document.getElementById("cad-fit");
const cadLengthInput = document.getElementById("cad-length");
const cadAngleInput = document.getElementById("cad-angle-value");
const cadWidthInput = document.getElementById("cad-width");
const cadHeightInput = document.getElementById("cad-height");
const cadGridInput = document.getElementById("cad-grid");
const cadSnapInput = document.getElementById("cad-snap");
const cadTargetOutlineBtn = document.getElementById("cad-target-outline");
const cadTargetFillBtn = document.getElementById("cad-target-fill");
const fillZoneHint = document.getElementById("fill-zone-hint");
const clearOutlineBtn = document.getElementById("clear-outline");
const cableEdgeBanner = document.getElementById("cable-edge-banner");
const cableEdgeBannerClose = document.getElementById("cable-edge-banner-close");

function resetView() { view.scale = 4; view.cx = 0; view.cy = 0; }
function fitViewTo(verts) {
  if (!verts.length) { resetView(); return; }
  const xs = verts.map((v) => v[0]), ys = verts.map((v) => v[1]);
  const minx = Math.min(...xs), maxx = Math.max(...xs);
  const miny = Math.min(...ys), maxy = Math.max(...ys);
  view.cx = (minx + maxx) / 2;
  view.cy = (miny + maxy) / 2;
  const w = Math.max(maxx - minx, 1e-3), h = Math.max(maxy - miny, 1e-3), margin = 26;
  view.scale = Math.max(0.05, Math.min((canvas.width - 2 * margin) / w, (canvas.height - 2 * margin) / h));
}

function zoomView(factor, anchorPx = null) {
  const anchor = anchorPx || [canvas.width / 2, canvas.height / 2];
  const before = pxToMm(anchor[0], anchor[1]);
  const nextScale = Math.max(0.03, Math.min(120, view.scale * factor));
  if (Math.abs(nextScale - view.scale) < 1e-9) return;
  view.scale = nextScale;
  const after = pxToMm(anchor[0], anchor[1]);
  view.cx += before[0] - after[0];
  view.cy += before[1] - after[1];
  redraw();
}

function fitOutlineView() {
  const combined = vertices.concat(fillVertices);
  fitViewTo(combined.length ? combined : activeArray());
  redraw();
}

function pxToMm(px, py) {
  return [(px - canvas.width / 2) / view.scale + view.cx,
          (canvas.height / 2 - py) / view.scale + view.cy];
}
function mmToPx(mx, my) {
  return [(mx - view.cx) * view.scale + canvas.width / 2,
          canvas.height / 2 - (my - view.cy) * view.scale];
}

function cloneVertices(src) {
  return src.map((p) => p.slice());
}

function pushUndo() {
  undoStack.push({
    vertices: cloneVertices(vertices),
    fillVertices: cloneVertices(fillVertices),
    foldLines: foldLines.map(([a, b]) => [a.slice(), b.slice()]),
    cableEdgeConfirmed,
  });
  if (undoStack.length > 80) undoStack.shift();
  redoStack = [];
  updateCadButtons();
  markResultsStale();
}

// Every outline/fill/cable-edge change goes through pushUndo (or an undo/redo
// restore), so this is the one choke point for flagging that the generated
// preview and stats no longer match the shape on the canvas. Without it a
// redrawn outline sits next to the previous design's board and stats, and a
// first-time user reads the stale result as the new shape's.
function markResultsStale() {
  if (!statsEl || !statsEl.innerHTML) return;
  genStatus.textContent = "Shape changed — press Generate PCB to refresh the preview and stats below.";
  pcbPreview.classList.add("pcb-preview--stale");
}

// The connector's contact face is built into the board's files at Generate
// (Download ZIP and route edits reuse them), so changing it afterwards needs a
// new Generate, as a changed outline does.
for (const el of document.querySelectorAll('input[name="contact_side"]')) {
  el.addEventListener("change", () => {
    if (!statsEl || !statsEl.innerHTML) return;
    genStatus.textContent = "Connector contacts changed — press Generate PCB to apply them to the board files.";
    pcbPreview.classList.add("pcb-preview--stale");
  });
}

function restoreSnapshot(snapshot) {
  vertices = cloneVertices(snapshot.vertices);
  fillVertices = cloneVertices(snapshot.fillVertices || []);
  foldLines = snapshot.foldLines.map(([a, b]) => [a.slice(), b.slice()]);
  cableEdgeConfirmed = snapshot.cableEdgeConfirmed ?? false;
  selected = null;
  previewPoint = null;
  arcState = null;
  angleReferenceEdge = null;
  redraw();
  setOutlineInfo();
  syncDimensionFields();
  updateCadButtons();
  markResultsStale();
}

function snapPoint(p) {
  if (!cadSnapInput.checked) return p;
  const grid = Math.max(0.001, parseFloat(cadGridInput.value) || 1);
  return [Math.round(p[0] / grid) * grid, Math.round(p[1] / grid) * grid];
}

function constrainPoint(start, raw, event) {
  let p = raw;
  if (mode === "hline") p = [raw[0], start[1]];
  if (mode === "vline") p = [start[0], raw[1]];
  if (event && event.shiftKey && !["hline", "vline"].includes(mode)) {
    const dx = raw[0] - start[0];
    const dy = raw[1] - start[1];
    p = Math.abs(dx) >= Math.abs(dy) ? [raw[0], start[1]] : [start[0], raw[1]];
  }
  return snapPoint(p);
}

function eventMm(event) {
  const [px, py] = eventCanvasPx(event);
  return snapPoint(pxToMm(px, py));
}

function eventCanvasPx(event) {
  const rect = canvas.getBoundingClientRect();
  const px = (event.clientX - rect.left) * (canvas.width / rect.width);
  const py = (event.clientY - rect.top) * (canvas.height / rect.height);
  return [px, py];
}

function startPan(event) {
  panState = {
    pointerId: event.pointerId,
    x: event.clientX,
    y: event.clientY,
    cx: view.cx,
    cy: view.cy,
  };
  canvas.classList.add("is-panning");
  canvas.setPointerCapture(event.pointerId);
  event.preventDefault();
}

function updatePan(event) {
  const rect = canvas.getBoundingClientRect();
  const dx = (event.clientX - panState.x) * (canvas.width / rect.width) / view.scale;
  const dy = (event.clientY - panState.y) * (canvas.height / rect.height) / view.scale;
  view.cx = panState.cx - dx;
  view.cy = panState.cy + dy;
  redraw();
}

function stopPan(event) {
  if (!panState || event.pointerId !== panState.pointerId) return false;
  panState = null;
  canvas.classList.remove("is-panning");
  canvas.releasePointerCapture(event.pointerId);
  return true;
}

function isTypingTarget(target) {
  const tag = target?.tagName?.toLowerCase();
  return tag === "input" || tag === "textarea" || tag === "select" || target?.isContentEditable;
}

function edgeAt(i) {
  const verts = activeArray();
  return [verts[i], verts[(i + 1) % verts.length]];
}

function edgeLength(i) {
  const [a, b] = edgeAt(i);
  return Math.hypot(b[0] - a[0], b[1] - a[1]);
}

function edgeAngle(i) {
  const [a, b] = edgeAt(i);
  return Math.atan2(b[1] - a[1], b[0] - a[0]) * 180 / Math.PI;
}

function normalizeAngleDeg(angle) {
  let a = ((angle + 180) % 360 + 360) % 360 - 180;
  return Object.is(a, -0) ? 0 : a;
}

function validEdgeIndex(i) {
  const verts = activeArray();
  const edgeCount = verts.length >= 3 ? verts.length : Math.max(0, verts.length - 1);
  return Number.isInteger(i) && i >= 0 && i < edgeCount;
}

function relativeEdgeAngle(refIndex, targetIndex) {
  if (!validEdgeIndex(refIndex) || !validEdgeIndex(targetIndex)) return null;
  return normalizeAngleDeg(edgeAngle(targetIndex) - edgeAngle(refIndex));
}

function sampleArcPoints(center, start, end) {
  const radius = Math.hypot(start[0] - center[0], start[1] - center[1]);
  if (radius < 1e-6) return [];
  const startAngle = Math.atan2(start[1] - center[1], start[0] - center[0]);
  const endAngle = Math.atan2(end[1] - center[1], end[0] - center[0]);
  let delta = endAngle - startAngle;
  while (delta <= -Math.PI) delta += Math.PI * 2;
  while (delta > Math.PI) delta -= Math.PI * 2;
  if (Math.abs(delta) < Math.PI / 180) return [];
  const segments = Math.max(8, Math.ceil(Math.abs(delta) / (Math.PI / 18)));
  const pts = [];
  for (let i = 0; i <= segments; i++) {
    const t = i / segments;
    const angle = startAngle + delta * t;
    pts.push(snapPoint([
      center[0] + radius * Math.cos(angle),
      center[1] + radius * Math.sin(angle),
    ]));
  }
  return pts;
}

function appendArcPoints(points) {
  if (!points.length) return;
  if (editTarget === "outline" && foldLines.length) foldLines = [];
  if (editTarget === "outline") cableEdgeConfirmed = false;
  const verts = activeArray();
  const merged = points.slice();
  if (verts.length && Math.hypot(verts[verts.length - 1][0] - merged[0][0], verts[verts.length - 1][1] - merged[0][1]) <= 1e-6) {
    merged.shift();
  }
  setActiveArray(verts.concat(merged));
}

function polygonBounds() {
  const verts = activeArray();
  if (!verts.length) return null;
  const xs = verts.map((p) => p[0]);
  const ys = verts.map((p) => p[1]);
  return { minx: Math.min(...xs), maxx: Math.max(...xs), miny: Math.min(...ys), maxy: Math.max(...ys) };
}

function pointSegDistance(p, a, b) {
  const vx = b[0] - a[0], vy = b[1] - a[1];
  const len2 = vx * vx + vy * vy;
  const t = len2 ? Math.max(0, Math.min(1, ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / len2)) : 0;
  return Math.hypot(p[0] - (a[0] + t * vx), p[1] - (a[1] + t * vy));
}

function pointInPoly(p) {
  const verts = activeArray();
  let inside = false;
  for (let i = 0, j = verts.length - 1; i < verts.length; j = i++) {
    const xi = verts[i][0], yi = verts[i][1];
    const xj = verts[j][0], yj = verts[j][1];
    if (((yi > p[1]) !== (yj > p[1])) &&
        (p[0] < (xj - xi) * (p[1] - yi) / ((yj - yi) || 1e-12) + xi)) inside = !inside;
  }
  return inside;
}

function hitTest(p) {
  const verts = activeArray();
  const tol = 8 / view.scale;
  for (let i = 0; i < verts.length; i++) {
    if (Math.hypot(verts[i][0] - p[0], verts[i][1] - p[1]) <= tol) return { type: "vertex", index: i };
  }
  if (verts.length >= 2) {
    const edgeCount = verts.length >= 3 ? verts.length : verts.length - 1;
    for (let i = 0; i < edgeCount; i++) {
      const [a, b] = edgeAt(i);
      if (pointSegDistance(p, a, b) <= tol) return { type: "edge", index: i };
    }
  }
  if (verts.length >= 3 && pointInPoly(p)) return { type: "outline" };
  return null;
}

function drawGrid() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const [mmL, mmT] = pxToMm(0, 0);
  const [mmR, mmB] = pxToMm(canvas.width, canvas.height);
  const step = 5; // mm
  const nx = Math.abs(mmR - mmL) / step, ny = Math.abs(mmT - mmB) / step;
  if (nx < 200 && ny < 200) {
    ctx.strokeStyle = "#1b2530";
    ctx.lineWidth = 1;
    for (let mx = Math.floor(Math.min(mmL, mmR) / step) * step; mx <= Math.max(mmL, mmR); mx += step) {
      const [px] = mmToPx(mx, 0);
      ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, canvas.height); ctx.stroke();
    }
    for (let my = Math.floor(Math.min(mmT, mmB) / step) * step; my <= Math.max(mmT, mmB); my += step) {
      const [, py] = mmToPx(0, my);
      ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(canvas.width, py); ctx.stroke();
    }
  }
  // origin axes
  ctx.strokeStyle = "#2c3a48";
  const [ox, oy] = mmToPx(0, 0);
  ctx.beginPath();
  ctx.moveTo(ox, 0); ctx.lineTo(ox, canvas.height);
  ctx.moveTo(0, oy); ctx.lineTo(canvas.width, oy);
  ctx.stroke();
}

function drawOutlinePolygon() {
  if (!vertices.length) return;
  ctx.beginPath();
  vertices.forEach(([mx, my], i) => {
    const [px, py] = mmToPx(mx, my);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  });
  if (vertices.length >= 3) ctx.closePath();
  ctx.fillStyle = "rgba(79,157,255,0.12)";
  ctx.strokeStyle = "#4f9dff";
  ctx.lineWidth = 1.5;
  if (vertices.length >= 3) ctx.fill();
  ctx.stroke();

  // The cable edge is highlighted only once the user has set it. Before
  // that, edge 0->1 is drawn like any other edge: a tentative highlight
  // would read as "already selected" and obscure the required step.
  if (vertices.length >= 2 && cableEdgeConfirmed) {
    const [ax, ay] = mmToPx(...vertices[0]);
    const [bx, by] = mmToPx(...vertices[1]);
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by);
    ctx.strokeStyle = "#ffb84d"; ctx.lineWidth = 3.5; ctx.stroke();
  }

  // Shared edges retained by the unfolding tree. These are bend/fold guides,
  // not cuts in the PCB outline.
  if (foldLines.length) {
    ctx.save();
    ctx.strokeStyle = "#ff9f43";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([7, 5]);
    for (const [a, b] of foldLines) {
      const [ax, ay] = mmToPx(a[0], a[1]);
      const [bx, by] = mmToPx(b[0], b[1]);
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
    }
    ctx.restore();
  }

  vertices.forEach(([mx, my], i) => {
    const [px, py] = mmToPx(mx, my);
    ctx.beginPath(); ctx.arc(px, py, 3.5, 0, Math.PI * 2);
    const isSelected = editTarget === "outline" && selected?.type === "vertex" && selected.index === i;
    ctx.fillStyle = isSelected ? "#ffd45a"
      : (cableEdgeConfirmed && i < 2 ? "#ffb84d" : "#4f9dff");
    ctx.fill();
  });
}

function drawFillPolygon() {
  if (!fillVertices.length) return;
  ctx.save();
  ctx.beginPath();
  fillVertices.forEach(([mx, my], i) => {
    const [px, py] = mmToPx(mx, my);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  });
  if (fillVertices.length >= 3) ctx.closePath();
  ctx.fillStyle = "rgba(45,125,70,0.18)";
  ctx.strokeStyle = "#2d7d46";
  ctx.setLineDash([4, 3]);
  ctx.lineWidth = 1.5;
  if (fillVertices.length >= 3) ctx.fill();
  ctx.stroke();
  ctx.restore();

  fillVertices.forEach(([mx, my], i) => {
    const [px, py] = mmToPx(mx, my);
    ctx.beginPath(); ctx.arc(px, py, 3.5, 0, Math.PI * 2);
    const isSelected = editTarget === "fill" && selected?.type === "vertex" && selected.index === i;
    ctx.fillStyle = isSelected ? "#ffd45a" : "#59c98a";
    ctx.fill();
  });
}

function redraw() {
  drawGrid();
  drawOutlinePolygon();
  drawFillPolygon();

  const verts = activeArray();
  if (verts.length) {
    drawDimensions(verts);
    if (validEdgeIndex(angleReferenceEdge)) drawReferenceEdge(angleReferenceEdge);
    if (selected?.type === "edge") drawSelectedEdge(selected.index);
    if (selected?.type === "outline") drawOutlineSelection();
  }

  if (previewPoint && verts.length && ["line", "hline", "vline"].includes(mode)) {
    const last = verts[verts.length - 1];
    const [ax, ay] = mmToPx(...last);
    const [bx, by] = mmToPx(...previewPoint);
    ctx.save();
    ctx.strokeStyle = "#ffd45a";
    ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
    ctx.restore();
  }

  if (arcState) {
    drawArcPreview();
  }

  if (dragState?.type === "rect-preview") {
    const a = dragState.start;
    const b = dragState.current;
    const rect = rectVertices(a, b);
    ctx.save();
    ctx.strokeStyle = "#ffd45a";
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    rect.forEach((p, i) => {
      const [x, y] = mmToPx(...p);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.stroke();
    ctx.restore();
  }
}

function drawReferenceEdge(i) {
  if (!validEdgeIndex(i)) return;
  const [a, b] = edgeAt(i);
  const [ax, ay] = mmToPx(...a);
  const [bx, by] = mmToPx(...b);
  ctx.save();
  ctx.strokeStyle = "#46e6a7";
  ctx.lineWidth = 4;
  ctx.setLineDash([8, 5]);
  ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
  ctx.restore();
}

function drawSelectedEdge(i) {
  if (activeArray().length < 2) return;
  const [a, b] = edgeAt(i);
  const [ax, ay] = mmToPx(...a);
  const [bx, by] = mmToPx(...b);
  ctx.save();
  ctx.strokeStyle = "#ffd45a";
  ctx.lineWidth = 4;
  ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
  ctx.restore();
}

function drawArcPreview() {
  const current = arcState.current || arcState.start || arcState.center;
  const [cx, cy] = mmToPx(...arcState.center);
  const [px, py] = mmToPx(...current);
  ctx.save();
  ctx.strokeStyle = "#ffd45a";
  ctx.fillStyle = "#ffd45a";
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  ctx.arc(cx, cy, 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(px, py);
  ctx.stroke();
  if (arcState.step === "sweep" && arcState.start && current) {
    const points = sampleArcPoints(arcState.center, arcState.start, current);
    if (points.length) {
      ctx.beginPath();
      points.forEach((point, i) => {
        const [x, y] = mmToPx(...point);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }
  }
  ctx.restore();
}

function drawOutlineSelection() {
  const b = polygonBounds();
  if (!b) return;
  const [x1, y1] = mmToPx(b.minx, b.maxy);
  const [x2, y2] = mmToPx(b.maxx, b.miny);
  ctx.save();
  ctx.strokeStyle = "#ffd45a";
  ctx.setLineDash([6, 4]);
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  ctx.restore();
}

function drawDimensions(verts) {
  if (verts.length < 2 || view.scale < 1.2) return;
  const edgeCount = verts.length >= 3 ? verts.length : verts.length - 1;
  ctx.save();
  ctx.font = "11px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (let i = 0; i < edgeCount; i++) {
    const [a, b] = edgeAt(i);
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
    if (len < 2) continue;
    const mx = (a[0] + b[0]) / 2;
    const my = (a[1] + b[1]) / 2;
    const [px, py] = mmToPx(mx, my);
    const text = `${len.toFixed(1)} mm`;
    const w = ctx.measureText(text).width + 8;
    ctx.fillStyle = "rgba(12,17,22,0.78)";
    ctx.fillRect(px - w / 2, py - 8, w, 16);
    ctx.fillStyle = "#dce8f5";
    ctx.fillText(text, px, py);
  }
  ctx.restore();
}

function setOutlineInfo(extra) {
  const verts = activeArray();
  const label = editTarget === "fill" ? "sensorize-zone" : "outline";
  const cableNote = editTarget === "outline" && verts.length >= 2
    ? (cableEdgeConfirmed
        ? " · cable edge = orange"
        : " · cable edge NOT SET — select an edge, press Set cable edge")
    : "";
  outlineInfo.textContent =
    `${verts.length} ${label} vertices` + cableNote + (extra ? ` · ${extra}` : "");
}

function setMode(nextMode) {
  mode = nextMode;
  selected = null;
  previewPoint = null;
  arcState = null;
  Object.entries(cadButtons).forEach(([name, button]) => button.classList.toggle("active", name === mode));
  canvas.style.cursor = mode === "select" ? "default" : "crosshair";
  syncDimensionFields();
  updateCadButtons();
  redraw();
}

function updateFillToggleAvailability() {
  if (!cadTargetFillBtn) return;
  const available = vertices.length >= 3;
  cadTargetFillBtn.disabled = !available;
  cadTargetFillBtn.title = available ? "" : "Draw a board outline first";
  if (!available && editTarget === "fill") setEditTarget("outline");
}

function updateTargetButtons() {
  cadTargetOutlineBtn?.classList.toggle("active", editTarget === "outline");
  cadTargetFillBtn?.classList.toggle("active", editTarget === "fill");
  if (fillZoneHint) fillZoneHint.hidden = editTarget !== "fill";
  if (clearOutlineBtn) clearOutlineBtn.textContent = editTarget === "fill" ? "Clear sensorize zone" : "Clear outline";
}

function setEditTarget(target) {
  if (target === editTarget) return;
  if (target === "fill" && vertices.length < 3) return;
  editTarget = target;
  selected = null;
  previewPoint = null;
  arcState = null;
  angleReferenceEdge = null;
  updateTargetButtons();
  syncDimensionFields();
  updateCadButtons();
  redraw();
  setOutlineInfo();
}

function updateCadButtons() {
  cadUndoBtn.disabled = undoStack.length === 0;
  cadRedoBtn.disabled = redoStack.length === 0;
  cadDeleteBtn.disabled = !selected;
  cadSetCableBtn.disabled = editTarget !== "outline" || selected?.type !== "edge";
  cadSetAngleRefBtn.disabled = selected?.type !== "edge";
  const canMirror = vertices.length >= 3;
  if (cadMirrorHBtn) cadMirrorHBtn.disabled = !canMirror;
  if (cadMirrorVBtn) cadMirrorVBtn.disabled = !canMirror;
  if (!validEdgeIndex(angleReferenceEdge)) angleReferenceEdge = null;
  updateFillToggleAvailability();
}

function syncDimensionFields() {
  const b = polygonBounds();
  cadWidthInput.value = b ? (b.maxx - b.minx).toFixed(2) : "";
  cadHeightInput.value = b ? (b.maxy - b.miny).toFixed(2) : "";
  if (selected?.type === "edge") {
    cadLengthInput.value = edgeLength(selected.index).toFixed(2);
    const relative = angleReferenceEdge !== null && selected.index !== angleReferenceEdge
      ? relativeEdgeAngle(angleReferenceEdge, selected.index)
      : null;
    cadAngleInput.value = (relative ?? edgeAngle(selected.index)).toFixed(1);
  } else {
    cadLengthInput.value = "";
  }
}

function rectVertices(a, b) {
  return [[a[0], a[1]], [b[0], a[1]], [b[0], b[1]], [a[0], b[1]]];
}

function addLinePoint(p, event) {
  if (editTarget === "outline" && foldLines.length) foldLines = [];
  if (editTarget === "outline") cableEdgeConfirmed = false;
  angleReferenceEdge = null;
  const verts = activeArray();
  if (!verts.length) {
    pushUndo();
    setActiveArray([p]);
  } else {
    const next = constrainPoint(verts[verts.length - 1], p, event);
    const closeTol = 8 / view.scale;
    if (verts.length >= 3 && Math.hypot(next[0] - verts[0][0], next[1] - verts[0][1]) <= closeTol) {
      setMode("select");
    } else {
      pushUndo();
      setActiveArray(verts.concat([next]));
    }
  }
  redraw();
  setOutlineInfo();
  syncDimensionFields();
}

canvas.addEventListener("pointerdown", (event) => {
  if (event.button === 1 || event.button === 2 || spacePanActive) {
    startPan(event);
    return;
  }
  const p = eventMm(event);
  if (mode === "arc") {
    if (!arcState) {
      arcState = { step: "radius", center: p, start: null, current: p };
      setOutlineInfo("arc center set");
    } else if (arcState.step === "radius") {
      arcState = { ...arcState, step: "sweep", start: p, current: p };
      setOutlineInfo("arc radius set");
    } else if (arcState.step === "sweep") {
      const points = sampleArcPoints(arcState.center, arcState.start, p);
      if (points.length) {
        pushUndo();
        appendArcPoints(points);
        selected = { type: "vertex", index: activeArray().length - 1 };
        angleReferenceEdge = null;
      }
      arcState = null;
      previewPoint = null;
      redraw();
      setOutlineInfo();
      syncDimensionFields();
      updateCadButtons();
    }
    redraw();
    return;
  }
  if (mode === "rect") {
    pushUndo();
    dragState = { type: "rect-preview", start: p, current: p };
    canvas.setPointerCapture(event.pointerId);
    return;
  }
  if (["line", "hline", "vline"].includes(mode)) {
    addLinePoint(p, event);
    return;
  }

  const hit = hitTest(p);
  selected = hit;
  syncDimensionFields();
  updateCadButtons();
  redraw();
  if (!hit) return;
  pushUndo();
  dragState = {
    type: hit.type,
    index: hit.index,
    start: p,
    original: cloneVertices(activeArray()),
  };
  canvas.setPointerCapture(event.pointerId);
});

canvas.addEventListener("pointermove", (event) => {
  if (panState && event.pointerId === panState.pointerId) {
    updatePan(event);
    return;
  }
  const raw = eventMm(event);
  if (dragState?.type === "rect-preview") {
    dragState.current = raw;
    redraw();
    return;
  }
  if (!dragState && mode === "arc" && arcState) {
    arcState.current = raw;
    redraw();
    return;
  }
  if (!dragState && ["line", "hline", "vline"].includes(mode) && activeArray().length) {
    previewPoint = constrainPoint(activeArray()[activeArray().length - 1], raw, event);
    redraw();
    return;
  }
  if (!dragState) return;

  const dx = raw[0] - dragState.start[0];
  const dy = raw[1] - dragState.start[1];
  if (dragState.type === "vertex") {
    const verts = activeArray().slice();
    verts[dragState.index] = snapPoint([
      dragState.original[dragState.index][0] + dx,
      dragState.original[dragState.index][1] + dy,
    ]);
    setActiveArray(verts);
  } else if (dragState.type === "outline") {
    setActiveArray(dragState.original.map((p) => snapPoint([p[0] + dx, p[1] + dy])));
  }
  redraw();
  syncDimensionFields();
});

canvas.addEventListener("pointerup", (event) => {
  if (stopPan(event)) return;
  if (!dragState) return;
  if (dragState.type === "rect-preview") {
    setActiveArray(rectVertices(dragState.start, snapPoint(dragState.current)));
    if (editTarget === "outline") {
      foldLines = [];
      cableEdgeConfirmed = false;
    }
    angleReferenceEdge = null;
    fitViewTo(activeArray());
    setMode("select");
  }
  dragState = null;
  canvas.releasePointerCapture(event.pointerId);
  redraw();
  setOutlineInfo();
  syncDimensionFields();
  updateCadButtons();
});

canvas.addEventListener("pointercancel", (event) => {
  stopPan(event);
});

canvas.addEventListener("contextmenu", (event) => {
  event.preventDefault();
});

canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  const [px, py] = eventCanvasPx(event);
  const factor = Math.exp(-event.deltaY * 0.0015);
  zoomView(factor, [px, py]);
}, { passive: false });

document.addEventListener("keydown", (event) => {
  if (isTypingTarget(event.target)) return;
  if (event.code === "Space" && !event.repeat) {
    spacePanActive = true;
    canvas.classList.add("space-pan");
    event.preventDefault();
    return;
  }
  if (event.key !== "Escape" || !arcState) return;
  arcState = null;
  previewPoint = null;
  redraw();
  setOutlineInfo("arc cancelled");
});

document.addEventListener("keyup", (event) => {
  if (event.code !== "Space") return;
  spacePanActive = false;
  canvas.classList.remove("space-pan");
});

window.addEventListener("blur", () => {
  spacePanActive = false;
  panState = null;
  canvas.classList.remove("space-pan", "is-panning");
});

Object.entries(cadButtons).forEach(([name, button]) => button.addEventListener("click", () => setMode(name)));

cadUndoBtn.addEventListener("click", () => {
  if (!undoStack.length) return;
  redoStack.push({
    vertices: cloneVertices(vertices),
    fillVertices: cloneVertices(fillVertices),
    foldLines: foldLines.map(([a, b]) => [a.slice(), b.slice()]),
  });
  restoreSnapshot(undoStack.pop());
});
cadRedoBtn.addEventListener("click", () => {
  if (!redoStack.length) return;
  undoStack.push({
    vertices: cloneVertices(vertices),
    fillVertices: cloneVertices(fillVertices),
    foldLines: foldLines.map(([a, b]) => [a.slice(), b.slice()]),
  });
  restoreSnapshot(redoStack.pop());
});
cadDeleteBtn.addEventListener("click", () => {
  if (!selected) return;
  pushUndo();
  let verts = activeArray();
  if (selected.type === "vertex" && verts.length > 3) {
    verts = verts.slice(); verts.splice(selected.index, 1);
    if (editTarget === "outline" && selected.index <= 1) cableEdgeConfirmed = false;
  } else if (selected.type === "edge" && verts.length > 3) {
    const gone = (selected.index + 1) % verts.length;
    verts = verts.slice(); verts.splice(gone, 1);
    if (editTarget === "outline" && gone <= 1) cableEdgeConfirmed = false;
  } else if (selected.type === "outline") {
    verts = [];
    if (editTarget === "outline") {
      foldLines = [];
      cableEdgeConfirmed = false;
    }
  }
  setActiveArray(verts);
  selected = null;
  angleReferenceEdge = null;
  redraw(); setOutlineInfo(); syncDimensionFields(); updateCadButtons();
});
cadSetAngleRefBtn.addEventListener("click", () => {
  if (selected?.type !== "edge") return;
  angleReferenceEdge = selected.index;
  redraw();
  setOutlineInfo(`angle reference = edge ${angleReferenceEdge + 1}`);
  syncDimensionFields();
  updateCadButtons();
});
cadSetCableBtn.addEventListener("click", () => {
  if (editTarget !== "outline" || selected?.type !== "edge") return;
  pushUndo();
  const i = selected.index;
  vertices = vertices.slice(i).concat(vertices.slice(0, i));
  selected = { type: "edge", index: 0 };
  angleReferenceEdge = null;
  cableEdgeConfirmed = true;
  hideCableEdgeBanner();
  redraw(); setOutlineInfo("cable edge set"); syncDimensionFields(); updateCadButtons();
});
cadZoomOutBtn.addEventListener("click", () => zoomView(0.82));
cadZoomInBtn.addEventListener("click", () => zoomView(1.22));
cadFitBtn.addEventListener("click", fitOutlineView);
cadTargetOutlineBtn?.addEventListener("click", () => setEditTarget("outline"));
cadTargetFillBtn?.addEventListener("click", () => setEditTarget("fill"));

cadLengthInput.addEventListener("change", () => {
  if (selected?.type !== "edge") return;
  const len = parseFloat(cadLengthInput.value);
  if (!Number.isFinite(len) || len <= 0) return;
  pushUndo();
  const verts = activeArray().slice();
  const i = selected.index;
  const a = verts[i];
  const angle = Math.atan2(verts[(i + 1) % verts.length][1] - a[1], verts[(i + 1) % verts.length][0] - a[0]);
  verts[(i + 1) % verts.length] = snapPoint([a[0] + len * Math.cos(angle), a[1] + len * Math.sin(angle)]);
  setActiveArray(verts);
  redraw(); setOutlineInfo(); syncDimensionFields();
});
cadAngleInput.addEventListener("change", () => {
  if (selected?.type !== "edge") return;
  const angleDeg = parseFloat(cadAngleInput.value);
  if (!Number.isFinite(angleDeg)) return;
  pushUndo();
  const verts = activeArray().slice();
  const i = selected.index;
  const a = verts[i];
  const len = edgeLength(i);
  const targetAngleDeg = angleReferenceEdge !== null && angleReferenceEdge !== i && validEdgeIndex(angleReferenceEdge)
    ? edgeAngle(angleReferenceEdge) + angleDeg
    : angleDeg;
  const angle = targetAngleDeg * Math.PI / 180;
  verts[(i + 1) % verts.length] = snapPoint([a[0] + len * Math.cos(angle), a[1] + len * Math.sin(angle)]);
  setActiveArray(verts);
  const extra = angleReferenceEdge !== null && angleReferenceEdge !== i && validEdgeIndex(angleReferenceEdge)
    ? `relative angle to edge ${angleReferenceEdge + 1}`
    : undefined;
  redraw(); setOutlineInfo(extra); syncDimensionFields();
});
function resizeBounds(axis, value) {
  const b = polygonBounds();
  if (!b || !Number.isFinite(value) || value <= 0) return;
  pushUndo();
  const cx = (b.minx + b.maxx) / 2;
  const cy = (b.miny + b.maxy) / 2;
  const sx = axis === "x" ? value / Math.max(1e-9, b.maxx - b.minx) : 1;
  const sy = axis === "y" ? value / Math.max(1e-9, b.maxy - b.miny) : 1;
  setActiveArray(activeArray().map(([x, y]) => snapPoint([cx + (x - cx) * sx, cy + (y - cy) * sy])));
  redraw(); setOutlineInfo(); syncDimensionFields();
}
cadWidthInput.addEventListener("change", () => resizeBounds("x", parseFloat(cadWidthInput.value)));
cadHeightInput.addEventListener("change", () => resizeBounds("y", parseFloat(cadHeightInput.value)));

clearOutlineBtn.addEventListener("click", () => {
  pushUndo();
  if (editTarget === "outline") {
    vertices = [];
    fillVertices = [];
    foldLines = [];
  } else {
    fillVertices = [];
  }
  selected = null;
  angleReferenceEdge = null;
  if (!vertices.length && !fillVertices.length) resetView();
  redraw();
  setOutlineInfo();
  syncDimensionFields();
  updateCadButtons();
});
// Preset outlines. The first edge (vertex 0 -> 1) becomes the cable edge, same
// rule as hand-drawn outlines.
const PRESET_SHAPES = {
  "preset-square": [[-20, -20], [20, -20], [20, 20], [-20, 20]],
  "preset-rect":   [[-25, -15], [25, -15], [25, 15], [-25, 15]],
  "preset-foot":   [[-30, 0], [16, 0], [16, 14], [-4, 22], [-4, 52], [-30, 52]],
};
for (const [id, coords] of Object.entries(PRESET_SHAPES)) {
  document.getElementById(id).addEventListener("click", () => {
    if (!confirmOverwrite(activeArray().length, editTarget === "fill" ? "sensorize zone" : "board outline")) return;
    pushUndo();
    setActiveArray(coords.map((p) => [p[0], p[1]]));
    if (editTarget === "outline") {
      foldLines = [];
      cableEdgeConfirmed = true;   // presets carry a designed cable edge
    }
    selected = null;
    angleReferenceEdge = null;
    fitViewTo(activeArray()); redraw(); setOutlineInfo(); syncDimensionFields(); updateCadButtons();
  });
}

function confirmOverwrite(arrLength, itemLabel) {
  if (!arrLength) return true;
  return window.confirm(`This will replace your current ${itemLabel} — continue?`);
}

function showCableEdgeBanner() {
  if (cableEdgeBanner) cableEdgeBanner.hidden = false;
}
function hideCableEdgeBanner() {
  if (cableEdgeBanner) cableEdgeBanner.hidden = true;
}
cableEdgeBannerClose?.addEventListener("click", hideCableEdgeBanner);

// Outline delivered from a selected STL face. Always targets the board
// outline, regardless of which array was being edited beforehand.
window.addEventListener("otc:face-outline", (e) => {
  if (!confirmOverwrite(vertices.length, "board outline")) return;
  pushUndo();
  editTarget = "outline";
  updateTargetButtons();
  vertices = e.detail.outline.map((p) => [p[0], p[1]]);
  foldLines = (e.detail.foldLines || []).map(([a, b]) => [a.slice(), b.slice()]);
  // The unfold flow rotates a deliberately chosen lowest edge into the cable
  // slot, so edge 0->1 is a sensible default, but it is still a guess about
  // where the cable exits on the real part. Unfolded outlines need the same
  // explicit Set cable edge as drawn and DXF outlines: the computed edge is
  // placed at 0->1 as the natural first candidate, but not pre-confirmed.
  cableEdgeConfirmed = false;
  selected = null;
  angleReferenceEdge = null;
  fitViewTo(vertices);
  redraw();
  const faceText = e.detail.faceCount > 1 ? `${e.detail.faceCount} unfolded faces` : "STL face";
  setOutlineInfo(`from ${faceText} · ${e.detail.w.toFixed(1)} × ${e.detail.h.toFixed(1)} mm · confirm the cable edge below`);
  syncDimensionFields();
  updateCadButtons();
  genStatus.textContent = "Unfolded net loaded — confirm the cable edge, then press Generate PCB.";
  showCableEdgeBanner();
});
window.addEventListener("otc:face-clear", () => {
  pushUndo();
  vertices = [];
  fillVertices = [];
  foldLines = [];
  cableEdgeConfirmed = false;
  selected = null;
  angleReferenceEdge = null;
  resetView();
  redraw();
  setOutlineInfo();
  syncDimensionFields();
  updateCadButtons();
});

// Outline delivered from a DXF or SVG import (dxf-import.js). Always targets
// the board outline, same convention as the STL handoff above.
window.addEventListener("otc:dxf-outline", (e) => {
  if (!confirmOverwrite(vertices.length, "board outline")) return;
  pushUndo();
  editTarget = "outline";
  updateTargetButtons();
  vertices = e.detail.outline.map((p) => [p[0], p[1]]);
  foldLines = [];
  // A DXF/SVG loop's first edge is an accident of the file's vertex order, so
  // the user must confirm the cable edge before Generate runs.
  cableEdgeConfirmed = false;
  selected = null;
  angleReferenceEdge = null;
  fitViewTo(vertices);
  redraw();
  const src = e.detail.source === "svg" ? "SVG" : "DXF";
  setOutlineInfo(`from ${src} · confirm the cable edge below`);
  syncDimensionFields();
  updateCadButtons();
  genStatus.textContent = (e.detail.warnings && e.detail.warnings.length)
    ? `${src} imported with warnings: ${e.detail.warnings.join(" ")}`
    : `${src} outline loaded — press Generate PCB.`;
  showCableEdgeBanner();
});

// Write a stored parameter set into the form. Returns the leftovers: values
// that are real /generate inputs but have no control on this page, so they
// have to ride along as overrides instead of being silently dropped.
//
// The type dispatch is the load-bearing part. A bare `el.value = String(value)`
// is a no-op on a checkbox (setting .value changes the submitted value, not
// the tick) and misses radio groups entirely, because a group is addressed by
// name, so checkbox and radio settings from a design file would be lost.
function applyParamsToForm(params) {
  const extras = {};
  for (const [key, value] of Object.entries(params || {})) {
    if (key === "board_mode") {
      const radio = boardModeInputs.find((el) => el.value === value);
      if (radio) radio.checked = true;
      continue;
    }
    // The hug router is the only one the page offers; a stored "legacy" is ignored.
    if (key === "router") continue;
    // A length the pipeline chose shows as a blank (automatic) Tail length, so
    // the next Generate chooses again for whatever the design has become.
    if (key === "cable_length_auto") continue;
    if (key === "cable_length_mm" && params.cable_length_auto) {
      const el = document.getElementById(key);
      if (el) el.value = "";
      continue;
    }
    const el = document.getElementById(key);
    if (el) {
      if (el.type === "checkbox") el.checked = !!value;
      else el.value = value === null ? "" : String(value);
      continue;
    }
    const radios = document.querySelectorAll(`input[type="radio"][name="${CSS.escape(key)}"]`);
    if (radios.length) {
      radios.forEach((r) => { r.checked = r.value === String(value); });
      continue;
    }
    // No control owns this one — pad_pitch_mm, wiring_keepout_floor_mm and
    // connector_pos_override are all real inputs with no field on the page.
    // Dropping them would bring a re-imported design back with the smallest
    // connector that fits instead of the one it was built with.
    if (value !== null && value !== undefined) extras[key] = value;
  }
  return extras;
}

// Finished example design from the landing gallery (examples.js), or a design
// file re-opened through design-import.js: outline + a complete parameter set,
// generated immediately. The parameters are written into the form first so the
// loaded design is fully editable afterwards and the fields show exactly what
// produced the result.
window.addEventListener("otc:load-example", (e) => {
  const {
    outline, params = {}, label = "example",
    // The sensorize zone, when the design carried one. Same frame as
    // `outline`; absent for every built-in example.
    fillRegion = null,
    // A real robot surface is a shape, not a finished design: which edge the
    // connector exits is a decision about the user's own build, so those
    // examples require the same explicit Set cable edge as a DXF import (and
    // so cannot auto-generate, since Generate is blocked until it is set).
    // The simple built-in shapes carry a designed cable edge and generate
    // immediately, which makes them a one-click demonstration.
    requireCableEdge = false,
    autoGenerate = true,
  } = e.detail || {};
  if (!Array.isArray(outline) || outline.length < 3) return;
  if (!confirmOverwrite(vertices.length, "board outline")) return;
  pushUndo();
  editTarget = "outline";
  updateTargetButtons();
  vertices = outline.map((p) => [p[0], p[1]]);
  fillVertices = Array.isArray(fillRegion) && fillRegion.length >= 3
    ? fillRegion.map((p) => [p[0], p[1]])
    : [];
  foldLines = [];
  cableEdgeConfirmed = !requireCableEdge;
  selected = null;
  angleReferenceEdge = null;
  const extras = applyParamsToForm(params);
  // Examples and design files both store explicit taxel sizes, never row/column
  // targets — a stored recipe carries the solved size, so re-solving it here
  // would be a second answer to a question that already has one.
  if (gridSizeModeInput) gridSizeModeInput.checked = true;
  enforcePitchFloor();
  syncGridMode();
  renderPixelPreview();
  fitViewTo(vertices);
  redraw();
  setOutlineInfo(fillVertices.length ? `${label} · with a sensorize zone` : `example: ${label}`);
  syncDimensionFields();
  updateCadButtons();
  if (requireCableEdge) {
    showCableEdgeBanner();
    genStatus.textContent =
      `${label} loaded — choose where the connector tail exits, then press Generate PCB.`;
  } else {
    hideCableEdgeBanner();
  }
  if (autoGenerate) runGenerate(extras);
});

// Exact restore of a previously downloaded design (design-import.js): the
// board comes back with its hand-dragged routes, tail extension and connector
// size intact, so nothing is re-routed and nothing can come out different.
//
// It arrives as a rebuilt board rather than as inputs, so this is deliberately
// not a variant of otc:load-example: the canvas and form are synced to it for
// further editing, but the preview and ZIP are the restored ones.
window.addEventListener("otc:restore-design", (e) => {
  const { outline, params = {}, fillRegion = null, editData, svg, stats, drc,
          zipB64, label = "design",
          // A board recovered from a pre-design-block manifest: its geometry is
          // exactly as fabricated, but the wiring was never in the file. The
          // case is geometry and still works; route editing
          // and a manufacturing re-export do not, and must not be offered as
          // if they did.
          geometryOnly = false, notes = [] } = e.detail || {};
  if (!editData || !svg) return;
  if (!confirmOverwrite(vertices.length, "board outline")) return;
  pushUndo();
  editTarget = "outline";
  updateTargetButtons();
  vertices = (outline || editData.outline).map((p) => [p[0], p[1]]);
  fillVertices = Array.isArray(fillRegion) && fillRegion.length >= 3
    ? fillRegion.map((p) => [p[0], p[1]])
    : [];
  foldLines = [];
  // A restored board was generated from this outline, so its cable edge was
  // settled when it was made — asking again would block Generate on a design
  // that is already built.
  cableEdgeConfirmed = true;
  selected = null;
  angleReferenceEdge = null;
  applyParamsToForm(params);
  if (gridSizeModeInput) gridSizeModeInput.checked = true;
  enforcePitchFloor();
  syncGridMode();
  renderPixelPreview();
  fitViewTo(vertices);
  redraw();
  setOutlineInfo(`reopened: ${label}`);
  syncDimensionFields();
  updateCadButtons();
  hideCableEdgeBanner();

  // Any case in the page belongs to whatever board was here before. Drop it
  // before the restored board lands.
  window.otcCaseInvalidate?.();
  setZipNote("");
  // The restored ZIP is the board as reopened, so Download ZIP can hand it
  // over unchanged until something is edited.
  routeEditor.load(editData, svg, geometryOnly ? null : (zipB64 || null));
  // Both paths enable the case: it is built from the board's geometry, which a
  // recovered design has in full.
  window.otcCaseBoardReady?.(true);

  if (geometryOnly) {
    // No ZIP and no DRC to show: nothing here was re-exported, because the
    // copper cannot be reproduced. Say that plainly instead of leaving a
    // Download button that would hand over an empty or wrong board.
    lastZipB64 = null;
    downloadBtn.disabled = true;
    setPrintPdf(null);
    renderRecoveredStats(editData, notes);
    genStatus.innerHTML =
      `Reopened <strong>${label}</strong> as a finished board. Build a case from it ` +
      `below — it is shaped from the board, so it fits the one you ` +
      `already had made. Downloading new manufacturing files is not possible: the wiring ` +
      `was never stored in this file.`;
    revealResult();
    return;
  }

  lastZipB64 = zipB64 || null;
  downloadBtn.disabled = !lastZipB64;
  setPrintPdf(stats);
  renderStats(stats, drc);
  genStatus.textContent = `Reopened ${label}. Edit and re-download, or press Generate PCB to rebuild it from its settings.`;
  revealResult();
});

// The stats panel for a recovered board. renderStats cannot be reused: it
// reads connector, board mode and DRC off a generate response, and none of
// those exist here — inventing them would be the one thing this whole path
// is trying to avoid.
function renderRecoveredStats(editData, notes) {
  const pts = editData.final_outline || editData.outline || [];
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const w = xs.length ? Math.max(...xs) - Math.min(...xs) : 0;
  const h = ys.length ? Math.max(...ys) - Math.min(...ys) : 0;
  const p = editData.params || {};
  const rows = [
    ["Taxels", `${(editData.layout?.placements_xy || []).length}`],
    ["Board size", `${w.toFixed(1)} × ${h.toFixed(1)} mm, exactly as manufactured`],
    ["Taxel", `${p.pixel_w_mm} × ${p.pixel_h_mm} mm on a ${p.pitch_x_mm} × ${p.pitch_y_mm} mm pitch`],
    ["Recovered from", `${editData.recovered_from || "an older export"}`],
  ];
  // Same <table> markup renderStats uses, so this inherits the panel's styling
  // instead of needing classes of its own.
  statsEl.innerHTML =
    "<table>" +
    rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("") +
    "</table>" +
    (notes.length
      ? "<details class=\"stats-details\" open><summary>About this recovered board</summary>" +
        `<ul>${notes.map((n) => `<li>${n}</li>`).join("")}</ul></details>`
      : "");
}

// ---- server status ----
// On the hosted site (GitHub Pages) API_BASE points to the hosted server. It
// sleeps after ~15 minutes without requests and takes 30-60 s to start, and a
// request sent meanwhile is held until it is up. A local `opentac cad` serves
// this page itself (API_BASE is ""), so it is already up and never sleeps, and
// the notice below never shows there.
//
// Generate waits for the server rather than firing into a sleeping one: the
// status says it is waiting for the server, and the request goes out when
// /health answers. A request sent into a sleeping server would otherwise sit
// for a minute with no explanation, which reads as a hang.
//
// Three rules prevent a false "unreachable" badge:
//  * It never latches: after the wake window it re-probes every 30 s, so a
//    server that comes up late still turns it green.
//  * A successful API response marks the server up: a response is proof of
//    life, whatever the last probe said.
//  * While the page's own request is in flight it does not probe: the single
//    worker is busy computing that request and cannot answer /health, which
//    is expected, not an outage.
const HOSTED = API_BASE !== "";
const WAKE_DEADLINE_MS = 90000;          // covers a cold start with room to spare
const IDLE_SLEEP_MS = 14 * 60 * 1000;    // a little under the server's 15-minute idle timeout
const NOTICE_DELAY_MS = 1500;            // an awake server answers well within this
const healthEl = document.getElementById("health");
const serverNoticeEl = document.getElementById("server-notice");
let serverUp = false;
let lastServerAnswer = 0;
let probing = false;
let wakeStarted = 0;
let noticeTimer = null;
let noticeTicker = null;
let upWaiters = [];
// A probe that fails at once, again and again, is not reaching the network at
// all: an ad or privacy blocker cancels requests to .../health
// (net::ERR_BLOCKED_BY_CLIENT). A sleeping server holds the request open while
// it boots instead. With the probe blocked the page cannot know the server's
// state, so it stops waiting on it: requests go straight out, and the first
// one after a quiet spell takes about a minute.
let probeBlocked = false;
let fastProbeFailures = 0;
const PROBE_FAST_FAIL_MS = 1500;
let generateInFlight = false;
// Bumped by every Generate press. A press that waited for the server runs only
// if no later press came in meanwhile.
let generateTicket = 0;

function setBadge(kind, text, title = "") {
  if (!healthEl) return;
  healthEl.textContent = text;
  healthEl.className = `health health--${kind}`;
  healthEl.title = title;
}

function wakeNoticeText() {
  const waited = Date.now() - wakeStarted;
  const s = Math.round(waited / 1000);
  if (waited < WAKE_DEADLINE_MS) {
    return `Starting the server, which takes about a minute after a quiet spell (${s} s so far). ` +
      "You can keep working; Generate runs once the server is up.";
  }
  return `The server has not answered for ${s} s. Still trying every 30 s; ` +
    "Generate runs once it answers. Reloading the page also retries.";
}

function showWakeNotice() {
  if (serverUp) return;
  if (healthEl?.textContent === "Server: checking…") {
    setBadge("unknown", "Server: starting…", "The server sleeps when idle and takes about a minute to start.");
  }
  if (!HOSTED || !serverNoticeEl) return;
  serverNoticeEl.textContent = wakeNoticeText();
  serverNoticeEl.classList.toggle("server-notice--late", Date.now() - wakeStarted >= WAKE_DEADLINE_MS);
  serverNoticeEl.hidden = false;
  if (!noticeTicker) noticeTicker = setInterval(showWakeNotice, 1000);
}

function hideWakeNotice() {
  clearTimeout(noticeTimer);
  noticeTimer = null;
  clearInterval(noticeTicker);
  noticeTicker = null;
  if (serverNoticeEl) serverNoticeEl.hidden = true;
}

function markBackendAlive() {
  serverUp = true;
  lastServerAnswer = Date.now();
  setBadge("ok", "Server: ready");
  hideWakeNotice();
  const waiting = upWaiters;
  upWaiters = [];
  waiting.forEach((resolve) => resolve());
}

function giveUpProbing() {
  probeBlocked = true;
  probing = false;
  hideWakeNotice();
  setBadge("unknown", "Server: status unknown",
    "This browser blocks the page's server check (often an ad or privacy blocker). " +
    "Generate still works; the first one after a quiet spell can take about a minute.");
  const waiting = upWaiters;
  upWaiters = [];
  waiting.forEach((resolve) => resolve());
}

// Probe /health until it answers. Retries rather than probing once, because
// the first visitor after a quiet spell always hits a booting server.
function startProbing() {
  if (probeBlocked) { giveUpProbing(); return; }
  if (probing) return;
  probing = true;
  serverUp = false;
  wakeStarted = Date.now();
  let delay = 1000;
  setBadge("unknown", "Server: checking…");
  clearTimeout(noticeTimer);
  // Not at once: an awake server answers within a few hundred ms, and a notice
  // that flashes up and away again is noise.
  noticeTimer = setTimeout(showWakeNotice, NOTICE_DELAY_MS);

  async function attempt() {
    if (serverUp) { probing = false; return; }   // another response got there first
    if (generateInFlight || zipInFlight) {
      setBadge("unknown", "Server: busy", zipInFlight
        ? "Adding the printed case to the download; the server answers again when it finishes."
        : "A generate is running; the server answers again when it finishes.");
      setTimeout(attempt, 5000);
      return;
    }
    const sent = Date.now();
    try {
      // Per-try timeout: a request to a waking instance can hang well past the
      // point where retrying is the better move.
      const ctl = new AbortController();
      const t = setTimeout(() => ctl.abort(), 10000);
      const r = await fetch(`${API_BASE}/health`, { signal: ctl.signal, cache: "no-store" });
      clearTimeout(t);
      if (r.ok) {
        probing = false;
        fastProbeFailures = 0;
        markBackendAlive();
        return;
      }
      fastProbeFailures = 0;
    } catch (err) {
      // A TypeError straight away: the request never left the browser.
      const instant = err instanceof TypeError && Date.now() - sent < PROBE_FAST_FAIL_MS;
      fastProbeFailures = instant ? fastProbeFailures + 1 : 0;
      if (HOSTED && fastProbeFailures >= 2) { giveUpProbing(); return; }
    }
    if (serverUp) { probing = false; return; }
    if (Date.now() - wakeStarted < WAKE_DEADLINE_MS) {
      setBadge("unknown", "Server: starting…",
        "The server sleeps when idle and takes about a minute to start.");
      setTimeout(attempt, delay);
      delay = Math.min(delay * 1.6, 8000);
      return;
    }
    setBadge("down", "Server: not reachable",
      `No answer from ${API_BASE || "the server"} for ${WAKE_DEADLINE_MS / 1000} s. Retrying every 30 s.`);
    setTimeout(attempt, 30000);   // never latch
  }
  attempt();
}

// Whether a request can go out now. On the hosted site a server this page has
// not heard from for longer than the server's idle timeout has most likely gone
// back to sleep, so it counts as not ready until /health answers again.
function serverReadyNow() {
  if (probeBlocked) return true;
  if (!serverUp) return false;
  if (!HOSTED || generateInFlight || zipInFlight) return true;
  return Date.now() - lastServerAnswer < IDLE_SLEEP_MS;
}

// Resolves once the server answers (at once if it is known to be up).
function ensureServer() {
  if (serverReadyNow()) return Promise.resolve();
  const ready = new Promise((resolve) => upWaiters.push(resolve));
  startProbing();
  return ready;
}

// For the other modules that call the server (design-import.js).
window.otcServer = { ready: ensureServer, answered: markBackendAlive };

// ---- generate ----
const genStatus = document.getElementById("gen-status");
const statsEl = document.getElementById("stats");
const pcbPreview = document.getElementById("pcb-preview");
const downloadBtn = document.getElementById("download");
const routeEditStatus = document.getElementById("route-edit-status");
let lastZipB64 = null;
const zipNoteEl = document.getElementById("zip-status");
// True while Download ZIP waits on the backend for the case. Like
// generateInFlight, it stops the health ping reading a busy worker as down.
let zipInFlight = false;
// The case takes a few seconds locally and far longer on the hosted server. Past
// this, give up on it and hand over the board ZIP rather than keep waiting.
const CASE_TIMEOUT_MS = 120000;

function saveZip(zipB64, filename) {
  const bin = atob(zipB64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const blob = new Blob([bytes], { type: "application/zip" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function setZipNote(text, warn = false) {
  if (!zipNoteEl) return;
  zipNoteEl.textContent = text || "";
  zipNoteEl.classList.toggle("download-bar__note--warn", !!warn);
}

// The ZIP the user downloads carries the printed case (case/: the bottom and
// all three lids). It is added here, at download time, by /zip-with-case, not
// by /generate: building three lids would slow every generate on the hosted
// server, and most generates are never downloaded. The case is built from the
// case panel's settings (its defaults if untouched).
//
// It never holds back the board: if the case cannot be built, or the request
// fails or times out, the board ZIP is saved as it is and the note says why.
async function addCaseToZip(zipB64) {
  const editData = routeEditor.getEditData();
  if (!editData) {
    return { zipB64, note: "The printed case was left out: there is no board geometry to build it from.", warn: true };
  }
  const settings = window.otcCaseZipSettings?.() || {};
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), CASE_TIMEOUT_MS);
  zipInFlight = true;
  try {
    const resp = await fetch(`${API_BASE}/zip-with-case`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        zip_b64: zipB64,
        edit_data: editData,
        params: settings.params || null,
        bump_params: settings.bump_params || null,
        primary: settings.primary || "bumps",
        name: downloadName("", "zip").replace(/\.zip$/, ""),
      }),
      signal: ctl.signal,
    });
    if (!resp.ok) {
      const detail = await resp.json().catch(() => ({}));
      throw new Error(detail.detail || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    markBackendAlive();
    const n = (data.case_files || []).length;
    if (data.case_error) {
      return { zipB64: data.zip_b64 || zipB64, note: `${data.case_error} The board files are complete.`, warn: true };
    }
    const w = (data.warnings || []).length;
    const warnings = w ? ` The case builder gave ${w} warning${w > 1 ? "s" : ""}, listed in case/README.txt.` : "";
    return { zipB64: data.zip_b64, note: `The printed case is in the ZIP's case/ folder: ${n} STL files and a README.${warnings}`, warn: false };
  } catch (err) {
    const why = err.name === "AbortError" ? "building it took too long" : err.message.replace(/\.$/, "");
    return { zipB64, note: `The printed case was left out: ${why}. The board files are complete.`, warn: true };
  } finally {
    clearTimeout(timer);
    zipInFlight = false;
  }
}

async function downloadZip(zipB64, filename) {
  if (!zipB64) return;
  setZipNote("Preparing download… adding the printed case.");
  const out = await addCaseToZip(zipB64);
  saveZip(out.zipB64, filename);
  setZipNote(out.note, out.warn);
}

const routeEditor = createRouteEditor({
  apiBase: API_BASE,
  previewEl: pcbPreview,
  statusEl: routeEditStatus,
  downloadZip,
  renderStats,
  readParams,
  onServerAnswer: markBackendAlive,
});



const boardModeInputs = Array.from(document.querySelectorAll('input[name="board_mode"]'));
function selectedBoardMode() {
  const picked = boardModeInputs.find((el) => el.checked);
  return picked ? picked.value : "expand";
}

// Copper layers: 2 or 4, independent of the board mode. Only the hug router
// wires the inner layers (the legacy router uses 2); the page always requests
// the hug router.
const copperLayerInputs = Array.from(document.querySelectorAll('input[name="copper_layers"]'));
function selectedCopperLayers() {
  const picked = copperLayerInputs.find((el) => el.checked);
  return picked && picked.value === "4" ? 4 : 2;
}


// Sensor grid: either the taxel size is given and the count follows, or the
// count is given and the backend solves the size (grow-board mode only — see
// _apply_target_grid in main.py).
const gridModeInputs = Array.from(document.querySelectorAll('input[name="grid_mode"]'));
const gridSizeInputIds = ["pixel_w_mm", "pixel_h_mm", "pitch_x_mm", "pitch_y_mm"];
const gridSizeFieldsNote = document.getElementById("grid-size-note");
const gridCountFields = document.getElementById("grid-count-fields");
const gridCountHint = document.getElementById("grid-count-hint");
const gridCountModeInput = gridModeInputs.find((el) => el.value === "count");
const gridSizeModeInput = gridModeInputs.find((el) => el.value === "size");

function selectedGridMode() {
  const picked = gridModeInputs.find((el) => el.checked);
  return picked ? picked.value : "size";
}

function syncGridMode() {
  // Rows/cols has no meaning once the outline is fixed and the keep-out band is
  // what decides the packable area, so fall back to size mode there.
  const countAllowed = selectedBoardMode() === "expand";
  if (gridCountModeInput) {
    gridCountModeInput.disabled = !countAllowed;
    if (!countAllowed && gridCountModeInput.checked && gridSizeModeInput) {
      gridSizeModeInput.checked = true;
    }
  }
  const countMode = selectedGridMode() === "count";
  // The size fields stay visible in count mode rather than being swapped out:
  // they are where the solved taxel size is reported back after Generate (see
  // showSolvedGridSize), and they feed the pixel footprint preview. Read-only,
  // since in this mode they are an output.
  for (const id of gridSizeInputIds) {
    const el = document.getElementById(id);
    if (!el) continue;
    el.readOnly = countMode;
    el.classList.toggle("is-derived", countMode);
  }
  if (gridSizeFieldsNote) gridSizeFieldsNote.hidden = !countMode;
  if (gridCountFields) gridCountFields.hidden = !countMode;
  if (gridCountHint) gridCountHint.hidden = !countMode;
}

// Write the taxel size the backend solved back into the (read-only) size
// fields, so the numbers and the footprint preview show what was actually
// built rather than the stale values that were sent and ignored.
function showSolvedGridSize(editData) {
  if (selectedGridMode() !== "count") return;
  const solved = editData?.params;
  if (!solved) return;
  for (const id of gridSizeInputIds) {
    const el = document.getElementById(id);
    if (el && Number.isFinite(solved[id])) el.value = String(Number(solved[id].toFixed(4)));
  }
  renderPixelPreview();
}

for (const el of [...gridModeInputs, ...boardModeInputs]) {
  el.addEventListener("change", syncGridMode);
}
syncGridMode();

// Pitch can never be smaller than the pixel — cells would overlap. Whenever
// either side of a pair changes, drag the pitch up to at least the pixel
// size ("change", not "input", so typing "2.5" isn't clamped mid-keystroke).
const pitchFloorPairs = [["pixel_w_mm", "pitch_x_mm"], ["pixel_h_mm", "pitch_y_mm"]];
function enforcePitchFloor() {
  for (const [pxId, pitchId] of pitchFloorPairs) {
    const px = parseFloat(document.getElementById(pxId)?.value);
    const pitchEl = document.getElementById(pitchId);
    if (!pitchEl || !Number.isFinite(px)) continue;
    const pitch = parseFloat(pitchEl.value);
    if (!Number.isFinite(pitch) || pitch < px) {
      pitchEl.value = String(px);
    }
    pitchEl.min = String(px);
  }
}
for (const [pxId, pitchId] of pitchFloorPairs) {
  document.getElementById(pxId)?.addEventListener("change", enforcePitchFloor);
  document.getElementById(pitchId)?.addEventListener("change", enforcePitchFloor);
}
enforcePitchFloor();
function readParams() {
  const ids = ["pixel_w_mm", "pixel_h_mm", "pitch_x_mm", "pitch_y_mm",
    "trace_w_mm", "gap_mm", "clearance_mm", "edge_keepout_mm", "board_edge_clear_mm",
    "center_clear_mm", "edge_clear_mm", "via_drill_mm", "via_dia_mm",
    "cable_length_mm", "tail_width_mm", "tail_side_padding_mm"];
  const p = {};
  for (const id of ids) {
    const value = document.getElementById(id).value;
    const parsed = parseFloat(value);
    if (id === "tail_width_mm" || id === "cable_length_mm") {
      // blank: automatic (the narrow tail's width; the tail's length)
      p[id] = Number.isFinite(parsed) && parsed > 0 ? parsed : null;
    } else {
      p[id] = Number.isFinite(parsed) ? parsed : 0;
    }
  }
  // Final guard for the pitch >= pixel rule (the inputs already clamp).
  p.pitch_x_mm = Math.max(p.pitch_x_mm, p.pixel_w_mm);
  p.pitch_y_mm = Math.max(p.pitch_y_mm, p.pixel_h_mm);
  p.board_mode = selectedBoardMode();
  p.copper_layers = selectedCopperLayers();
  p.readout_board = document.querySelector('input[name="readout_board"]:checked')?.value || "auto";
  p.contact_side = document.querySelector('input[name="contact_side"]:checked')?.value || "bottom";
  p.router = "hug";
  // In rows/cols mode the backend solves pixel size and pitch from the outline,
  // so the pixel_*/pitch_* values above are sent but ignored. Only send the
  // targets in that mode — sending them always would override the size fields.
  if (selectedGridMode() === "count") {
    p.target_cols = Math.max(1, Math.round(parseFloat(document.getElementById("target_cols").value) || 1));
    p.target_rows = Math.max(1, Math.round(parseFloat(document.getElementById("target_rows").value) || 1));
  }
  // Also sent for a backend that predates board_mode; the current backend
  // derives both from board_mode (_params_for_board_mode in main.py).
  p.preserve_sensors = true;
  p.auto_expand_board = p.board_mode === "expand";
  // Printed on the front silkscreen just inboard of the connector. The backend
  // re-sanitises it and drops it if it will not fit.
  p.design_name = designStem();
  // Routing margin, follow padding and the connector shoulder fillet are not on
  // the page: the backend's defaults apply, and fixed-outline mode turns the
  // follow padding off itself (_without_expansion_padding in main.py).
  return p;
}

// ---- pixel footprint preview ----
// Mirrors export/kicad_footprint.py's pixel_geometry() exactly (same
// variable names/formulas) so this preview always matches the footprint a
// real Generate call would produce.
const pixelPreviewIds = ["pixel_w_mm", "pixel_h_mm", "trace_w_mm", "gap_mm", "center_clear_mm", "edge_clear_mm", "via_drill_mm", "via_dia_mm"];
const pixelPreviewEl = document.getElementById("pixel-preview");

function getNum(id) {
  const el = document.getElementById(id);
  return el ? parseFloat(el.value) : 0;
}

function renderPixelPreview() {
  if (!pixelPreviewEl) return;
  const w = getNum("pixel_w_mm");
  const h = getNum("pixel_h_mm");
  const tw = getNum("trace_w_mm");
  const gp = getNum("gap_mm");
  const cc = getNum("center_clear_mm");
  const ec = getNum("edge_clear_mm");
  const vDrill = getNum("via_drill_mm");
  const vDia = getNum("via_dia_mm");

  const x1 = -w / 2, y1 = -h / 2, x2 = w / 2, y2 = h / 2;
  const ix1 = x1 + ec, iy1 = y1 + ec;
  const ix2 = x2 - ec, iy2 = y2 - ec;
  const innerW = ix2 - ix1;
  const innerH = iy2 - iy1;

  if (!(innerW > tw) || !(innerH > tw)) {
    pixelPreviewEl.innerHTML = '<div class="pixel-info">Invalid: taxel edge clearance too large for this taxel size and trace width</div>';
    return;
  }

  const xL = ix1 + tw / 2;
  const xR = ix2 - tw / 2;
  const ymid = (iy1 + iy2) / 2;
  const pitch = tw + gp;
  const totalSlots = Math.max(1, Math.floor((innerH + gp) / pitch));
  const span = totalSlots * tw + (totalSlots - 1) * gp;
  const y0 = (iy1 + iy2 - span) / 2 + tw / 2;
  const spineYlo = iy1 + tw / 2;
  const spineYhi = iy2 - tw / 2;
  const xEndLeftMax = xR - (cc + tw);
  const xEndRightMin = xL + (cc + tw);
  const viaX = ix1 + vDia / 2;
  const viaY = ymid;

  const pad = Math.max(w, h) * 0.12;
  const labelSpace = 0.5;
  const svgX1 = x1 - pad;
  const svgW = w + 2 * pad;
  const svgH = h + 2 * pad + labelSpace;
  const fy = (v) => (pad + labelSpace) + (y2 - v);

  const col1 = "#2673d9";
  const col2 = "#d94026";
  const strokeW = tw;

  let svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${svgX1.toFixed(3)} 0 ${svgW.toFixed(3)} ${svgH.toFixed(3)}">`;
  svg += `<rect x="${x1}" y="${fy(y2)}" width="${w}" height="${h}" fill="none" stroke="#666" stroke-width="0.06"/>`;
  svg += `<rect x="${ix1}" y="${fy(iy2)}" width="${innerW}" height="${innerH}" fill="none" stroke="#aaa" stroke-width="0.03" stroke-dasharray="0.1"/>`;
  svg += `<line x1="${xL}" y1="${fy(spineYlo)}" x2="${xL}" y2="${fy(spineYhi)}" stroke="${col1}" stroke-width="${strokeW}" stroke-linecap="butt"/>`;
  svg += `<line x1="${xR}" y1="${fy(spineYlo)}" x2="${xR}" y2="${fy(spineYhi)}" stroke="${col2}" stroke-width="${strokeW}" stroke-linecap="butt"/>`;
  for (let i = 0; i < totalSlots; i++) {
    const yy = y0 + i * pitch;
    if (i % 2 === 0) {
      svg += `<line x1="${xL}" y1="${fy(yy)}" x2="${xEndLeftMax}" y2="${fy(yy)}" stroke="${col1}" stroke-width="${strokeW}" stroke-linecap="butt"/>`;
    } else {
      svg += `<line x1="${xR}" y1="${fy(yy)}" x2="${xEndRightMin}" y2="${fy(yy)}" stroke="${col2}" stroke-width="${strokeW}" stroke-linecap="butt"/>`;
    }
  }
  svg += `<circle cx="${viaX}" cy="${fy(viaY)}" r="${vDia / 2}" fill="${col1}" fill-opacity="0.3" stroke="${col1}" stroke-width="0.02"/>`;
  svg += `<circle cx="${viaX}" cy="${fy(viaY)}" r="${vDrill / 2}" fill="white" stroke="#555" stroke-width="0.02"/>`;
  const labelY = fy(y2) - 0.15;
  svg += `<text x="${xL}" y="${labelY}" text-anchor="middle" font-size="0.35" fill="${col1}" font-weight="bold">Pad 1</text>`;
  svg += `<text x="${xR}" y="${labelY}" text-anchor="middle" font-size="0.35" fill="${col2}" font-weight="bold">Pad 2</text>`;
  svg += "</svg>";
  svg += `<div class="pixel-info">${w.toFixed(1)} × ${h.toFixed(1)} mm · ${totalSlots} fingers · trace ${tw} gap ${gp}</div>`;

  pixelPreviewEl.innerHTML = svg;
}

pixelPreviewIds.forEach((id) => {
  document.getElementById(id)?.addEventListener("input", renderPixelPreview);
});

// The /generate body for what is on the canvas and in the form, or null with
// the reason shown in the status line.
function generateBody(overrides) {
  if (vertices.length < 3) {
    genStatus.innerHTML = `<span class="error">Need at least 3 outline vertices.</span>`;
    return null;
  }
  if (!cableEdgeConfirmed) {
    genStatus.innerHTML =
      `<span class="error">Cable edge not set — click the edge where the connector ` +
      `tail should exit, then press <strong>Set cable edge</strong>.</span>`;
    showCableEdgeBanner();
    return null;
  }
  const body = {
    outline: vertices,
    ...(fillVertices.length >= 3 ? { fill_region: fillVertices } : {}),
    ...readParams(),
    ...overrides,
  };
  // Catch a hopeless size mismatch before the round trip: an outline smaller
  // than a single taxel (common after unfolding one tiny STL face) would come
  // back as a bare "No pixels packed" with no hint at which number is wrong.
  if (!body.target_cols) {
    const xs = vertices.map((v) => v[0]), ys = vertices.map((v) => v[1]);
    const w = Math.max(...xs) - Math.min(...xs);
    const h = Math.max(...ys) - Math.min(...ys);
    const needW = body.pixel_w_mm + 2 * body.edge_keepout_mm;
    const needH = body.pixel_h_mm + 2 * body.edge_keepout_mm;
    if (w < needW || h < needH) {
      genStatus.innerHTML =
        `<span class="error">Your outline is ${w.toFixed(1)} × ${h.toFixed(1)} mm, but a single taxel plus ` +
        `its edge keepout needs ${needW.toFixed(1)} × ${needH.toFixed(1)} mm. Make the outline bigger, or ` +
        `reduce Taxel W/H or Edge keepout.</span>`;
      return null;
    }
  }
  return body;
}

// Backend messages that use the internal "pixel" terminology, restated in the
// page's "taxel" wording. The backend's own text is left alone: the pipeline
// matches on it.
function readableError(err) {
  const msg = String(err?.message || err);
  if (/^No pixels packed/.test(msg)) {
    return "No taxels fit in the outline. Make the outline larger, or reduce the taxel size, pitch or edge keepout.";
  }
  if (/^edgeClearMm too large/.test(msg)) {
    return "Taxel edge clearance is too large for this taxel size and trace width.";
  }
  // fetch() rejects with a TypeError when no answer came back at all.
  if (err instanceof TypeError) {
    return "No answer from the server. See the Server badge at the top right, and try again when it says ready.";
  }
  return msg;
}

async function runGenerate(overrides = {}) {
  if (!generateBody(overrides)) return;
  const ticket = ++generateTicket;
  if (!serverReadyNow()) {
    genStatus.textContent = HOSTED
      ? "Waiting for the server to start (about a minute after a quiet spell). The board is generated as soon as it is up."
      : "Waiting for the server to answer. The board is generated as soon as it does.";
    await ensureServer();
    // Pressed again while waiting: the later press is the one that runs.
    if (ticket !== generateTicket) return;
  }
  // Built after the wait, so a queued Generate uses the outline and settings
  // as they are when it runs, not as they were when it was pressed.
  const body = generateBody(overrides);
  if (!body) return;
  genStatus.textContent = "Generating… (may take a few seconds)";
  generateInFlight = true;
  pcbPreview.classList.remove("pcb-preview--stale");
  statsEl.innerHTML = "";
  downloadBtn.disabled = true;
  setPrintPdf(null);
  lastZipB64 = null;
  setZipNote("");
  clearPrinted();
  // Any case already built belongs to the board being replaced. Drop it before
  // the new one lands, so a stale STL can never be downloaded as if it fitted.
  window.otcCaseInvalidate?.();
  try {
    const resp = await fetch(`${API_BASE}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const detail = await resp.json().catch(() => ({}));
      throw new Error(detail.detail || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    markBackendAlive();
    showSolvedGridSize(data.edit_data);
    if (data.edit_data) {
      // The generate ZIP goes with it: Download ZIP uses it as it is unless the
      // board is edited first (route-editor.js currentZip).
      routeEditor.load(data.edit_data, data.svg, data.zip_b64);
    } else {
      pcbPreview.innerHTML = data.svg;
      routeEditStatus.textContent = "";
    }
    lastZipB64 = data.zip_b64;
    markPrinted(body.design_name ?? "");
    downloadBtn.disabled = false;
    setPrintPdf(data.stats);
    window.otcCaseBoardReady?.(!!data.edit_data);
    renderStats(data.stats, data.drc);
    genStatus.textContent = "Done.";
    revealResult();
  } catch (err) {
    genStatus.innerHTML = `<span class="error">Error: ${readableError(err)}</span>`;
  } finally {
    generateInFlight = false;
  }
}

// After Generate (or Open), bring the result into view: the status and stats,
// with the PCB preview under them. Only when the preview is mostly off screen,
// so someone already looking at it is not moved.
function revealResult() {
  const toolView = document.getElementById("view-tool");
  const panel = document.getElementById("pcb-preview-panel");
  const target = document.querySelector(".generate-panel") || panel;
  if (!toolView || toolView.hidden || !panel || !target) return;
  // After layout: the stats and the preview SVG were only just inserted.
  requestAnimationFrame(() => {
    const r = panel.getBoundingClientRect();
    const vh = window.innerHeight || document.documentElement.clientHeight;
    const shown = Math.max(0, Math.min(r.bottom, vh) - Math.max(r.top, 0));
    if (shown >= Math.min(r.height, vh) * 0.5) return;
    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    target.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
  });
}

// ---- mirror ----
// Flips the whole design in place: outline, sensorize zone and any fold lines
// together, about the outline's bounding-box centre. Mirroring only the piece
// you happen to be editing would slide the sensorize zone off the board.
//
// The winding fix is the subtle part. Mirroring reverses a polygon's
// orientation, and the pipeline requires counter-clockwise: geometry.ensure_ccw
// pins vertex 0 and reverses the rest, so handing it a clockwise polygon does
// not merely flip the winding back -- it silently swaps the cable edge for the
// other edge meeting vertex 0, and the board generates fine with its tail in
// the wrong place. Reversing the cycle while rotating it to start at the
// pre-mirror vertex 1 restores counter-clockwise and keeps edge 0->1 the same
// physical edge, traversed the other way, so the cable edge survives the flip.
function mirrorCycleCCW(pts, flip) {
  const m = pts.map(flip);
  if (m.length < 3) return m;
  return [m[1], m[0], ...m.slice(2).reverse()];
}

function mirrorDesign(axis) {
  if (vertices.length < 3) return;
  const xs = vertices.map((v) => v[0]);
  const ys = vertices.map((v) => v[1]);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
  const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  const flip = axis === "h"
    ? (p) => [2 * cx - p[0], p[1]]
    : (p) => [p[0], 2 * cy - p[1]];

  pushUndo();
  vertices = mirrorCycleCCW(vertices, flip);
  // The fill zone carries no cable edge, so a plain reverse keeps it CCW.
  fillVertices = fillVertices.length >= 3
    ? fillVertices.map(flip).reverse()
    : fillVertices.map(flip);
  foldLines = foldLines.map(([a, b]) => [flip(a), flip(b)]);
  selected = null;
  previewPoint = null;
  arcState = null;
  angleReferenceEdge = null;
  redraw();
  setOutlineInfo(axis === "h" ? "mirrored left-right" : "mirrored top-bottom");
  updateCadButtons();
  syncDimensionFields();
}

cadMirrorHBtn?.addEventListener("click", () => mirrorDesign("h"));
cadMirrorVBtn?.addEventListener("click", () => mirrorDesign("v"));

// ---- 1:1 printable PDF ----
// The PDF is built by the same call that builds the ZIP and arrives inside the
// generate response, so this only hands over bytes it already has. Rebuilding
// the layout separately from edit_data would not match what the export does
// (tail extension, post-route containment, follow-padding and connector
// shoulder) and could produce a plausible-looking template of the wrong
// shape. One artefact, one producer.
const downloadPdfBtn = document.getElementById("download-pdf");
const pdfStatus = document.getElementById("pdf-status");
let lastPdf = null;

function setPrintPdf(stats) {
  lastPdf = stats?.print_pdf || null;
  if (downloadPdfBtn) downloadPdfBtn.disabled = !lastPdf;
  setVelostat(stats);
  if (!pdfStatus) return;
  pdfStatus.textContent = lastPdf
    ? (lastPdf.pages > 1
        ? `Printable: ${lastPdf.board_w_mm} × ${lastPdf.board_h_mm} mm over ${lastPdf.pages} sheets, at 100%.`
        : `Printable: ${lastPdf.board_w_mm} × ${lastPdf.board_h_mm} mm on one sheet, at 100%.`)
    : "";
}

// Route edits rebuild the board, so the template returned by the original
// generate is then for the wrong outline.
window.otcPrintPdfUpdated = (stats) => setPrintPdf(stats);

downloadPdfBtn?.addEventListener("click", () => {
  if (!lastPdf) return;
  const bin = atob(lastPdf.b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
  const url = URL.createObjectURL(new Blob([bytes], { type: "application/pdf" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = downloadName("1to1", "pdf");
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
});

// ---- cut files for the piezoresistive layer (Velostat, LINQSTAT, ...) ----
// Same producer as the printable PDF (the export call), same lifecycle: they
// ride on stats and are refreshed wherever setPrintPdf is. Too big to fold
// gives one un-mirrored copy (fold null); too big for even that, no PDF.
const downloadVeloDxfBtn = document.getElementById("download-velostat-dxf");
const downloadVeloPdfBtn = document.getElementById("download-velostat-pdf");
let lastVelostat = null;

function setVelostat(stats) {
  lastVelostat = stats?.velostat || null;
  if (downloadVeloDxfBtn) downloadVeloDxfBtn.disabled = !lastVelostat?.dxf_b64;
  if (downloadVeloPdfBtn) {
    const hasPdf = !!lastVelostat?.fold_pdf_b64;
    const folds = !lastVelostat || !hasPdf || !!lastVelostat.fold;
    downloadVeloPdfBtn.disabled = !hasPdf;
    downloadVeloPdfBtn.textContent = folds ? "Piezo layer fold PDF" : "Piezo layer PDF";
    downloadVeloPdfBtn.title = !lastVelostat || hasPdf
      ? (folds
          ? "The shape of the piezoresistive layer (e.g. Velostat, LINQSTAT) printed twice, mirrored across a fold line. Fold the paper with the sheet inside and cut both together."
          : "The shape of the piezoresistive layer (e.g. Velostat, LINQSTAT), too large to print twice on one sheet, so printed once at 1:1. Tape the sheet under the paper and cut both together.")
      : "This sensor is too large for one A3 sheet. Use the piezo layer DXF instead.";
  }
}

function saveB64(b64, type, filename) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
  const url = URL.createObjectURL(new Blob([bytes], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

downloadVeloDxfBtn?.addEventListener("click", () => {
  if (lastVelostat?.dxf_b64) saveB64(lastVelostat.dxf_b64, "application/dxf", downloadName("velostat", "dxf"));
});
downloadVeloPdfBtn?.addEventListener("click", () => {
  if (lastVelostat?.fold_pdf_b64) saveB64(lastVelostat.fold_pdf_b64, "application/pdf", downloadName("velostat_fold", "pdf"));
});

window.otcGetEditData = () => routeEditor.getEditData();

document.getElementById("generate").addEventListener("click", () => runGenerate());

// The OpenTac readout board a connector size belongs to (30-pin or 60-pin).
function readoutBoardText(connectorPos) {
  if (connectorPos === 30) return "30-pin board (30-position FPC connector)";
  if (connectorPos === 60) return "60-pin board (60-position FPC connector)";
  return `${connectorPos}-position FPC connector (no OpenTac readout board)`;
}

// "2", or "4 — N rows wired on the inner layers" (and how many stayed on
// the bottom copper, if any did).
function copperLayersText(stats) {
  if (stats.copper_layers !== 4) return "2";
  const inner = stats.rows_on_inner_layers || {};
  const n = (inner["In1.Cu"] || 0) + (inner["In2.Cu"] || 0);
  if (stats.router !== "hug") return "4 — inner layers unused (legacy router)";
  return `4 — ${n} row${n === 1 ? "" : "s"} wired on the inner layers`;
}

// "18.5 mm (auto: the printed case clears the connector)"
const TAIL_LENGTH_REASONS = {
  "printed case clearance": "the printed case clears the connector",
  "fit width": "long enough for the tail to narrow to its traces",
  "minimum 12 mm": "the 12 mm minimum",
};
function tailLengthText(stats) {
  const mm = `${stats.tail_length_mm} mm`;
  if (!stats.tail_length_auto) return mm;
  const why = TAIL_LENGTH_REASONS[stats.tail_length_reason] || stats.tail_length_reason || "";
  return why ? `${mm} (auto: ${why})` : `${mm} (auto)`;
}

function renderStats(stats, drc) {
  const tailWidth = `${stats.tail_width_mm} mm`;
  // Headline first: the handful of numbers a first-time user actually needs
  // to judge the result. The full engineering table (every routing/tail
  // metric) stays available under a collapsible "All details".
  const summary = [
    ["Taxels", `${stats.active_pixels}`],
    ["Readout board", readoutBoardText(stats.connector_pos)],
    ["Board mode", stats.board_mode === "fixed_keepout"
      ? "outline kept exactly as drawn"
      : "board grown to fit the wiring"],
    ["Copper layers", copperLayersText(stats)],
    ["Tail length", tailLengthText(stats)],
    ["Design check (DRC)", drc.violations === 0
      ? "0 violations — ready to manufacture"
      : `${drc.violations} violations — see details below`],
  ];
  if (stats.removed_pixels > 0) {
    summary.push(["Taxels dropped", `${stats.removed_pixels} (couldn't be wired cleanly — try a larger pitch or a simpler shape)`]);
  }
  if (stats.board_mode === "fixed_keepout" && (stats.dropped_by_keepout || 0) > 0) {
    summary.push(["Taxels dropped for wiring room", `${stats.dropped_by_keepout} of ${stats.max_pack_pixels} — grow-board mode keeps them all`]);
  }
  if (stats.connector_export_warning) {
    summary.push(["Connector warning", stats.connector_export_warning]);
  }
  if (stats.mask_note) {
    summary.push(["Solder mask", stats.mask_note]);
  }
  if (stats.routing_warning_nets > 0 || stats.routing_problem_pixels > 0) {
    summary.push(["Routing warnings", `${stats.routing_warning_nets || 0} nets / ${stats.routing_problem_pixels || 0} taxels`]);
  }
  const rows = [
    ["Total taxels", stats.total_pixels],
    ["Active taxels", stats.active_pixels],
    ["Removed taxels", stats.removed_pixels],
    ["Column routes", stats.col_routes],
    ["Row routes", stats.row_routes],
    ["Connector", `${stats.connector_pos}-pos${stats.connector_swap ? " (swapped row/column pad groups)" : ""}`],
    ["Connector export", stats.connector_export_enabled ? "on" : "off"],
    ...(stats.connector_export_warning ? [["Connector warning", stats.connector_export_warning]] : []),
    ["Board mode", stats.board_mode === "fixed_keepout" ? "keep outline fixed" : "grow to fit wiring"],
    ["Copper layers", copperLayersText(stats)],
    ...(stats.copper_layers === 4 ? [
      ["Rows on In1 / In2", `${(stats.rows_on_inner_layers || {})["In1.Cu"] || 0} / ${(stats.rows_on_inner_layers || {})["In2.Cu"] || 0}`],
      ["Rows linked across notches on In2", `${stats.rows_linked_on_in2 || 0}`],
    ] : []),
    ["Router", stats.router === "hug" ? "hug (wires follow the sensor shape)" : "legacy"],
    ...(stats.board_mode === "fixed_keepout" ? [
      ["Taxels dropped for wiring", `${stats.dropped_by_keepout || 0} of ${stats.max_pack_pixels || 0} max-packed`],
      ["Keep-out width", `${stats.fixed_keepout_min_mm || 0} mm thinnest → ${stats.fixed_keepout_max_mm || 0} mm at the tail`],
      ["Keep-out passes", `${stats.fixed_keepout_iterations || 0} carve, ${stats.fixed_keepout_route_attempts || 0} routing`],
      ["Keep-out routing", stats.fixed_keepout_routing_clean ? "clean"
        : `warnings — copper ${stats.fixed_keepout_worst_escape_mm || 0} mm past the edge`],
    ] : []),
    ["Fixed-board routing", stats.fixed_board_routing ? "on" : "off"],
    ["Perimeter routing attempted", stats.perimeter_routing_attempted ? "yes" : "no"],
    ...(stats.removal_reason ? [["Removal reason", stats.removal_reason]] : []),
    ["Removed by routing failure", stats.removed_by_routing_failure || 0],
    ["Bad nets before removal", stats.bad_nets_before_removal || 0],
    ["Routing margin", `${stats.routing_margin_mm} mm`],
    ["Applied expansion", `${stats.routing_margin_applied_mm} mm`],
    ["Left routing padding", `${stats.routing_left_padding_mm || 0} mm`],
    ["Right routing padding", `${stats.routing_right_padding_mm || 0} mm`],
    ["Side routing padding", `${stats.routing_side_padding_mm || 0} mm`],
    ["Top routing padding", `${stats.routing_top_padding_mm || 0} mm`],
    ["Bottom routing padding", `${stats.routing_bottom_padding_mm || 0} mm`],
    ["Routing routes counted", stats.routing_route_count || 0],
    ["Follow padding", stats.follow_main_padding ? "on" : "off"],
    ["Follow distance", `${stats.follow_main_padding_mm || 0} mm`],
    ["Smooth follow edge", stats.smooth_follow_padding ? "on" : "off"],
    ["Final outline mode", stats.final_outline_mode || "rectangular"],
    ["Tail shoulder expansion", stats.tail_shoulder_expansion ? `${stats.tail_shoulder_width_mm} mm` : "off"],
    ["Tail pocket depth", `${stats.tail_pocket_depth_mm || 0} mm`],
    ["Post-route containment", stats.post_route_containment_expansion ? `${stats.post_route_containment_buffer_mm} mm` : "off"],
    ["Max route outside", `${stats.max_route_outside_distance_mm || 0} mm`],
    ["Row routes left", stats.row_routes_left || 0],
    ["Row routes right", stats.row_routes_right || 0],
    ["Routing warning taxels", stats.routing_problem_pixels],
    ["Routing warning nets", stats.routing_warning_nets],
    ["Tail length", tailLengthText(stats)],
    ["Tail width", tailWidth],
    ["Tail side padding", `${stats.tail_side_padding_mm} mm`],
    ["Pad center span", `${stats.connector_pad_center_span_mm || 0} mm`],
    ["Copper span", `${stats.connector_copper_span_mm || 0} mm`],
    ["Tab tip width", `${stats.connector_tab_tip_width_mm || 0} mm`],
    ["Ear/stiffener width", `${stats.connector_ear_stiffener_width_mm || 0} mm`],
    ["Tail extension", stats.tail_extension_enabled ? `${stats.tail_extension_length_mm} mm` : "off"],
    ["Tail extension width", stats.tail_extension_enabled ? `${stats.tail_extension_width_mm} mm` : "off"],
    ["DRC violations", drc.violations],
  ];
  statsEl.innerHTML =
    "<table>" +
    summary.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("") +
    "</table>" +
    renderDrcDetails(drc) +
    "<details class=\"stats-details\"><summary>All routing &amp; tail details</summary><table>" +
    rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("") +
    "</table></details>";
}

function renderDrcDetails(drc) {
  if (!drc) return "";
  const routeWarnings = (drc.routing_warning_nets || []).slice(0, 8);
  const warningHtml = routeWarnings.length
    ? `<p><strong>Routing warnings:</strong> ${routeWarnings.join(", ")}</p>`
    : "";
  if (!drc.details || !drc.details.length) return warningHtml;
  const items = drc.details.slice(0, 6).map((v) => {
    const names = v.net2 === "BOARD_EDGE" ? `${v.net1} → edge` : `${v.net1} ↔ ${v.net2}`;
    const req = v.min_required == null ? "" : ` / ${v.min_required} mm required`;
    return `<li>${v.layer} ${v.type}: ${names}, ${v.distance} mm${req}</li>`;
  }).join("");
  return `<div class="drc-details">${warningHtml}<strong>DRC details</strong><ul>${items}</ul></div>`;
}

// ---- download zip (base64 -> Blob) ----
// Every path ends in downloadZip, which adds the printed case. The board ZIP
// itself is rebuilt by /export-edited only when the board was changed after
// Generate or Open (a moved or chamfered route, a tail extension, Redo
// padding); otherwise the ZIP already in hand is the same board, and
// rebuilding it would be a slow extra round trip on the hosted server. The
// button says it is working for the whole wait.
downloadBtn.addEventListener("click", async () => {
  if (zipInFlight || downloadBtn.disabled) return;
  const label = downloadBtn.textContent;
  downloadBtn.disabled = true;
  downloadBtn.textContent = "Preparing download…";
  setZipNote(serverReadyNow()
    ? "Preparing download…"
    : "Waiting for the server to start (about a minute after a quiet spell), then preparing the download…");
  try {
    await ensureServer();
    setZipNote("Preparing download…");
    if (routeEditor.hasEditableRoutes()) {
      const ready = routeEditor.currentZip();
      if (ready) {
        await downloadZip(ready, zipName({ edited: routeEditor.isEdited() }));
        return;
      }
      try {
        await routeEditor.exportEdited({ download: true });
      } catch (error) {
        routeEditStatus.innerHTML = `<span class="error">Edited export failed: ${readableError(error)}</span>`;
        setZipNote("");
      }
      return;
    }
    await downloadZip(lastZipB64, zipName({ edited: false }));
  } finally {
    downloadBtn.textContent = label;
    // A generate started meanwhile owns the button now.
    downloadBtn.disabled = generateInFlight || !lastZipB64;
  }
});

// First probe of the server; see "server status" above.
startProbing();

redraw();
setOutlineInfo();
syncDimensionFields();
updateCadButtons();
updateTargetButtons();
renderPixelPreview();
