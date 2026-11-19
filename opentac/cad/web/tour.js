// Interactive guided tour: a small dependency-free coachmark/spotlight
// engine driven by a declarative step list. One shared "spine" of steps
// covers Parameters/Generate/Preview/Download regardless of mode; only the
// opening step (how to get a shape) branches on which mode was chosen.
import { showView } from "./views.js";
import { applyModeVisibility, getSelectedMode } from "./landing.js";

const BASE_STEPS = [
  {
    view: "landing",
    title: "Welcome to OpenTac-CAD",
    body: "OpenTac-CAD lays out a piezoresistive tactile sensor on a flex PCB from a 2D outline: the taxels, the wiring to the connector, and the fabrication files. A taxel is one sensing element of the skin. This tour walks through the steps.",
  },
  {
    view: "landing",
    selector: ".landing-modes",
    title: "Pick how to start",
    body: "Every design begins with a shape: draw it by hand, import a DXF or SVG, or upload an STL of the actual part and unfold its surface at true scale. This tour continues with Draw mode as the example — pick DXF / SVG or STL from this screen instead to see those steps, or switch modes anytime with the switcher inside the tool.",
  },
];

function modeStep(mode) {
  if (mode === "dxf") {
    return [{
      view: "tool", mode: "dxf", selector: "#dxf-import-group",
      title: "Mode: Import a DXF or SVG",
      body: "This is what \"Import a DXF or SVG\" from the start screen opens. Upload a closed 2D outline as .dxf or .svg. We pick the largest closed shape automatically and scale it to millimetres using the file's units. Since an imported shape has no inherent cable edge, confirm or change it afterward with Set cable edge.",
    }];
  }
  if (mode === "stl") {
    return [{
      view: "tool", mode: "stl", selector: "#stl-sidebar-group",
      title: "Mode: Import an STL",
      body: "This is what \"Import an STL\" from the start screen opens. Upload a 3D model of the real part your sensor will wrap around, click a face (or a smoothly curved patch), then press Unfold selection — it flattens that surface to true scale, like unwrapping paper from the part, so the resulting flex PCB fits it exactly. Selecting several surfaces that don't share an edge? The tool walks you through clicking matching corner points on each, so you control exactly how they join. On a textured or oddly-shaped surface, try Draw patch instead: click points directly on the model to outline your sensor by hand, and texture like tread patterns gets smoothed away. Either way, confirm the cable edge once the outline loads — the app never picks it for you.",
    }];
  }
  return [
    {
      view: "tool", mode: "draw", selector: "#cad-line",
      title: "Mode: Draw an outline",
      body: "This is what \"Draw an outline\" from the start screen opens. Use Line, Rectangle, H/V, or Arc to sketch your board outline on the canvas.",
    },
    {
      view: "tool", selector: "#cad-set-cable",
      title: "Choose the cable edge",
      body: "Every sensor needs a flat ribbon cable running to your electronics. After drawing, click the edge where that cable should exit and press Set cable edge — it turns orange. Generate stays blocked until you've chosen one, so a stray first stroke can never silently become the connector edge.",
    },
    {
      view: "tool", selector: ".cad-fields",
      title: "Draw precisely, not just freehand",
      body: "Select a vertex or edge, then type an exact Length and Angle here — just like a real CAD program, you're not limited to dragging by eye. Width/Height resize the whole outline, and Grid/Snap keep every point on a fixed spacing.",
    },
  ];
}

const TAIL_STEPS = [
  {
    view: "tool", selector: "#cad-target-fill",
    title: "Sensorize zone (optional)",
    body: "Once you have an outline, this toggle lets you draw a second shape that controls where taxels are placed — it won't change your board's outer edge or connector.",
  },
  {
    view: "tool", selector: "#params-group",
    title: "Parameters",
    body: "Taxel size, pitch, and routing clearances live here — watch the live taxel footprint preview below update as you adjust them. The \"Board sizing\" group controls how the board grows to fit its own wiring — click the (?) link there for a full explanation.",
  },
  {
    view: "tool", selector: "#generate",
    title: "Generate",
    body: "With an outline drawn (and parameters you're happy with), press Generate PCB to place the taxels, route the wiring, and run the design-rule check.",
  },
  {
    view: "tool", selector: "#pcb-preview",
    title: "Preview & fine-tune",
    body: "After generating, this preview becomes editable — drag routes, add a custom tail/cable path, chamfer corners, or re-check design rules.",
  },
  {
    view: "tool", selector: "#download",
    title: "Download",
    body: "Download a ZIP of manufacturing files — a KiCad board file, footprints, Gerbers, a drill file, and a placement manifest — ready for a fab or for KiCad. It also has the printed case as STL files, in its case/ folder.",
  },
  {
    view: "tool", selector: "#nav-help",
    title: "Need more detail?",
    body: "The Help page has a full glossary of every parameter plus a troubleshooting FAQ. You can restart this tour anytime from here too.",
  },
];

let overlayEl, spotlightEl, cardEl;
let steps = [];
let stepIndex = 0;

function buildSteps(fromLanding) {
  const list = [];
  if (fromLanding) list.push(...BASE_STEPS);
  list.push(...modeStep(getSelectedMode()));
  list.push(...TAIL_STEPS);
  return list;
}

function ensureOverlay() {
  if (overlayEl) return;
  overlayEl = document.createElement("div");
  overlayEl.className = "tour-overlay";
  overlayEl.innerHTML = `
    <div class="tour-spotlight" id="tour-spotlight"></div>
    <div class="tour-card panel" id="tour-card">
      <div class="tour-card__step" id="tour-step-counter"></div>
      <h3 id="tour-title"></h3>
      <p id="tour-body"></p>
      <div class="row tour-card__actions">
        <button type="button" class="secondary" id="tour-skip">Skip tour</button>
        <span class="tour-card__spacer"></span>
        <button type="button" class="secondary" id="tour-back">Back</button>
        <button type="button" class="primary" id="tour-next">Next</button>
      </div>
    </div>
  `;
  document.body.appendChild(overlayEl);
  spotlightEl = overlayEl.querySelector("#tour-spotlight");
  cardEl = overlayEl.querySelector("#tour-card");
  overlayEl.querySelector("#tour-skip").addEventListener("click", endTour);
  overlayEl.querySelector("#tour-back").addEventListener("click", () => goToStep(stepIndex - 1));
  overlayEl.querySelector("#tour-next").addEventListener("click", () => {
    if (stepIndex >= steps.length - 1) { endTour(); return; }
    goToStep(stepIndex + 1);
  });
  window.addEventListener("resize", () => {
    if (overlayEl.classList.contains("is-active")) positionStep();
  });
}

function startTour({ fromLanding = true } = {}) {
  ensureOverlay();
  steps = buildSteps(fromLanding);
  overlayEl.classList.add("is-active");
  goToStep(0);
}

function endTour() {
  if (!overlayEl) return;
  overlayEl.classList.remove("is-active");
  document.querySelectorAll(".tour-highlight").forEach((el) => el.classList.remove("tour-highlight"));
}

function goToStep(index) {
  if (index < 0 || index >= steps.length) return;
  document.querySelectorAll(".tour-highlight").forEach((el) => el.classList.remove("tour-highlight"));
  stepIndex = index;
  const step = steps[stepIndex];
  if (step.view) showView(step.view);
  if (step.mode) applyModeVisibility(step.mode);
  overlayEl.querySelector("#tour-step-counter").textContent = `${stepIndex + 1} / ${steps.length}`;
  overlayEl.querySelector("#tour-title").textContent = step.title;
  overlayEl.querySelector("#tour-body").textContent = step.body;
  overlayEl.querySelector("#tour-back").disabled = stepIndex === 0;
  overlayEl.querySelector("#tour-next").textContent = stepIndex === steps.length - 1 ? "Done" : "Next";
  // View/mode toggles above can change layout; wait a couple of frames before measuring.
  requestAnimationFrame(() => requestAnimationFrame(positionStep));
}

function positionStep() {
  const step = steps[stepIndex];
  const target = step.selector ? document.querySelector(step.selector) : null;
  if (!target) {
    spotlightEl.style.display = "none";
    centerCard();
    return;
  }
  target.classList.add("tour-highlight");
  // Instant, not smooth: an animated scroll would not have finished by the
  // time the rAF below re-measures getBoundingClientRect(), so the card and
  // spotlight would be positioned against a still-scrolling target.
  target.scrollIntoView({ behavior: "auto", block: "center" });
  requestAnimationFrame(() => {
    const rect = target.getBoundingClientRect();
    spotlightEl.style.display = "block";
    spotlightEl.style.left = `${rect.left - 6}px`;
    spotlightEl.style.top = `${rect.top - 6}px`;
    spotlightEl.style.width = `${rect.width + 12}px`;
    spotlightEl.style.height = `${rect.height + 12}px`;
    positionCard(rect);
  });
}

function centerCard() {
  cardEl.style.transform = "translate(-50%, -50%)";
  cardEl.style.left = "50%";
  cardEl.style.top = "50%";
}

function positionCard(rect) {
  const margin = 14;
  const cardRect = cardEl.getBoundingClientRect();
  let top = rect.bottom + margin;
  let left = Math.max(margin, rect.left);
  if (top + cardRect.height > window.innerHeight) top = Math.max(margin, rect.top - cardRect.height - margin);
  if (left + cardRect.width > window.innerWidth) left = Math.max(margin, window.innerWidth - cardRect.width - margin);
  cardEl.style.transform = "none";
  cardEl.style.left = `${left}px`;
  cardEl.style.top = `${top}px`;
}

window.addEventListener("otc:start-tour", (e) => startTour(e.detail || {}));
document.getElementById("landing-tour-btn")?.addEventListener("click", () => startTour({ fromLanding: true }));
document.getElementById("nav-tour")?.addEventListener("click", () => startTour({ fromLanding: !getSelectedMode() }));
