// Landing page: mode selection (Draw / DXF-SVG / STL / Describe / Open), the
// in-tool mode switcher, and the "take the tour" / "browse help" entry points.
import { showView } from "./views.js";

const stlGroup = document.getElementById("stl-sidebar-group");
const dxfGroup = document.getElementById("dxf-import-group");
const chatGroup = document.getElementById("chat-sidebar-group");
const openGroup = document.getElementById("design-import-group");
const modeSwitcherButtons = document.querySelectorAll(".mode-switcher .mode-button");

let selectedMode = null;

export function applyModeVisibility(mode) {
  selectedMode = mode;
  if (stlGroup) stlGroup.hidden = mode !== "stl";
  if (dxfGroup) dxfGroup.hidden = mode !== "dxf";
  if (chatGroup) chatGroup.hidden = mode !== "chat";
  if (openGroup) openGroup.hidden = mode !== "open";
  modeSwitcherButtons.forEach((btn) => btn.classList.toggle("active", btn.dataset.mode === mode));
  // Nudge stl-viewer.js's resize logic in case the STL panel just became visible.
  window.dispatchEvent(new CustomEvent("otc:view-shown"));
}

export function getSelectedMode() {
  return selectedMode;
}

function selectMode(mode, { openPicker = false } = {}) {
  applyModeVisibility(mode);
  showView("tool");
  if (openPicker) {
    if (mode === "dxf") document.getElementById("upload-dxf")?.click();
    if (mode === "stl") document.getElementById("upload-stl")?.click();
    if (mode === "open") document.getElementById("open-design")?.click();
  }
  window.dispatchEvent(new CustomEvent("otc:mode-selected", { detail: { mode } }));
}

document.querySelectorAll(".mode-card__cta").forEach((btn) => {
  btn.addEventListener("click", () => selectMode(btn.dataset.mode, { openPicker: true }));
});

modeSwitcherButtons.forEach((btn) => {
  btn.addEventListener("click", () => selectMode(btn.dataset.mode));
});

document.getElementById("landing-help-btn")?.addEventListener("click", () => showView("help"));

// If the user jumps straight to the Tool view via the header nav without
// ever picking a mode from the landing cards, default to Draw so the tool
// is not showing every input section at once.
window.addEventListener("otc:view-shown", (e) => {
  if (e.detail?.view === "tool" && !selectedMode) applyModeVisibility("draw");
});

// Deep-link help sections: any element with data-help-target scrolls the
// Help view to the matching section (used by parameter "(?)" links and the
// help page's own in-page nav).
document.querySelectorAll("[data-help-target]").forEach((el) => {
  el.addEventListener("click", (event) => {
    event.preventDefault();
    const targetId = el.getAttribute("data-help-target");
    showView("help");
    const target = document.getElementById(targetId);
    // Wait a tick so the (just-unhidden) help view has a layout to scroll within.
    requestAnimationFrame(() => target?.scrollIntoView({ behavior: "smooth", block: "start" }));
  });
});
