// 2D outline import: uploads a .dxf or .svg file to POST /import-dxf or
// /import-svg and hands the chosen closed loop off to app.js via the same
// otc:dxf-outline event convention stl-viewer.js already uses for STL face
// unfolding. SVG is the drawing-app counterpart of DXF (Inkscape, Illustrator,
// Figma); both become one millimetre-space polygon.
import { API_BASE } from "./config.js";
import { setSource } from "./design-name.js";

const uploadBtn = document.getElementById("upload-dxf");
const fileInput = document.getElementById("dxf-input");
const infoEl = document.getElementById("dxf-info");

uploadBtn?.addEventListener("click", () => fileInput.click());

fileInput?.addEventListener("change", async (event) => {
  const file = event.target.files && event.target.files[0];
  fileInput.value = "";
  if (!file) return;

  const kind = file.name.toLowerCase().endsWith(".svg") ? "svg" : "dxf";
  const endpoint = kind === "svg" ? "/import-svg" : "/import-dxf";
  const label = kind === "svg" ? "SVG" : "DXF";

  infoEl.textContent = `Importing ${file.name}…`;
  setSource(file.name);
  infoEl.classList.remove("error");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const resp = await fetch(`${API_BASE}${endpoint}`, { method: "POST", body: formData });
    if (!resp.ok) {
      const detail = await resp.json().catch(() => ({}));
      throw new Error(detail.detail || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    const chosen = data.loops[data.chosen_index];
    infoEl.textContent = data.warnings && data.warnings.length
      ? data.warnings.join(" ")
      : `Imported ${chosen.length}-point outline from ${file.name}.`;
    window.dispatchEvent(new CustomEvent("otc:dxf-outline", {
      detail: { outline: chosen, warnings: data.warnings || [], source: kind },
    }));
  } catch (err) {
    infoEl.innerHTML = `<span class="error">${label} import failed: ${err.message}</span>`;
  }
});
