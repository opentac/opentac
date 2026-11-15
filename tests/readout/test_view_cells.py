"""How the live viewer draws each taxel."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from opentac.view import _skin_cells

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "manifests"


def _cell_size(manifest):
    fig, ax = plt.subplots()
    try:
        return _skin_cells(ax, manifest)[2:]
    finally:
        plt.close(fig)


def test_a_taxel_as_wide_as_its_pitch_is_drawn_without_a_seam():
    """The manifest stores the taxel size to 0.01 mm: a 4.162 mm taxel on a 4.162 mm
    pitch reads back as 4.16, and the 2 um gap showed as a hairline between squares.
    Such taxels are drawn at the pitch; a taxel with a real gap keeps its size."""
    raw = json.loads((EXAMPLES / "square_9x9_60pin.json").read_text())
    assert _cell_size(raw) == (4.0, 4.0)          # 4.0 mm taxels at 4.2 mm: the gap is real

    m = json.loads(json.dumps(raw))
    x0, y0 = m["pixels"][0][0], m["pixels"][0][1]
    m["pixels"] = [[x0 + (x - x0) * 4.162 / 4.2, y0 + (y - y0) * 4.162 / 4.2, r, c]
                   for x, y, r, c in (p[:4] for p in m["pixels"])]
    m["w"] = m["h"] = 4.16
    w, h = _cell_size(m)
    assert abs(w - 4.162) < 1e-6 and abs(h - 4.162) < 1e-6
