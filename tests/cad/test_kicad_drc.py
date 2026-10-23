"""The exported KiCad board passes KiCad's own DRC with no errors.

board.kicad_pro carries the rules the board was generated to and
board.kicad_dru the scoped exceptions (gold fingers at the tab edge, the
taxel electrode gap). Without them KiCad checks the flex board against its
rigid-board defaults and reports hundreds of errors on a correct design.
"""
import base64
import io
import json
import shutil
import subprocess
import zipfile

import pytest
from fastapi.testclient import TestClient

from opentac.cad import main

KICAD = shutil.which("kicad-cli")
client = TestClient(main.app)

CASES = {
    "square": {"outline": [[-20, -20], [20, -20], [20, 20], [-20, 20]]},
    "tight_gap_4layer": {"outline": [[-15, -15], [15, -15], [15, 15], [-15, 15]],
                         "gap_mm": 0.1, "copper_layers": 4},
    # Electrode gap wider than the 0.2 mm centre clearance, as in the
    # calibration set's gap 0.3-0.5 designs: the finger tips set the minimum.
    "wide_gap_2mm": {"outline": [[-10, 0], [10, 0], [10, 20], [-10, 20]],
                     "pixel_w_mm": 2.0, "pixel_h_mm": 2.0, "pitch_x_mm": 2.0, "pitch_y_mm": 2.0,
                     "gap_mm": 0.4},
    "fixed": {"outline": [[0, 0], [50, 0], [50, 20], [25, 20], [25, 40], [0, 40]],
              "board_mode": "fixed_keepout"},
    # Gold fingers on the top face: the rows change layer through vias in the tail.
    "square_top_contacts": {"outline": [[-20, -20], [20, -20], [20, 20], [-20, 20]],
                            "contact_side": "top"},
    "tight_gap_4layer_top_contacts": {"outline": [[-15, -15], [15, -15], [15, 15], [-15, 15]],
                                      "gap_mm": 0.1, "copper_layers": 4, "contact_side": "top"},
}


@pytest.mark.skipif(KICAD is None, reason="kicad-cli not installed")
@pytest.mark.parametrize("name", sorted(CASES))
def test_exported_board_has_no_kicad_drc_errors(name, tmp_path):
    r = client.post("/generate", json=CASES[name])
    assert r.status_code == 200, r.text[:300]
    zipfile.ZipFile(io.BytesIO(base64.b64decode(r.json()["zip_b64"]))).extractall(tmp_path)
    assert (tmp_path / "board.kicad_pro").exists() and (tmp_path / "board.kicad_dru").exists()
    out = tmp_path / "drc.json"
    subprocess.run([KICAD, "pcb", "drc", "--format", "json", "-o", str(out),
                    str(tmp_path / "board.kicad_pcb")], capture_output=True, check=False)
    report = json.loads(out.read_text())
    errors = [v["type"] + ": " + v["description"] for v in report["violations"] if v["severity"] == "error"]
    assert errors == []
    assert report["unconnected_items"] == []
