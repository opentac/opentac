"""Replay of recorded takes: frames round-trip, peak pick, CLI snapshot."""
from pathlib import Path

import numpy as np
import pytest

from opentac.cli import main
from opentac.device.protocol import BoardFrame
from opentac.manifest import load
from opentac.record import Take, iter_frames, load_take, peak_frame_index
from opentac.view import replay

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples" / "manifests"


def _take(tmp_path, n=5, peak_at=3, xampling=False, description="press"):
    m = load(EXAMPLES / "square_9x9_60pin.json")
    take = Take(tmp_path, m, description, context={"settings": {
        "mode": "xamp_grow" if xampling else "raster", "mux_settle_us": 8, "imu": False}})
    for i in range(n):
        taxels = np.full(m.n, 40.0)
        taxels[10] = 100.0 + 10 * i
        if i == peak_at:
            taxels[10] = 2500.0
        mask = np.ones(m.n, dtype=bool)
        if xampling:
            mask[:] = False
            mask[8:13] = True
            taxels = np.where(mask, taxels, np.nan)
        imu = {"accel_mg": [i, 0, 1000], "gyro_mdps": [0, 0, 0],
               "quat_wxyz": [1.0, 0.0, 0.0, 0.0], "euler_deg": [0.0, 0.0, float(i)]}
        take.add(BoardFrame(taxels=taxels, t=1000.0 + 0.01 * i, mask=mask, imu=imu,
                            index=i, generation=1, mode=2 if xampling else 0, dt_us=10000))
    path = take.save()
    assert path is not None
    return path, m


def test_iter_frames_matches_what_was_saved(tmp_path):
    path, m = _take(tmp_path, n=4, peak_at=2)
    loaded = load_take(path)
    frames = list(iter_frames(path))
    assert len(frames) == 4 == loaded["info"]["n_frames"]
    assert frames[2].taxels[10] == pytest.approx(2500.0)
    assert frames[0].taxels[10] == pytest.approx(100.0)
    assert frames[0].mask.all()
    assert frames[1].index == 1
    assert frames[1].imu["euler_deg"][2] == pytest.approx(1.0)
    np.testing.assert_allclose(frames[2].taxels, np.where(loaded["raw"][2] >= 0, loaded["raw"][2], np.nan))
    assert peak_frame_index(path) == 2
    assert m.n == 81


def test_iter_frames_restores_xampling_nans(tmp_path):
    path, m = _take(tmp_path, n=3, peak_at=1, xampling=True)
    frames = list(iter_frames(path))
    assert frames[0].mode == 2
    assert frames[0].units == "counts above zero"
    assert int(frames[0].mask.sum()) == 5
    assert np.isnan(frames[0].taxels[0])
    assert np.isfinite(frames[0].taxels[10])
    assert peak_frame_index(path) == 1


def test_replay_snapshot_png(tmp_path):
    path, _ = _take(tmp_path, n=6, peak_at=4)
    png = tmp_path / "peak.png"
    assert replay(path, snapshot=str(png), shear=True, log=lambda *_: None) == 0
    assert png.stat().st_size > 10_000
    named = tmp_path / "frame0.png"
    assert replay(path, snapshot=str(named), frame=0, log=lambda *_: None) == 0
    assert named.stat().st_size > 10_000


def test_cli_replay_snapshot(tmp_path, capsys):
    path, _ = _take(tmp_path, n=4, peak_at=2)
    png = tmp_path / "cli.png"
    assert main(["replay", str(path), "--snapshot", str(png)]) == 0
    out = capsys.readouterr().out
    assert "4 frames, 81 taxels" in out
    assert "saved" in out and "frame 2" in out
    assert png.exists()
    assert main(["replay", str(tmp_path / "missing")]) == 1
    assert "not a take folder" in capsys.readouterr().err


def test_cli_replay_rejects_bad_speed(tmp_path, capsys):
    path, _ = _take(tmp_path, n=2, peak_at=0)
    assert main(["replay", str(path), "--snapshot", str(tmp_path / "x.png"), "--speed", "0"]) == 1
    assert "speed must be positive" in capsys.readouterr().err


def test_cli_help_lists_replay(capsys):
    assert main([]) == 0
    assert "replay" in capsys.readouterr().out
