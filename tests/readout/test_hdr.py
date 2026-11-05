"""HDR raster on the 60-pin board, against the stand-in board (tests/readout/fakeboard.py)."""
import warnings
from pathlib import Path

import numpy as np
import pytest

from opentac.device import DeviceError, Sensor
from opentac.device.hdr import LADDER, HdrTable
from opentac.device.link import UsbLink
from opentac.record import Take, hdr_counts, load_take

from .fakeboard import FakeBoard, feedback_ohm

ROOT = Path(__file__).resolve().parents[2]
SKIN = ROOT / "examples" / "manifests" / "square_9x9_60pin.json"      # 81 taxels


@pytest.fixture(autouse=True)
def opentac_home(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENTAC_HOME", str(tmp_path / "home"))


def hdr_board():
    fake = FakeBoard("READOUT60", hdr=True)
    s = Sensor(_link=UsbLink(link=fake))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s.setup(SKIN)
    fake.load = np.zeros(81)
    return fake, s


def test_table_converts_each_value_with_the_gain_of_its_pass():
    gain = np.array([1.0, 0.25])
    table = HdrTable(gain_sw=(0, 1), gain=gain, zero=np.array([[100.0, 100.0], [110.0, 120.0]]),
                     noise=np.ones((2, 2)))
    assert table.code == 0x1
    counts, g = table.decode(np.array([150, (1 << 12) | 3920]))
    assert counts.tolist() == [50.0, (3920 - 120) / 0.25] and g.tolist() == [1.0, 0.25]
    assert table.saturated(np.array([4000, (1 << 12) | 3900])).tolist() == [False, True]
    assert HdrTable(LADDER, np.array([1, 0.3197, 0.0991, 0.0263, 0.00675]), np.zeros((5, 1)),
                    np.zeros((5, 1))).labels() == ["1", "1/3", "1/10", "1/38", "1/148"]
    assert HdrTable(LADDER, np.ones(5), np.zeros((5, 1)), np.zeros((5, 1))).code == 0x8421


def test_firmware_or_board_without_hdr_says_so():
    s = Sensor(_link=UsbLink(link=FakeBoard("READOUT60")))         # gain switch, no HDR
    with pytest.raises(DeviceError, match="firmware has no HDR"):
        s.hdr()
    s = Sensor(_link=UsbLink(link=FakeBoard("NODE16", hdr=True)))  # no gain switch
    with pytest.raises(DeviceError, match="no gain switch"):
        s.hdr()


def test_light_and_heavy_taxels_in_one_frame():
    fake, s = hdr_board()
    table = s.hdr()
    assert fake.values[36] == 0x8421 and fake.values[35] == 0 and s.hdr_on
    # the zero at each gain is the board's offset there
    assert np.allclose(table.zero[:, 0], [100 + 10 * sw for sw in LADDER])
    assert np.allclose(table.gain, [feedback_ohm(sw) / 10000 for sw in LADDER])
    fake.load[3], fake.load[40] = 50.0, 300_000.0      # a light touch and a heavy press
    frame = s.read()
    assert frame.gain[3] == 1.0 and frame.gain[40] == pytest.approx(table.gain[-1])
    assert frame.taxels[3] == pytest.approx(50, abs=0.5)
    assert frame.taxels[40] == pytest.approx(300_000, rel=0.002)
    assert np.allclose(np.delete(frame.taxels, [3, 40]), 0)
    assert int(frame.hdr_raw[40]) >> 12 == 4 and frame.units.startswith("counts above zero")


def test_off_close_and_open_leave_plain_frames():
    fake, s = hdr_board()
    s.set("GAIN_SW", 2)
    s.hdr()
    assert fake.values[35] == 0                         # HDR scans from the highest gain
    assert s.hdr(False) is None
    assert fake.values[36] == 0 and fake.values[35] == 2    # the gain as it was
    frame = s.read()
    assert frame.gain is None and frame.hdr_raw is None and frame.units == "counts"
    s.hdr(zero=False)                                   # on again with the zero from before
    s.close()
    assert fake.values[36] == 0                         # closing turns it off
    fake.values[36] = 0x8421                            # left on by a program that crashed
    Sensor(_link=UsbLink(link=fake))
    assert fake.values[36] == 0


def test_hdr_in_xampling():
    fake, s = hdr_board()
    s.hdr()
    s.zero()                                            # Xampling's zero, at HDR's first gain
    s.xampling(64)
    fake.load[3], fake.load[40] = 50.0, 300_000.0       # then a light touch and a heavy press
    frame = s.read()
    assert frame.mode == 2 and not frame.mask.all()     # a sparse frame...
    assert frame.taxels[3] == pytest.approx(50, abs=0.5) and frame.gain[3] == 1.0
    assert frame.taxels[40] == pytest.approx(300_000, rel=0.002)    # ...with a re-read taxel
    assert int(frame.hdr_raw[40]) >> 12 == 4
    assert np.isnan(frame.taxels[~frame.mask]).all() and np.isnan(frame.gain[~frame.mask]).all()
    # off and on again in Xampling: no new zero needed while GAIN_SW stays the same
    zeros = sum(c == "Z" for c in fake.commands)
    s.hdr(False)
    assert s.read().gain is None and fake.values[2] == 2
    s.hdr(zero=False)
    assert s.read().gain is not None and sum(c == "Z" for c in fake.commands) == zeros
    # a fresh zero from Xampling: measured in raster (hands off), then Xampling resumes
    fake.load[:] = 0
    s.hdr()
    assert fake.values[2] == 2 and s.read().mode == 2


def test_a_take_keeps_what_the_board_sent(tmp_path):
    fake, s = hdr_board()
    table = s.hdr()
    fake.load[40] = 300_000.0
    take = Take(tmp_path, s.manifest)
    for frame in s.frames(limit=3):
        take.add(frame)
    s.hdr(False)
    take.add(s.read())
    s.hdr(zero=False)
    s.zero()
    s.xampling(64)
    sparse = s.read()
    take.add(sparse)
    t = load_take(take.save())
    assert t["raw"].dtype == np.int16 and t["raw"][0, 40] >> 12 == 4
    assert t["hdr_table"].tolist() == [0, 0, 0, -1, 0] and t["hdr_mode"].tolist() == [0, 0, 0, 0, 2]
    counts, _ = hdr_counts(t)
    assert np.allclose(counts[4], sparse.taxels, equal_nan=True)      # Xampling, converted alike
    assert np.allclose(t["hdr_zero"][0], table.zero)
    counts, gain = hdr_counts(t)
    assert counts[0, 40] == pytest.approx(300_000, rel=0.002) and gain[0, 40] < 0.01
    assert np.isnan(counts[3]).all()
    assert "HDR" in t["info"]["notes"]["hdr"]


def test_viewer_snapshot_in_hdr(tmp_path):
    fake, s = hdr_board()
    fake.load[40] = 300_000.0
    from opentac.view import view
    png = tmp_path / "snap.png"
    assert view([s], snapshot=str(png), hdr=True, log=lambda *_: None) == 0
    assert png.stat().st_size > 10_000 and s.hdr_on


def test_floor_only_changes_what_is_drawn():
    from opentac.view import floored
    v = np.array([5.0, 30.0, np.nan, 400.0])
    out = floored(v, 30)
    assert out[0] == 0 and out[1] == 30 and np.isnan(out[2]) and out[3] == 400
    assert floored(np.array([-3.0]), 0)[0] == -3     # off by default: nothing changes


def test_the_gain_belongs_to_hdr_while_it_is_on():
    fake, s = hdr_board()
    s.hdr()
    for change in (lambda: s.set("GAIN_SW", 2), lambda: s.set_feedback(1000),
                   lambda: s.set("HDR_CLIP", 3000), lambda: s.set("HDR_GAINS", 0)):
        with pytest.raises(DeviceError, match="hdr\\(False\\) first"):
            change()
    assert fake.values[35] == 0 and fake.values[36] == 0x8421      # nothing changed
    s.hdr(False)
    assert s.set("GAIN_SW", 2) == 2 and s.set("HDR_CLIP", 3500) == 3500
    with pytest.raises(DeviceError, match="turn HDR on with hdr"):
        s.set("HDR_GAINS", 0x8421)                                  # bypassing the conversion
    assert s.set("HDR_GAINS", 0) == 0
    assert s.hdr().clip == 3500 and fake.values[35] == 0            # its clip, from the top gain
    with pytest.raises(DeviceError, match="turn HDR on with hdr"):
        Sensor(_link=UsbLink(link=FakeBoard("READOUT60", hdr=True)), settings={"HDR_GAINS": 0x21})
