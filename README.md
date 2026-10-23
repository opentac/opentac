# OpenTac

Design, build, read and simulate flex-PCB tactile sensors for any robot surface.

OpenTac-CAD turns a drawn outline, a DXF, or a surface of an STL part into a manufacturable
tactile sensor: a grid of taxels on a flex PCB, wired to a connector, with Gerbers, a printed
case and a manifest that describes the sensor. The same manifest configures the readout board,
the simulator and transfer between sensors.

**Website and docs:** https://opentac.github.io  ·  **Try it online:** https://opentac.github.io/cad/

## Quick start

You need [uv](https://docs.astral.sh/uv/getting-started/installation/) (the launchers offer to
install it).

```bash
git clone --depth 1 https://github.com/opentac/opentac opentac
cd opentac
uv run opentac cad          # design a sensor in your browser, locally
```

Or double-click `Start_OpenTac.command` (macOS) or `Start_OpenTac.bat` (Windows). On Linux, run
`bash Start_OpenTac.command`. The first run downloads what it needs, once.

With a readout board plugged in over USB:

```bash
uv run opentac flash                 # put the firmware on the board
uv run opentac setup my_design.zip   # tell the board which sensor is attached (once)
uv run opentac view                  # live heatmap
uv run opentac record takes/         # write a take; `opentac replay takes/<folder>` plays it back
```

`opentac record DIR` writes each take as a folder of `tactile.npz`, `manifest.json` and
`take.json`. `opentac replay` opens that folder in the same heatmap as `opentac view`, with
no board attached: `--snapshot out.png` draws one frame (the strongest taxel, or `--frame N`)
and exits; without `--snapshot` the window plays the take, pause with space, step with the
arrow keys.

A new board has no firmware yet: hold its BOOT0 button while plugging it in, then run
`uv run opentac flash --board readout60` (or `node16` for the 30-pin pair). See
[Flash the firmware](https://opentac.github.io/use/flash/), which also covers Linux permissions.

```python
import opentac
sensor = opentac.Sensor()     # finds the board; it already knows its sensor
frame = sensor.read()         # one value per taxel; frame.image(sensor.manifest) lays it out
```

## Design with an AI agent

The repo ships a skill at `.claude/skills/opentac/`, which Claude Code loads automatically when
you open this folder. Other agents: see [AGENTS.md](AGENTS.md).

## Script a design

The skill's worked example builds a sensor without the browser:

```bash
uv run python .claude/skills/opentac/examples/insole.py insole_out
```

It draws a size 9 insole, checks the outline and generates the board: 200 taxels (10 columns x
28 rows) for the 60-pin readout board, DRC clean, in about 5 s on a laptop. `insole_out/` gets
the KiCad board, Gerbers, `manifest.json`, `design.json`, a preview SVG, a 1:1 printable PDF and
the cut files for the piezoresistive layer.

From your own script, call the function the web app's `/generate` route calls, as the skill's
helpers do (`.claude/skills/opentac/helpers/generate.py`):

```python
import base64
from opentac.cad import main

# Outline in mm, counter-clockwise; edge 0->1 is where the cable leaves.
resp = main.generate(main.GenerateRequest(
    outline=[[0, 0], [40, 0], [40, 30], [0, 30]],
    pixel_w_mm=3.0, pixel_h_mm=3.0, pitch_x_mm=3.5, pitch_y_mm=3.5,
    board_mode="expand", router="hug"))
print(resp.stats["total_pixels"], "taxels,", resp.drc["violations"], "DRC violations")

# The ZIP the web page downloads: board, Gerbers, manifest and the printed case.
full = main.zip_with_case_endpoint(main.ZipWithCaseRequest(zip_b64=resp.zip_b64, edit_data=resp.edit_data))
open("square.zip", "wb").write(base64.b64decode(full.zip_b64))
```

This prints `88 taxels, 0 DRC violations`, in about 2 s. `GenerateRequest` in
`opentac/cad/main.py` lists every parameter; [SKILL.md](.claude/skills/opentac/SKILL.md)
explains how to choose them.

## Simulate without a board

```python
from opentac import manifest, sim
m = manifest.load("examples/manifests/finger_8x6_60pin.json")
frame = sim.Sensor(m).press("sphere", force_n=5.0, radius_mm=5.0)   # 10 mm ball, 5 N
print(frame.image(m).round(2))                                      # newtons per taxel, as laid out
```

This is the foam model, for a skin with no bump sheet; it needs only numpy. The frame has the
same type a readout board returns. See [Simulation](https://opentac.github.io/sim/) for the bump
model, shear and ADC counts, and [Transfer](https://opentac.github.io/transfer/) for moving
frames between sensors.

## What is here

| Folder | Contents |
|---|---|
| `opentac/` | the Python package: CAD tool (`opentac.cad`), manifests, readout, viewer, simulation, transfer |
| `firmware/` | readout board firmware (60-pin and 30-pin boards) |
| `hardware/` | readout board KiCad files, fabrication files and BOMs |
| `examples/` | small manifests, example scripts, example STL parts, and the sources of the ready-to-order designs |
| `tests/` | the test suite (see [CONTRIBUTING.md](CONTRIBUTING.md)) |
| `scripts/` | maintenance: regenerate the example designs and board media, check the website against the package |
| `deploy/` | the Dockerfile for the hosted OpenTac-CAD backend (Render, configured in `render.yaml`) |

The website source is maintained separately in [opentac.github.io](https://github.com/opentac/opentac.github.io).

## Readout boards

| Board | Reads up to | Size |
|---|---|---|
| 60-pin | 30 rows x 30 columns | 37 x 34.3 mm |
| 30-pin (two stacked boards) | 15 rows x 15 columns | 16 x 22.8 mm |

Both plug straight into USB and carry an IMU. 30-pin boards can be daisy-chained over I2C,
behind a Teensy 4.0 ([Daisy chain](https://opentac.github.io/use/daisy-chain/)).
Where a command takes `--board`, the 60-pin board is `readout60` or `60pin`, the 30-pin pair
`node16` or `30pin`.

## Development

```bash
uv run --extra sim pytest -q     # about 6 min on a laptop
```

`--extra sim` installs MuJoCo, which four tests need. On a fresh clone 12 tests skip because they
need data that is not in the repository, and a few more without `node` or a C++ compiler.
[CONTRIBUTING.md](CONTRIBUTING.md) lists what each optional test needs, and how to report a
problem.

## Citing

The paper is on arXiv: [arXiv:2610.31661](https://arxiv.org/abs/2610.31661). See
[CITATION.cff](CITATION.cff) and the [paper page](https://opentac.github.io/paper/) for the citation.

## License

Code: MIT ([LICENSE](LICENSE)). Hardware designs: CERN-OHL-P-2.0 ([LICENSE-hardware](LICENSE-hardware)).
Documentation and media: CC BY 4.0 ([LICENSE-docs](LICENSE-docs)).
Third-party robot meshes in `examples/stl/` keep their own licenses, some of them non-commercial;
see [examples/stl/end_effectors/SOURCES.md](examples/stl/end_effectors/SOURCES.md).
