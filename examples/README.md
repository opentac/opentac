# Examples

Run each from the repository root.

| Script | What it does | Needs a board | Command |
|---|---|---|---|
| `read_sensor.py` | Prints the frame rate and the strongest taxel, five times a second, for five seconds | yes, set up with `opentac setup` | `uv run python examples/read_sensor.py` |
| `record_take.py` | Records a three-second take into `recordings/`, then loads it back and prints its size | yes, set up with `opentac setup` | `uv run python examples/record_take.py` |
| `xampling.py` | Compares frame rates: full raster scan, then Xampling with a fixed and a growing read budget | yes, set up with `opentac setup`; keep the sensor untouched while it zeroes | `uv run python examples/xampling.py` |
| `sim_press.py` | Simulates a 10 mm ball pressed at 5 N into a flat sensor (foam model) and plots the reading | no | `uv run python examples/sim_press.py [manifest.json]` |
| `transfer.py` | Simulates a press on one sensor and moves the reading onto a sensor with different taxels | no | `uv run python examples/transfer.py` |

The three board scripts need a readout board with the OpenTac firmware
([Flash the firmware](https://opentac.github.io/use/flash/)) and a sensor that the
board has been told about, once:

```sh
uv run opentac setup my_design.zip
```

Without a board they print one line saying what is missing and exit.

A take folder from `record_take.py` or `opentac record` / `opentac view --save` can be
opened later with no board:

```sh
uv run opentac replay recordings/<take>                 # heatmap playback
uv run opentac replay recordings/<take> --snapshot a.png
```

## Data

| Folder | Contents |
|---|---|
| `manifests/` | Manifests of small sensors for both boards (`*_60pin.json`, `*_30pin.json`). `uv run opentac manifest check examples/manifests/square_9x9_60pin.json` shows which board one fits. |
| `designs/` | `sources.json`: outlines and settings of the 31 sensors made for the paper, the input to `scripts/make_example_designs.py`. |
| `stl/` | Robot gripper and hand parts as STL, in millimetres, to place sensors on in OpenTac-CAD; `stl/end_effectors/SOURCES.md` says where each came from. |
