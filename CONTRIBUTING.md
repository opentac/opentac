# Contributing to OpenTac

## Set up

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), clone the repository and
run everything through `uv run` from the repository root. The first `uv run` creates `.venv/`
from `uv.lock`.

```bash
git clone --depth 1 https://github.com/opentac/opentac opentac
cd opentac
uv run opentac --version
```

## Where things live

| Path | What |
|---|---|
| `opentac/cad/` | OpenTac-CAD: the FastAPI backend (`main.py`), the layout and routing pipeline (`pipeline/`), the exporters (`export/`) and the browser frontend (`web/`) |
| `opentac/manifest/`, `opentac/frame.py` | the sensor manifest and the `Frame` every reader returns |
| `opentac/device/` | readout boards over USB and the I2C bus: protocol, config, flashing |
| `opentac/sim/`, `opentac/transfer.py` | OpenTac-Sim (foam and bump models) and OpenTac-Transfer |
| `opentac/cli.py`, `opentac/view.py`, `opentac/record.py` | the `opentac` command, live viewer and recording |
| `firmware/` | readout board firmware, with prebuilt binaries in `firmware/prebuilt/` |
| `hardware/` | readout board KiCad projects and fabrication files |
| `.claude/skills/opentac/` | the agent skill: how to design a sensor from a description |
| `examples/` | example manifests, scripts, STL parts and the sources of the ready-to-order designs |
| `tests/` | the test suite: `cad/`, `readout/` (manifests and boards), `sim/` (simulation and transfer), `hardware/` (fabrication files) |
| `scripts/` | maintenance scripts (see each file's docstring) |

## Run the tests

```bash
uv run --extra sim pytest -q
```

About 6 minutes on a laptop; most of it is the CAD tests, which generate and route real boards.
`uv run pytest tests/sim -q` runs one folder. On a fresh clone 12 tests skip, because they need
data that is not in the repository. Optional tests and what they need:

| Tests | Need | Without it |
|---|---|---|
| the MuJoCo tests in `tests/sim/test_sensor.py` (3) and `tests/sim/test_bump_core.py` (1) | `--extra sim` (MuJoCo) | skipped |
| design-chat tests in `tests/cad/test_chat_*.py` | `--extra demo` (google-genai) | skipped |
| `tests/test_qr_page.py` | `node` on the PATH | skipped |
| the firmware's config parser, in `tests/readout/test_config.py` | a C++ compiler (`c++`) | skipped |
| the published example designs in `tests/sim/test_bump_core.py` (1) | the website checkout: `OPENTAC_SITE_ROOT`, or `../opentac.github.io` | skipped |

CI (`.github/workflows/tests.yml`) installs `--extra demo --extra sim` and runs the whole suite on
every push.

The website source is in [opentac.github.io](https://github.com/opentac/opentac.github.io).
For website integration tests or asset-generation scripts, clone it beside this repository,
or set `OPENTAC_SITE_ROOT` to its absolute path. CI checks the revision recorded in
`website-revision.txt`; it does not silently follow the website's latest changes.

```bash
git clone https://github.com/opentac/opentac.github.io ../opentac.github.io
uv run --extra docs python ../opentac.github.io/scripts/preview_site.py
```

The QR integration test, and the test that loads the website's example designs into the bump
model (`tests/sim/test_bump_core.py`), are skipped when no website checkout is available. CI always
provides one and runs both the QR and transfer consistency checks.

## Commit messages

One line that says what changed, in plain words, led by the area it touches. Add a body only when
the reason is not obvious. Examples from the history:

- `Readout boards: the Tag-Connect pads J2 are excluded from the position files, so JLCPCB's BOM step no longer stops on them`
- `Narrow tail: the cable is cut to its traces after routing and flares back to the connector`
- `Docs: 60-pin cable is A type (contacts down at both ends)`

Documentation is plain: what to do, what was measured, with numbers. If a page states a number,
say where it comes from.

## Ask a question

Ask questions and share designs in [GitHub Discussions](https://github.com/opentac/opentac/discussions).

## Report a problem

Open an issue with the bug report form. For a problem with a board or a sensor, include:

- the output of `uv run opentac info` with the board plugged in (or the error it prints);
- which board: 60-pin, 30-pin pair, or 30-pin boards on a Teensy bus;
- your operating system;
- the design ZIP from OpenTac-CAD, if the problem involves a sensor or its manifest;
- what you expected and what you saw: a screenshot of `uv run opentac view`, and a photo of the
  sensor's tail in the connector if readings are missing or mirrored. A take folder from
  `opentac record` or `opentac view --save` can be played later with `opentac replay`, which
  needs no board.

For a design that will not generate or fails DRC, the design ZIP (or the `design.json` inside it)
is enough to reproduce it.
