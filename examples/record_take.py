"""Record a three-second take, then read it back.

    uv run python examples/record_take.py

Run `uv run opentac setup my_design.zip` once first. The take goes into
recordings/ in the current folder.
"""
import sys

from opentac.device import DeviceError, Sensor
from opentac.record import load_take, record


def main():
    with Sensor() as s:
        if s.manifest is None:
            sys.exit("this computer has no manifest for the board's sensor: run "
                     "`uv run opentac setup my_design.zip` first")
        path = record(s, "recordings", seconds=3, description="example take")

    take = load_take(path)
    raw = take["raw"]                      # frames x taxels, int16
    print(f"{raw.shape[0]} frames of {raw.shape[1]} taxels, "
          f"{take['info']['fps']:.0f} fps, sensor {take['manifest'].name}")
    print("taxel 0 is at", take["xy"][0], "mm")
    print(f"play it back with: uv run opentac replay {path}")


if __name__ == "__main__":
    try:
        main()
    except DeviceError as exc:          # no board, no skin, no answer: say so in one line
        sys.exit(f"error: {exc}")
    except KeyboardInterrupt:
        sys.exit(130)
