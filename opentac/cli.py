"""The `opentac` command. Run `opentac COMMAND --help` for each command's options."""

import argparse
import sys
import warnings

from opentac import __version__


def _cad(args) -> int:
    from opentac.cad import launcher

    launcher.main(host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


# ----- readout boards -----------------------------------------------------------

def _name_value(text):
    """argparse type for --set: 'NAME=VALUE' -> (NAME, VALUE)."""
    name, sep, value = text.partition("=")
    if not sep or not name.strip() or not value.strip():
        raise argparse.ArgumentTypeError(f"takes NAME=VALUE, e.g. MUX_SETTLE_US=10; got {text!r}")
    return name.strip(), value.strip()


def _node(text):
    """argparse type for --node: a number, `pick` or `all`."""
    if text in ("pick", "all") or text.isdigit():
        return text
    raise argparse.ArgumentTypeError(f"takes a node number, `pick` or `all`; got {text!r}")


def _settings(pairs):
    return dict(pairs or [])


def _board_arg(p, help_text):
    """--board, taking the names the docs use (60pin, 30pin) and the firmware's (readout60, ...)."""
    from opentac.device.config import ALIASES
    p.add_argument("--board", type=str.lower, choices=list(ALIASES), metavar="BOARD",
                   help=f"{help_text}: readout60 or 60pin (the 60-pin board), node16 or 30pin "
                        f"(the 30-pin pair), readout32 or 70pin (the earlier 70-pin board)")


#: Open a board without changing anything on it.
READ_ONLY = dict(raster=False, mux_settle_us=None, imu=None)


def _session(args) -> dict:
    return dict(imu=getattr(args, "imu", False), settings=_settings(getattr(args, "set", None)))


def _open(args, all_nodes=False, session=None):
    """(sensors, bus) for the board or bus the arguments point at."""
    from opentac.device import Bus, Sensor
    from opentac.device.link import find_any

    session = _session(args) if session is None else session
    node = getattr(args, "node", None)
    if node is None and all_nodes and not args.port:
        kind, port = find_any()
        if kind == "bus":
            bus = Bus(port, **session)
            return list(bus), bus
        return [Sensor(port, **session)], None
    if node is None:
        return [Sensor(args.port, **session)], None
    bus = Bus(args.port, **session)
    if node == "pick":
        print("which board? its LED is lit red")
        s = bus.pick()
        if s is None:
            raise SystemExit("cancelled")
        return [s], bus
    if node == "all":
        return list(bus), bus
    return [bus[int(node)]], bus


def _close(sensors, bus):
    if bus is not None:
        bus.close()
    else:
        for s in sensors:
            s.close()


def _describe_skin(stored, indent="  "):
    shear = f"{stored['shear']} taxels grouped" if stored["shear"] else "none (pressure only)"
    return "\n".join(indent + line for line in (
        f"array       {stored['rows']} x {stored['cols']}, {stored['active']} taxels",
        f"pitch       {stored['pitch_x_um'] / 1000:.3f} x {stored['pitch_y_um'] / 1000:.3f} mm",
        f"shear       {shear}",
        f"skin id     {stored['config_id']:08X}  (crc {stored['crc']:08X}, {stored['bytes']} bytes)"))


def _setup(args) -> int:
    from opentac import manifest
    from opentac.device import board, build_config, cache_path, extract
    from opentac.device.config import describe_lines, infer_board, label

    m = manifest.load(args.manifest, scan=True)
    if args.dry_run:
        if args.board:
            b = board(args.board)
        else:
            name = infer_board(m)
            if name is None:
                args.parser.error("--dry-run could not tell the board from the manifest's pads "
                                  "(`opentac manifest check` says which boards it fits): "
                                  "pass --board readout60 or --board node16")
            b = board(name)
            print(f"board: the {label(b.name)}, from the manifest's pads")
        spec = extract(m.raw, b)
        blob = build_config(spec, m.config_id)
        for note in m.warnings + spec["warnings"]:
            print(f"warning: {note}")
        print(f"{m.name}: {spec['rows']} x {spec['cols']}, {len(spec['active'])} taxels on "
              f"the {label(b.name)}")
        print(f"  rows    -> {describe_lines(spec['row_order'])}")
        print(f"  columns -> {describe_lines(spec['col_order'])}")
        print(f"  config  {len(blob)} bytes, skin id {m.config_id:08X}; nothing sent")
        if args.out:
            with open(args.out, "wb") as fh:
                fh.write(blob)
            print(f"  written to {args.out}")
        return 0
    sensors, bus = _open(args, session=READ_ONLY)
    try:
        s = sensors[0]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stored = s.setup(m)
        for w in caught:
            print(f"warning: {w.message}")
        where = f"node {s.node}" if s.node is not None else s.link.port
        print(f"{s.board_name} on {where} now holds {m.name}:")
        print(_describe_skin(stored))
        print(f"  manifest kept at {cache_path(stored['config_id'])}")
    finally:
        _close(sensors, bus)
    return 0


def _info(args) -> int:
    from opentac.device import Bus
    from opentac.device.config import label
    from opentac.device.link import find_any

    if args.node is None and not args.port:
        kind, port = find_any()
        if kind == "bus":
            with Bus(port, **READ_ONLY) as bus:
                print(f"bus controller, protocol v{bus.version}: {len(bus.nodes)} node(s)")
                for n in bus.nodes:
                    line = f"  node {n['node']:2}  {n['board']:<9}  uid {n['uid']}  "
                    if not n["online"]:
                        print(line + "OFFLINE")
                        continue
                    s = bus[n["node"]]
                    skin = (f"skin {s.stored['config_id']:08X}, {s.stored['active']} taxels"
                            if s.stored else "no skin")
                    print(line + skin + ("" if s.manifest or not s.stored else
                                         "  (no manifest on this computer)"))
            return 0
        args.port = port
    # Read-only: do not assert a session on the board just to look at it.
    sensors, bus = _open(args, session=READ_ONLY)
    try:
        for s in sensors:
            info = s.info()
            print(f"{label(s.board_name)}, firmware API {s.caps['api_version']}, up to "
                  f"{s.caps['steps']} driven x {s.caps['channels']} sensed lines")
            if info["skin"] is None:
                print("no skin: run `opentac setup <manifest>`")
            else:
                print("skin:")
                print(_describe_skin(info["skin"]))
                print(f"  manifest    {info['manifest'] or 'NOT on this computer: run opentac setup <manifest> (or recover it from the QR code)'}")
            print("parameters (reset when the board reboots):")
            for name, p in sorted(s.params.items(), key=lambda kv: kv[1]["id"]):
                extra = f"  ({s.feedback_ohm():.0f} ohm feedback)" if name == "GAIN_SW" else ""
                print(f"  {name:<16} {p['value']:>6}   [{p['lo']}..{p['hi']}]{extra}")
    finally:
        _close(sensors, bus)
    return 0


def _start_xampling(sensors, args):
    if not args.xampling:
        return
    print("zeroing: nothing may touch the skin...")
    for s in sensors:
        z = s.zero()
        print(f"  node {s.node}: " if s.node is not None else "  ", end="")
        print(f"noise median <= {z['noise_median']}, max {z['noise_max']}, "
              f"{z['noisy']} taxels >= 100 counts")
        got = s.xampling(args.xampling, grow=not args.fixed)
        print(f"  xampling {'fixed' if args.fixed else 'grow'}, M = {got}")


def _view(args) -> int:
    from opentac.view import view

    sensors, bus = _open(args, all_nodes=True)
    try:
        _start_xampling(sensors, args)
        return view(sensors, bus=bus, shear=args.shear, vmax=args.vmax, save=args.save,
                    description=args.description, snapshot=args.snapshot)
    finally:
        _close(sensors, bus)


def _record(args) -> int:
    from opentac.record import record

    sensors, bus = _open(args)
    try:
        _start_xampling(sensors, args)
        record(sensors[0], args.dir, seconds=args.seconds, description=args.description)
    finally:
        _close(sensors, bus)
    return 0


def _replay(args) -> int:
    from opentac.view import replay

    return replay(args.take, shear=args.shear, vmax=args.vmax, snapshot=args.snapshot,
                  frame=args.frame, speed=args.speed)


def _flash(args) -> int:
    from opentac.device.flash import flash

    return flash(board=args.board, port=args.port, binary=args.bin, force=args.force,
                 dry_run=args.dry_run)


def _manifest_check(args) -> int:
    from opentac import manifest
    from opentac.device import fits
    from opentac.device.config import CURRENT, label

    m = manifest.load(args.file, scan=True)
    for note in m.warnings:
        print(f"warning: {note}")
    pads = sorted(set(m.raw["rp"]) | set(m.raw["cp"]))
    print(f"{m.name}: manifest v{m.version}, {m.raw['r']} rows x {m.raw['c']} columns, "
          f"{m.n} taxels, pads {pads[0]}..{pads[-1]}, "
          f"{'shear bumps' if m.shear_groups is not None else 'no shear bumps'}, "
          f"skin id {m.config_id:08X}")
    result = fits(m)
    current = [name for name in CURRENT if name in result]
    for name in current:
        reason = result[name]
        print(f"  fits the {label(name)}" if reason is None
              else f"  does not fit the {label(name)}: {reason}")
    # Older boards are named only when none of the current ones can read the skin.
    if not any(result[name] is None for name in current):
        for name, reason in result.items():
            if name not in current and reason is None:
                print(f"  fits the {label(name)} only")
    return 0 if any(reason is None for reason in result.values()) else 1


def _manifest_from_qr(args) -> int:
    from opentac import manifest

    m = manifest.from_qr(args.text)
    if args.out:
        m.save(args.out)
        print(f"wrote {args.out}: {m.n} taxels, {m.raw['r']} x {m.raw['c']}; no outline, "
              f"edge cuts, design or shear (the QR does not carry them)")
    else:
        sys.stdout.write(m.source_bytes.decode("utf-8"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="opentac", description="OpenTac: design, build, read "
                                     "and simulate custom tactile sensors.")
    parser.add_argument("--version", action="version", version=f"opentac {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    cad = sub.add_parser("cad", help="design a sensor: start the CAD tool and open it in the browser")
    cad.add_argument("--host", default="127.0.0.1", help="address to serve on (default: this computer only)")
    cad.add_argument("--port", type=int, default=None, help="port (default: first free one from 8000)")
    cad.add_argument("--no-browser", action="store_true", help="don't open a browser window")
    cad.set_defaults(func=_cad)

    def board_args(p, nodes="a node number, `pick` to choose by LED"):
        p.add_argument("--port", help="serial port (found automatically if omitted)")
        p.add_argument("--node", type=_node, help=f"a sensor behind a bus controller: {nodes}")

    def session_args(p):
        p.add_argument("--set", action="append", metavar="NAME=VALUE", type=_name_value,
                       help="set a board parameter for this session, e.g. MUX_SETTLE_US=10 or "
                            "GAIN_SW=2 (repeatable; `opentac info` lists them)")
        p.add_argument("--imu", action="store_true", help="keep the IMU on (off by default)")
        p.add_argument("--xampling", type=int, metavar="M",
                       help="Xampling: read M taxels per frame (zeroes first: hands off)")
        p.add_argument("--fixed", action="store_true",
                       help="with --xampling: exactly M reads per frame, for a steady frame "
                            "rate (default: M search reads, then grow around every contact)")

    p = sub.add_parser("setup", help="tell the board which sensor is plugged in (once per sensor)")
    p.add_argument("manifest", help="the sensor's manifest.json or design .zip")
    board_args(p)
    p.add_argument("--dry-run", action="store_true", help="check and build the config, send nothing")
    _board_arg(p, "with --dry-run, the board to build for (default: the one the manifest fits)")
    p.add_argument("--out", help="with --dry-run: also write the config bytes here")
    p.set_defaults(func=_setup, parser=p)

    p = sub.add_parser("info", help="what is attached: board, stored sensor, parameters")
    board_args(p)
    p.set_defaults(func=_info)

    p = sub.add_parser("view", help="live heatmap (every node, on a bus controller)")
    board_args(p, nodes="a node number, `pick`, or `all` (default on a bus)")
    session_args(p)
    p.add_argument("--shear", action="store_true", help="start with shear arrows on (key s)")
    p.add_argument("--vmax", type=float, default=4095, help="top of the colour scale, counts")
    p.add_argument("--save", metavar="DIR", help="record takes into DIR: Enter starts and stops")
    p.add_argument("--description", default="", help="free text stored with each take")
    p.add_argument("--snapshot", metavar="PNG", help="save one frame to a PNG and exit")
    p.set_defaults(func=_view)

    p = sub.add_parser("record", help="record one take into DIR (Ctrl-C stops)")
    p.add_argument("dir", help="folder for takes; each take is a subfolder")
    board_args(p)
    session_args(p)
    p.add_argument("--seconds", type=float, help="stop after this long")
    p.add_argument("--description", default="", help="free text stored with the take")
    p.set_defaults(func=_record)

    p = sub.add_parser("replay", help="play a recorded take in the heatmap viewer (no board)")
    p.add_argument("take", help="take folder written by `opentac record` or `opentac view --save`")
    p.add_argument("--shear", action="store_true", help="start with shear arrows on (key s)")
    p.add_argument("--vmax", type=float, default=4095, help="top of the colour scale, counts")
    p.add_argument("--snapshot", metavar="PNG", help="save one frame to a PNG and exit")
    p.add_argument("--frame", type=int, metavar="N",
                   help="with --snapshot: 0-based frame to draw (default: the strongest)")
    p.add_argument("--speed", type=float, default=1.0,
                   help="playback speed against the recorded timestamps (1 is realtime)")
    p.set_defaults(func=_replay)

    p = sub.add_parser("flash", help="put the prebuilt firmware on the board over USB (a new "
                       "board: hold BOOT0 while plugging it in, and pass --board)")
    _board_arg(p, "which board (default: the one the running firmware reports; needed for a "
                  "new board)")
    p.add_argument("--port", help="serial port (found automatically if omitted)")
    p.add_argument("--bin", help="flash this binary instead of the prebuilt one")
    p.add_argument("--force", action="store_true", help="flash even if the board reports another name")
    p.add_argument("--dry-run", action="store_true", help="say what would happen, change nothing")
    p.set_defaults(func=_flash)

    p = sub.add_parser("manifest", help="check a manifest, or recover one from a sensor's QR code")
    msub = p.add_subparsers(dest="manifest_command", metavar="ACTION", required=True)
    q = msub.add_parser("check", help="validate a manifest and say which readout boards it fits")
    q.add_argument("file", help="manifest.json or design .zip")
    q.set_defaults(func=_manifest_check)
    q = msub.add_parser("from-qr", help="rebuild a manifest from the text of the sensor's QR code")
    q.add_argument("text", help="what a QR scanner reads off the back of the sensor: the link "
                   "(https://opentac.github.io/q/...) or, on older boards, the bare code; quote it")
    q.add_argument("-o", "--out", help="write the manifest here (default: print it)")
    q.set_defaults(func=_manifest_from_qr)
    return parser


def main(argv=None) -> int:
    # Line by line even when output goes to a file or a pipe, so a log shows the
    # startup banner (`opentac cad > log.txt`) and progress as they happen.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except (ValueError, OSError):
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    from opentac.device.protocol import DeviceError
    from opentac.manifest import ManifestError
    try:
        return args.func(args)
    except (DeviceError, ManifestError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
