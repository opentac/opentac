"""A stand-in for a readout board's USB serial port, speaking the firmware's line protocol.

Built from the firmware source's rules (opentac_readout.ino, ot_api.h, ot_devcfg.h),
so the host library can be driven end to end without hardware: capabilities and
the parameter table, provisioning with the CRC check, parameters with clamping,
refusals, zeroing, and F / X frames. It is a test double, not a simulator: frame
values are a fixed pattern.
"""
import struct
import zlib

PARAMS = [  # id, name, lo, hi, default  (ot_api.h)
    (2, "TAC_MODE", 0, 15, 0), (3, "LED_MODE", 0, 5, 3), (4, "LED_BRIGHT", 0, 255, 255),
    (5, "ROW_SETTLE_US", 1, 50, 2), (6, "MUX_SETTLE_US", 1, 50, 1), (7, "TAC_ENABLE", 0, 1, 1),
    (8, "VIB_ENABLE", 0, 1, 1), (9, "ORIENT_ENABLE", 0, 1, 1), (19, "LED_RATE_HZ", 1, 100, 1),
    (24, "FRAME_MAX_HZ", 0, 1000, 0), (25, "VIB_ODR_HZ", 15, 1920, 960),
    (26, "ORIENT_ODR_HZ", 15, 480, 240), (27, "ACC_FS_G", 2, 16, 4),
    (28, "GYRO_FS_DPS", 125, 4000, 2000), (29, "XAMP_M", 1, 1024, 128),
    (30, "XAMP_THRESH", 1, 4095, 30), (31, "XAMP_SEARCH_MIN", 0, 1024, 30),
    (32, "XAMP_ZERO_FRAMES", 16, 1024, 64), (33, "XAMP_BUMP_FILL", 0, 1, 0),
    (34, "XAMP_CONFIRM_US", 0, 500, 50),
]
GAIN = (35, "GAIN_SW", 0, 15, 0)
HDR = [(36, "HDR_GAINS", 0, 0xFFFF, 0), (37, "HDR_CLIP", 1, 4095, 3800)]
SWITCH_OHM = (4700, 1100, 270, 68)     # readout60.json: each across the 10 kOhm feedback


def feedback_ohm(sw):
    return 1.0 / (1.0 / 10000 + sum(1.0 / r for k, r in enumerate(SWITCH_OHM) if sw >> k & 1))
IMU_TAIL = "-12,1003,980,0,0,0,1.00000,0.00000,0.00000,0.00000,0.00,0.00,0.00"


class FakeBoard:
    """pyserial-like: write(), read(), in_waiting, flush(), reset_input_buffer(), close()."""

    def __init__(self, board="READOUT60", gain_ok=True, hdr=False):
        self.board = board
        self.steps = 16 if board == "NODE16" else 32
        table = PARAMS + ([GAIN] + (HDR if hdr else []) if board == "READOUT60" else [])
        self.spec = {pid: (name, lo, hi) for pid, name, lo, hi, _ in table}
        self.values = {pid: d for pid, _, _, _, d in table}
        self.order = [pid for pid, *_ in table]
        self.gain_ok = gain_ok
        self.flash = None              # the stored config bytes
        self.staging = None
        self.baseline = False
        self.generation = 0
        self.frame = 0
        self.out = bytearray()
        self.commands = []             # every line received
        self.commits = 0
        self.dfu = False
        # With hdr: raster frames come from `load` (counts per taxel at the highest gain,
        # None = the fixed pattern) on an offset of 100 + 10 x GAIN_SW raw counts.
        self.load = None

    # --- pyserial surface ---------------------------------------------------------
    def write(self, data):
        for line in data.decode("ascii").splitlines():
            self.commands.append(line)
            self._handle(line.strip())
        return len(data)

    def flush(self):
        pass

    @property
    def in_waiting(self):
        return len(self.out)

    def read(self, n=1):
        if not self.out and self._streaming():
            self._emit_frame()
        chunk, self.out = bytes(self.out[:max(n, 1)]), self.out[max(n, 1):]
        return chunk

    def reset_input_buffer(self):
        self.out.clear()

    def close(self):
        pass

    # --- the firmware -------------------------------------------------------------
    def _say(self, line):
        self.out += (line + "\n").encode("ascii")

    def _param(self, pid):
        name, lo, hi = self.spec[pid]
        self._say(f"R,PARAM,{pid},{name},{lo},{hi},{self.values[pid]}")

    def _config(self):
        if self.flash is None:
            return None
        raw = self.flash
        rows, cols = raw[6], raw[7]
        px, py, active, shear, cid, plen = struct.unpack_from("<IIHHII", raw, 8)
        crc = struct.unpack_from("<I", raw, 28 + plen)[0]
        return rows, cols, active, px, py, shear, cid, crc, len(raw)

    def _cfg_reply(self):
        c = self._config()
        if c is None:
            self._say("R,NOCFG")
            return
        rows, cols, active, px, py, shear, cid, crc, n = c
        self._say(f"R,CFG,2,{rows},{cols},{active},{px},{py},{shear},{cid:08X},{crc:08X},{n}")

    def _streaming(self):
        mode = self.values[2]
        return (self.flash is not None and self.values[7] == 1 and not self.staging
                and (mode == 0 or self.baseline) and not self.dfu)

    def _emit_frame(self):
        active = self._config()[2]
        mode = self.values[2]
        head = f"{self.frame},{self.generation},{mode},3150,0"
        if mode == 0 and self.load is not None:
            values = ",".join(str(v) for v in self._hdr_scan(active))
            self._say(f"F,{head},{values},{IMU_TAIL}")
        elif mode == 0:
            values = ",".join(str((k * 37 + self.frame) % 4096) for k in range(active))
            self._say(f"V,{self.frame},1,2,3")
            self._say(f"F,{head},{values},{IMU_TAIL}")
        else:
            read = list(range(0, active, 3))[: self.values[29]]
            vals = [k % 100 for k in read]
            if self.load is not None:          # the loaded taxels too, then the HDR passes
                read = sorted(set(read) | {k for k in range(active) if self.load[k] > 0})
                vals = self._hdr_xamp(read)
            mask = bytearray((active + 7) // 8)
            for k in read:
                mask[k >> 3] |= 1 << (k & 7)
            values = ",".join(str(v) for v in vals)
            self._say(f"X,{head},{len(read)},{mask.hex().upper()},{values},{IMU_TAIL}")
        self.frame += 1

    def _read_at(self, sw, k):
        g = feedback_ohm(sw) / 10000
        return min(4095, int(round(100 + 10 * sw + self.load[k] * g)))

    def _hdr_scan(self, active):
        """The firmware's HDR raster: a pass at GAIN_SW, then one per nibble of HDR_GAINS
        over the taxels that reached HDR_CLIP in the pass before."""
        vals = [self._read_at(self.values[35], k) for k in range(active)]
        gains, clip, p = self.values.get(36, 0), self.values.get(37, 3800), 1
        while gains & 0xF:
            due = [k for k in range(active) if vals[k] >> 12 == p - 1 and vals[k] & 0xFFF >= clip]
            if not due:
                break
            for k in due:
                vals[k] = self._read_at(gains & 0xF, k) | (p << 12)
            gains, p = gains >> 4, p + 1
        return vals

    def _hdr_xamp(self, read):
        """Xampling with HDR: values above the zero at GAIN_SW (the offset there), then
        each pass re-reads, as raw readings, the sampled taxels that reached HDR_CLIP."""
        sw0 = self.values[35]
        base = 100 + 10 * sw0
        raw0 = {k: self._read_at(sw0, k) for k in read}
        vals = {k: max(0, raw0[k] - base) for k in read}
        gains, clip, p = self.values.get(36, 0), self.values.get(37, 3800), 1
        while gains & 0xF:
            due = [k for k in read if vals[k] >> 12 == p - 1
                   and ((raw0[k] if p == 1 else vals[k] & 0xFFF) >= clip)]
            if not due:
                break
            for k in due:
                vals[k] = self._read_at(gains & 0xF, k) | (p << 12)
            gains, p = gains >> 4, p + 1
        return [vals[k] for k in read]

    def _handle(self, line):
        if not line:
            return
        cmd, rest = line[0], line[1:].replace(",", " ").split()
        if line == "?":
            self._say(f"R,CAPS,1,{self.steps},{self.steps},{self.steps * self.steps},00000007,"
                      f"0000001B,{len(self.order)},00000000,{self.board}")
            for pid in self.order:
                self._param(pid)
            self._say("R,END")
        elif cmd == "p":
            pid, value = int(rest[0]), int(rest[1])
            if pid not in self.spec:
                self._say("R,ERR,NOPARAM")
                return
            _, lo, hi = self.spec[pid]
            value = min(max(value, lo), hi)
            if pid == 2 and value not in (0, 1, 2):
                self._say("R,ERR,NOTIMPL")
                return
            if pid == 2 and value in (1, 2) and not self.baseline:
                self._say("R,ERR,NOBASE")
                return
            if pid == 35 and not self.gain_ok:
                self._say("R,ERR,GAIN")
                return
            if self.values[pid] != value:
                self.values[pid] = value
                self.generation += 1
                if pid in (5, 6, 34, 35):
                    self.baseline = False
            self._param(pid)
        elif cmd == "P":
            if rest:
                self._param(int(rest[0]))
            else:
                for pid in self.order:
                    self._param(pid)
                self._say("R,END")
        elif cmd == "G":
            self._cfg_reply()
        elif cmd == "S":
            self.staging = bytearray(b"\xff" * int(line[1:]))
            self._say(f"R,OK,S,{len(self.staging)}")
        elif cmd == "D":
            off, data = line[1:].split(",")
            data = bytes.fromhex(data)
            self.staging[int(off):int(off) + len(data)] = data
            self._say(f"R,OK,D,{off},{len(data)}")
        elif cmd == "E":
            blob = bytes(self.staging)
            self.staging = None
            body, crc = blob[:-4], struct.unpack("<I", blob[-4:])[0]
            if blob[:4] != b"OTC1" or zlib.crc32(body) & 0xFFFFFFFF != crc:
                self._say("R,ERR,INVALID")
                return
            self.flash, self.baseline = blob, False
            self.commits += 1
            self.generation += 1
            self._cfg_reply()
        elif cmd == "A":
            self.staging = None
            self._say("R,OK,A")
        elif cmd == "X":
            self.flash = None
            self._say("R,OK,X")
        elif cmd == "Z":
            if self.flash is None:
                self._say("R,ERR,NOCFG")
                return
            self.baseline = True
            self._say(f"R,OK,Z,{self.values[32]},310,402,16,27,0")
        elif cmd == "t":
            self._say(f"R,STAT,{self.frame},0,1200,1500,{self.generation}")
        elif cmd == "c":
            self._say("R,OK,C")
        elif cmd in ("B", "b"):
            self._say("R,OK,B")
            self.dfu = True
        else:
            self._say("R,ERR,CMD")


class FakeBus:
    """A bus controller with FakeBoard nodes behind it: `@<n> <line>` in, `<n>|` out."""

    def __init__(self, nodes=3, board="NODE16"):
        self.nodes = [FakeBoard(board) for _ in range(nodes)]
        for b in self.nodes:            # bus nodes boot with the IMU off
            b.values[8] = b.values[9] = 0
        self.out = bytearray()
        self.t_us = 1000
        self._turn = 0

    def write(self, data):
        for line in data.decode("ascii").splitlines():
            if line.startswith("@"):
                node, _, text = line[1:].partition(" ")
                b = self.nodes[int(node)]
                b.write((text + "\n").encode())
                for reply in bytes(b.out).decode().splitlines():
                    self.out += f"{node}|{reply}\n".encode()
                b.out.clear()
            elif line == "?":
                self.out += b"R,MCAPS,5,18,1000000\nR,END\n"
            elif line == "N":
                self.out += f"R,NODES,{len(self.nodes)}\n".encode()
                for k in range(len(self.nodes)):
                    self.out += f"R,NODE,{k},{0x10 + k:02X},UID{k},NODE16,1\n".encode()
                self.out += b"R,END\n"
        return len(data)

    def flush(self):
        pass

    @property
    def in_waiting(self):
        return len(self.out)

    def read(self, n=1):
        if not self.out:
            # Frames from the streaming nodes, in turn, stamped on one clock.
            for _ in range(len(self.nodes)):
                k = self._turn % len(self.nodes)
                self._turn += 1
                b = self.nodes[k]
                if b._streaming():
                    b._emit_frame()
                    for line in bytes(b.out).decode().splitlines():
                        if line[0] in "FX":
                            self.out += f"{k},{self.t_us}|{line}\n".encode()
                    b.out.clear()
                    self.t_us += 2500
                    break
        chunk, self.out = bytes(self.out[:max(n, 1)]), self.out[max(n, 1):]
        return chunk

    def reset_input_buffer(self):
        self.out.clear()

    def close(self):
        pass
