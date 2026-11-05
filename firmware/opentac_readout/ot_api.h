// OpenTac single-board API surface -- modes, parameters, capabilities.
//
// Commands and parameter ids are the same on USB and on the OpenTac bus (a bus
// node runs this firmware), so one host library, opentac.device, drives both.
// Ids 0, 1, 10-18 and 20-23 are reserved for bus-node features (demand
// reference, vibration projection weights, per-modality LED channels, binary
// thresholds) and are left unallocated rather than reused.
//
// ----- The split that matters: what persists ---------------------------------
//
// State is divided by what it belongs to:
//
//   Board identity -- the DeviceConfig: array size, ordering, active mask,
//   pitch, shear grouping. Belongs to the physical skin, written once when it
//   is provisioned. Persists in flash (ot_devcfg.h).
//
//   Session config -- mode, per-modality enables, scan timing, LED behaviour,
//   IMU rates. Belongs to the experiment, written every session. Does not
//   persist. The board boots to known-safe defaults and the host asserts what
//   it wants.
//
// The asymmetry is deliberate. A board that silently retains a mode from an
// earlier session, which the current host never chose, would mislead that
// host; a host that holds per-board skin tables keyed by serial number would
// fail as soon as a board is swapped. Each piece of state lives with whatever
// owns it.
//
// ----- Config generation -----------------------------------------------------
//
// A u16 that is 0 at boot and increments on every accepted config write. Every
// frame record echoes it, so a recording says which configuration produced it,
// and a host that sees the generation go backwards knows the board rebooted
// and its session config is gone, and can assert it again.

#pragma once

#include <stdint.h>

#define OT_API_VERSION 1

// ----- Tactile modes ---------------------------------------------------------
//
// RASTER reads every active taxel. The two XAMPLE modes are binary
// compressive sampling (ot_xampling.h) and emit sparse X records instead of F.
// The remaining slots are reserved, not stubbed: selecting one is refused with
// NOTIMPL rather than quietly falling back to a raster scan. Every response
// states the mode that produced it, so that a host cannot decode one mode's
// data as another's and obtain plausible but wrong values; silently
// substituting a mode would break that guarantee at the source.

#define OT_TAC_RASTER        0   // every active taxel, shift-register order
#define OT_TAC_XAMPLE_FIRST  1   // 1..15 reserved for xampling schemes
#define OT_TAC_XAMPLE_FIXED  1   // M samples per frame, growth included
#define OT_TAC_XAMPLE_GROW   2   // M pattern samples; growth on top, unbounded
#define OT_TAC_MODE_MAX      15

// Bit n set = tactile mode n is implemented on this build.
#define OT_TAC_MODES_SUPPORTED  ((1UL << OT_TAC_RASTER) | \
                                 (1UL << OT_TAC_XAMPLE_FIXED) | \
                                 (1UL << OT_TAC_XAMPLE_GROW))

static inline bool otTacModeIsXampling(uint16_t mode) {
  return mode == OT_TAC_XAMPLE_FIXED || mode == OT_TAC_XAMPLE_GROW;
}

// Bit n set = tactile mode n is meaningless without shear bumps.
//
// Whether a skin has shear is a property of the skin, not of the firmware, so
// a mode can be implemented and still be unusable on the sensor currently
// provisioned. Selecting one on a pressure-only skin is refused with NOSHEAR
// rather than run over taxels that share no bump -- the answer would be a
// well-formed number computed from a grouping that does not exist. Neither
// raster nor xampling needs shear -- xampling fills whole bumps when a skin has
// them and grows by neighbours alone when it does not -- so this is 0.
#define OT_TAC_MODES_NEED_SHEAR 0UL

// ----- LED modes -------------------------------------------------------------
//
// The numbers are shared with the bus-node LED modes.
// IDENTITY and BINARY are bus-node concepts (showing an assigned address,
// per-modality thresholds) and are not implemented here.

#define OT_LED_OFF        0
#define OT_LED_MANUAL     1   // host drives the colour with `c r g b`
#define OT_LED_IDENTITY   2   // not implemented on a single board
#define OT_LED_HEARTBEAT  3   // red, green, blue cycle
#define OT_LED_SENSOR     4   // brightness follows peak tactile level
#define OT_LED_BINARY     5   // not implemented here
#define OT_LED_MODE_MAX   5

#define OT_LED_MODES_SUPPORTED \
  ((1UL << OT_LED_OFF) | (1UL << OT_LED_MANUAL) | \
   (1UL << OT_LED_HEARTBEAT) | (1UL << OT_LED_SENSOR))

// ----- Parameters ------------------------------------------------------------
//
// Unallocated here, reserved for bus-node features: 0 DEMAND_REF, 1 VIB_AXIS,
// 10-12 projection weights, 13-18 per-modality LED channels and gains, 20-23
// binary thresholds and the transition counter. 24-28: frame rate and IMU;
// 29-34: xampling; 35: the 60-pin board's gain switch; 36-37: its HDR re-read.

#define OT_P_TAC_MODE       2
#define OT_P_LED_MODE       3
#define OT_P_LED_BRIGHT     4
#define OT_P_ROW_SETTLE_US  5
#define OT_P_MUX_SETTLE_US  6
#define OT_P_TAC_ENABLE     7
#define OT_P_VIB_ENABLE     8
#define OT_P_ORIENT_ENABLE  9
#define OT_P_LED_RATE_HZ   19
#define OT_P_FRAME_MAX_HZ  24
#define OT_P_VIB_ODR_HZ    25
#define OT_P_ORIENT_ODR_HZ 26
#define OT_P_ACC_FS_G      27
#define OT_P_GYRO_FS_DPS   28
#define OT_P_XAMP_M        29   // budget; see OT_TAC_XAMPLE_FIXED / _GROW
#define OT_P_XAMP_THRESH   30   // counts above a taxel's own `Z` noise peak
#define OT_P_XAMP_SEARCH_MIN 31 // pattern samples taken before any growth
#define OT_P_XAMP_ZERO_FRAMES 32 // raster frames `Z` measures noise over
#define OT_P_XAMP_BUMP_FILL 33  // 1 = growth fills whole shear bumps
#define OT_P_XAMP_CONFIRM_US 34 // extra settle before re-reading a trigger
#define OT_P_GAIN_SW       35   // 60-pin board: gain switches closed, bit k = SW(k+1)
#define OT_P_HDR_GAINS     36   // 60-pin board: re-read passes, one GAIN_SW per nibble
#define OT_P_HDR_CLIP      37   // 60-pin board: raw reading that sends a taxel to the next pass

// 60-pin board: the gain at boot. 0 closes no switch, which leaves the 10 kOhm
// feedback resistor alone: the highest gain, and the gain switch's own
// power-on state, so the board reads the same before and after the firmware
// first writes it.
#ifndef OT_GAIN_SW_DEFAULT
#define OT_GAIN_SW_DEFAULT 0
#endif

// Whether VIB_ENABLE and ORIENT_ENABLE boot on. A USB board boots with both
// on; a bus node (-DOT_BUS_NODE) defines this 0 before including this file, so
// its IMU stays off until a host asks for it.
#ifndef OT_IMU_DEFAULT
#define OT_IMU_DEFAULT 1
#endif

struct OtParamSpec {
  uint8_t id;
  const char *name;
  uint16_t lo;
  uint16_t hi;
  uint16_t def;
};

// The single source of truth for the parameter surface. `?` dumps this table
// over the wire -- id, name, range and current value -- so the host discovers
// what a board supports instead of carrying its own copy to drift out of sync.
// Ranges are enforced here, on the board: a host must not be able to set scan
// timing that breaks the board's own real-time behaviour.
static const OtParamSpec OT_PARAMS[] = {
  {OT_P_TAC_MODE,       "TAC_MODE",       0, OT_TAC_MODE_MAX, OT_TAC_RASTER},
  {OT_P_LED_MODE,       "LED_MODE",       0, OT_LED_MODE_MAX, OT_LED_HEARTBEAT},
  {OT_P_LED_BRIGHT,     "LED_BRIGHT",     0, 255,   255},
  {OT_P_ROW_SETTLE_US,  "ROW_SETTLE_US",  1, 50,    2},
  {OT_P_MUX_SETTLE_US,  "MUX_SETTLE_US",  1, 50,    1},
  {OT_P_TAC_ENABLE,     "TAC_ENABLE",     0, 1,     1},
  {OT_P_VIB_ENABLE,     "VIB_ENABLE",     0, 1,     OT_IMU_DEFAULT},
  {OT_P_ORIENT_ENABLE,  "ORIENT_ENABLE",  0, 1,     OT_IMU_DEFAULT},
  {OT_P_LED_RATE_HZ,    "LED_RATE_HZ",    1, 100,   1},
  {OT_P_FRAME_MAX_HZ,   "FRAME_MAX_HZ",   0, 1000,  0},   // 0 = free-running
  {OT_P_VIB_ODR_HZ,     "VIB_ODR_HZ",     15, 1920, 960},
  {OT_P_ORIENT_ODR_HZ,  "ORIENT_ODR_HZ",  15, 480,  240},
  {OT_P_ACC_FS_G,       "ACC_FS_G",       2, 16,    4},   // snaps to 2/4/8/16
  {OT_P_GYRO_FS_DPS,    "GYRO_FS_DPS",    125, 4000, 2000},  // snaps to 125<<n
  // Clamped again to the skin's taxel count when a frame runs.
  {OT_P_XAMP_M,         "XAMP_M",         1, 1024,  128},
  // Defaults from 30 s hands-off on the 32x32 readout: a 64-frame noise
  // capture plus a 30-count margin.
  {OT_P_XAMP_THRESH,    "XAMP_THRESH",    1, 4095,  30},
  {OT_P_XAMP_SEARCH_MIN, "XAMP_SEARCH_MIN", 0, 1024, 30},
  {OT_P_XAMP_ZERO_FRAMES, "XAMP_ZERO_FRAMES", 16, 1024, 64},
  // Off by default: whole shear bumps are only needed when shear is computed.
  {OT_P_XAMP_BUMP_FILL, "XAMP_BUMP_FILL",   0, 1,     0},
  // 0 = off. A reading that would trigger is taken again after this much more
  // settling, and the second reading stands; `Z` measures at the same settle.
  // 50 us, hands-off on the 32x32 at MUX_SETTLE_US 8: false growth 92% of
  // frames -> 0%, and every taxel's noise <= 27 counts (without it, up to ~900).
  {OT_P_XAMP_CONFIRM_US, "XAMP_CONFIRM_US", 0, 500,   50},
#if defined(OT_BOARD_READOUT60)
  // Which MAX14662 switches are closed. Each puts a resistor across the
  // amplifier's 10k feedback resistor: SW1 4.7k, SW2 1.1k, SW3 270, SW4 68 ohm,
  // so alone they give 3.2k, 990, 263 and 68 ohm. Less feedback, fewer counts
  // for the same press. Any combination works (they add in parallel).
  {OT_P_GAIN_SW,        "GAIN_SW",          0, 15,    OT_GAIN_SW_DEFAULT},
  // HDR raster. 0 = off. Otherwise each nibble, lowest first, up to the first
  // 0, is the GAIN_SW of one more pass that re-reads only the taxels whose
  // reading in the pass before reached HDR_CLIP: 0x82 re-reads at 991 ohm, then
  // at 67.5 ohm. A re-read value carries its pass in bits 12-15:
  // value = raw + 4096 x pass. In xampling the passes re-read only the taxels
  // the frame sampled, and a re-read value is that raw reading, not counts
  // above the zero; values from pass 0 stay above the zero. Z ignores it.
  {OT_P_HDR_GAINS,      "HDR_GAINS",        0, 0xFFFF, 0},
  {OT_P_HDR_CLIP,       "HDR_CLIP",         1, 4095,  3800},
#endif
};

#define OT_PARAM_COUNT ((uint8_t)(sizeof(OT_PARAMS) / sizeof(OT_PARAMS[0])))

// Table slot for a parameter id, or 0xFF. Linear over at most 21 entries: a
// switch would have to repeat the ids that the table already states.
static inline uint8_t otParamSlot(uint8_t id) {
  for (uint8_t i = 0; i < OT_PARAM_COUNT; ++i) {
    if (OT_PARAMS[i].id == id) return i;
  }
  return 0xFFu;
}

// Clamp to the declared range, then snap the two set-valued parameters up to a
// legal member. The caller always reports back what was actually stored, so a
// host never has to guess whether its value survived.
static inline uint16_t otParamCoerce(uint8_t slot, uint32_t requested) {
  const OtParamSpec *spec = &OT_PARAMS[slot];
  uint16_t value = (requested < spec->lo) ? spec->lo
                 : (requested > spec->hi) ? spec->hi
                 : (uint16_t)requested;

  if (spec->id == OT_P_ACC_FS_G) {
    uint16_t legal = 2;
    while (legal < value && legal < 16) legal = (uint16_t)(legal * 2u);
    value = legal;
  } else if (spec->id == OT_P_GYRO_FS_DPS) {
    uint16_t legal = 125;
    while (legal < value && legal < 4000) legal = (uint16_t)(legal * 2u);
    value = legal;
  }
  return value;
}
