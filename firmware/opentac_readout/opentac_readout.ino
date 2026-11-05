// OpenTac readout firmware -- free-running stream from a provisioned sensor
// Target: STM32G431KBU6
//
// The skin need not be a full 32x32 rectangle. The array size, the row and
// column hardware ordering, which taxels exist and their physical pitch all
// come from a DeviceConfig held in flash, which the host derives from the
// OpenTac manifest and sends once over this same USB CDC link. A provisioned
// board keeps its config across power cycles, so no sensor needs its own
// build.
//
// Records out, distinguished by their first field:
//   V,<seq>,<ax,ay,az mg>                      one per FIFO sample
//   F,<frame>,<gen>,<mode>,<dt_us>,<dropped>,<N tactile>,
//     <ax,ay,az mg>,<gx,gy,gz mdps>,<qw,qx,qy,qz>,<roll,pitch,yaw deg>
//   X,<frame>,<gen>,<mode>,<dt_us>,<dropped>,<n>,<mask hex>,<n values>,
//     <same IMU fields as F>                   xampling modes, sparse
//   M,<gen>,<mode>,<tac>,<vib>,<ori>,<active>  status, ~1 Hz and on any change
//   R,...                                      reply to a command
//   # ...                                      human-readable notice
//
// Every F record states the config generation and the tactile mode that
// produced it, so a recording is self-describing and a host decoding one
// mode's data as another's is not a silent failure. The M record carries the
// same facts when tactile is switched off.
//
// N is the config's active taxel count, and the samples are in the manifest's
// `pixels` order, so sample k, pixels[k] and shear.b[k] are the same taxel.
// An X record carries only the n taxels xampling read this frame. <mask hex> is
// a bitmask over `pixels` order, ceil(N/8) bytes as hex, bit k of byte j set
// when taxel 8j+k was read (LSB first, like the config's active_mask). The n
// values follow in ascending taxel order, as counts above the `Z` baseline
// (clamped at 0). A mask costs N/4 characters however many taxels were read,
// where a per-sample index cost ~5 characters each.
// With no valid config the board stays fully communicative -- V records, the
// command set, RGB -- but emits no F records at all. Guessing a geometry would
// produce a plausible-looking frame that silently means nothing.
//
// Commands in, one per line, '\n' terminated (host side: opentac/device). A bus
// node takes the same commands (see the bus node build below):
//   ?            capabilities, then the whole parameter table
//   G            report the stored sensor config
//   X            erase the stored sensor config
//   S<len>       begin staging a <len>-byte config; suspends acquisition
//   D<off>,<hex> stage <hex> at byte offset <off>
//   E            validate, commit to flash, reload, resume
//   A            abort staging, resume
//   p <id> <v>   set a parameter; the reply states what was actually stored
//   P [id]       read a parameter, or all of them
//   m <mode>     shortcut for `p 2 <mode>`  (tactile mode)
//   l <mode> [b] shortcut for `p 3 <mode>` and `p 4 <b>`  (LED)
//   c <r> <g> <b> manual LED colour
//   t            statistics
//   z            zero statistics
//   Z            capture the xampling baseline and per-taxel noise --
//                nothing touching the skin
//   B or b       reboot into the DFU bootloader (no BOOT0 button needed)
//
// ----- Bus node build (-DOT_BUS_NODE) ----------------------------------------
//
// The same firmware as a node on the OpenTac bus: many boards behind one
// Teensy 4.0 on I2C1 (SCL PA15, SDA PB7), auto-addressed by SMBus ARP. Only
// the transport changes. The command lines above arrive over the bus
// instead of USB and the replies go back the same way, so provisioning,
// parameters and xampling are this file's code, not a copy of it. Frames are
// published as binary (libraries/OpenTactileBus/src/ot_bus.h) and the
// controller prints them as the F / X / V records above. The IMU moves to I2C2
// on its own TwoWire, and boots off (VIB_ENABLE = ORIENT_ENABLE = 0) -- a host
// that wants it asks. USB still answers `B` for reflashing.
//
// Build with -DOT_BUS_NODE -DOT_BOARD_NODE16 -DI2C_TXRX_BUFFER_SIZE=160
// (`firmware/build.sh node16_bus`). Without OT_BUS_NODE this file compiles to
// the USB firmware.
//
// ----- Hardware this assumes ------------------------------------------------
//
// Three boards share this file; see "Which board" below and ot_devcfg.h. The
// description below is that of the 70-pin readout (READOUT32). The 60-pin
// board (READOUT60) has the same drive and mux, differs in its '595 pin order
// and adds a gain switch (GAIN_SW). The 30-pin board (NODE16) has two '595
// stages and a 16:1 mux on MUX0..MUX3; PB5 is not part of it.
//
// Drive: four cascaded 74HC595 stages, walked by a single travelling 1. Because
// the 1 can only shift forward, the scan visits steps in order and reads, at
// each, every taxel with a line on that step; steps no taxel uses are clocked
// past without being sampled. Which side of the skin -- rows or columns -- sits
// on the shift register is set by the readout board's connector and carried in
// the config (see "Connector line codes" in ot_devcfg.h); on the 32x32 readout
// it is the skin's columns that are driven.
//
// Sense: a 32:1 mux fed by MUX0..MUX4:
//   MUX0 = PA6, MUX1 = PA7, MUX2 = PA10, MUX3 = PB3, MUX4 = PB5,
// wired straight through to A0..A4. PB5 also carries CAN_RX on boards that
// populate the transceiver; leave CAN unused.
//
// The order of the lines comes from the config's row_order and col_order, so
// a board that needs a different mapping is provisioned differently, not
// rebuilt.

// ----- Which board ------------------------------------------------------------
//
// A build is for one board. firmware/build.sh passes the flag for each. In the
// Arduino IDE, delete the // at the start of your board's line:
//
// #define OT_BOARD_READOUT60   // 60-pin readout board
// #define OT_BOARD_NODE16      // 30-pin board
// #define OT_BOARD_READOUT32   // 70-pin readout board (older boards)

#include <Wire.h>
#include <math.h>
#include <LSM6DSV16BXSensor.h>
#include <Adafruit_NeoPixel.h>

#if defined(OT_BUS_NODE) && defined(OT_BOARD_READOUT60)
// A 1024-taxel node needs more RAM for its frame buffers than the G431 has
// left, so the bus node build is for the 30-pin board only.
#error "The bus node build is for OT_BOARD_NODE16 only"
#endif

#if defined(OT_BUS_NODE)
// A bus node boots with the IMU off: the host asks for it when it wants it.
// Must precede ot_api.h, whose parameter table takes its defaults from it.
#define OT_IMU_DEFAULT 0
#endif

#include "ot_api.h"
#include "ot_devcfg.h"
#include "ot_xampling.h"

#if defined(OT_BUS_NODE)
#include <ot_bus.h>
#endif

// ----- Pinout ---------------------------------------------------------------

#define TACTILE_ADC  PA0
#if defined(OT_BOARD_READOUT60)
// The 60-pin board wires the '595 control lines to PA1..PA5 in a different
// order (read off its KiCad netlist at the '595 pins). The rest is the same.
#define SR_MR        PA1
#define SR_SHCP      PA2
#define SR_STCP      PA3
#define SR_OE        PA4
#define SR_DS        PA5
#define SR_MR_BIT    GPIO_PIN_1
#define SR_SHCP_BIT  GPIO_PIN_2
#define SR_STCP_BIT  GPIO_PIN_3
#define SR_DS_BIT    GPIO_PIN_5
#else
#define SR_DS        PA1
#define SR_OE        PA2
#define SR_STCP      PA3
#define SR_SHCP      PA4
#define SR_MR        PA5
#define SR_DS_BIT    GPIO_PIN_1
#define SR_STCP_BIT  GPIO_PIN_3
#define SR_SHCP_BIT  GPIO_PIN_4
#define SR_MR_BIT    GPIO_PIN_5
#endif
// The *_BIT masks are the same pins on GPIOA, for the direct BSRR writes the
// scan makes; keep the two lists in step.
#define MUX0_PIN     PA6
#define MUX1_PIN     PA7
#define MUX2_PIN     PA10
#define MUX3_PIN     PB3
#define MUX4_PIN     PB5
#define IMU_SDA      PA_8
#define IMU_SCL      PA_9
#define RGB_DATA     PB6
#if defined(OT_BUS_NODE)
#define BUS_SCL      PA15
#define BUS_SDA      PB7
#endif

// Sets the max source impedance the input can resolve; an untouched cell is far
// above what the 12.5-cycle default handles. Raise to 247CYCLES_5 for less noise.
constexpr uint32_t TACTILE_SAMPLE_TIME = ADC_SAMPLETIME_92CYCLES_5;

// Each drain pays a fixed ~90 us I2C poll, so drain rarely. Counted in row
// steps, so a small sparse skin drains more often per taxel than a big one --
// which is the right way round, since its frames are quicker.
constexpr uint8_t FIFO_DRAIN_ROW_INTERVAL = 16;

// Xampling changes step far more often than raster -- every batch walks the
// register again -- so it drains on time instead, at about the cadence a
// 32x32 raster scan drains at. Counting step changes would drain 3-5x as often.
constexpr uint32_t XAMP_FIFO_DRAIN_US = 8000;

constexpr float RAD_TO_DEG_F = 57.2957795131f;

// How often the status record repeats when nothing has changed.
constexpr uint32_t STATUS_PERIOD_MS = 1000;

// How often to remind the operator that the board has nothing to scan.
constexpr uint32_t NO_CONFIG_NOTICE_MS = 2000;

// ----- Runtime parameters ---------------------------------------------------
//
// Session config: these are not persisted. The board boots to the defaults in
// OT_PARAMS and the host asserts whatever it wants, so a board never silently
// carries someone else's session settings. Only the sensor DeviceConfig, which
// describes the physical skin, survives a power cycle.

uint16_t paramValue[OT_PARAM_COUNT];

// Incremented on every accepted config write, 0 at boot, echoed in every F and
// M record. A host that sees it go backwards knows the board rebooted and its
// session config went with it.
uint16_t configGeneration = 0;

static inline uint16_t param(uint8_t id) { return paramValue[otParamSlot(id)]; }

// The IMU has the same address on a USB board and on a bus node. On a bus
// node, Wire is the bus, so the IMU gets I2C2 to itself.
#if defined(OT_BUS_NODE)
TwoWire WireImu(IMU_SDA, IMU_SCL);
#define IMU_WIRE WireImu
#else
#define IMU_WIRE Wire
#endif
LSM6DSV16BXSensor imu(&IMU_WIRE, LSM6DSV16BX_I2C_ADD_L);
Adafruit_NeoPixel rgbPixel(1, RGB_DATA, NEO_GRB + NEO_KHZ800);

ADC_HandleTypeDef tactileAdc = {};
bool imuReady = false;

// ----- Sensor configuration -------------------------------------------------

OtDeviceConfig deviceConfig;

// Both defined in ot_devcfg.h, so the host tests exercise the same builder
// the firmware runs rather than a copy of it that can drift.
ScanPlan scanPlan;
uint16_t tactileFrame[OTCFG_MAX_TAXELS];

// ----- Xampling state -------------------------------------------------------
//
// The baseline is session state, like every parameter: RAM only, gone on
// reboot, and cleared whenever what it was measured under changes -- a new
// config, or a different ROW/MUX settle, since the untouched-taxel offset it
// removes is a settling artifact. Xampling emits nothing until `Z` is sent.

XampGrid xampGrid;
bool xampReady = false;          // this skin can be xampled
XampState xampState;
uint16_t xampBaseline[OTCFG_MAX_TAXELS];
uint16_t xampNoise[OTCFG_MAX_TAXELS];   // hands-off peak above baseline
uint16_t xampBaselineMax = 4095;        // the highest baseline, for HDR's early out
bool baselineValid = false;

// The baseline is the mean of the first 16 zero frames: 16 x 4095 = 65520
// still fits the u16 it is summed in. The noise peak is taken over all
// XAMP_ZERO_FRAMES of them.
constexpr uint8_t BASELINE_FRAMES = 16;

// ----- Provisioning state ---------------------------------------------------

uint8_t stagingBuffer[OTCFG_MAX_BYTES];
uint32_t stagingLength = 0;       // 0 = no session open
uint32_t stagingReceived = 0;
bool provisioning = false;

// Longest legal line is D<off>,<128 hex>, so this has ample slack.
char commandLine[192];
uint8_t commandLength = 0;
bool commandOverflow = false;

int32_t latestAccel[3] = {0, 0, 0};
int32_t latestGyro[3] = {0, 0, 0};
// The STM32duino library returns quaternion elements in [x,y,z,w] order.
float latestQuat[4] = {0.0f, 0.0f, 0.0f, 1.0f};
float latestEuler[3] = {0.0f, 0.0f, 0.0f};

struct VibrationSample {
  uint32_t sequence;
  int32_t acceleration[3];
};

// Absorbs USB/scan stalls; drops oldest when full so the stream stays current.
constexpr uint16_t VIBRATION_QUEUE_CAPACITY = 256;
VibrationSample vibrationQueue[VIBRATION_QUEUE_CAPACITY];
uint16_t vibrationQueueHead = 0;
uint16_t vibrationQueueTail = 0;
uint16_t vibrationQueueCount = 0;
uint32_t vibrationSequence = 0;
uint32_t vibrationDropped = 0;

// Peak of the last frame, the one summary LED_SENSOR needs.
uint16_t peakTactile = 0;

// Statistics, zeroed by `z` so a host can measure over a window rather than
// since boot.
uint32_t statScanUsMax = 0;
uint32_t statLoopUsMax = 0;
uint32_t statFrames = 0;

uint8_t manualRgb[3] = {0, 0, 0};
uint32_t lastStatusMs = 0;
bool statusDue = true;

uint32_t frameCounter = 0;
uint32_t lastFrameUs = 0;
uint32_t lastRgbMs = 0;
uint32_t lastNoConfigMs = 0;
uint32_t lastNoBaselineMs = 0;
uint8_t rgbStep = 0;

// ----- DFU ------------------------------------------------------------------
//
// `B` over CDC reboots into the STM32 system bootloader, so reflashing never
// needs the BOOT0 button. The jump happens first thing after the reset, before
// any peripheral is configured, which is what makes it reliable.

#define BOOT_MAGIC_DFU 0xB00710ADUL
__attribute__((section(".noinit"))) static uint32_t bootMagic;

static void enterDfu() {
  bootMagic = BOOT_MAGIC_DFU;
  NVIC_SystemReset();
}

static void checkDfuRequest() {
  if (bootMagic != BOOT_MAGIC_DFU) return;
  bootMagic = 0;
  __disable_irq();
  HAL_RCC_DeInit();
  SysTick->CTRL = 0;
  SysTick->LOAD = 0;
  SysTick->VAL  = 0;
  for (uint8_t i = 0; i < 8; i++) { NVIC->ICER[i] = 0xFFFFFFFF; NVIC->ICPR[i] = 0xFFFFFFFF; }
  __HAL_SYSCFG_REMAPMEMORY_SYSTEMFLASH();
  const uint32_t sysMem = 0x1FFF0000UL;
  uint32_t sp = *(volatile uint32_t *)sysMem;
  uint32_t pc = *(volatile uint32_t *)(sysMem + 4);
  __set_MSP(sp);
  ((void (*)(void))pc)();
  while (1) {}
}

// ----- Tactile --------------------------------------------------------------

// analogRead() re-inits and recalibrates every call; configure ADC1 once.
static bool tactileAdcBegin() {
  // Everything omitted here is zero, which is the default this needs
  // (right-aligned, single conversion, no scan/DMA/oversampling/trigger).
  tactileAdc.Instance = ADC1;
  tactileAdc.Init.ClockPrescaler = ADC_CLOCK_SYNC_PCLK_DIV4;
  tactileAdc.Init.Resolution = ADC_RESOLUTION_12B;
  tactileAdc.Init.EOCSelection = ADC_EOC_SINGLE_CONV;
  tactileAdc.Init.NbrOfConversion = 1;
  tactileAdc.Init.Overrun = ADC_OVR_DATA_OVERWRITTEN;
  if (HAL_ADC_Init(&tactileAdc) != HAL_OK) return false;
  if (HAL_ADCEx_Calibration_Start(&tactileAdc, ADC_SINGLE_ENDED) != HAL_OK) {
    return false;
  }

  ADC_ChannelConfTypeDef channel = {};
  channel.Channel = ADC_CHANNEL_1;  // PA0 = ADC1_IN1
  channel.Rank = ADC_REGULAR_RANK_1;
  channel.SamplingTime = TACTILE_SAMPLE_TIME;
  channel.SingleDiff = ADC_SINGLE_ENDED;
  channel.OffsetNumber = ADC_OFFSET_NONE;
  if (HAL_ADC_ConfigChannel(&tactileAdc, &channel) != HAL_OK) return false;

  // Enable the ADC once here so adcRead() never has to.
  if ((ADC1->CR & ADC_CR_ADEN) == 0u) {
    ADC1->ISR = ADC_ISR_ADRDY;
    ADC1->CR |= ADC_CR_ADEN;
    uint32_t guard = 0;
    while ((ADC1->ISR & ADC_ISR_ADRDY) == 0u) {
      if (++guard > 100000u) return false;
    }
  }
  return true;
}

// Direct ADSTART/EOC/DR; HAL's call overhead dwarfs the conversion itself.
static inline uint16_t adcRead() {
  ADC1->CR |= ADC_CR_ADSTART;
  uint32_t guard = 0;
  while ((ADC1->ISR & ADC_ISR_EOC) == 0u) {
    if (++guard > 20000u) return 0;
  }
  return (uint16_t)ADC1->DR;
}

// `channel` is what the five nets carry: bit i drives MUXi. It is already the
// hardware channel -- col_order did the mapping when the plan was built.
static inline void selectMuxChannel(uint8_t channel) {
  GPIOA->BSRR = ((channel & 0x01u) ? GPIO_PIN_6 : (GPIO_PIN_6 << 16)) |
                ((channel & 0x02u) ? GPIO_PIN_7 : (GPIO_PIN_7 << 16)) |
                ((channel & 0x04u) ? GPIO_PIN_10 : (GPIO_PIN_10 << 16));
#if defined(OT_BOARD_NODE16)
  // 16:1 mux: MUX3 is the top address bit, and PB5 belongs to something else.
  GPIOB->BSRR = ((channel & 0x08u) ? GPIO_PIN_3 : (GPIO_PIN_3 << 16));
#else
  GPIOB->BSRR = (((channel & 0x08u) ? GPIO_PIN_3 : (GPIO_PIN_3 << 16)) |
                 ((channel & 0x10u) ? GPIO_PIN_5 : (GPIO_PIN_5 << 16)));
#endif
}

// MR clears the shift register; the latch pushes all-zero to the outputs.
static void rowsOff() {
  GPIOA->BSRR = SR_MR_BIT << 16;
  __NOP(); __NOP(); __NOP(); __NOP();
  GPIOA->BSRR = SR_MR_BIT;
  GPIOA->BSRR = SR_STCP_BIT << 16;
  __NOP(); __NOP();
  GPIOA->BSRR = SR_STCP_BIT;
}

static void tactileBegin() {
  pinMode(SR_DS, OUTPUT);   digitalWrite(SR_DS, LOW);
  pinMode(SR_SHCP, OUTPUT); digitalWrite(SR_SHCP, LOW);
  pinMode(SR_STCP, OUTPUT); digitalWrite(SR_STCP, LOW);
  pinMode(SR_OE, OUTPUT);   digitalWrite(SR_OE, HIGH);
  pinMode(SR_MR, OUTPUT);

  rowsOff();
  digitalWrite(SR_OE, LOW);

  pinMode(MUX0_PIN, OUTPUT); digitalWrite(MUX0_PIN, LOW);
  pinMode(MUX1_PIN, OUTPUT); digitalWrite(MUX1_PIN, LOW);
  pinMode(MUX2_PIN, OUTPUT); digitalWrite(MUX2_PIN, LOW);
  pinMode(MUX3_PIN, OUTPUT); digitalWrite(MUX3_PIN, LOW);
#if !defined(OT_BOARD_NODE16)
  pinMode(MUX4_PIN, OUTPUT); digitalWrite(MUX4_PIN, LOW);
#endif

  pinMode(TACTILE_ADC, INPUT_ANALOG);
  analogReadResolution(12);
  tactileAdcBegin();
}

// ----- IMU ------------------------------------------------------------------

static void updateEuler() {
  const float qx = latestQuat[0];
  const float qy = latestQuat[1];
  const float qz = latestQuat[2];
  const float qw = latestQuat[3];

  latestEuler[0] = atan2f(2.0f * (qw * qx + qy * qz),
                          1.0f - 2.0f * (qx * qx + qy * qy)) * RAD_TO_DEG_F;
  float sinPitch = 2.0f * (qw * qy - qz * qx);
  if (sinPitch > 1.0f) sinPitch = 1.0f;
  if (sinPitch < -1.0f) sinPitch = -1.0f;
  latestEuler[1] = asinf(sinPitch) * RAD_TO_DEG_F;
  latestEuler[2] = atan2f(2.0f * (qw * qz + qx * qy),
                          1.0f - 2.0f * (qy * qy + qz * qz)) * RAD_TO_DEG_F;
}

// Applies every IMU-facing parameter. Called at boot and whenever one of them
// changes -- always from loop(), never from inside a scan, since these are I2C
// transactions of hundreds of microseconds.
static void imuApplyParams() {
  if (!imuReady) return;
  const float vibOdr = (float)param(OT_P_VIB_ODR_HZ);
  const float oriOdr = (float)param(OT_P_ORIENT_ODR_HZ);

  imu.Set_X_FS((int32_t)param(OT_P_ACC_FS_G));
  imu.Set_G_FS((int32_t)param(OT_P_GYRO_FS_DPS));
  // High-performance = wide-bandwidth accel; FIFO batching preserves every
  // sample regardless of frame rate.
  imu.Set_X_ODR(vibOdr, LSM6DSV16BX_ACC_HIGH_PERFORMANCE_MODE);
  imu.Set_G_ODR(oriOdr);
  imu.Set_SFLP_ODR(oriOdr);

  // Disabling a modality has to stop the work at its source, or "disabled"
  // means something different for each one.
  if (param(OT_P_ORIENT_ENABLE)) imu.Enable_Rotation_Vector();
  else imu.Disable_Rotation_Vector();

  // BDR 0 keeps the accelerometer out of the FIFO entirely. The FIFO itself is
  // still drained: it also carries the rotation vector, and gating the drain on
  // the vibration flag would silently stop orientation as well.
  imu.FIFO_Set_X_BDR(param(OT_P_VIB_ENABLE) ? vibOdr : 0.0f);
  imu.FIFO_Set_Mode(LSM6DSV16BX_STREAM_MODE);
}

static bool imuBegin() {
  if (imu.begin() != LSM6DSV16BX_OK) return false;
  imuReady = true;

  // Fusion setup resets control registers, so it goes first.
  imu.Enable_Sensor_Fusion();
  imu.Enable_X();
  imu.Enable_G();
  imuApplyParams();
  return true;
}

#if defined(OT_BOARD_READOUT60)
// ----- Gain (60-pin board) ----------------------------------------------------
//
// A MAX14662 eight-switch chip at I2C 0x4C (AD0 = AD1 = GND), on the IMU's
// bus, puts resistors across the transimpedance amplifier's 10k feedback
// resistor. Its only register, 0x00, holds the switch states: bit k is SW(k+1),
// 1 closed. It powers up all open. Written from loop(), never mid-scan.
#define GAIN_SWITCH_ADDR 0x4C

static bool gainWrite(uint8_t want) {
  IMU_WIRE.beginTransmission(GAIN_SWITCH_ADDR);
  IMU_WIRE.write((uint8_t)0x00);
  IMU_WIRE.write(want);
  if (IMU_WIRE.endTransmission() != 0) return false;
  // Read it back: the chip always answers with that one register.
  if (IMU_WIRE.requestFrom((uint8_t)GAIN_SWITCH_ADDR, (uint8_t)1) != 1) return false;
  return (uint8_t)IMU_WIRE.read() == want;
}

static bool gainApply() { return gainWrite((uint8_t)param(OT_P_GAIN_SW)); }
#endif

// Extra wait after each IMU FIFO drain, in microseconds; off by default (0).
// On one 70-pin board the reads just after a drain spiked by 600-750 counts,
// within ~30-90 us of it. Building with -DOT_POST_DRAIN_US=100 gives those
// reads time to settle before the scan continues, at the cost of the wait on
// every drain.
#ifndef OT_POST_DRAIN_US
#define OT_POST_DRAIN_US 0
#endif

static void drainFifo() {
  if (!imuReady) return;
  // With both IMU streams off nothing is batched, and the FIFO status poll
  // alone is an I2C transaction of a few hundred microseconds -- called several
  // times a frame, it would be a fixed cost that keeps small xampling frames
  // from scaling with M.
  if (!param(OT_P_VIB_ENABLE) && !param(OT_P_ORIENT_ENABLE)) return;

  uint16_t fifoSamples = 0;
  if (imu.FIFO_Get_Num_Samples(&fifoSamples) != LSM6DSV16BX_OK) return;

  for (uint16_t i = 0; i < fifoSamples; ++i) {
    uint8_t tag = 0;
    if (imu.FIFO_Get_Tag(&tag) != LSM6DSV16BX_OK) break;

    if (tag == 0x02u) {
      int32_t acceleration[3];
      if (imu.FIFO_Get_X_Axes(acceleration) != LSM6DSV16BX_OK) continue;
      if (!param(OT_P_VIB_ENABLE)) continue;
      latestAccel[0] = acceleration[0];
      latestAccel[1] = acceleration[1];
      latestAccel[2] = acceleration[2];

      if (vibrationQueueCount == VIBRATION_QUEUE_CAPACITY) {
        vibrationQueueTail = (vibrationQueueTail + 1u) % VIBRATION_QUEUE_CAPACITY;
        --vibrationQueueCount;
        ++vibrationDropped;
      }
      VibrationSample &sample = vibrationQueue[vibrationQueueHead];
      sample.sequence = vibrationSequence++;
      sample.acceleration[0] = acceleration[0];
      sample.acceleration[1] = acceleration[1];
      sample.acceleration[2] = acceleration[2];
      vibrationQueueHead = (vibrationQueueHead + 1u) % VIBRATION_QUEUE_CAPACITY;
      ++vibrationQueueCount;
    } else if (tag == 0x13u) {
      if (imu.FIFO_Get_Rotation_Vector(latestQuat) == LSM6DSV16BX_OK) {
        updateEuler();
      }
    } else {
      uint8_t discarded[6];
      imu.FIFO_Get_Data(discarded);
    }
  }
#if OT_POST_DRAIN_US > 0
  delayMicroseconds(OT_POST_DRAIN_US);
#endif
}

// Clear, then clock in a single 1 so the first storage stage is active.
static void rowWalkReset() {
  GPIOA->BSRR = SR_MR_BIT << 16;     // MR low
  __NOP(); __NOP(); __NOP(); __NOP();
  GPIOA->BSRR = SR_MR_BIT;           // MR high
  GPIOA->BSRR = SR_STCP_BIT << 16;     // STCP low
  GPIOA->BSRR = SR_DS_BIT;           // DS high
  __NOP(); __NOP();
  GPIOA->BSRR = SR_SHCP_BIT;           // SHCP rising
  __NOP(); __NOP(); __NOP();
  GPIOA->BSRR = SR_SHCP_BIT << 16;     // SHCP falling
  GPIOA->BSRR = SR_DS_BIT << 16;     // DS low: every later shift injects 0
  GPIOA->BSRR = SR_STCP_BIT;           // STCP high latches P0 active
}

// One clock walks the active bit to the next stage.
static inline void rowWalkAdvance() {
  GPIOA->BSRR = SR_STCP_BIT << 16;     // STCP low
  __NOP(); __NOP();
  GPIOA->BSRR = SR_SHCP_BIT;           // SHCP rising
  __NOP(); __NOP(); __NOP();
  GPIOA->BSRR = SR_SHCP_BIT << 16;     // SHCP falling
  __NOP(); __NOP();
  GPIOA->BSRR = SR_STCP_BIT;           // STCP high latches the next row
}

// The walking 1 only moves forward, so shift-register steps are visited in
// order, and at each step the plan lists every active taxel with a line on it:
// the mux channel to read and the frame slot the sample belongs in. Whether the
// driven line is the skin's row or its column is already resolved in the plan.
// `extraSettleUs` lengthens the mux settle for this scan only -- `Z` uses it to
// measure at the settle a confirmed xampling reading gets.
static void scanTactileFrame(uint16_t extraSettleUs = 0) {
  // Hoisted: the mux settle is paid per taxel, so a table lookup inside that
  // loop would be one of the few things fast enough to matter.
  const uint16_t rowSettleUs = param(OT_P_ROW_SETTLE_US);
  const uint16_t muxSettleUs = (uint16_t)(param(OT_P_MUX_SETTLE_US) + extraSettleUs);
  peakTactile = 0;

  rowWalkReset();

  for (uint8_t step = 0; step < scanPlan.hwRowSteps; ++step) {
    if (step != 0) rowWalkAdvance();

    const uint16_t begin = scanPlan.stepStart[step];
    const uint16_t end = scanPlan.stepStart[step + 1u];
    if (begin != end) {
      delayMicroseconds(rowSettleUs);
      for (uint16_t e = begin; e < end; ++e) {
        selectMuxChannel(scanPlan.entryMux[e]);
        delayMicroseconds(muxSettleUs);
        tactileFrame[scanPlan.entryOut[e]] = adcRead();
      }
    }

    if (((step + 1u) % FIFO_DRAIN_ROW_INTERVAL) == 0u) drainFifo();
  }
  rowsOff();

  // Cheapest possible summary, for LED_SENSOR. Kept out of the sampling loop
  // so switching the LED mode cannot perturb scan timing.
  for (uint16_t i = 0; i < deviceConfig.activeCount; ++i) {
    if (tactileFrame[i] > peakTactile) peakTactile = tactileFrame[i];
  }
}

#if defined(OT_BOARD_READOUT60)
// ----- HDR raster (60-pin board) ----------------------------------------------
//
// After a raster scan at GAIN_SW, each nibble of HDR_GAINS (lowest first, up
// to the first 0) is one more pass, at that GAIN_SW, over only the taxels whose
// reading in the pass before reached HDR_CLIP. The new reading replaces the
// old one, tagged with its pass in bits 12-15 (value = raw + 4096 x pass), so
// the host converts every taxel with the gain it was actually read at. A pass
// with nothing to re-read ends the frame, so a frame with nothing clipped
// costs nothing extra; otherwise each pass costs one switch (~0.13 ms, written
// and read back) plus one read per taxel, and the gain is back at GAIN_SW
// before the frame is sent. A switch that does not take ends the passes: the
// readings so far stand, and hdrFailures counts it.
#define HDR_SETTLE_US 20    // after a switch: the first read 15 us after one is within 0.2 %
uint32_t hdrFailures = 0;

static void hdrRereadPasses() {
  uint16_t gains = param(OT_P_HDR_GAINS);
  if (gains == 0u) return;
  const uint16_t clip = param(OT_P_HDR_CLIP);
  if (peakTactile < clip) return;      // nothing reached the clip: no pass, no cost
  const uint16_t rowSettleUs = param(OT_P_ROW_SETTLE_US);
  const uint16_t muxSettleUs = param(OT_P_MUX_SETTLE_US);
  const uint16_t active = deviceConfig.activeCount;
  bool switched = false;

  for (uint16_t pass = 1; (gains & 0x0Fu) != 0u; ++pass, gains = (uint16_t)(gains >> 4)) {
    // Due: read in the pass before, and at or above the clip level there.
    const uint16_t due = (uint16_t)((pass - 1u) << 12);
    bool any = false;
    for (uint16_t k = 0; k < active && !any; ++k) {
      any = (tactileFrame[k] & 0xF000u) == due && (tactileFrame[k] & 0x0FFFu) >= clip;
    }
    if (!any) break;
    if (!gainWrite((uint8_t)(gains & 0x0Fu))) { ++hdrFailures; break; }
    switched = true;
    delayMicroseconds(HDR_SETTLE_US);

    rowWalkReset();
    for (uint8_t step = 0; step < scanPlan.hwRowSteps; ++step) {
      if (step != 0) rowWalkAdvance();
      bool settled = false;
      for (uint16_t e = scanPlan.stepStart[step]; e < scanPlan.stepStart[step + 1u]; ++e) {
        const uint16_t slot = scanPlan.entryOut[e];
        if ((tactileFrame[slot] & 0xF000u) != due || (tactileFrame[slot] & 0x0FFFu) < clip) continue;
        if (!settled) { delayMicroseconds(rowSettleUs); settled = true; }
        selectMuxChannel(scanPlan.entryMux[e]);
        delayMicroseconds(muxSettleUs);
        tactileFrame[slot] = (uint16_t)(adcRead() | (pass << 12));
      }
    }
    rowsOff();
  }
  if (switched && !gainApply()) ++hdrFailures;
}
#endif

// Xampling reads each batch in scan-plan order, so within a batch the walking 1
// only moves forward; the next batch starts by clearing the register and
// clocking forward again, which costs microseconds, not a settle.
//
// Unlike raster it jumps several steps at a time, and the jump must not drive
// the lines it passes. A '595 shifts and latches separately, so the bit is
// clocked to its step with the latch held low and latched once: the outputs go
// straight from the old line to the new one. (Walking with rowWalkAdvance()
// would latch every step in between -- a ~20 ns pulse on each skipped line,
// and step 0 on every rewind -- which raster, moving one step at a time, never
// does.) MR clears only the shift register, so the old line stays driven until
// that single latch.
// ROW_SETTLE is paid once per change of step, MUX_SETTLE once per taxel -- the
// same two costs the raster scan pays.
uint8_t xampStep = 0xFF;          // 0xFF = register cleared
uint32_t xampLastDrainUs = 0;

static void xampWalkTo(uint8_t step) {
  uint8_t at = xampStep;
  GPIOA->BSRR = SR_STCP_BIT << 16;     // STCP low: outputs hold until the end
  if (at == 0xFF || step < at) {
    GPIOA->BSRR = SR_MR_BIT << 16;   // MR low: clear the shift register
    __NOP(); __NOP(); __NOP(); __NOP();
    GPIOA->BSRR = SR_MR_BIT;         // MR high
    GPIOA->BSRR = SR_DS_BIT;         // DS high
    __NOP(); __NOP();
    GPIOA->BSRR = SR_SHCP_BIT;         // SHCP rising: the 1 enters stage 0
    __NOP(); __NOP(); __NOP();
    GPIOA->BSRR = SR_SHCP_BIT << 16;   // SHCP falling
    GPIOA->BSRR = SR_DS_BIT << 16;   // DS low: every later shift injects 0
    at = 0;
  }
  while (at < step) {
    __NOP(); __NOP();
    GPIOA->BSRR = SR_SHCP_BIT;         // SHCP rising
    __NOP(); __NOP(); __NOP();
    GPIOA->BSRR = SR_SHCP_BIT << 16;   // SHCP falling
    ++at;
  }
  __NOP(); __NOP();
  GPIOA->BSRR = SR_STCP_BIT;           // STCP high: one latch, straight to `step`
  xampStep = step;
}
uint16_t xampRowSettleUs = 0;
bool xampImuOn = false;          // skip the drain clock check when nothing is batched
uint16_t xampMuxSettleUs = 0;

static uint16_t xampReadTaxel(uint8_t step, uint8_t mux) {
  if (step != xampStep) {
    // Drain between steps, as raster does: before the next line is driven, so
    // it is not held for the ~2 ms of I2C ahead of its first read.
    if (xampImuOn) {
      const uint32_t now = micros();
      if ((uint32_t)(now - xampLastDrainUs) >= XAMP_FIFO_DRAIN_US) {
        drainFifo();
        xampLastDrainUs = now;
      }
    }
    xampWalkTo(step);
    delayMicroseconds(xampRowSettleUs);
  }
  selectMuxChannel(mux);
  delayMicroseconds(xampMuxSettleUs);
  return adcRead();
}

uint16_t xampConfirmUs = 0;

// Same step, same mux channel: only more settling before a second conversion.
static uint16_t xampRereadTaxel() {
  delayMicroseconds(xampConfirmUs);
  return adcRead();
}

static void scanXampleFrame() {
  xampRowSettleUs = param(OT_P_ROW_SETTLE_US);
  xampConfirmUs = param(OT_P_XAMP_CONFIRM_US);
  xampImuOn = imuReady && (param(OT_P_VIB_ENABLE) || param(OT_P_ORIENT_ENABLE));
  xampMuxSettleUs = param(OT_P_MUX_SETTLE_US);
  const XampSettings settings = {
    (uint8_t)param(OT_P_TAC_MODE), param(OT_P_XAMP_M),
    param(OT_P_XAMP_THRESH), param(OT_P_XAMP_SEARCH_MIN),
    param(OT_P_XAMP_BUMP_FILL) != 0u};

  xampStep = 0xFF;
  xampFrame(&xampGrid, &settings, xampBaseline, xampNoise, tactileFrame,
            &xampState, xampReadTaxel, xampConfirmUs ? xampRereadTaxel : nullptr);
  rowsOff();

  peakTactile = xampState.peak;
}

#if defined(OT_BOARD_READOUT60)
// ----- HDR in Xampling (60-pin board) -------------------------------------------
//
// The passes of hdrRereadPasses(), over only the taxels this frame sampled.
// Their values are counts above the board's zero (`Z`, measured at GAIN_SW), so
// a taxel is due for pass 1 when its value plus its zero reached HDR_CLIP. A
// re-read value is the raw reading at that pass's gain, tagged with the pass as
// in raster (raw + 4096 x pass): the board keeps a zero at one gain only, so
// the host, which measured the zero at every gain, converts it. Reads go through
// xampReadTaxel(), so each change of driven line costs one walk and ROW_SETTLE,
// as the frame's own reads do.
static inline bool hdrSampledDue(uint16_t taxel, uint16_t pass, uint16_t clip) {
  if (((xampState.taxelMask[taxel >> 5] >> (taxel & 31u)) & 1u) == 0u) return false;
  const uint16_t v = tactileFrame[taxel];
  if ((uint16_t)(v >> 12) != (uint16_t)(pass - 1u)) return false;
  const uint32_t raw = (pass == 1u) ? (uint32_t)v + xampBaseline[taxel] : (uint32_t)(v & 0x0FFFu);
  return raw >= clip;
}

// Whether any sampled taxel is due: only the frame's samples are visited, so a
// frame with nothing to re-read costs a few microseconds.
static bool hdrSampledAny(uint16_t pass, uint16_t clip) {
  for (uint16_t word = 0; word < XAMP_WORDS; ++word) {
    uint32_t bits = xampState.taxelMask[word];
    while (bits) {
      const uint16_t taxel = (uint16_t)((word << 5) | (uint16_t)__builtin_ctz(bits));
      bits &= bits - 1u;
      if (hdrSampledDue(taxel, pass, clip)) return true;
    }
  }
  return false;
}

static void hdrRereadSampled() {
  uint16_t gains = param(OT_P_HDR_GAINS);
  if (gains == 0u) return;
  const uint16_t clip = param(OT_P_HDR_CLIP);
  // No sampled reading can have reached the clip: none is above the frame's peak
  // value plus the highest zero. So a frame with nothing pressed costs nothing.
  if ((uint32_t)xampState.peak + xampBaselineMax < clip) return;
  bool switched = false;

  for (uint16_t pass = 1; (gains & 0x0Fu) != 0u; ++pass, gains = (uint16_t)(gains >> 4)) {
    if (!hdrSampledAny(pass, clip)) break;
    if (!gainWrite((uint8_t)(gains & 0x0Fu))) { ++hdrFailures; break; }
    switched = true;
    delayMicroseconds(HDR_SETTLE_US);

    xampStep = 0xFF;
    for (uint8_t step = 0; step < scanPlan.hwRowSteps; ++step) {
      for (uint16_t e = scanPlan.stepStart[step]; e < scanPlan.stepStart[step + 1u]; ++e) {
        const uint16_t slot = scanPlan.entryOut[e];
        if (!hdrSampledDue(slot, pass, clip)) continue;
        tactileFrame[slot] = (uint16_t)(xampReadTaxel(step, scanPlan.entryMux[e]) | (pass << 12));
      }
    }
    rowsOff();
  }
  if (switched && !gainApply()) ++hdrFailures;
}
#endif

// ----- RGB ------------------------------------------------------------------

static void rgbBegin() {
  // PB6 doubles as UCPD1_CC1; clear its dead-battery pull-down explicitly.
#if defined(PWR_CR3_UCPD_DBDIS)
  SET_BIT(PWR->CR3, PWR_CR3_UCPD_DBDIS);
#endif
  rgbPixel.begin();
  rgbPixel.setBrightness((uint8_t)param(OT_P_LED_BRIGHT));
  rgbPixel.clear();
  rgbPixel.show();
}

// Mode-driven, but every mode still goes through one rate limit and one
// brightness, so switching mode cannot accidentally hammer the NeoPixel.
static void updateRgb() {
  const uint16_t mode = param(OT_P_LED_MODE);
  const uint8_t bright = (uint8_t)param(OT_P_LED_BRIGHT);
  const uint32_t now = millis();
  const uint32_t periodMs = 1000u / param(OT_P_LED_RATE_HZ);
  if ((uint32_t)(now - lastRgbMs) < periodMs) return;
  lastRgbMs = now;

  uint8_t r = 0, g = 0, b = 0;
  switch (mode) {
    case OT_LED_MANUAL:
      r = manualRgb[0]; g = manualRgb[1]; b = manualRgb[2];
      break;
    case OT_LED_HEARTBEAT:
      r = (rgbStep == 0) ? 255 : 0;
      g = (rgbStep == 1) ? 255 : 0;
      b = (rgbStep == 2) ? 255 : 0;
      rgbStep = (uint8_t)((rgbStep + 1u) % 3u);
      break;
    case OT_LED_SENSOR: {
      // 12-bit ADC down to 8 bits. Deliberately the crudest possible mapping:
      // anything cleverer is feature extraction, which belongs on the host.
      const uint8_t level = (uint8_t)(peakTactile >> 4);
      r = level; g = (uint8_t)(255u - level); b = 0;
      break;
    }
    case OT_LED_OFF:
    default:
      break;
  }

  rgbPixel.setBrightness(bright);
  rgbPixel.setPixelColor(0, rgbPixel.Color(r, g, b));
  rgbPixel.show();
}

// ----- Transport ------------------------------------------------------------
//
// Where command lines come from and where replies and records go. On USB that
// is Serial. On a bus node it is the two text rings in ot_bus.h, filled and drained by the
// I2C interrupt below, and frames are published as binary instead of printed.

#if !defined(OT_BUS_NODE)

#define OT_OUT Serial
#define OT_IN_AVAILABLE() Serial.available()
#define OT_IN_READ() Serial.read()

#else

// The tx ring holds the longest burst of replies -- `?` is about 1 KB -- with
// room to spare; the rx ring, a few full command lines.
static uint8_t busTxMem[2048];
static uint8_t busRxMem[512];
static OtbTxRing busTx = {busTxMem, sizeof(busTxMem), 0, 0};
static OtbRxRing busRx = {busRxMem, sizeof(busRxMem), 0, 0};

#define OT_IN_AVAILABLE() (busRx.head != busRx.tail)
#define OT_IN_READ() otbRxRead(&busRx)

static volatile uint32_t busLastSeenMs = 0;   // last transaction that reached us
static uint32_t busTextDropped = 0;

static bool busControllerPresent() {
  return busLastSeenMs != 0 && (uint32_t)(millis() - busLastSeenMs) < 1000u;
}

// Replies go into the tx ring. A controller drains it every few milliseconds,
// so a full ring waits for it -- but only while one is actually talking to us.
// A node on a bench with no controller would otherwise stall on every notice.
class BusOut : public Print {
 public:
  size_t write(uint8_t c) override { return write(&c, 1); }
  size_t write(const uint8_t *bytes, size_t n) override {
    size_t done = 0;
    const uint32_t start = millis();
    while (done < n) {
      const size_t left = n - done;
      const uint16_t piece = (uint16_t)(left < busTx.size ? left : busTx.size);
      if (otbTxPush(&busTx, bytes + done, piece)) {
        done += piece;
        continue;
      }
      if (!busControllerPresent() || (uint32_t)(millis() - start) > 50u) {
        busTextDropped += (uint32_t)left;
        break;
      }
      delayMicroseconds(50);
    }
    return n;
  }
  // `B` flushes before rebooting into DFU, so its reply gets out first.
  void flush() override {
    const uint32_t start = millis();
    while (otbTxUsed(&busTx) && busControllerPresent() && (uint32_t)(millis() - start) < 200u) {
      delayMicroseconds(100);
    }
  }
};
static BusOut busOut;
#define OT_OUT busOut

#if defined(OT_BOARD_NODE16)
#define OTB_THIS_BOARD OTB_BOARD_NODE16
#else
#define OTB_THIS_BOARD OTB_BOARD_READOUT32
#endif

// Addressing: SMBus ARP (see ot_bus.h).
static uint8_t busUid[OTB_UID_BYTES];
static uint8_t busUdid[OTB_UDID_LEN];
static volatile uint8_t busAddr = 0;          // 0 = unassigned: answering 0x61 only
static volatile bool busResolved = false;
static volatile uint8_t busPendingAddr = 0;   // set by the interrupt, applied by loop()
static volatile uint8_t busCmd = 0;           // what the next read answers
static volatile uint8_t busTextMax = 0;
static volatile uint8_t busChunk = 0;
static uint8_t busResp[OTB_CHUNK_RESP_LEN];   // the longest response

// Frames: a triple buffer. One is the newest published, one may be the one the
// controller is partway through reading, and the loop always finds a third
// that is neither -- so publishing never waits for the bus, and a read in
// progress is never overwritten underneath it.
#define BUS_BLOB_BYTES OTB_BLOB_MAX(OTCFG_MAX_TAXELS)
static uint8_t busBlob[3][BUS_BLOB_BYTES];
static uint16_t busBlobLen[3];
static uint16_t busBlobSeq[3];
static uint32_t busBlobUs[3];
static uint32_t busBlobVibEnd[3];              // vibration seq just past the blob's samples
static volatile uint8_t busReady = 0xFF;       // newest published, or none yet
static volatile uint8_t busServed = 0xFF;      // latched by a chunk-0 read
static volatile uint32_t busVibDelivered = 0;  // samples before this seq reached the controller
static uint16_t busPublishSeq = 0;
static uint32_t busLastPublishUs = 0;

// Vibration samples ride along in blobs and leave the queue only once a blob
// carrying them has been read to its last chunk. Most blobs are never read --
// the node publishes faster than the bus drains -- and a sample must not be
// lost just because the blob it happened to land in was.
static void busDropDelivered() {
  const uint32_t upTo = busVibDelivered;
  while (vibrationQueueCount != 0 &&
         (int32_t)(vibrationQueue[vibrationQueueTail].sequence - upTo) < 0) {
    vibrationQueueTail = (uint16_t)((vibrationQueueTail + 1u) % VIBRATION_QUEUE_CAPACITY);
    --vibrationQueueCount;
  }
}

// `flags` OTB_FF_TACTILE (+ OTB_FF_SPARSE for xampling), or 0 for an IMU-only
// blob. The IMU block goes in whenever either IMU stream is on.
static void busPublish(uint32_t deltaUs, uint8_t flags) {
  busDropDelivered();
  uint8_t w = 0;
  while (w == busReady || w == busServed) ++w;

  static int32_t vib[OTB_MAX_VIB * 3];
  uint8_t vibCount = 0;
  const uint32_t vibSeq = vibrationQueueCount ? vibrationQueue[vibrationQueueTail].sequence
                                              : vibrationSequence;
  for (uint16_t i = 0, q = vibrationQueueTail; i < vibrationQueueCount && vibCount < OTB_MAX_VIB; ++i) {
    for (uint8_t a = 0; a < 3; ++a) vib[vibCount * 3u + a] = vibrationQueue[q].acceleration[a];
    ++vibCount;
    q = (uint16_t)((q + 1u) % VIBRATION_QUEUE_CAPACITY);
  }

  OtbFrame f = {};
  f.flags = flags;
  if (imuReady && (param(OT_P_VIB_ENABLE) || param(OT_P_ORIENT_ENABLE))) f.flags |= OTB_FF_IMU;
  f.frame = frameCounter;
  f.generation = configGeneration;
  f.mode = (uint8_t)param(OT_P_TAC_MODE);
  f.dtUs = deltaUs;
  f.vibDropped = vibrationDropped;
  f.active = deviceConfig.valid ? deviceConfig.activeCount : 0;
  f.values = tactileFrame;
  f.mask = xampState.taxelMask;
  for (uint8_t i = 0; i < 3; ++i) {
    f.accel[i] = latestAccel[i];
    f.gyro[i] = latestGyro[i];
    f.euler[i] = latestEuler[i];
  }
  f.quat[0] = latestQuat[3];      // w, x, y, z out; stored x, y, z, w
  f.quat[1] = latestQuat[0];
  f.quat[2] = latestQuat[1];
  f.quat[3] = latestQuat[2];
  f.vibCount = vibCount;
  f.vibSeq = vibSeq;
  f.vib = vib;

  const uint16_t len = otbBuildFrame(busBlob[w], BUS_BLOB_BYTES, &f);
  if (len == 0) return;
  busBlobLen[w] = len;
  busBlobSeq[w] = ++busPublishSeq;
  busBlobUs[w] = micros();
  busBlobVibEnd[w] = vibSeq + vibCount;
  __disable_irq();
  busReady = w;
  __enable_irq();
  busLastPublishUs = busBlobUs[w];
}

// With tactile off (or not yet provisioned) the IMU still has to reach the
// host, so it is published alone every 10 ms while either stream is on.
static void busPublishImuOnly() {
  if (!imuReady || !(param(OT_P_VIB_ENABLE) || param(OT_P_ORIENT_ENABLE))) return;
  if ((uint32_t)(micros() - busLastPublishUs) < 10000u) return;
  busPublish(0, 0);
}

// ----- the I2C interrupt: nothing here waits, scans or formats text ----------

static void busOnReceive(int count) {
  (void)count;
  uint8_t b[OTB_WIRE_LIMIT + 8];
  uint16_t n = 0;
  while (Wire.available() && n < sizeof(b)) b[n++] = (uint8_t)Wire.read();
  while (Wire.available()) Wire.read();
  if (n == 0) return;
  busLastSeenMs = millis();
  const uint8_t addr = busAddr;

  switch (b[0]) {
    case OTB_ARP_PREPARE:
      if (n >= 2 && b[1] != otbPecSendByte(OTB_ADDR_ARP, OTB_ARP_PREPARE)) return;
      busResolved = false;
      break;
    case OTB_ARP_RESET:
      if (n >= 2 && b[1] != otbPecSendByte(OTB_ADDR_ARP, OTB_ARP_RESET)) return;
      busResolved = false;
      busPendingAddr = 0xFF;
      break;
    case OTB_ARP_GET_UDID:
      break;                                    // answered by the read
    case OTB_ARP_ASSIGN: {
      if (n < OTB_ARP_WRITE_LEN || b[1] != OTB_ARP_BLOCK_LEN) return;
      if (b[OTB_ARP_WRITE_LEN - 1] !=
          otbPecBlockWrite(OTB_ADDR_ARP, OTB_ARP_ASSIGN, OTB_ARP_BLOCK_LEN, &b[2])) return;
      if (memcmp(&b[2], busUdid, OTB_UDID_LEN) != 0) return;
      const uint8_t a = (uint8_t)(b[2 + OTB_UDID_LEN] >> 1);
      if (a >= OTB_ADDR_BASE && a < 0x78) {
        busResolved = true;
        busPendingAddr = a;
      }
      break;
    }
    case OTB_CMD_TEXT: {
      uint16_t off;
      const uint8_t *bytes;
      uint8_t len;
      if (otbParseText(b, n, addr, &off, &bytes, &len)) otbRxAccept(&busRx, off, bytes, len);
      busCmd = 0;
      return;
    }
    case OTB_CMD_GET_TEXT: {
      uint16_t ack;
      uint8_t max;
      if (!otbParseGetTextReq(b, n, addr, &ack, &max)) { busCmd = 0; return; }
      otbTxAck(&busTx, ack);
      busTextMax = max;
      break;
    }
    case OTB_CMD_GET_FRAME: {
      uint8_t chunk;
      if (!otbParseChunkReq(b, n, addr, &chunk)) { busCmd = 0; return; }
      busChunk = chunk;
      if (chunk == 0) busServed = busReady;     // latch the newest blob
      break;
    }
    default:
      break;                                    // STATUS, or a plain INFO read
  }
  busCmd = b[0];
}

static void busOnRequest() {
  const uint8_t addr = busAddr;
  const uint8_t cmd = busCmd;
  busCmd = 0;          // a plain read with no command before it means INFO

  switch (cmd) {
    case OTB_ARP_GET_UDID: {
      uint8_t r[OTB_ARP_READ_LEN];
      // A resolved node must lose the arbitration without going silent: SMBDEN
      // already ACKed the address, and an empty transmit would clock-stretch
      // forever. All ones is the wired-AND identity, so it always loses.
      if (busResolved) {
        memset(r, 0xFF, sizeof(r));
      } else {
        r[0] = OTB_ARP_BLOCK_LEN;
        memcpy(&r[1], busUdid, OTB_UDID_LEN);
        r[1 + OTB_UDID_LEN] = addr ? (uint8_t)((addr << 1) | 1) : (uint8_t)(OTB_ADDR_ARP << 1);
        r[OTB_ARP_READ_LEN - 1] =
            otbPecBlockRead(OTB_ADDR_ARP, OTB_ARP_GET_UDID, OTB_ARP_BLOCK_LEN, &r[1]);
      }
      Wire.write(r, OTB_ARP_READ_LEN);
      return;
    }
    case OTB_CMD_STATUS: {
      OtbStatus st;
      const uint8_t ready = busReady;
      st.flags = (uint8_t)((deviceConfig.valid ? OTB_SF_CONFIG : 0) |
                           (ready != 0xFF ? OTB_SF_FRAME : 0) |
                           (provisioning ? OTB_SF_BUSY : 0) | (imuReady ? OTB_SF_IMU : 0));
      st.frameSeq = ready != 0xFF ? busBlobSeq[ready] : 0;
      const uint32_t used = otbTxUsed(&busTx);
      st.txTail = (uint16_t)busTx.tail;
      st.txPending = (uint16_t)(used > 0xFFFFu ? 0xFFFFu : used);
      st.rxAccepted = (uint16_t)busRx.head;
      const uint16_t space = otbRxFree(&busRx);
      st.rxFree = (uint8_t)(space > 255u ? 255u : space);
      st.generation = configGeneration;
      st.board = OTB_THIS_BOARD;
      otbBuildStatus(busResp, addr, &st);
      Wire.write(busResp, OTB_STATUS_LEN);
      return;
    }
    case OTB_CMD_GET_TEXT: {
      uint8_t bytes[OTB_TEXT_MAX];
      uint16_t off;
      const uint8_t max = busTextMax;
      const uint8_t n = otbTxPeek(&busTx, bytes, max, &off);
      Wire.write(busResp, otbBuildGetTextResp(busResp, addr, off, bytes, n, max));
      return;
    }
    case OTB_CMD_GET_FRAME: {
      const uint8_t s = busServed, chunk = busChunk;
      if (s == 0xFF) {
        otbBuildChunkResp(busResp, addr, 0, chunk, NULL, 0, 0);
      } else {
        const uint32_t age = micros() - busBlobUs[s];
        otbBuildChunkResp(busResp, addr, busBlobSeq[s], chunk, busBlob[s], busBlobLen[s],
                          (uint16_t)(age > 0xFFFFu ? 0xFFFFu : age));
        // The last chunk is the blob delivered; its vibration samples may go.
        if ((uint16_t)(chunk + 1u) == otbChunkCount(busBlobLen[s]) &&
            (int32_t)(busBlobVibEnd[s] - busVibDelivered) > 0) {
          busVibDelivered = busBlobVibEnd[s];
        }
      }
      Wire.write(busResp, OTB_CHUNK_RESP_LEN);
      return;
    }
    default:
      otbBuildInfo(busResp, addr, OTB_THIS_BOARD, busUid);
      Wire.write(busResp, OTB_INFO_LEN);
      return;
  }
}

// `addr` 0 answers only the ARP default address. begin() resets the clock and
// the callbacks, so everything after it is redone on every change of address.
static void busBegin(uint8_t addr) {
  Wire.setSCL(BUS_SCL);
  Wire.setSDA(BUS_SDA);
  Wire.begin(addr ? addr : (uint8_t)OTB_ADDR_ARP, false);
  Wire.setClock(OTB_BUS_HZ);
  Wire.onReceive(busOnReceive);
  Wire.onRequest(busOnRequest);
  // SMBDEN makes the peripheral answer 0x61 in hardware alongside its own
  // address. Only writable while the peripheral is disabled.
  const uint32_t cr1 = I2C1->CR1;
  __disable_irq();
  I2C1->CR1 = cr1 & ~I2C_CR1_PE;
  (void)I2C1->CR1;
  I2C1->CR1 = (cr1 & ~I2C_CR1_PE) | I2C_CR1_SMBDEN;
  I2C1->CR1 = cr1 | I2C_CR1_SMBDEN;
  __enable_irq();
  busAddr = addr;
}

static void busSetup() {
  // Fast-mode-plus drive on the bus pins: 1 MHz over a harness needs it, and
  // the core does not set it.
  __HAL_RCC_SYSCFG_CLK_ENABLE();
  SET_BIT(SYSCFG->CFGR1, SYSCFG_CFGR1_I2C_PB7_FMP | SYSCFG_CFGR1_I2C1_FMP);
  const uint32_t *w = (const uint32_t *)UID_BASE;
  for (uint8_t i = 0; i < 3; i++) {
    busUid[i * 4 + 0] = (uint8_t)(w[i]);
    busUid[i * 4 + 1] = (uint8_t)(w[i] >> 8);
    busUid[i * 4 + 2] = (uint8_t)(w[i] >> 16);
    busUid[i * 4 + 3] = (uint8_t)(w[i] >> 24);
  }
  otbBuildUdid(busUdid, busUid);
}

// Once per loop: an address the interrupt accepted, and `B` on USB.
static void busService() {
  if (busPendingAddr) {
    const uint8_t a = (busPendingAddr == 0xFF) ? 0 : busPendingAddr;
    busPendingAddr = 0;
    Wire.end();
    busBegin(a);
  }
  while (Serial.available()) {
    const int c = Serial.read();
    if (c == 'B' || c == 'b') enterDfu();
  }
}

#endif  // OT_BUS_NODE

// ----- Output ---------------------------------------------------------------

// Serial.print() per value is far costlier than the formatting; build lines in
// RAM and hand the USB stack a few large writes. A 1024-taxel record is ~5 KB,
// more than RAM can spare for one buffer once xampling's tables are in, so
// F and X records go out in 1.5 KB pieces (flushIfFull); everything else fits.
static char lineBuffer[1536];

// The host reads a stream and splits on newlines, and nothing else writes to
// Serial mid-record, so a record written in pieces is the same bytes.
static char *flushIfFull(char *cursor, size_t reserve) {
  if ((size_t)(lineBuffer + sizeof(lineBuffer) - cursor) >= reserve) return cursor;
  OT_OUT.write((const uint8_t *)lineBuffer, (size_t)(cursor - lineBuffer));
  return lineBuffer;
}

static char *appendInt(char *cursor, int32_t value) {
  if (value < 0) {
    *cursor++ = '-';
    value = -value;
  }
  char reversed[10];
  uint8_t digits = 0;
  do {
    reversed[digits++] = (char)('0' + (value % 10));
    value /= 10;
  } while (value != 0);
  while (digits != 0) *cursor++ = reversed[--digits];
  return cursor;
}

static char *appendHex32(char *cursor, uint32_t value) {
  for (int8_t shift = 28; shift >= 0; shift -= 4) {
    const uint8_t nibble = (uint8_t)((value >> shift) & 0xFu);
    *cursor++ = (char)(nibble < 10 ? ('0' + nibble) : ('A' + nibble - 10));
  }
  return cursor;
}

static char *appendFixed(char *cursor, float value, uint8_t decimals) {
  if (!isfinite(value)) {
    *cursor++ = '0';
    return cursor;
  }
  if (value < 0.0f) {
    *cursor++ = '-';
    value = -value;
  }
  uint32_t scale = 1u;
  for (uint8_t i = 0; i < decimals; ++i) scale *= 10u;
  const uint32_t scaled = (uint32_t)(value * (float)scale + 0.5f);
  cursor = appendInt(cursor, (int32_t)(scaled / scale));
  *cursor++ = '.';
  uint32_t divisor = scale;
  const uint32_t fraction = scaled % scale;
  for (uint8_t i = 0; i < decimals; ++i) {
    divisor /= 10u;
    *cursor++ = (char)('0' + ((fraction / divisor) % 10u));
  }
  return cursor;
}

#if defined(OT_BUS_NODE)
// On the bus, vibration samples travel inside the frame blobs (busPublish).
static void flushVibration() {}
#else
// Batch queued samples into as few USB writes as possible.
static void flushVibration() {
  while (vibrationQueueCount != 0) {
    char *cursor = lineBuffer;
    const char *safeEnd = lineBuffer + sizeof(lineBuffer) - 48;
    while (vibrationQueueCount != 0 && cursor < safeEnd) {
      const VibrationSample &sample = vibrationQueue[vibrationQueueTail];
      *cursor++ = 'V';
      *cursor++ = ',';
      cursor = appendInt(cursor, (int32_t)sample.sequence);
      for (uint8_t i = 0; i < 3; ++i) {
        *cursor++ = ',';
        cursor = appendInt(cursor, sample.acceleration[i]);
      }
      *cursor++ = '\n';
      vibrationQueueTail = (vibrationQueueTail + 1u) % VIBRATION_QUEUE_CAPACITY;
      --vibrationQueueCount;
    }
    OT_OUT.write((const uint8_t *)lineBuffer, (size_t)(cursor - lineBuffer));
  }
}

// The IMU tail shared by F and X records. At most ~160 characters.
static char *appendImuFields(char *cursor) {
  for (uint8_t i = 0; i < 3; ++i) {
    *cursor++ = ',';
    cursor = appendInt(cursor, latestAccel[i]);
  }
  for (uint8_t i = 0; i < 3; ++i) {
    *cursor++ = ',';
    cursor = appendInt(cursor, latestGyro[i]);
  }
  // w,x,y,z out; stored x,y,z,w.
  const uint8_t quatOrder[4] = {3, 0, 1, 2};
  for (uint8_t i = 0; i < 4; ++i) {
    *cursor++ = ',';
    cursor = appendFixed(cursor, latestQuat[quatOrder[i]], 5);
  }
  for (uint8_t i = 0; i < 3; ++i) {
    *cursor++ = ',';
    cursor = appendFixed(cursor, latestEuler[i], 2);
  }
  return cursor;
}

static void printFrame(uint32_t deltaUs) {
  char *cursor = lineBuffer;
  *cursor++ = 'F';
  *cursor++ = ',';
  cursor = appendInt(cursor, (int32_t)frameCounter);
  *cursor++ = ',';
  cursor = appendInt(cursor, (int32_t)configGeneration);
  *cursor++ = ',';
  cursor = appendInt(cursor, (int32_t)param(OT_P_TAC_MODE));
  *cursor++ = ',';
  cursor = appendInt(cursor, (int32_t)deltaUs);
  *cursor++ = ',';
  cursor = appendInt(cursor, (int32_t)vibrationDropped);
  for (uint16_t i = 0; i < deviceConfig.activeCount; ++i) {
    cursor = flushIfFull(cursor, 16);
    *cursor++ = ',';
    cursor = appendInt(cursor, tactileFrame[i]);
  }
  cursor = flushIfFull(cursor, 256);
  cursor = appendImuFields(cursor);
  *cursor++ = '\n';
  OT_OUT.write((const uint8_t *)lineBuffer, (size_t)(cursor - lineBuffer));
}

static void printXampleFrame(uint32_t deltaUs) {
  char *cursor = lineBuffer;
  *cursor++ = 'X';
  const uint32_t head[6] = {frameCounter, configGeneration, param(OT_P_TAC_MODE),
                            deltaUs, vibrationDropped, xampState.count};
  for (uint8_t i = 0; i < 6; ++i) {
    *cursor++ = ',';
    cursor = appendInt(cursor, (int32_t)head[i]);
  }
  *cursor++ = ',';
  const uint16_t active = deviceConfig.activeCount;
  static const char HEX_DIGITS[] = "0123456789ABCDEF";
  for (uint16_t byte = 0; byte < (uint16_t)((active + 7u) / 8u); ++byte) {
    cursor = flushIfFull(cursor, 16);
    const uint8_t bits = (uint8_t)(xampState.taxelMask[byte >> 2] >> ((byte & 3u) * 8u));
    *cursor++ = HEX_DIGITS[bits >> 4];
    *cursor++ = HEX_DIGITS[bits & 0x0Fu];
  }
  for (uint16_t word = 0; word < XAMP_WORDS; ++word) {
    uint32_t bits = xampState.taxelMask[word];
    while (bits) {
      const uint16_t taxel = (uint16_t)((word << 5) | (uint16_t)__builtin_ctz(bits));
      bits &= bits - 1u;
      cursor = flushIfFull(cursor, 16);
      *cursor++ = ',';
      cursor = appendInt(cursor, tactileFrame[taxel]);
    }
  }
  cursor = flushIfFull(cursor, 256);
  cursor = appendImuFields(cursor);
  *cursor++ = '\n';
  OT_OUT.write((const uint8_t *)lineBuffer, (size_t)(cursor - lineBuffer));
}
#endif  // !OT_BUS_NODE: on the bus, frames go out through busPublish

static void printLine(const char *text) {
  OT_OUT.write((const uint8_t *)text, strlen(text));
  OT_OUT.write((const uint8_t *)"\n", 1);
}

// R,CFG is the whole of GET_CONFIG: the host already holds the bytes it sent,
// so echoing them back proves nothing that the CRC does not prove more cheaply.
static void printConfigReply() {
  if (!deviceConfig.valid) {
    printLine("R,NOCFG");
    return;
  }
  char *cursor = lineBuffer;
  const char *prefix = "R,CFG,";
  memcpy(cursor, prefix, strlen(prefix));
  cursor += strlen(prefix);
  cursor = appendInt(cursor, (int32_t)deviceConfig.version);
  *cursor++ = ',';
  cursor = appendInt(cursor, (int32_t)deviceConfig.rows);
  *cursor++ = ',';
  cursor = appendInt(cursor, (int32_t)deviceConfig.cols);
  *cursor++ = ',';
  cursor = appendInt(cursor, (int32_t)deviceConfig.activeCount);
  *cursor++ = ',';
  cursor = appendInt(cursor, (int32_t)deviceConfig.pitchXUm);
  *cursor++ = ',';
  cursor = appendInt(cursor, (int32_t)deviceConfig.pitchYUm);
  *cursor++ = ',';
  cursor = appendInt(cursor, (int32_t)deviceConfig.shearCount);
  *cursor++ = ',';
  cursor = appendHex32(cursor, deviceConfig.configId);
  *cursor++ = ',';
  cursor = appendHex32(cursor, deviceConfig.crc);
  *cursor++ = ',';
  cursor = appendInt(cursor, (int32_t)deviceConfig.totalBytes);
  *cursor++ = '\n';
  OT_OUT.write((const uint8_t *)lineBuffer, (size_t)(cursor - lineBuffer));
}

// The frame width changes with the config, so restate it whenever it could
// have moved: at boot, and after any successful commit or erase.
static void printFrameHeader() {
  if (!deviceConfig.valid) {
#if defined(OT_BOARD_READOUT60)
    printLine("# no device config -- run: opentac setup <manifest>");
#else
    // The released 30-pin and 70-pin binaries print this older wording; the
    // command is `opentac setup <manifest>`.
    printLine("# no device config -- run: opentac_device.py provision <manifest>");
#endif
    return;
  }
  char *cursor = lineBuffer;
  const char *prefix = "# F,frame,gen,mode,dt_us,dropped,tactile[";
  memcpy(cursor, prefix, strlen(prefix));
  cursor += strlen(prefix);
  cursor = appendInt(cursor, (int32_t)deviceConfig.activeCount);
  const char *suffix = "],accel_mg[3],gyro_mdps[3],quat_wxyz[4],euler_deg[3]";
  memcpy(cursor, suffix, strlen(suffix));
  cursor += strlen(suffix);
  *cursor++ = '\n';
  OT_OUT.write((const uint8_t *)lineBuffer, (size_t)(cursor - lineBuffer));
  printLine(xampReady
      ? "# X,frame,gen,mode,dt_us,dropped,n,sampled_mask_hex,value_above_baseline[n],accel_mg[3],gyro_mdps[3],quat_wxyz[4],euler_deg[3]"
      : "# xampling unavailable on this skin: a shear group is too spread out to fill");
}

// Says which configuration and mode the stream is currently produced under.
// Emitted on any change and once a second otherwise, so the fact survives in a
// recording even when tactile is switched off and no F records are flowing.
static void printStatus() {
  char *cursor = lineBuffer;
  const char *prefix = "M,";
  memcpy(cursor, prefix, strlen(prefix));
  cursor += strlen(prefix);
  const uint16_t fields[7] = {
    configGeneration, param(OT_P_TAC_MODE), param(OT_P_TAC_ENABLE),
    param(OT_P_VIB_ENABLE), param(OT_P_ORIENT_ENABLE),
    deviceConfig.valid ? deviceConfig.activeCount : (uint16_t)0,
    (deviceConfig.valid && otCfgHasShear(&deviceConfig)) ? (uint16_t)1 : (uint16_t)0};
  for (uint8_t i = 0; i < 7; ++i) {
    if (i) *cursor++ = ',';
    cursor = appendInt(cursor, (int32_t)fields[i]);
  }
  *cursor++ = '\n';
  OT_OUT.write((const uint8_t *)lineBuffer, (size_t)(cursor - lineBuffer));
  lastStatusMs = millis();
  statusDue = false;
}

// ----- Config load ----------------------------------------------------------

// Reads the config straight out of memory-mapped flash. A config that parses
// but whose plan will not build is refused outright rather than half-applied.
static void loadDeviceConfig() {
  OtDeviceConfig candidate;
  if (!otCfgParse(otCfgFlashBase(), OTCFG_FLASH_BYTES, &candidate)) {
    deviceConfig.valid = false;
    return;
  }
  deviceConfig = candidate;
  if (!buildScanPlan(&deviceConfig, &scanPlan)) deviceConfig.valid = false;
}

// Whatever config just loaded, xampling starts over: grid positions, trigger
// points and the baseline all belonged to the previous skin.
static void loadXampling() {
  xampReady = deviceConfig.valid && xampBuildGrid(&deviceConfig, &scanPlan, &xampGrid);
  xampReset(&xampState);
  baselineValid = false;
}

// ----- Commands -------------------------------------------------------------

static int8_t hexDigit(char c) {
  if (c >= '0' && c <= '9') return (int8_t)(c - '0');
  if (c >= 'a' && c <= 'f') return (int8_t)(c - 'a' + 10);
  if (c >= 'A' && c <= 'F') return (int8_t)(c - 'A' + 10);
  return -1;
}

// Parses a decimal run in place; returns the first unconsumed character.
static const char *parseUint(const char *text, uint32_t *out) {
  uint32_t value = 0;
  const char *cursor = text;
  while (*cursor >= '0' && *cursor <= '9') {
    value = value * 10u + (uint32_t)(*cursor - '0');
    cursor++;
  }
  *out = value;
  return cursor;
}

// Every accepted config write moves the generation on, whichever command made
// it -- that is what lets a host notice a reboot from the data stream alone.
static void configChanged() {
  ++configGeneration;
  statusDue = true;
}

static void replyError(const char *reason) {
  char *cursor = lineBuffer;
  const char *prefix = "R,ERR,";
  memcpy(cursor, prefix, strlen(prefix));
  cursor += strlen(prefix);
  memcpy(cursor, reason, strlen(reason));
  cursor += strlen(reason);
  *cursor++ = '\n';
  OT_OUT.write((const uint8_t *)lineBuffer, (size_t)(cursor - lineBuffer));
}

// R,ERR,FLASH,<stage>,<hal_error>,<flash_sr>,<addr>,<dev_id>,<flash_kb>
// stage: 1 erase, 2 program, 3 verify. The chip id and flash size ride along
// because page layout differs across the G4 family, and "which chip is this
// really" is the first question any flash failure raises.
static void replyFlashError() {
  char detail[96];
  char *cursor = detail;
  const char *prefix = "FLASH,";
  memcpy(cursor, prefix, strlen(prefix));
  cursor += strlen(prefix);
  cursor = appendInt(cursor, otCfgFlashDiag.stage);
  *cursor++ = ',';
  cursor = appendHex32(cursor, otCfgFlashDiag.halError);
  *cursor++ = ',';
  cursor = appendHex32(cursor, otCfgFlashDiag.sr);
  *cursor++ = ',';
  cursor = appendHex32(cursor, otCfgFlashDiag.addr);
  *cursor++ = ',';
  cursor = appendHex32(cursor, DBGMCU->IDCODE & 0xFFFu);
  *cursor++ = ',';
  cursor = appendInt(cursor, *(const uint16_t *)FLASHSIZE_BASE);
  *cursor = '\0';
  replyError(detail);
}

static void endStaging() {
  provisioning = false;
  stagingLength = 0;
  stagingReceived = 0;
  // Whatever the IMU banked while the stream was suspended is stale by now.
  vibrationQueueHead = vibrationQueueTail = vibrationQueueCount = 0;
  lastFrameUs = micros();
}

static void handleBegin(const char *args) {
  uint32_t length = 0;
  parseUint(args, &length);
  if (length < OTCFG_HEADER_BYTES + OTCFG_CRC_BYTES ||
      length > OTCFG_MAX_BYTES) {
    replyError("LEN");
    return;
  }
  // 0xFF is the erased value, so an offset never written stays erased-looking
  // and fails the CRC rather than reading as a plausible zero.
  memset(stagingBuffer, 0xFF, sizeof(stagingBuffer));
  stagingLength = length;
  stagingReceived = 0;
  provisioning = true;

  char *cursor = lineBuffer;
  const char *prefix = "R,OK,S,";
  memcpy(cursor, prefix, strlen(prefix));
  cursor += strlen(prefix);
  cursor = appendInt(cursor, (int32_t)length);
  *cursor++ = '\n';
  OT_OUT.write((const uint8_t *)lineBuffer, (size_t)(cursor - lineBuffer));
}

static void handleData(const char *args) {
  if (!provisioning) {
    replyError("NOSESSION");
    return;
  }
  uint32_t offset = 0;
  const char *cursor = parseUint(args, &offset);
  if (*cursor != ',') {
    replyError("SYNTAX");
    return;
  }
  cursor++;

  uint32_t written = 0;
  while (cursor[0] != '\0') {
    const int8_t high = hexDigit(cursor[0]);
    const int8_t low = hexDigit(cursor[1]);
    if (high < 0 || low < 0) {
      replyError("HEX");
      return;
    }
    if (offset + written >= stagingLength) {
      replyError("RANGE");
      return;
    }
    stagingBuffer[offset + written] = (uint8_t)((high << 4) | low);
    written++;
    cursor += 2;
  }
  stagingReceived += written;

  char *out = lineBuffer;
  const char *prefix = "R,OK,D,";
  memcpy(out, prefix, strlen(prefix));
  out += strlen(prefix);
  out = appendInt(out, (int32_t)offset);
  *out++ = ',';
  out = appendInt(out, (int32_t)written);
  *out++ = '\n';
  OT_OUT.write((const uint8_t *)lineBuffer, (size_t)(out - lineBuffer));
}

// Validate before erasing: a rejected config leaves the working one in place.
static void handleCommit() {
  if (!provisioning) {
    replyError("NOSESSION");
    return;
  }
  if (stagingReceived != stagingLength) {
    replyError("SHORT");
    return;
  }

  OtDeviceConfig candidate;
  if (!otCfgParse(stagingBuffer, stagingLength, &candidate)) {
    replyError("INVALID");
    return;
  }
  if (candidate.totalBytes != stagingLength) {
    replyError("LEN");
    return;
  }
  if (!buildScanPlan(&candidate, NULL)) {
    replyError("PLAN");
    return;
  }
  if (!otCfgFlashWrite(stagingBuffer, stagingLength)) {
    // The erase may already have run, so the config the scan is using may be
    // gone. Reload rather than keep scanning from a pointer into erased flash.
    endStaging();
    loadDeviceConfig();
    loadXampling();
  loadXampling();
    statusDue = true;
    replyFlashError();
    return;
  }

  endStaging();
  loadDeviceConfig();
  loadXampling();
  if (!deviceConfig.valid) {
    replyError("READBACK");
    return;
  }
  configChanged();
  printConfigReply();
  printFrameHeader();
}

static void handleErase() {
  const bool ok = otCfgFlashErase();
  if (provisioning) endStaging();
  loadDeviceConfig();
  loadXampling();
  if (!ok) {
    replyFlashError();
    return;
  }
  configChanged();
  printLine("R,OK,X");
  printFrameHeader();
}

// R,PARAM lines are the parameter table itself, range included, so the host
// discovers the surface from the board rather than carrying a copy of it.
static void printParam(uint8_t slot) {
  const OtParamSpec *spec = &OT_PARAMS[slot];
  char *cursor = lineBuffer;
  const char *prefix = "R,PARAM,";
  memcpy(cursor, prefix, strlen(prefix));
  cursor += strlen(prefix);
  cursor = appendInt(cursor, (int32_t)spec->id);
  *cursor++ = ',';
  memcpy(cursor, spec->name, strlen(spec->name));
  cursor += strlen(spec->name);
  const uint16_t rest[3] = {spec->lo, spec->hi, paramValue[slot]};
  for (uint8_t i = 0; i < 3; ++i) {
    *cursor++ = ',';
    cursor = appendInt(cursor, (int32_t)rest[i]);
  }
  *cursor++ = '\n';
  OT_OUT.write((const uint8_t *)lineBuffer, (size_t)(cursor - lineBuffer));
}

// Skips spaces and commas so `p 2 0`, `p2,0` and `p 2,0` all work.
static const char *skipSeparators(const char *cursor) {
  while (*cursor == ' ' || *cursor == ',' || *cursor == '\t') cursor++;
  return cursor;
}

static void handleSetParam(const char *args) {
  uint32_t id = 0, requested = 0;
  const char *cursor = skipSeparators(args);
  const char *after = parseUint(cursor, &id);
  if (after == cursor) { replyError("SYNTAX"); return; }
  cursor = skipSeparators(after);
  after = parseUint(cursor, &requested);
  if (after == cursor) { replyError("SYNTAX"); return; }

  const uint8_t slot = otParamSlot((uint8_t)id);
  if (slot == 0xFFu) { replyError("NOPARAM"); return; }

  const uint16_t value = otParamCoerce(slot, requested);

  // A reserved mode is refused outright rather than silently falling back to
  // raster: a host must never be able to believe it is reading xampled data
  // while the board is producing something else.
  if (OT_PARAMS[slot].id == OT_P_TAC_MODE) {
    if (((OT_TAC_MODES_SUPPORTED >> value) & 1u) == 0u) {
      replyError("NOTIMPL");
      return;
    }
    // Implemented is not the same as usable: a shear mode on a pressure-only
    // skin would produce numbers from a grouping that does not exist.
    if (((OT_TAC_MODES_NEED_SHEAR >> value) & 1u) != 0u &&
        !(deviceConfig.valid && otCfgHasShear(&deviceConfig))) {
      replyError("NOSHEAR");
      return;
    }
    // Xampling thresholds against a baseline, so it has to exist first: a
    // stream that starts before `Z` would grow over every settling offset.
    if (otTacModeIsXampling(value)) {
      if (!deviceConfig.valid) { replyError("NOCFG"); return; }
      if (!xampReady) { replyError("GROUPS"); return; }
      if (!baselineValid) { replyError("NOBASE"); return; }
    }
  }
  if (OT_PARAMS[slot].id == OT_P_LED_MODE &&
      ((OT_LED_MODES_SUPPORTED >> value) & 1u) == 0u) {
    replyError("NOTIMPL");
    return;
  }

  const bool changed = (paramValue[slot] != value);
#if defined(OT_BOARD_READOUT60)
  const uint16_t previous = paramValue[slot];
#endif
  paramValue[slot] = value;
  if (changed) {
    switch (OT_PARAMS[slot].id) {
#if defined(OT_BOARD_READOUT60)
      case OT_P_GAIN_SW:
        // A switch that did not take keeps the old value, so the table never
        // claims a gain the hardware is not at.
        if (!gainApply()) {
          paramValue[slot] = previous;
          gainApply();
          replyError("GAIN");
          return;
        }
        // The baseline was measured at the old gain.
        baselineValid = false;
        break;
#endif
      case OT_P_VIB_ENABLE:
      case OT_P_ORIENT_ENABLE:
      case OT_P_VIB_ODR_HZ:
      case OT_P_ORIENT_ODR_HZ:
      case OT_P_ACC_FS_G:
      case OT_P_GYRO_FS_DPS:
        imuApplyParams();
        break;
      case OT_P_ROW_SETTLE_US:
      case OT_P_MUX_SETTLE_US:
      case OT_P_XAMP_CONFIRM_US:
        // The baseline measured the settling offset at the old timing.
        baselineValid = false;
        break;
      default:
        break;
    }
    configChanged();
  }
  // Always report the stored value, so a clamp or a snap is visible rather
  // than something the host has to infer.
  printParam(slot);
}

static void handleGetParam(const char *args) {
  const char *cursor = skipSeparators(args);
  if (*cursor == '\0') {
    for (uint8_t i = 0; i < OT_PARAM_COUNT; ++i) printParam(i);
    printLine("R,END");
    return;
  }
  uint32_t id = 0;
  parseUint(cursor, &id);
  const uint8_t slot = otParamSlot((uint8_t)id);
  if (slot == 0xFFu) { replyError("NOPARAM"); return; }
  printParam(slot);
}

// One shortcut per thing a user changes constantly, expanded into the same
// parameter writes so there is only one code path to be right. A shortcut can
// therefore emit two R,PARAM replies, so it closes with R,END the way `?` and
// bare `P` do -- a command whose reply count varies must say where it stops, or
// a host reading one reply per command silently falls a reply behind.
static void handleShortcut(uint8_t id, const char *args, uint8_t secondId) {
  const char *cursor = skipSeparators(args);
  uint32_t value = 0;
  const char *after = parseUint(cursor, &value);
  if (after == cursor) { replyError("SYNTAX"); return; }

  char expanded[24];
  char *out = expanded;
  out = appendInt(out, (int32_t)id);
  *out++ = ' ';
  out = appendInt(out, (int32_t)value);
  *out = '\0';
  handleSetParam(expanded);

  cursor = skipSeparators(after);
  if (secondId != 0xFFu && *cursor != '\0') {
    uint32_t second = 0;
    parseUint(cursor, &second);
    out = expanded;
    out = appendInt(out, (int32_t)secondId);
    *out++ = ' ';
    out = appendInt(out, (int32_t)second);
    *out = '\0';
    handleSetParam(expanded);
  }
  printLine("R,END");
}

static void handleColor(const char *args) {
  const char *cursor = args;
  for (uint8_t i = 0; i < 3; ++i) {
    cursor = skipSeparators(cursor);
    uint32_t value = 0;
    const char *after = parseUint(cursor, &value);
    if (after == cursor) { replyError("SYNTAX"); return; }
    manualRgb[i] = (uint8_t)(value > 255u ? 255u : value);
    cursor = after;
  }
  lastRgbMs = 0;   // show it now rather than at the next tick
  printLine("R,OK,C");
}

// Capabilities first, then the table. A host that can read this can drive the
// board without knowing anything about this firmware version in advance.
static void handleCaps() {
  char *cursor = lineBuffer;
  const char *prefix = "R,CAPS,";
  memcpy(cursor, prefix, strlen(prefix));
  cursor += strlen(prefix);
  cursor = appendInt(cursor, OT_API_VERSION);
  *cursor++ = ',';
  cursor = appendInt(cursor, OTCFG_SR_STEPS);
  *cursor++ = ',';
  cursor = appendInt(cursor, OTCFG_MUX_CHANNELS);
  *cursor++ = ',';
  cursor = appendInt(cursor, OTCFG_SR_STEPS * OTCFG_MUX_CHANNELS);
  *cursor++ = ',';
  cursor = appendHex32(cursor, OT_TAC_MODES_SUPPORTED);
  *cursor++ = ',';
  cursor = appendHex32(cursor, OT_LED_MODES_SUPPORTED);
  *cursor++ = ',';
  cursor = appendInt(cursor, OT_PARAM_COUNT);
  *cursor++ = ',';
  cursor = appendHex32(cursor, OT_TAC_MODES_NEED_SHEAR);
  *cursor++ = ',';
  // Which board this is, because the host's pad-to-line map is per board.
  memcpy(cursor, OT_BOARD_NAME, strlen(OT_BOARD_NAME));
  cursor += strlen(OT_BOARD_NAME);
  *cursor++ = '\n';
  OT_OUT.write((const uint8_t *)lineBuffer, (size_t)(cursor - lineBuffer));

  for (uint8_t i = 0; i < OT_PARAM_COUNT; ++i) printParam(i);
  printLine("R,END");
}

static void handleStats() {
  char *cursor = lineBuffer;
  const char *prefix = "R,STAT,";
  memcpy(cursor, prefix, strlen(prefix));
  cursor += strlen(prefix);
  const uint32_t fields[5] = {statFrames, vibrationDropped, statScanUsMax,
                              statLoopUsMax, configGeneration};
  for (uint8_t i = 0; i < 5; ++i) {
    if (i) *cursor++ = ',';
    cursor = appendInt(cursor, (int32_t)fields[i]);
  }
  *cursor++ = '\n';
  OT_OUT.write((const uint8_t *)lineBuffer, (size_t)(cursor - lineBuffer));
}

// R,OK,Z,<frames>,<baseline mean>,<baseline max>,<noise median>,<noise max>,
// <noisy taxels>: enough to see what the thresholds will be without dumping
// them. "Noisy" is a hands-off peak of 100 counts or more -- on the 32x32 those
// are whole logical columns, which is worth knowing before trusting a trigger.
// Blocks for XAMP_ZERO_FRAMES raster scans: 1.6 s at the default on a full
// 32x32 at MUX_SETTLE_US 8 with the IMU on.
static void handleZero() {
  if (!deviceConfig.valid) {
    replyError("NOCFG");
    return;
  }
  const uint16_t active = deviceConfig.activeCount;
  const uint16_t frames = param(OT_P_XAMP_ZERO_FRAMES);
  memset(xampBaseline, 0, sizeof(xampBaseline[0]) * active);
  memset(xampNoise, 0, sizeof(xampNoise[0]) * active);
  // xampNoise holds each taxel's highest raw reading until the loop ends; the
  // baseline is summed over the first 16 frames and divided once they are in.
  // With XAMP_CONFIRM_US on, measure at the settle a confirmed reading gets:
  // a trigger is judged on its confirmed value, so that is the noise that
  // matters. At the fast settle, settling artifacts that confirmation removes
  // would otherwise count as noise and desensitise those taxels.
  const uint16_t confirmUs = param(OT_P_XAMP_CONFIRM_US);
  for (uint16_t f = 0; f < frames; ++f) {
    scanTactileFrame(confirmUs);
    for (uint16_t k = 0; k < active; ++k) {
      if (f < BASELINE_FRAMES) xampBaseline[k] = (uint16_t)(xampBaseline[k] + tactileFrame[k]);
      if (tactileFrame[k] > xampNoise[k]) xampNoise[k] = tactileFrame[k];
    }
    if (f + 1u == BASELINE_FRAMES) {
      for (uint16_t k = 0; k < active; ++k) xampBaseline[k] /= BASELINE_FRAMES;
    }
    flushVibration();
  }
  uint32_t sum = 0;
  uint16_t baseMax = 0, noiseMax = 0, noisy = 0;
  uint16_t histogram[8] = {0};    // noise in 16-count bins, for the median
  for (uint16_t k = 0; k < active; ++k) {
    xampNoise[k] = (xampNoise[k] > xampBaseline[k]) ? (uint16_t)(xampNoise[k] - xampBaseline[k]) : 0u;
    sum += xampBaseline[k];
    if (xampBaseline[k] > baseMax) baseMax = xampBaseline[k];
    if (xampNoise[k] > noiseMax) noiseMax = xampNoise[k];
    if (xampNoise[k] >= 100u) ++noisy;
    ++histogram[xampNoise[k] >= 112u ? 7u : xampNoise[k] / 16u];
  }
  uint16_t medianBin = 0;
  for (uint16_t below = histogram[0]; below * 2u < active && medianBin < 7u; ) {
    below = (uint16_t)(below + histogram[++medianBin]);
  }
  xampBaselineMax = baseMax;
  baselineValid = true;
  xampReset(&xampState);
  configChanged();

  char *cursor = lineBuffer;
  const char *prefix = "R,OK,Z,";
  memcpy(cursor, prefix, strlen(prefix));
  cursor += strlen(prefix);
  // Median as the upper edge of its 16-count bin: "half the taxels are below".
  const uint32_t fields[6] = {frames, sum / active, baseMax,
                              (uint32_t)(medianBin + 1u) * 16u, noiseMax, noisy};
  for (uint8_t i = 0; i < 6; ++i) {
    if (i) *cursor++ = ',';
    cursor = appendInt(cursor, (int32_t)fields[i]);
  }
  *cursor++ = '\n';
  OT_OUT.write((const uint8_t *)lineBuffer, (size_t)(cursor - lineBuffer));
}

static void handleCommand(const char *line) {
  switch (line[0]) {
    case '?': handleCaps(); break;
    case 'B':
    case 'b':   // either case: on a board with a dead BOOT0 this is the only
                // way into DFU, so it must not depend on the shift key
      printLine("R,OK,B");
      OT_OUT.flush();
      delay(50);             // let the reply leave before the USB stack goes
      enterDfu();
      break;
    case 'G': printConfigReply(); break;
    case 'X': handleErase(); break;
    case 'S': handleBegin(line + 1); break;
    case 'D': handleData(line + 1); break;
    case 'E': handleCommit(); break;
    case 'A':
      endStaging();
      printLine("R,OK,A");
      break;
    case 'p': handleSetParam(line + 1); break;
    case 'P': handleGetParam(line + 1); break;
    case 'm': handleShortcut(OT_P_TAC_MODE, line + 1, 0xFFu); break;
    case 'l': handleShortcut(OT_P_LED_MODE, line + 1, OT_P_LED_BRIGHT); break;
    case 'c': handleColor(line + 1); break;
    case 't': handleStats(); break;
    case 'z':
      statScanUsMax = statLoopUsMax = statFrames = 0;
      vibrationDropped = 0;
      printLine("R,OK,Z");
      break;
    case 'Z': handleZero(); break;
    default: replyError("CMD"); break;
  }
}

// Non-blocking: one call reads whatever the CDC has and dispatches complete
// lines. Bounded per call so a flood cannot starve the scan.
static void pollCommands() {
  for (uint16_t budget = 0; budget < 512u && OT_IN_AVAILABLE(); ++budget) {
    const int raw = OT_IN_READ();
    if (raw < 0) break;
    const char c = (char)raw;

    if (c == '\n' || c == '\r') {
      if (commandLength == 0 && !commandOverflow) continue;
      if (commandOverflow) {
        replyError("LONG");
      } else {
        commandLine[commandLength] = '\0';
        handleCommand(commandLine);
      }
      commandLength = 0;
      commandOverflow = false;
      continue;
    }

    if (commandLength + 1u >= sizeof(commandLine)) {
      commandOverflow = true;
      continue;
    }
    commandLine[commandLength++] = c;
  }
}

// ----- Arduino entry points -------------------------------------------------

void setup() {
  checkDfuRequest();     // must run before anything is initialised
  Serial.begin(115200);  // baud is ignored on USB CDC

  // Session config starts from the table's defaults every boot; only the
  // sensor DeviceConfig survives a power cycle.
  for (uint8_t i = 0; i < OT_PARAM_COUNT; ++i) paramValue[i] = OT_PARAMS[i].def;

  tactileBegin();
  rgbBegin();

#if defined(OT_BUS_NODE)
  busSetup();
  WireImu.begin();
  WireImu.setClock(400000);
#else
  Wire.setSDA(IMU_SDA);
  Wire.setSCL(IMU_SCL);
  Wire.begin();
  Wire.setClock(400000);
#endif

  imuReady = imuBegin();
#if defined(OT_BOARD_READOUT60)
  const bool gainOk = gainApply();
#endif

  loadDeviceConfig();
  loadXampling();

  OT_OUT.println(F("# OpenTac single-board API, board " OT_BOARD_NAME " -- `?` for capabilities"));
  OT_OUT.println(F("# V,seq,accel_mg[3]"));
  OT_OUT.println(F("# M,gen,mode,tac_en,vib_en,ori_en,active,shear"));
#if defined(OT_BOARD_READOUT60)
  if (!gainOk) printLine("# gain switch did not answer at I2C 0x4C; GAIN_SW has no effect");
#endif
  printFrameHeader();
  printConfigReply();
  printStatus();
  lastFrameUs = micros();
  lastNoConfigMs = millis();
#if defined(OT_BUS_NODE)
  // Last, so the controller only finds the node once there is a config, an IMU
  // state and a boot banner for it to read.
  busBegin(0);
#endif
}

void loop() {
  const uint32_t loopStartUs = micros();
#if defined(OT_BUS_NODE)
  busService();
#endif
  pollCommands();

  // A flash erase stalls the CPU and the USB stack with it, so nothing else
  // runs while a config is being staged or written.
  if (provisioning) {
    updateRgb();
    return;
  }

  // The FIFO is drained unconditionally: it carries the rotation vector as
  // well as the accelerometer batch, and it overflows if left alone. The
  // per-modality gating happens inside, at the source.
  drainFifo();
  flushVibration();

  const bool xampling = otTacModeIsXampling(param(OT_P_TAC_MODE));
  const bool scanning = deviceConfig.valid && param(OT_P_TAC_ENABLE) &&
                        (!xampling || (xampReady && baselineValid));
  if (scanning) {
    const uint32_t scanStartUs = micros();
    if (xampling) {
      scanXampleFrame();
#if defined(OT_BOARD_READOUT60)
      hdrRereadSampled();
#endif
    } else {
      scanTactileFrame();
#if defined(OT_BOARD_READOUT60)
      hdrRereadPasses();
#endif
    }
    const uint32_t scanUs = micros() - scanStartUs;
    if (scanUs > statScanUsMax) statScanUsMax = scanUs;

    if (imuReady && param(OT_P_ORIENT_ENABLE)) imu.Get_G_Axes(latestGyro);

    drainFifo();
    flushVibration();

    // FRAME_MAX_HZ caps the frame rate and hands USB bandwidth back to the V
    // records; 0 means free-running.
    const uint16_t maxHz = param(OT_P_FRAME_MAX_HZ);
    const uint32_t minPeriodUs = maxHz ? (1000000u / maxHz) : 0u;
    const uint32_t now = micros();
    const uint32_t deltaUs = now - lastFrameUs;
    if (deltaUs >= minPeriodUs) {
      lastFrameUs = now;
#if defined(OT_BUS_NODE)
      busPublish(deltaUs, (uint8_t)(OTB_FF_TACTILE | (xampling ? OTB_FF_SPARSE : 0)));
#else
      if (xampling) printXampleFrame(deltaUs);
      else printFrame(deltaUs);
#endif
      ++frameCounter;
      ++statFrames;
    }
  } else if (!deviceConfig.valid) {
    // Communicative, but silent about tactile data: a frame scanned from a
    // guessed geometry would look fine and mean nothing.
    const uint32_t now = millis();
    if ((uint32_t)(now - lastNoConfigMs) >= NO_CONFIG_NOTICE_MS) {
      lastNoConfigMs = now;
      printLine("# no device config -- tactile acquisition disabled");
    }
  } else if (xampling && param(OT_P_TAC_ENABLE)) {
    // The mode is still selected but its baseline went stale (a settle change,
    // or a new config). Say so rather than stream against the wrong zero.
    const uint32_t now = millis();
    if ((uint32_t)(now - lastNoBaselineMs) >= NO_CONFIG_NOTICE_MS) {
      lastNoBaselineMs = now;
      printLine("# xampling paused: no baseline -- send Z with nothing touching");
    }
  }
#if defined(OT_BUS_NODE)
  if (!scanning) busPublishImuOnly();
#endif

  updateRgb();

#if defined(OT_BOARD_READOUT60)
  // At most once a second, so a switch that keeps failing cannot flood the link.
  static uint32_t hdrFailuresSaid = 0, hdrSaidMs = 0;
  if (hdrFailures != hdrFailuresSaid && (uint32_t)(millis() - hdrSaidMs) >= 1000u) {
    hdrFailuresSaid = hdrFailures;
    hdrSaidMs = millis();
    printLine("# HDR: a gain switch did not take; that frame kept the readings it had");
  }
#endif

  if (statusDue || (uint32_t)(millis() - lastStatusMs) >= STATUS_PERIOD_MS) {
    printStatus();
  }

  const uint32_t loopUs = micros() - loopStartUs;
  if (loopUs > statLoopUsMax) statLoopUsMax = loopUs;
}
