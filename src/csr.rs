//! Control/status register offsets and bit fields for the Beagle chip (the
//! Edge TPU on the Coral Dev Board Micro).
//!
//! Only the registers the driver touches are ported. Fields are `(lsb, width)`
//! pairs, applied with [`get_field`] and [`set_field`].
//!
//! A field write is a read-modify-write of the whole 64-bit word, so a wide
//! field's mask can wipe bits the caller did not mean to touch. See
//! [`IDLE_REGISTER_RUN`].

/// Extracts `bits` bits starting at `lsb`, right-aligned.
///
/// An `lsb` of 64 or more yields `0`; a `bits` of 64 or more takes the whole
/// word.
#[must_use]
pub const fn get_field(raw: u64, lsb: u32, bits: u32) -> u64 {
    if lsb >= 64 {
        return 0;
    }
    (raw >> lsb) & mask(bits)
}

/// Replaces `bits` bits starting at `lsb` with `value`, preserving the rest.
///
/// An `lsb` of 64 or more returns `raw` unchanged.
#[must_use]
pub const fn set_field(raw: u64, lsb: u32, bits: u32, value: u64) -> u64 {
    if lsb >= 64 {
        return raw;
    }
    let mask = mask(bits);
    (raw & !(mask << lsb)) | ((value & mask) << lsb)
}

const fn mask(bits: u32) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

// ---------------------------------------------------------------------------
// Apex CSRs
// ---------------------------------------------------------------------------

/// `omc0_00`: chip ID and scratch test register.
pub const OMC0_00: u64 = 0x1_a000;
/// `omc0_d0`: temperature-sensor clock and trim.
pub const OMC0_D0: u64 = 0x1_a0d0;
/// `omc0_d8`: temperature-sensor analog enables.
pub const OMC0_D8: u64 = 0x1_a0d8;
/// `omc0_dc`: temperature-sensor enable and reading.
pub const OMC0_DC: u64 = 0x1_a0dc;

/// `omc0_00.chip_id`, bits `[14:0]`.
pub const OMC0_00_CHIP_ID: (u32, u32) = (0, 15);
/// `omc0_00.test_reg0`, bits `[38:16]`.
///
/// The field is declared 23 bits wide and overlaps `test_reg1` at bit 31;
/// that is how the vendor header is written, and it is harmless because the
/// register is only ever accessed 32 bits at a time.
pub const OMC0_00_TEST_REG0: (u32, u32) = (16, 23);

/// The chip ID a Beagle Edge TPU reports.
pub const CHIP_ID: u64 = 0x89A;
/// The value `Initialize` writes to `test_reg0` and reads back as a bus
/// self-test.
pub const TEST_REG0_PATTERN: u64 = 0xAA;

/// `omc0_d0.clk_div`, bits `[2:0]`.
pub const OMC0_D0_CLK_DIV: (u32, u32) = (0, 2);
/// `omc0_d0.clk_en`, bit 7.
pub const OMC0_D0_CLK_EN: (u32, u32) = (7, 1);
/// `omc0_d0.adr`, bits `[12:8]`.
pub const OMC0_D0_ADR: (u32, u32) = (8, 5);
/// `omc0_d0.tref`, bits `[19:16]`.
pub const OMC0_D0_TREF: (u32, u32) = (16, 4);
/// `omc0_d0.tslope`, bits `[23:20]`.
pub const OMC0_D0_TSLOPE: (u32, u32) = (20, 4);
/// `omc0_d0.t_setting`, bits `[27:24]`.
pub const OMC0_D0_T_SETTING: (u32, u32) = (24, 4);

/// `omc0_d8.enbg`, bit 0.
pub const OMC0_D8_ENBG: (u32, u32) = (0, 1);
/// `omc0_d8.envr`, bit 1.
pub const OMC0_D8_ENVR: (u32, u32) = (1, 1);
/// `omc0_d8.enad`, bit 2.
pub const OMC0_D8_ENAD: (u32, u32) = (2, 1);

/// `omc0_dc.enthmc`, bit 0.
pub const OMC0_DC_ENTHMC: (u32, u32) = (0, 1);
/// `omc0_dc.data`, bits `[25:16]`, the raw temperature reading.
pub const OMC0_DC_DATA: (u32, u32) = (16, 10);

// ---------------------------------------------------------------------------
// SCU CSRs
// ---------------------------------------------------------------------------

/// `scu_ctrl_0`: PHY inactivity modes.
pub const SCU_CTRL_0: u64 = 0x1_a30c;
/// `scu_ctrl_2`: block resets and clock gating.
pub const SCU_CTRL_2: u64 = 0x1_a314;
/// `scu_ctrl_3`: power state and clock rates.
pub const SCU_CTRL_3: u64 = 0x1_a318;

/// `scu_ctrl_0.rg_pcie_inact_phy_mode`, bits `[10:8]`.
pub const SCU_CTRL_0_PCIE_INACT_PHY_MODE: (u32, u32) = (8, 3);
/// `scu_ctrl_0.rg_usb_inact_phy_mode`, bits `[13:11]`.
pub const SCU_CTRL_0_USB_INACT_PHY_MODE: (u32, u32) = (11, 3);

/// `scu_ctrl_2.rg_gated_gcb`, bits `[19:18]`.
pub const SCU_CTRL_2_GATED_GCB: (u32, u32) = (18, 2);

/// `scu_ctrl_3.cur_pwr_state`, bits `[9:8]`, read-only.
pub const SCU_CTRL_3_CUR_PWR_STATE: (u32, u32) = (8, 2);
/// `scu_ctrl_3.rg_force_sleep`, bits `[23:22]`.
pub const SCU_CTRL_3_FORCE_SLEEP: (u32, u32) = (22, 2);
/// `scu_ctrl_3.rg_gcb_clkdiv`, bits `[29:28]`.
pub const SCU_CTRL_3_GCB_CLKDIV: (u32, u32) = (28, 2);
/// `scu_ctrl_3.rg_axi_clk_125m`, bit 30.
pub const SCU_CTRL_3_AXI_CLK_125M: (u32, u32) = (30, 1);
/// `scu_ctrl_3.rg_8051_clk_250m`, bit 31.
pub const SCU_CTRL_3_8051_CLK_250M: (u32, u32) = (31, 1);

/// `rg_force_sleep` value that holds the GCB in reset.
pub const FORCE_SLEEP_RESET: u64 = 0x3;
/// `rg_force_sleep` value that releases the GCB from reset.
pub const FORCE_SLEEP_RUN: u64 = 0x2;
/// `cur_pwr_state` observed once the chip has entered reset.
pub const PWR_STATE_SLEEP: u64 = 0x2;
/// `cur_pwr_state` observed once the chip is fully awake.
pub const PWR_STATE_ACTIVE: u64 = 0x0;

// ---------------------------------------------------------------------------
// CB bridge CSRs
// ---------------------------------------------------------------------------

/// `gcbb_credit0`: bridge credit counter, pulsed while entering reset.
pub const GCBB_CREDIT0: u64 = 0x1_907c;

// ---------------------------------------------------------------------------
// Misc CSRs
// ---------------------------------------------------------------------------

/// `idleRegister`: idle-detection counter.
pub const IDLE_REGISTER: u64 = 0x4_a000;

/// The value bring-up writes to [`IDLE_REGISTER`].
///
/// `counter = 1` through a 31-bit field mask, which clears the register's
/// `0x9000` reset value rather than merging with it. `0x1` reaches the wire.
pub const IDLE_REGISTER_RUN: u64 = 0x1;

// ---------------------------------------------------------------------------
// Scalar core CSRs
// ---------------------------------------------------------------------------

/// `scalarCoreRunControl`.
pub const SCALAR_CORE_RUN_CONTROL: u64 = 0x4_4018;
/// `avDataPopRunControl`.
pub const AV_DATA_POP_RUN_CONTROL: u64 = 0x4_4158;
/// `parameterPopRunControl`.
pub const PARAMETER_POP_RUN_CONTROL: u64 = 0x4_4198;
/// `infeedRunControl`.
pub const INFEED_RUN_CONTROL: u64 = 0x4_41d8;
/// `outfeedRunControl`.
pub const OUTFEED_RUN_CONTROL: u64 = 0x4_4218;

// ---------------------------------------------------------------------------
// Tile config CSRs
// ---------------------------------------------------------------------------

/// `tileconfig0`: selects which tile subsequent tile-CSR writes reach.
pub const TILECONFIG0: u64 = 0x4_8788;

/// `tileconfig0` value that broadcasts to every tile.
///
/// All 7 bits of the `tile` field.
pub const TILECONFIG_BROADCAST: u64 = 0x7F;

// ---------------------------------------------------------------------------
// Tile CSRs
// ---------------------------------------------------------------------------

/// `deepSleep`: tile memory sleep/wake delays.
pub const DEEP_SLEEP: u64 = 0x4_0020;
/// `opRunControl`.
pub const OP_RUN_CONTROL: u64 = 0x4_00c0;
/// `wideToNarrowRunControl`.
pub const WIDE_TO_NARROW_RUN_CONTROL: u64 = 0x4_0110;
/// `narrowToWideRunControl`.
pub const NARROW_TO_WIDE_RUN_CONTROL: u64 = 0x4_0150;
/// `ringBusConsumer0RunControl`.
pub const RING_BUS_CONSUMER0_RUN_CONTROL: u64 = 0x4_0190;
/// `ringBusConsumer1RunControl`.
pub const RING_BUS_CONSUMER1_RUN_CONTROL: u64 = 0x4_01d0;
/// `ringBusProducerRunControl`.
pub const RING_BUS_PRODUCER_RUN_CONTROL: u64 = 0x4_0210;
/// `meshBus0RunControl`.
pub const MESH_BUS0_RUN_CONTROL: u64 = 0x4_0250;
/// `meshBus1RunControl`.
pub const MESH_BUS1_RUN_CONTROL: u64 = 0x4_0298;
/// `meshBus2RunControl`.
pub const MESH_BUS2_RUN_CONTROL: u64 = 0x4_02e0;
/// `meshBus3RunControl`.
pub const MESH_BUS3_RUN_CONTROL: u64 = 0x4_0328;

/// `deepSleep.to_sleep_delay`, bits `[7:0]`.
pub const DEEP_SLEEP_TO_SLEEP_DELAY: (u32, u32) = (0, 8);
/// `deepSleep.to_wake_delay`, bits `[15:8]`.
pub const DEEP_SLEEP_TO_WAKE_DELAY: (u32, u32) = (8, 8);

/// The exact 64-bit value `Initialize` writes to [`DEEP_SLEEP`]:
/// `to_sleep_delay = 2`, `to_wake_delay = 30`.
pub const DEEP_SLEEP_INIT: u64 = 0x1E02;

// ---------------------------------------------------------------------------
// USB CSRs
// ---------------------------------------------------------------------------

/// `outfeed_chunk_length`.
pub const USB_OUTFEED_CHUNK_LENGTH: u64 = 0x4_c058;
/// `descr_ep`.
pub const USB_DESCR_EP: u64 = 0x4_c148;
/// `multi_bo_ep`.
pub const USB_MULTI_BO_EP: u64 = 0x4_c160;

/// Value written to [`USB_DESCR_EP`].
pub const USB_DESCR_EP_INIT: u64 = 0xF0;
/// Value written to [`USB_MULTI_BO_EP`].
///
/// Zero selects the "single endpoint" framing that the `apex_latest_single_ep`
/// firmware implements: every outbound stream shares bulk OUT endpoint 1 and is
/// delimited by the 8-byte descriptor header instead of getting an endpoint of
/// its own.
pub const USB_MULTI_BO_EP_INIT: u64 = 0x0;
/// Value written to [`USB_OUTFEED_CHUNK_LENGTH`].
pub const USB_OUTFEED_CHUNK_LENGTH_INIT: u64 = 0x20;

// ---------------------------------------------------------------------------
// Run control
// ---------------------------------------------------------------------------

/// Run-control state written to the scalar-core and tile run-control CSRs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
pub enum RunControl {
    /// Stop at the next instruction boundary.
    MoveToIdle = 0,
    /// Execute. The only state bring-up requests.
    MoveToRun = 1,
    /// Halt.
    MoveToHalt = 2,
    /// Single-step.
    MoveToSingleStep = 3,
}

/// The tile run-control CSRs that `DoRunControl` writes, in the order it writes
/// them.
///
/// Written after the broadcast `tileconfig0`. The per-tile-index variants and
/// `narrowToNarrowRunControl` do not exist on this chip.
pub const TILE_RUN_CONTROL_SEQUENCE: [u64; 10] = [
    OP_RUN_CONTROL,
    NARROW_TO_WIDE_RUN_CONTROL,
    WIDE_TO_NARROW_RUN_CONTROL,
    MESH_BUS0_RUN_CONTROL,
    MESH_BUS1_RUN_CONTROL,
    MESH_BUS2_RUN_CONTROL,
    MESH_BUS3_RUN_CONTROL,
    RING_BUS_CONSUMER0_RUN_CONTROL,
    RING_BUS_CONSUMER1_RUN_CONTROL,
    RING_BUS_PRODUCER_RUN_CONTROL,
];

/// The scalar-core run-control CSRs, in the order `DoRunControl` writes them.
pub const SCALAR_RUN_CONTROL_SEQUENCE: [u64; 5] = [
    SCALAR_CORE_RUN_CONTROL,
    AV_DATA_POP_RUN_CONTROL,
    PARAMETER_POP_RUN_CONTROL,
    INFEED_RUN_CONTROL,
    OUTFEED_RUN_CONTROL,
];
