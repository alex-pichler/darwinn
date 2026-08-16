//! The runtime driver: CSR access, bring-up, bulk stream framing and the
//! inference path.

use embedded_hal::delay::DelayNs;

use crate::csr;
use crate::executable::{Description, Executable, Hint, Layer};
use crate::{Error, Transport};

/// USB vendor ID of the Edge TPU after DFU.
pub const RUNTIME_VID: u16 = 0x18D1;
/// USB product ID of the Edge TPU after DFU.
pub const RUNTIME_PID: u16 = 0x9302;

/// Endpoint number carrying every host-to-device stream.
///
/// Instructions, parameters and input activations all go out here, each
/// preceded by an 8-byte header. This is the "single ep" the firmware blob is
/// named for.
pub const BULK_OUT_ENDPOINT: u8 = 1;

/// Endpoint number carrying output activations back.
///
/// The same number as [`BULK_OUT_ENDPOINT`]. Endpoint 1 is one bidirectional
/// channel: `0x01` OUT, `0x81` IN.
pub const BULK_IN_ENDPOINT: u8 = 1;

/// Endpoint number of the completion-event pipe.
///
/// A *bulk* IN endpoint, despite the name "event".
pub const EVENT_ENDPOINT: u8 = 2;

/// Endpoint number of the interrupt IN pipe.
///
/// Enumerated but never read: completion is signalled on [`EVENT_ENDPOINT`]
/// instead. Kept so [`Driver::poll_interrupt`] has a default, not because
/// anything is known to arrive here.
pub const INTERRUPT_ENDPOINT: u8 = 3;

/// Size of the framing header prefixed to every outbound stream.
pub const HEADER_SIZE: usize = 8;

/// Size of a completion event.
pub const EVENT_SIZE: usize = 16;

/// Largest bulk transfer handed to the transport at once.
///
/// The caller's slice is passed straight down; any bouncing is the transport's
/// business. Unrelated to `wMaxPacketSize` (512 out, 256 in), which the host
/// controller splits on.
pub const MAX_BULK_CHUNK: usize = 32 * 1024;

// USB request-type bits.
const DIR_OUT: u8 = 0x00;
const DIR_IN: u8 = 0x80;
const TYPE_VENDOR: u8 = 0x40;
const RECIPIENT_DEVICE: u8 = 0x00;

/// `bmRequestType` for a CSR read.
const CSR_READ_REQ_TYPE: u8 = TYPE_VENDOR | RECIPIENT_DEVICE | DIR_IN;
/// `bmRequestType` for a CSR write.
const CSR_WRITE_REQ_TYPE: u8 = TYPE_VENDOR | RECIPIENT_DEVICE | DIR_OUT;
/// `bRequest` selecting a 64-bit register.
const CSR_REQUEST_64: u8 = 0;
/// `bRequest` selecting a 32-bit register.
const CSR_REQUEST_32: u8 = 1;

/// Which stream a bulk-OUT payload belongs to; the low nibble of header byte 4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DescriptorTag {
    /// Instruction bitstream.
    Instructions = 0,
    /// Input activations.
    InputActivations = 1,
    /// Parameters.
    Parameters = 2,
    /// Output activations. Never sent: outputs come back on the IN direction
    /// with no header at all.
    OutputActivations = 3,
    /// Interrupt tag 0. Never sent.
    Interrupt0 = 4,
    /// Interrupt tag 1. Never sent.
    Interrupt1 = 5,
    /// Interrupt tag 2. Never sent.
    Interrupt2 = 6,
    /// Interrupt tag 3. Never sent.
    Interrupt3 = 7,
}

/// Clock rates selected during bring-up.
///
/// | mode | GCB | AXI | USB 8051 |
/// |---|---|---|---|
/// | [`PerformanceMode::Max`] | 500 MHz | 250 MHz | 500 MHz |
/// | [`PerformanceMode::High`] | 250 MHz | 125 MHz | 500 MHz |
/// | [`PerformanceMode::Medium`] | 125 MHz | 125 MHz | 500 MHz |
/// | [`PerformanceMode::Low`] | 63 MHz | 125 MHz | 250 MHz |
///
/// `High` is the vendor default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PerformanceMode {
    /// 63 MHz GCB, 125 MHz AXI, 250 MHz USB.
    Low,
    /// 125 MHz GCB, 125 MHz AXI, 500 MHz USB.
    Medium,
    /// 250 MHz GCB, 125 MHz AXI, 500 MHz USB. The vendor default.
    #[default]
    High,
    /// 500 MHz GCB, 250 MHz AXI, 500 MHz USB.
    Max,
}

impl PerformanceMode {
    /// `(rg_gcb_clkdiv, rg_axi_clk_125m, rg_8051_clk_250m)` for this mode.
    ///
    /// GCB divider 0/1/2/3 selects 500/250/125/63 MHz; `rg_axi_clk_125m` is 1
    /// for 125 MHz; `rg_8051_clk_250m` is 1 for 250 MHz.
    const fn clock_bits(self) -> (u64, u64, u64) {
        match self {
            PerformanceMode::Max => (0, 0, 0),
            PerformanceMode::High => (1, 1, 0),
            PerformanceMode::Medium => (2, 1, 0),
            PerformanceMode::Low => (3, 1, 1),
        }
    }
}

/// A completion event read off [`EVENT_ENDPOINT`].
///
/// The arrival of 16 bytes is the entire completion signal; the fields are
/// informational. Bytes 13-15 are unaccounted for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Event {
    /// Bytes 0-7: a device address.
    pub address: u64,
    /// Bytes 8-11: a length.
    pub length: u32,
    /// Byte 12, low nibble: a [`DescriptorTag`] value.
    pub tag: u8,
}

/// Timeouts and poll bounds. Every blocking operation in this crate derives its
/// limit from one of these fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timeouts {
    /// Timeout for one CSR control transfer, in microseconds.
    pub control_us: u32,
    /// Timeout for one bulk chunk, in microseconds.
    pub bulk_us: u32,
    /// Timeout for the completion-event read, in microseconds.
    pub event_us: u32,
    /// Maximum reads a bounded register poll may perform before failing with
    /// [`Error::PollTimeout`].
    ///
    /// The default of 1000 is generous against a control transfer costing on
    /// the order of a millisecond.
    pub poll_attempts: u32,
}

impl Timeouts {
    /// 200 ms per transfer, and a 1000-read poll bound.
    pub const DEFAULT: Timeouts = Timeouts {
        control_us: 200_000,
        bulk_us: 200_000,
        event_us: 200_000,
        poll_attempts: 1000,
    };
}

impl Default for Timeouts {
    fn default() -> Self {
        Timeouts::DEFAULT
    }
}

/// The Edge TPU runtime driver.
pub struct Driver<T: Transport> {
    transport: T,
    timeouts: Timeouts,
}

impl<T: Transport> Driver<T> {
    /// Wraps a transport that is already talking to a post-DFU Edge TPU
    /// ([`RUNTIME_VID`]`:`[`RUNTIME_PID`]).
    pub fn new(transport: T, timeouts: Timeouts) -> Self {
        Driver {
            transport,
            timeouts,
        }
    }

    /// Borrows the underlying transport.
    pub fn transport(&mut self) -> &mut T {
        &mut self.transport
    }

    /// Returns the underlying transport.
    pub fn release(self) -> T {
        self.transport
    }

    // -----------------------------------------------------------------------
    // CSR access
    // -----------------------------------------------------------------------

    /// Reads a 32-bit CSR.
    pub fn csr_read32(&mut self, reg: u64) -> Result<u32, Error<T::Error>> {
        let mut buf = [0u8; 4];
        self.csr_in(reg, CSR_REQUEST_32, &mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    /// Writes a 32-bit CSR.
    pub fn csr_write32(&mut self, reg: u64, value: u32) -> Result<(), Error<T::Error>> {
        self.csr_out(reg, CSR_REQUEST_32, &value.to_le_bytes())
    }

    /// Reads a 64-bit CSR.
    pub fn csr_read64(&mut self, reg: u64) -> Result<u64, Error<T::Error>> {
        let mut buf = [0u8; 8];
        self.csr_in(reg, CSR_REQUEST_64, &mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    /// Writes a 64-bit CSR.
    pub fn csr_write64(&mut self, reg: u64, value: u64) -> Result<(), Error<T::Error>> {
        self.csr_out(reg, CSR_REQUEST_64, &value.to_le_bytes())
    }

    /// The address split shared by every CSR transfer.
    ///
    /// Only bits 0-31 are transmitted: bits 0-15 become `wValue`, bits 16-31
    /// become `wIndex`. No offset used here exceeds `0x4c160`.
    const fn address_split(reg: u64) -> (u16, u16) {
        ((reg & 0xFFFF) as u16, ((reg >> 16) & 0xFFFF) as u16)
    }

    fn csr_in(&mut self, reg: u64, request: u8, buf: &mut [u8]) -> Result<(), Error<T::Error>> {
        let (value, index) = Self::address_split(reg);
        let want = buf.len();
        let n = self
            .transport
            .control_in(
                CSR_READ_REQ_TYPE,
                request,
                value,
                index,
                buf,
                self.timeouts.control_us,
            )
            .map_err(Error::Transport)?;
        if n != want {
            return Err(Error::ShortTransfer {
                expected: want,
                actual: n,
            });
        }
        Ok(())
    }

    fn csr_out(&mut self, reg: u64, request: u8, data: &[u8]) -> Result<(), Error<T::Error>> {
        let (value, index) = Self::address_split(reg);
        self.transport
            .control_out(
                CSR_WRITE_REQ_TYPE,
                request,
                value,
                index,
                data,
                self.timeouts.control_us,
            )
            .map_err(Error::Transport)
    }

    /// Read-modify-write of bit fields of a 32-bit CSR.
    fn modify32(&mut self, reg: u64, fields: &[((u32, u32), u64)]) -> Result<u32, Error<T::Error>> {
        let mut raw = u64::from(self.csr_read32(reg)?);
        for ((lsb, bits), value) in fields {
            raw = csr::set_field(raw, *lsb, *bits, *value);
        }
        let raw = raw as u32;
        self.csr_write32(reg, raw)?;
        Ok(raw)
    }

    /// Reads `reg` until `predicate` holds, at most
    /// [`Timeouts::poll_attempts`] times.
    fn poll32(
        &mut self,
        reg: u64,
        what: &'static str,
        predicate: impl Fn(u32) -> bool,
    ) -> Result<u32, Error<T::Error>> {
        for _ in 0..self.timeouts.poll_attempts {
            let v = self.csr_read32(reg)?;
            if predicate(v) {
                return Ok(v);
            }
        }
        Err(Error::PollTimeout(what))
    }

    /// Reads `reg` until `predicate` holds, at most
    /// [`Timeouts::poll_attempts`] times.
    fn poll64(
        &mut self,
        reg: u64,
        what: &'static str,
        predicate: impl Fn(u64) -> bool,
    ) -> Result<u64, Error<T::Error>> {
        for _ in 0..self.timeouts.poll_attempts {
            let v = self.csr_read64(reg)?;
            if predicate(v) {
                return Ok(v);
            }
        }
        Err(Error::PollTimeout(what))
    }

    // -----------------------------------------------------------------------
    // Bring-up
    // -----------------------------------------------------------------------

    /// Brings the chip out of reset and into the run state.
    ///
    /// In order:
    ///
    /// 1. Read `omc0_00`, require `chip_id == 0x89A`.
    /// 2. Write `test_reg0 = 0xAA` and read it back, a round-trip self-test of
    ///    the control path.
    /// 3. `scu_ctrl_0`: clear `rg_pcie_inact_phy_mode` and
    ///    `rg_usb_inact_phy_mode` (disable inactive PHY mode).
    /// 4. `scu_ctrl_2`: `rg_gated_gcb = 2` (clock gating off for bring-up).
    /// 5. If `scu_ctrl_3.rg_force_sleep != 3`, force sleep, poll
    ///    `cur_pwr_state == 2`, then pulse `gcbb_credit0` `0xF` then `0`.
    /// 6. `scu_ctrl_3`: `rg_force_sleep = 2` plus the clock triplet for `mode`;
    ///    poll `cur_pwr_state == 0`.
    /// 7. Poll `scalarCoreRunControl` (64-bit) until it reads `0`.
    /// 8. `idleRegister = 1`, `tileconfig0 = 0x7F` (poll for read-back),
    ///    `deepSleep = 0x1E02`.
    /// 9. `scu_ctrl_2`: `rg_gated_gcb = 1` (clock gating back on).
    /// 10. USB HIB: `descr_ep = 0xF0`, `multi_bo_ep = 0`,
    ///     `outfeed_chunk_length = 0x20`.
    /// 11. Temperature sensor: `omc0_d0`, `omc0_d8`, a 100 µs settle, then
    ///     `omc0_dc.enthmc = 1`.
    /// 12. [`Driver::run_control`] with [`csr::RunControl::MoveToRun`].
    ///
    /// Steps 3 and 4 each end in a read whose result is discarded. It is on
    /// the wire in the known-good sequence and may act as a write barrier, so
    /// it stays.
    ///
    /// `delay` supplies the one settling wait: 100 µs between enabling the
    /// temperature sensor's input ports and its measurement flow.
    pub fn init<D: DelayNs>(
        &mut self,
        mode: PerformanceMode,
        delay: &mut D,
    ) -> Result<(), Error<T::Error>> {
        // 1. Chip ID.
        let omc0_00 = u64::from(self.csr_read32(csr::OMC0_00)?);
        let chip_id = csr::get_field(omc0_00, csr::OMC0_00_CHIP_ID.0, csr::OMC0_00_CHIP_ID.1);
        if chip_id != csr::CHIP_ID {
            return Err(Error::ChipId { found: chip_id });
        }

        // 2. Scratch-register round trip.
        let patched = csr::set_field(
            omc0_00,
            csr::OMC0_00_TEST_REG0.0,
            csr::OMC0_00_TEST_REG0.1,
            csr::TEST_REG0_PATTERN,
        );
        self.csr_write32(csr::OMC0_00, patched as u32)?;
        let read_back = u64::from(self.csr_read32(csr::OMC0_00)?);
        let found = csr::get_field(
            read_back,
            csr::OMC0_00_TEST_REG0.0,
            csr::OMC0_00_TEST_REG0.1,
        );
        if found != csr::TEST_REG0_PATTERN {
            return Err(Error::SelfTest { found });
        }

        // 3. Disable inactive PHY mode.
        self.modify32(
            csr::SCU_CTRL_0,
            &[
                (csr::SCU_CTRL_0_PCIE_INACT_PHY_MODE, 0),
                (csr::SCU_CTRL_0_USB_INACT_PHY_MODE, 0),
            ],
        )?;
        let _ = self.csr_read32(csr::SCU_CTRL_0)?;

        // 4. Disable clock gating.
        self.modify32(csr::SCU_CTRL_2, &[(csr::SCU_CTRL_2_GATED_GCB, 0x2)])?;
        let _ = self.csr_read32(csr::SCU_CTRL_2)?;

        // 5. Enter reset, if not already there.
        let scu3 = u64::from(self.csr_read32(csr::SCU_CTRL_3)?);
        if csr::get_field(
            scu3,
            csr::SCU_CTRL_3_FORCE_SLEEP.0,
            csr::SCU_CTRL_3_FORCE_SLEEP.1,
        ) != csr::FORCE_SLEEP_RESET
        {
            let forced = csr::set_field(
                scu3,
                csr::SCU_CTRL_3_FORCE_SLEEP.0,
                csr::SCU_CTRL_3_FORCE_SLEEP.1,
                csr::FORCE_SLEEP_RESET,
            );
            self.csr_write32(csr::SCU_CTRL_3, forced as u32)?;
            self.poll32(csr::SCU_CTRL_3, "scu_ctrl_3.cur_pwr_state == sleep", |v| {
                csr::get_field(
                    u64::from(v),
                    csr::SCU_CTRL_3_CUR_PWR_STATE.0,
                    csr::SCU_CTRL_3_CUR_PWR_STATE.1,
                ) == csr::PWR_STATE_SLEEP
            })?;
            self.csr_write32(csr::GCBB_CREDIT0, 0xF)?;
            self.csr_write32(csr::GCBB_CREDIT0, 0x0)?;
        }

        // 6. Clock rates, then exit reset.
        let (gcb_div, axi_125m, usb_250m) = mode.clock_bits();
        self.modify32(
            csr::SCU_CTRL_3,
            &[
                (csr::SCU_CTRL_3_FORCE_SLEEP, csr::FORCE_SLEEP_RUN),
                (csr::SCU_CTRL_3_GCB_CLKDIV, gcb_div),
                (csr::SCU_CTRL_3_AXI_CLK_125M, axi_125m),
                (csr::SCU_CTRL_3_8051_CLK_250M, usb_250m),
            ],
        )?;
        self.poll32(csr::SCU_CTRL_3, "scu_ctrl_3.cur_pwr_state == active", |v| {
            csr::get_field(
                u64::from(v),
                csr::SCU_CTRL_3_CUR_PWR_STATE.0,
                csr::SCU_CTRL_3_CUR_PWR_STATE.1,
            ) == csr::PWR_STATE_ACTIVE
        })?;

        // 7. Confirm reset exit through a known register.
        self.poll64(
            csr::SCALAR_CORE_RUN_CONTROL,
            "scalarCoreRunControl == 0",
            |v| v == 0,
        )?;

        // 8. Idle, tile config, deep sleep.
        self.csr_write64(csr::IDLE_REGISTER, csr::IDLE_REGISTER_RUN)?;
        self.csr_write64(csr::TILECONFIG0, csr::TILECONFIG_BROADCAST)?;
        self.poll64(csr::TILECONFIG0, "tileconfig0 broadcast", |v| {
            v == csr::TILECONFIG_BROADCAST
        })?;
        self.csr_write64(csr::DEEP_SLEEP, csr::DEEP_SLEEP_INIT)?;

        // 9. Re-enable clock gating.
        self.modify32(csr::SCU_CTRL_2, &[(csr::SCU_CTRL_2_GATED_GCB, 1)])?;

        // 10. USB HIB framing configuration.
        self.csr_write64(csr::USB_DESCR_EP, csr::USB_DESCR_EP_INIT)?;
        self.csr_write64(csr::USB_MULTI_BO_EP, csr::USB_MULTI_BO_EP_INIT)?;
        self.csr_write64(
            csr::USB_OUTFEED_CHUNK_LENGTH,
            csr::USB_OUTFEED_CHUNK_LENGTH_INIT,
        )?;

        // 11. Temperature sensor.
        self.modify32(
            csr::OMC0_D0,
            &[
                (csr::OMC0_D0_CLK_EN, 0x1),
                (csr::OMC0_D0_ADR, 0xC),
                (csr::OMC0_D0_TREF, 0),
                (csr::OMC0_D0_TSLOPE, 0),
                (csr::OMC0_D0_T_SETTING, 0),
            ],
        )?;
        self.modify32(
            csr::OMC0_D8,
            &[
                (csr::OMC0_D8_ENBG, 0x1),
                (csr::OMC0_D8_ENVR, 0x1),
                (csr::OMC0_D8_ENAD, 0x1),
            ],
        )?;
        delay.delay_us(100);
        self.modify32(csr::OMC0_DC, &[(csr::OMC0_DC_ENTHMC, 0x1)])?;

        // 12. Start everything running.
        self.run_control(csr::RunControl::MoveToRun)
    }

    /// Writes `state` to every scalar-core and tile run-control register.
    ///
    /// The five scalar-core registers, then a re-broadcast of `tileconfig0`
    /// with a read-back poll, then the ten tile registers.
    ///
    /// The re-broadcast is not redundant: tile writes are routed by whatever
    /// `tileconfig0` selects, and the hardware does not order them against the
    /// previous write, so the poll is what makes them land.
    pub fn run_control(&mut self, state: csr::RunControl) -> Result<(), Error<T::Error>> {
        let value = state as u64;
        for reg in csr::SCALAR_RUN_CONTROL_SEQUENCE {
            self.csr_write64(reg, value)?;
        }
        self.csr_write64(csr::TILECONFIG0, csr::TILECONFIG_BROADCAST)?;
        self.poll64(
            csr::TILECONFIG0,
            "tileconfig0 broadcast (run control)",
            |v| v == csr::TILECONFIG_BROADCAST,
        )?;
        for reg in csr::TILE_RUN_CONTROL_SEQUENCE {
            self.csr_write64(reg, value)?;
        }
        Ok(())
    }

    /// Reads the on-die temperature in millidegrees Celsius.
    ///
    /// `(662 - omc0_dc.data) * 250 + 550`. Millidegrees so the crate needs no
    /// floating point; divide by 1000 for degrees.
    ///
    /// Only meaningful after [`Driver::init`] has enabled the sensor.
    pub fn temperature_millicelsius(&mut self) -> Result<i32, Error<T::Error>> {
        let raw = u64::from(self.csr_read32(csr::OMC0_DC)?);
        let data = csr::get_field(raw, csr::OMC0_DC_DATA.0, csr::OMC0_DC_DATA.1) as i32;
        Ok((662 - data) * 250 + 550)
    }

    // -----------------------------------------------------------------------
    // Bulk streams
    // -----------------------------------------------------------------------

    /// Builds the 8-byte framing header.
    ///
    /// Byte layout: `length` as a little-endian `u32` in bytes 0-3, the
    /// descriptor tag's low nibble in byte 4, zero padding in bytes 5-7.
    #[must_use]
    pub fn build_header(tag: DescriptorTag, length: u32) -> [u8; HEADER_SIZE] {
        let mut header = [0u8; HEADER_SIZE];
        header[..4].copy_from_slice(&length.to_le_bytes());
        header[4] = (tag as u8) & 0xF;
        header
    }

    /// Sends a header followed by `data` on [`BULK_OUT_ENDPOINT`].
    ///
    /// Header and payload are two bulk transfers back to back, not one.
    pub fn send_data(&mut self, tag: DescriptorTag, data: &[u8]) -> Result<(), Error<T::Error>> {
        let header = Self::build_header(tag, data.len() as u32);
        self.bulk_out(&header)?;
        self.bulk_out(data)
    }

    /// Sends an instruction bitstream.
    pub fn send_instructions(&mut self, data: &[u8]) -> Result<(), Error<T::Error>> {
        self.send_data(DescriptorTag::Instructions, data)
    }

    /// Sends input activations.
    pub fn send_inputs(&mut self, data: &[u8]) -> Result<(), Error<T::Error>> {
        self.send_data(DescriptorTag::InputActivations, data)
    }

    /// Sends parameters.
    pub fn send_parameters(&mut self, data: &[u8]) -> Result<(), Error<T::Error>> {
        self.send_data(DescriptorTag::Parameters, data)
    }

    /// Reads output activations.
    ///
    /// No header is sent or expected: the device knows how much to return, and
    /// in what order, purely from the DMA hints already streamed to it.
    pub fn get_outputs(&mut self, buf: &mut [u8]) -> Result<(), Error<T::Error>> {
        self.bulk_in(buf)
    }

    /// Writes `data` to [`BULK_OUT_ENDPOINT`] in [`MAX_BULK_CHUNK`] pieces.
    fn bulk_out(&mut self, data: &[u8]) -> Result<(), Error<T::Error>> {
        for chunk in data.chunks(MAX_BULK_CHUNK) {
            self.transport
                .bulk_out(BULK_OUT_ENDPOINT, chunk, self.timeouts.bulk_us)
                .map_err(Error::Transport)?;
        }
        Ok(())
    }

    /// Fills `buf` from [`BULK_IN_ENDPOINT`] in [`MAX_BULK_CHUNK`] pieces.
    ///
    /// Reads straight into the caller's slice, so a short read cannot leave
    /// stale bytes behind.
    fn bulk_in(&mut self, buf: &mut [u8]) -> Result<(), Error<T::Error>> {
        let total = buf.len();
        let mut done = 0usize;
        // Bounded by construction: a zero-length read is rejected below, so
        // `done` strictly increases and the loop runs at most `total` times.
        while done < total {
            let end = (done + MAX_BULK_CHUNK).min(total);
            let n = self
                .transport
                .bulk_in(BULK_IN_ENDPOINT, &mut buf[done..end], self.timeouts.bulk_us)
                .map_err(Error::Transport)?;
            if n == 0 || done + n > total {
                return Err(Error::ShortTransfer {
                    expected: end - done,
                    actual: n,
                });
            }
            done += n;
        }
        Ok(())
    }

    /// Waits for the completion event on [`EVENT_ENDPOINT`].
    ///
    /// One 16-byte bulk read on [`EVENT_ENDPOINT`], bounded by
    /// [`Timeouts::event_us`]. Called once per inference, after every outbound
    /// stream and before any output is read back: it is a completion barrier,
    /// and the [`Event`] fields are informational.
    ///
    /// Deliberately not an interrupt transfer; see [`Driver::poll_interrupt`].
    pub fn read_event(&mut self) -> Result<Event, Error<T::Error>> {
        let mut buf = [0u8; EVENT_SIZE];
        let n = self
            .transport
            .bulk_in(EVENT_ENDPOINT, &mut buf, self.timeouts.event_us)
            .map_err(Error::Transport)?;
        if n != EVENT_SIZE {
            return Err(Error::ShortTransfer {
                expected: EVENT_SIZE,
                actual: n,
            });
        }
        Ok(Event {
            address: u64::from_le_bytes([
                buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
            ]),
            length: u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]),
            tag: buf[12] & 0xF,
        })
    }

    /// Polls the interrupt IN endpoint once, returning the byte count if
    /// anything arrived.
    ///
    /// Nothing is known about what, if anything, the device sends here, or
    /// whether the pipe survives the USB bandwidth check at all. Provided for
    /// bring-up experiments; use [`Driver::read_event`] for completion.
    pub fn poll_interrupt(
        &mut self,
        buf: &mut [u8],
        timeout_us: u32,
    ) -> Result<Option<usize>, Error<T::Error>> {
        self.transport
            .interrupt_in(INTERRUPT_ENDPOINT, buf, timeout_us)
            .map_err(Error::Transport)
    }

    // -----------------------------------------------------------------------
    // Inference
    // -----------------------------------------------------------------------

    /// Runs one inference.
    ///
    /// Walks the executable's DMA hints in array order and, for each one:
    ///
    /// * `BASE_ADDRESS_PARAMETER`: sends `parameters[offset..offset + size]`.
    /// * `BASE_ADDRESS_INPUT_ACTIVATION`: if the named input layer has a
    ///   signed fixed-point type, flips the sign bit of every element of
    ///   `input` first, then sends `input[offset..offset + size]`.
    /// * `BASE_ADDRESS_OUTPUT_ACTIVATION`: reads `size` bytes into the staging
    ///   buffer of the named output layer.
    /// * an instruction hint: sends that instruction bitstream chunk.
    /// * `BASE_ADDRESS_SCRATCH`, interrupt and fence hints: skipped.
    ///
    /// Once the last hint has been handled it waits for the completion event.
    ///
    /// `outputs` is indexed by output-layer index. Each buffer receives the
    /// device's *padded, tile-scattered* bytes and must be at least
    /// [`crate::Layer::padded_size_bytes`] long; run
    /// [`crate::Layer::relayout_into`] and
    /// [`crate::Layer::transform_signed_data_type`] afterwards for the flat
    /// tensor.
    ///
    /// `input` is `&mut` because the sign-bit fixup is applied in place. It is
    /// applied once per matching hint, so an executable that split one input
    /// layer across several hints would transform it more than once.
    ///
    /// A parameter-caching sibling executable is run through this same method
    /// first; the token bookkeeping is caller policy.
    pub fn invoke(
        &mut self,
        exe: &Executable<'_>,
        input: &mut [u8],
        outputs: &mut [&mut [u8]],
    ) -> Result<(), Error<T::Error>> {
        for hint in exe.hints() {
            match hint {
                Hint::Dma {
                    desc: Description::Parameter,
                    offset,
                    size,
                    ..
                } => {
                    let params = exe.parameters();
                    let slice = params
                        .get(offset..offset.saturating_add(size))
                        .ok_or(Error::Malformed("parameter hint out of range"))?;
                    self.send_parameters(slice)?;
                }
                Hint::Dma {
                    desc: Description::InputActivation,
                    name,
                    offset,
                    size,
                } => {
                    // Applied to the whole tensor, per matching hint.
                    if let Some(idx) = exe.find_input_layer(name) {
                        if let Some(layer) = exe.input_layer(idx) {
                            if layer.data_type().is_signed() {
                                Layer::transform_signed_data_type_raw(
                                    input,
                                    layer.data_type().size_bytes(),
                                    layer.x_dim(),
                                    layer.y_dim(),
                                    layer.z_dim(),
                                );
                            }
                        }
                    }
                    let slice = input
                        .get(offset..offset.saturating_add(size))
                        .ok_or(Error::Malformed("input hint out of range"))?;
                    self.send_inputs(slice)?;
                }
                Hint::Dma {
                    desc: Description::OutputActivation,
                    name,
                    size,
                    ..
                } => {
                    let idx = exe.find_output_layer(name).ok_or(Error::UnknownLayer)?;
                    let buf = outputs.get_mut(idx).ok_or(Error::UnknownLayer)?;
                    if buf.len() < size {
                        return Err(Error::BufferTooSmall {
                            needed: size,
                            given: buf.len(),
                        });
                    }
                    // offset_in_bytes is ignored: reads always land at the
                    // start of the layer's staging buffer.
                    self.get_outputs(&mut buf[..size])?;
                }
                Hint::Instruction { chunk_index } => {
                    let bitstream = exe
                        .instruction_bitstream(chunk_index)
                        .ok_or(Error::Malformed("instruction chunk index out of range"))?;
                    self.send_instructions(bitstream)?;
                }
                // Scratch, unknown descriptions, interrupt and fence hints.
                _ => {}
            }
        }
        self.read_event()?;
        Ok(())
    }
}
