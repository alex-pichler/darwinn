#![no_std]
#![deny(missing_docs)]
#![forbid(unsafe_code)]
//! A `no_std` driver for the Google Edge TPU (DarwiNN) attached over USB.
//!
//! Ported from the TPU half of Google's coralmicro SDK (`libs/tpu`).
//!
//! The crate holds no USB host driver and no chip firmware. It reaches the
//! device through the [`Transport`] trait, which the host-controller side
//! implements, and the Apex firmware blob stays in the application.
//!
//! # The three stages
//!
//! 1. DFU. A cold Edge TPU enumerates as [`DFU_VID`]`:`[`DFU_PID`] with no
//!    functional endpoints. [`Dfu`] downloads the firmware in
//!    [`DFU_BLOCK_SIZE`] blocks, verifies it by read-back and detaches; the
//!    device re-enumerates as [`RUNTIME_VID`]`:`[`RUNTIME_PID`].
//! 2. Bring-up. [`Driver::init`] takes the chip from cold to the run state:
//!    chip-ID check, control-path self-test, PHY and clock-gating setup, a trip
//!    through reset, clock-rate selection, tile configuration, temperature
//!    sensor, then [`csr::RunControl::MoveToRun`] across the scalar core and
//!    every tile.
//! 3. Inference. [`Driver::invoke`] walks an [`Executable`]'s DMA hints,
//!    streams instructions, parameters and input activations out on bulk
//!    endpoint 1, reads output activations back, and waits for the completion
//!    event.
//!
//! # Bounded waits
//!
//! Every register poll is bounded by [`Timeouts::poll_attempts`] and fails with
//! [`Error::PollTimeout`] naming the register; every transfer carries a
//! microsecond timeout handed to the transport. No loop in this crate is
//! unbounded.
//!
//! # Example
//!
//! ```no_run
//! # fn example<T: darwinn::Transport, D: embedded_hal::delay::DelayNs>(
//! #     mut transport: T, mut delay: D, firmware: &[u8], model: &[u8],
//! #     input: &mut [u8], staging: &mut [u8], readback: &mut [u8],
//! # ) -> Result<(), darwinn::Error<T::Error>> {
//! use darwinn::{Dfu, Driver, Package, PerformanceMode, Timeouts};
//!
//! // 1. Cold device: push the Apex firmware, then let it re-enumerate.
//! Dfu::new(&mut transport, 0).download(firmware, readback)?;
//!
//! // 2. Post-DFU device: bring the chip out of reset.
//! let mut tpu = Driver::new(transport, Timeouts::DEFAULT);
//! tpu.init(PerformanceMode::Max, &mut delay)?;
//!
//! // 3. Run the inference executable out of a TFLite custom-op payload.
//! let package = Package::from_custom_op(model).unwrap();
//! let exe = package.inference_executable().unwrap();
//! tpu.invoke(&exe, input, &mut [staging])?;
//!
//! // 4. De-tile the staging buffer into the flat output tensor.
//! let layer = exe.output_layer(0).unwrap();
//! let mut tensor = [0u8; 64];
//! layer.relayout_into(staging, &mut tensor)?;
//! layer.transform_signed_data_type(&mut tensor);
//! # Ok(())
//! # }
//! ```

pub mod csr;
mod dfu;
mod driver;
mod executable;
mod fb;

#[cfg(test)]
extern crate std;

#[cfg(test)]
mod tests;

pub use csr::RunControl;
pub use dfu::{
    Dfu, DfuReport, DfuStatus, DETACH_TIMEOUT_MS, DFU_BLOCK_SIZE, DFU_PID, DFU_TIMEOUT_US, DFU_VID,
};
pub use driver::{
    DescriptorTag, Driver, Event, PerformanceMode, Timeouts, BULK_IN_ENDPOINT, BULK_OUT_ENDPOINT,
    EVENT_ENDPOINT, EVENT_SIZE, HEADER_SIZE, INTERRUPT_ENDPOINT, MAX_BULK_CHUNK, RUNTIME_PID,
    RUNTIME_VID,
};
pub use executable::{
    DataType, Description, Executable, ExecutableType, Hint, Layer, LayoutError, MultiExecutable,
    Numerics, Package,
};

/// The USB transport this driver runs on.
///
/// Implemented by the host-controller side. The crate reaches the device
/// through these five calls and nothing else, which is what makes it testable
/// on a desktop against a mock.
///
/// All timeouts are in microseconds and are upper bounds on how long the
/// implementation may block. `interrupt_in` returns `Ok(None)` when the timeout
/// expires with no data, because an idle polled endpoint is not an error; every
/// other method treats a timeout as an error.
pub trait Transport {
    /// Transport-specific error type.
    type Error: core::fmt::Debug;
    /// Performs a device-to-host control transfer, returning the byte count.
    fn control_in(
        &mut self,
        req_type: u8,
        request: u8,
        value: u16,
        index: u16,
        buf: &mut [u8],
        timeout_us: u32,
    ) -> Result<usize, Self::Error>;
    /// Performs a host-to-device control transfer.
    fn control_out(
        &mut self,
        req_type: u8,
        request: u8,
        value: u16,
        index: u16,
        data: &[u8],
        timeout_us: u32,
    ) -> Result<(), Self::Error>;
    /// Writes to a bulk OUT endpoint.
    fn bulk_out(&mut self, ep: u8, data: &[u8], timeout_us: u32) -> Result<(), Self::Error>;
    /// Reads from a bulk IN endpoint, returning the byte count.
    fn bulk_in(&mut self, ep: u8, buf: &mut [u8], timeout_us: u32) -> Result<usize, Self::Error>;
    /// Polls an interrupt IN endpoint. `Ok(None)` means the poll timed out.
    fn interrupt_in(
        &mut self,
        ep: u8,
        buf: &mut [u8],
        timeout_us: u32,
    ) -> Result<Option<usize>, Self::Error>;
}

/// Errors returned by this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error<E> {
    /// The underlying USB transport failed.
    Transport(E),
    /// A transfer moved fewer bytes than were asked for.
    ShortTransfer {
        /// Bytes requested.
        expected: usize,
        /// Bytes actually moved.
        actual: usize,
    },
    /// `omc0_00.chip_id` was not [`csr::CHIP_ID`].
    ///
    /// The device answered on the control pipe but is not a Beagle Edge TPU, or
    /// the firmware download did not take.
    ChipId {
        /// The chip ID actually read.
        found: u64,
    },
    /// The `omc0_00.test_reg0` write/read-back self-test failed: the control
    /// path works in one direction but not the other.
    SelfTest {
        /// The pattern actually read back.
        found: u64,
    },
    /// A bounded register poll never reached its target value. The payload
    /// names the register, so it points at the stage of bring-up that stalled.
    PollTimeout(&'static str),
    /// The device reported a DFU error in a `DFU_GETSTATUS` response.
    DfuStatus {
        /// `bStatus` from the response.
        status: u8,
        /// `bState` from the response.
        state: u8,
    },
    /// Firmware read back after download did not match what was sent.
    ReadbackMismatch {
        /// Offset of the first differing byte.
        offset: usize,
    },
    /// The executable FlatBuffer could not be walked. The payload names the
    /// field that was missing or out of bounds.
    Malformed(&'static str),
    /// A caller-provided buffer was too small for the transfer the executable
    /// asked for.
    BufferTooSmall {
        /// Bytes the executable requires.
        needed: usize,
        /// Bytes the caller provided.
        given: usize,
    },
    /// A DMA hint named a layer that is not in the executable's layer list.
    UnknownLayer,
}

impl<E: core::fmt::Debug> core::fmt::Display for Error<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::Transport(e) => write!(f, "USB transport error: {e:?}"),
            Error::ShortTransfer { expected, actual } => {
                write!(f, "short transfer: wanted {expected} bytes, got {actual}")
            }
            Error::ChipId { found } => write!(
                f,
                "not an Edge TPU: chip ID is {found:#x}, expected {:#x}",
                csr::CHIP_ID
            ),
            Error::SelfTest { found } => {
                write!(
                    f,
                    "CSR self-test read back {found:#x}, expected {:#x}",
                    csr::TEST_REG0_PATTERN
                )
            }
            Error::PollTimeout(what) => write!(f, "timed out waiting for {what}"),
            Error::DfuStatus { status, state } => {
                write!(f, "DFU error: bStatus={status:#04x}, bState={state:#04x}")
            }
            Error::ReadbackMismatch { offset } => {
                write!(f, "firmware read-back differs at offset {offset}")
            }
            Error::Malformed(what) => write!(f, "malformed executable: {what}"),
            Error::BufferTooSmall { needed, given } => {
                write!(f, "buffer too small: need {needed} bytes, have {given}")
            }
            Error::UnknownLayer => write!(f, "DMA hint names a layer the executable does not have"),
        }
    }
}

impl<E: core::fmt::Debug> core::error::Error for Error<E> {}
