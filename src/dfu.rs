//! DFU firmware download.
//!
//! A cold Edge TPU comes up in DFU mode with no functional endpoints and has to
//! be handed its Apex firmware image before it can do anything. DFU 1.1.
//!
//! The firmware blob is not in this crate: it is Google's binary, and the
//! application embeds it and passes the bytes to [`Dfu::download`].

use crate::{Error, Transport};

/// USB vendor ID of the Edge TPU in DFU mode.
pub const DFU_VID: u16 = 0x1A6E;
/// USB product ID of the Edge TPU in DFU mode.
pub const DFU_PID: u16 = 0x089A;

/// DFU transfer block size, in bytes.
///
/// The DFU functional descriptor's `wTransferSize` is not read; this is the
/// known-good value. See [`Dfu::with_block_size`].
pub const DFU_BLOCK_SIZE: usize = 256;

/// Default timeout for a DFU control transfer, in microseconds.
///
/// The same 200 ms the runtime driver uses for every control transfer.
pub const DFU_TIMEOUT_US: u32 = 200_000;

/// The `wDetachTimeOut` sent with `DFU_DETACH`, in milliseconds.
pub const DETACH_TIMEOUT_MS: u16 = 1000;

// DFU 1.1 class request codes.
const DFU_DETACH: u8 = 0x00;
const DFU_DNLOAD: u8 = 0x01;
const DFU_UPLOAD: u8 = 0x02;
const DFU_GETSTATUS: u8 = 0x03;

// USB request-type bits.
const DIR_OUT: u8 = 0x00;
const DIR_IN: u8 = 0x80;
const TYPE_CLASS: u8 = 0x20;
const RECIPIENT_INTERFACE: u8 = 0x01;

// Standard request.
const SET_INTERFACE: u8 = 0x0B;

/// `bmRequestType` for the host-to-device DFU class requests.
const CLASS_OUT: u8 = DIR_OUT | TYPE_CLASS | RECIPIENT_INTERFACE;
/// `bmRequestType` for the device-to-host DFU class requests.
const CLASS_IN: u8 = DIR_IN | TYPE_CLASS | RECIPIENT_INTERFACE;

/// Length of a `DFU_GETSTATUS` response.
const STATUS_LEN: usize = 6;

/// A parsed `DFU_GETSTATUS` response.
///
/// Parsed and checked; see [`Dfu::download`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DfuStatus {
    /// `bStatus`: `0` is `OK`; anything else is an error code.
    pub status: u8,
    /// `bwPollTimeout`: milliseconds the device asks the host to wait before
    /// the next `DFU_GETSTATUS`.
    ///
    /// Not honoured here; surfaced so a caller that wants to can.
    pub poll_timeout_ms: u32,
    /// `bState`: the state the device will be in after this response.
    pub state: u8,
    /// `iString`: index of a status description string.
    pub string_index: u8,
}

impl DfuStatus {
    fn parse(buf: &[u8; STATUS_LEN]) -> Self {
        DfuStatus {
            status: buf[0],
            poll_timeout_ms: u32::from(buf[1])
                | (u32::from(buf[2]) << 8)
                | (u32::from(buf[3]) << 16),
            state: buf[4],
            string_index: buf[5],
        }
    }

    /// Whether `bStatus` is `OK` (`0`).
    #[must_use]
    pub fn is_ok(&self) -> bool {
        self.status == 0
    }
}

/// A summary of one completed firmware download, for logging and tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DfuReport {
    /// Number of `DFU_DNLOAD` requests carrying data.
    pub download_blocks: u16,
    /// Number of `DFU_UPLOAD` requests issued during read-back verification.
    pub upload_blocks: u16,
    /// Number of `DFU_GETSTATUS` requests issued in total.
    pub status_requests: u32,
    /// Bytes downloaded, which equals the firmware length on success.
    pub bytes: usize,
}

/// Drives an Edge TPU in DFU mode through a firmware download.
///
/// Borrows the transport rather than consuming it: the same USB stack usually
/// has to keep talking to the device after it re-enumerates, and the driver
/// object that does that is built separately.
pub struct Dfu<'t, T: Transport> {
    transport: &'t mut T,
    interface: u8,
    block_size: usize,
    timeout_us: u32,
}

impl<'t, T: Transport> Dfu<'t, T> {
    /// Wraps a transport that is already talking to the DFU interface.
    ///
    /// `interface` is the `bInterfaceNumber` of the DFU interface; it becomes
    /// `wIndex` on every class request. Finding it means walking the
    /// configuration for class `0xFE` subclass `0x01`, which is the USB
    /// host stack's job, so it is passed in.
    pub fn new(transport: &'t mut T, interface: u8) -> Self {
        Dfu {
            transport,
            interface,
            block_size: DFU_BLOCK_SIZE,
            timeout_us: DFU_TIMEOUT_US,
        }
    }

    /// Overrides the transfer block size.
    ///
    /// For a caller that does read `wTransferSize` out of the DFU functional
    /// descriptor. The default [`DFU_BLOCK_SIZE`] is the only value known to
    /// work on this part.
    #[must_use]
    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size.max(1);
        self
    }

    /// Overrides the per-transfer timeout, in microseconds.
    #[must_use]
    pub fn with_timeout_us(mut self, timeout_us: u32) -> Self {
        self.timeout_us = timeout_us;
        self
    }

    /// Downloads `firmware`, verifies it by read-back, and detaches.
    ///
    /// In order:
    ///
    /// 1. `SET_INTERFACE` alternate setting 0.
    /// 2. `DFU_GETSTATUS`, then a `DFU_DNLOAD` of up to [`DFU_BLOCK_SIZE`]
    ///    bytes, repeated until the whole image has been sent. The status
    ///    request comes *before* each block, so an `n`-block image
    ///    produces `n + 1` of them.
    /// 3. A zero-length `DFU_DNLOAD` to terminate the download.
    /// 4. The whole image read back with `DFU_UPLOAD` and compared byte
    ///    for byte, not by checksum. Here the status request comes
    ///    *after* each block, so `n` blocks produce `n`.
    /// 5. `DFU_DETACH`.
    ///
    /// The caller then resets the bus and re-enumerates the device, which
    /// reappears as [`crate::RUNTIME_VID`]`:`[`crate::RUNTIME_PID`].
    ///
    /// `readback` is scratch space for step 4 and must be at least
    /// `firmware.len()` bytes.
    ///
    /// A `DFU_GETSTATUS` reporting a non-zero `bStatus` fails with
    /// [`Error::DfuStatus`].
    pub fn download(
        &mut self,
        firmware: &[u8],
        readback: &mut [u8],
    ) -> Result<DfuReport, Error<T::Error>> {
        if readback.len() < firmware.len() {
            return Err(Error::BufferTooSmall {
                needed: firmware.len(),
                given: readback.len(),
            });
        }
        let readback = &mut readback[..firmware.len()];
        let mut report = DfuReport::default();

        self.set_interface(0)?;

        // --- Download -------------------------------------------------------
        let mut block: u16 = 0;
        let mut sent = 0usize;
        while sent < firmware.len() {
            self.get_status(&mut report)?;
            let len = self.block_size.min(firmware.len() - sent);
            self.dnload(block, &firmware[sent..sent + len])?;
            block = block.wrapping_add(1);
            sent += len;
            report.download_blocks = report.download_blocks.wrapping_add(1);
        }
        // The final status request is the one whose byte-count test sends the
        // state machine to kZeroLengthTransfer rather than to another block.
        self.get_status(&mut report)?;
        self.dnload(block, &[])?;
        report.bytes = sent;

        // --- Read-back verification ----------------------------------------
        let mut block: u16 = 0;
        let mut received = 0usize;
        while received < readback.len() {
            let len = self.block_size.min(readback.len() - received);
            self.upload(block, &mut readback[received..received + len])?;
            block = block.wrapping_add(1);
            received += len;
            report.upload_blocks = report.upload_blocks.wrapping_add(1);
            self.get_status(&mut report)?;
        }
        if let Some(offset) = first_difference(firmware, readback) {
            return Err(Error::ReadbackMismatch { offset });
        }

        // --- Detach ---------------------------------------------------------
        self.detach(DETACH_TIMEOUT_MS)?;
        Ok(report)
    }

    /// `SET_INTERFACE`, alternate setting `alt`.
    ///
    /// A standard request, not a DFU class request: `bmRequestType` is bare
    /// `RECIPIENT_INTERFACE` with no type bits.
    fn set_interface(&mut self, alt: u16) -> Result<(), Error<T::Error>> {
        self.transport
            .control_out(
                RECIPIENT_INTERFACE,
                SET_INTERFACE,
                alt,
                u16::from(self.interface),
                &[],
                self.timeout_us,
            )
            .map_err(Error::Transport)
    }

    /// `DFU_GETSTATUS`.
    fn get_status(&mut self, report: &mut DfuReport) -> Result<DfuStatus, Error<T::Error>> {
        let mut buf = [0u8; STATUS_LEN];
        let n = self
            .transport
            .control_in(
                CLASS_IN,
                DFU_GETSTATUS,
                0,
                u16::from(self.interface),
                &mut buf,
                self.timeout_us,
            )
            .map_err(Error::Transport)?;
        report.status_requests = report.status_requests.wrapping_add(1);
        if n != STATUS_LEN {
            return Err(Error::ShortTransfer {
                expected: STATUS_LEN,
                actual: n,
            });
        }
        let status = DfuStatus::parse(&buf);
        if !status.is_ok() {
            return Err(Error::DfuStatus {
                status: status.status,
                state: status.state,
            });
        }
        Ok(status)
    }

    /// `DFU_DNLOAD` of one block.
    ///
    /// `block` is a full 16-bit `wValue`, per the DFU 1.1 spec.
    fn dnload(&mut self, block: u16, data: &[u8]) -> Result<(), Error<T::Error>> {
        self.transport
            .control_out(
                CLASS_OUT,
                DFU_DNLOAD,
                block,
                u16::from(self.interface),
                data,
                self.timeout_us,
            )
            .map_err(Error::Transport)
    }

    /// `DFU_UPLOAD` of one block.
    fn upload(&mut self, block: u16, buf: &mut [u8]) -> Result<(), Error<T::Error>> {
        let want = buf.len();
        let n = self
            .transport
            .control_in(
                CLASS_IN,
                DFU_UPLOAD,
                block,
                u16::from(self.interface),
                buf,
                self.timeout_us,
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

    /// `DFU_DETACH`.
    fn detach(&mut self, timeout_ms: u16) -> Result<(), Error<T::Error>> {
        self.transport
            .control_out(
                CLASS_OUT,
                DFU_DETACH,
                timeout_ms,
                u16::from(self.interface),
                &[],
                self.timeout_us,
            )
            .map_err(Error::Transport)
    }
}

/// Index of the first byte at which `a` and `b` differ.
fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    a.iter().zip(b.iter()).position(|(x, y)| x != y)
}
