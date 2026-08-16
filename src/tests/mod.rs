//! Host-side unit tests.
//!
//! Everything runs against [`Mock`], a fake [`Transport`] that records every
//! transfer and answers reads out of a small behavioural model of the chip: a
//! CSR register file whose `scu_ctrl_3` power-state bits track what is written
//! to its sleep bits, a DFU "flash" that stores what is downloaded and serves
//! it back on upload, and queues for bulk and event data. No hardware is
//! involved:
//!
//! ```text
//! cargo test
//! ```

use std::collections::BTreeMap;
use std::vec::Vec;

use crate::csr;
use crate::Transport;

mod dfu;
mod driver;
mod executable;
mod model;

// ---------------------------------------------------------------------------
// Recorded transfers
// ---------------------------------------------------------------------------

/// One recorded USB transfer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Xfer {
    ControlIn {
        req_type: u8,
        request: u8,
        value: u16,
        index: u16,
        len: usize,
        timeout_us: u32,
    },
    ControlOut {
        req_type: u8,
        request: u8,
        value: u16,
        index: u16,
        data: Vec<u8>,
        timeout_us: u32,
    },
    BulkOut {
        ep: u8,
        data: Vec<u8>,
        timeout_us: u32,
    },
    BulkIn {
        ep: u8,
        len: usize,
        timeout_us: u32,
    },
    InterruptIn {
        ep: u8,
        len: usize,
        timeout_us: u32,
    },
}

/// A CSR transfer decoded back out of a [`Xfer`], for golden-sequence asserts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CsrOp {
    R32(u64),
    W32(u64, u32),
    R64(u64),
    W64(u64, u64),
}

impl Xfer {
    /// Rebuilds the CSR register address from `wValue`/`wIndex`.
    fn csr_reg(value: u16, index: u16) -> u64 {
        u64::from(value) | (u64::from(index) << 16)
    }

    /// Decodes this transfer as a CSR access, if it is one.
    pub fn as_csr(&self) -> Option<CsrOp> {
        match self {
            Xfer::ControlIn {
                req_type: 0xC0,
                request,
                value,
                index,
                ..
            } => Some(match request {
                1 => CsrOp::R32(Self::csr_reg(*value, *index)),
                _ => CsrOp::R64(Self::csr_reg(*value, *index)),
            }),
            Xfer::ControlOut {
                req_type: 0x40,
                request,
                value,
                index,
                data,
                ..
            } => {
                let reg = Self::csr_reg(*value, *index);
                Some(if *request == 1 {
                    CsrOp::W32(reg, u32::from_le_bytes(data[..4].try_into().unwrap()))
                } else {
                    CsrOp::W64(reg, u64::from_le_bytes(data[..8].try_into().unwrap()))
                })
            }
            _ => None,
        }
    }
}

/// Error type of [`Mock`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MockError;

// ---------------------------------------------------------------------------
// The mock device
// ---------------------------------------------------------------------------

pub struct Mock {
    /// Every transfer, in order.
    pub log: Vec<Xfer>,
    /// CSR register file. Unlisted registers read as zero.
    pub regs: BTreeMap<u64, u64>,
    /// When false, writes to `scu_ctrl_3` do not update `cur_pwr_state`, which
    /// is what a wedged chip looks like to the bring-up polls.
    pub power_state_follows: bool,
    /// Device-side DFU storage: what has been downloaded so far.
    pub dfu_flash: Vec<u8>,
    /// If set, `DFU_UPLOAD` serves these bytes instead of `dfu_flash`, so a
    /// read-back mismatch can be provoked.
    pub dfu_upload_override: Option<Vec<u8>>,
    /// `bStatus` returned by `DFU_GETSTATUS`.
    pub dfu_status: u8,
    /// `bState` returned by `DFU_GETSTATUS`.
    pub dfu_state: u8,
    /// Bytes served on the next bulk IN reads of the data endpoint, in order.
    pub bulk_in_data: Vec<Vec<u8>>,
    /// Bytes served on the event endpoint.
    pub event_data: Vec<u8>,
    /// What `interrupt_in` returns.
    pub interrupt_data: Option<Vec<u8>>,
    /// Block size the device-side DFU model uses to place blocks.
    pub dfu_block_size: usize,
    /// Registers that ignore writes, for modelling a broken control path.
    pub readonly_regs: Vec<u64>,
    /// Bytes to withhold from every control IN, for testing short transfers.
    pub control_in_short_by: usize,
}

impl Default for Mock {
    fn default() -> Self {
        let mut regs = BTreeMap::new();
        // A healthy part reports the Beagle chip ID.
        regs.insert(csr::OMC0_00, csr::CHIP_ID);
        // Arbitrary non-zero power-on values, so read-modify-writes are visible
        // in the recorded trace rather than being indistinguishable from zero.
        regs.insert(csr::SCU_CTRL_0, 0x0000_0941);
        regs.insert(csr::SCU_CTRL_2, 0x0000_0000);
        regs.insert(csr::SCU_CTRL_3, 0x8005_0410);
        regs.insert(csr::SCALAR_CORE_RUN_CONTROL, 0);
        Mock {
            log: Vec::new(),
            regs,
            power_state_follows: true,
            dfu_flash: Vec::new(),
            dfu_upload_override: None,
            dfu_status: 0,
            dfu_state: 2,
            bulk_in_data: Vec::new(),
            event_data: Vec::new(),
            interrupt_data: None,
            dfu_block_size: crate::DFU_BLOCK_SIZE,
            readonly_regs: Vec::new(),
            control_in_short_by: 0,
        }
    }
}

impl Mock {
    pub fn new() -> Self {
        Self::default()
    }

    /// Every recorded transfer decoded as a CSR access, non-CSR ones dropped.
    pub fn csr_ops(&self) -> Vec<CsrOp> {
        self.log.iter().filter_map(Xfer::as_csr).collect()
    }

    /// Every recorded bulk-OUT payload, in order.
    pub fn bulk_writes(&self) -> Vec<Vec<u8>> {
        self.log
            .iter()
            .filter_map(|x| match x {
                Xfer::BulkOut { data, .. } => Some(data.clone()),
                _ => None,
            })
            .collect()
    }

    fn reg(&self, addr: u64) -> u64 {
        self.regs.get(&addr).copied().unwrap_or(0)
    }

    /// Applies a CSR write, modelling the side effects the bring-up polls wait
    /// on.
    fn write_reg(&mut self, addr: u64, value: u64) {
        if self.readonly_regs.contains(&addr) {
            return;
        }
        self.regs.insert(addr, value);
        if addr == csr::SCU_CTRL_3 && self.power_state_follows {
            let force_sleep = csr::get_field(
                value,
                csr::SCU_CTRL_3_FORCE_SLEEP.0,
                csr::SCU_CTRL_3_FORCE_SLEEP.1,
            );
            let state = match force_sleep {
                csr::FORCE_SLEEP_RESET => csr::PWR_STATE_SLEEP,
                csr::FORCE_SLEEP_RUN => csr::PWR_STATE_ACTIVE,
                _ => return,
            };
            let updated = csr::set_field(
                value,
                csr::SCU_CTRL_3_CUR_PWR_STATE.0,
                csr::SCU_CTRL_3_CUR_PWR_STATE.1,
                state,
            );
            self.regs.insert(addr, updated);
        }
    }

    fn dfu_upload(&self, block: u16, buf: &mut [u8]) -> usize {
        let src = self.dfu_upload_override.as_ref().unwrap_or(&self.dfu_flash);
        let start = usize::from(block) * self.dfu_block_size;
        let end = (start + buf.len()).min(src.len());
        if start >= end {
            return 0;
        }
        let n = end - start;
        buf[..n].copy_from_slice(&src[start..end]);
        n
    }
}

impl Transport for Mock {
    type Error = MockError;

    fn control_in(
        &mut self,
        req_type: u8,
        request: u8,
        value: u16,
        index: u16,
        buf: &mut [u8],
        timeout_us: u32,
    ) -> Result<usize, Self::Error> {
        self.log.push(Xfer::ControlIn {
            req_type,
            request,
            value,
            index,
            len: buf.len(),
            timeout_us,
        });
        match req_type {
            // Vendor IN: a CSR read.
            0xC0 => {
                let raw = self.reg(Xfer::csr_reg(value, index));
                let bytes = raw.to_le_bytes();
                buf.copy_from_slice(&bytes[..buf.len()]);
                Ok(buf.len() - self.control_in_short_by.min(buf.len()))
            }
            // Class IN: DFU_GETSTATUS or DFU_UPLOAD.
            0xA1 => match request {
                0x03 => {
                    let status = [self.dfu_status, 0, 0, 0, self.dfu_state, 0];
                    buf.copy_from_slice(&status[..buf.len()]);
                    Ok(buf.len())
                }
                0x02 => Ok(self.dfu_upload(value, buf)),
                _ => Err(MockError),
            },
            _ => Err(MockError),
        }
    }

    fn control_out(
        &mut self,
        req_type: u8,
        request: u8,
        value: u16,
        index: u16,
        data: &[u8],
        timeout_us: u32,
    ) -> Result<(), Self::Error> {
        self.log.push(Xfer::ControlOut {
            req_type,
            request,
            value,
            index,
            data: data.to_vec(),
            timeout_us,
        });
        match req_type {
            // Vendor OUT: a CSR write.
            0x40 => {
                let mut raw = [0u8; 8];
                raw[..data.len()].copy_from_slice(data);
                self.write_reg(Xfer::csr_reg(value, index), u64::from_le_bytes(raw));
                Ok(())
            }
            // Class OUT: DFU_DNLOAD or DFU_DETACH.
            0x21 => {
                // The zero-length DNLOAD that terminates a download must not
                // extend the image.
                if request == 0x01 && !data.is_empty() {
                    let start = usize::from(value) * self.dfu_block_size;
                    if self.dfu_flash.len() < start + data.len() {
                        self.dfu_flash.resize(start + data.len(), 0);
                    }
                    self.dfu_flash[start..start + data.len()].copy_from_slice(data);
                }
                Ok(())
            }
            // Standard OUT to an interface: SET_INTERFACE.
            0x01 => Ok(()),
            _ => Err(MockError),
        }
    }

    fn bulk_out(&mut self, ep: u8, data: &[u8], timeout_us: u32) -> Result<(), Self::Error> {
        self.log.push(Xfer::BulkOut {
            ep,
            data: data.to_vec(),
            timeout_us,
        });
        Ok(())
    }

    fn bulk_in(&mut self, ep: u8, buf: &mut [u8], timeout_us: u32) -> Result<usize, Self::Error> {
        self.log.push(Xfer::BulkIn {
            ep,
            len: buf.len(),
            timeout_us,
        });
        if ep == crate::EVENT_ENDPOINT {
            let n = self.event_data.len().min(buf.len());
            buf[..n].copy_from_slice(&self.event_data[..n]);
            return Ok(n);
        }
        if self.bulk_in_data.is_empty() {
            // Default: fill with a recognisable ramp so relayout tests have
            // deterministic source bytes.
            for (i, b) in buf.iter_mut().enumerate() {
                *b = i as u8;
            }
            return Ok(buf.len());
        }
        let chunk = self.bulk_in_data.remove(0);
        let n = chunk.len().min(buf.len());
        buf[..n].copy_from_slice(&chunk[..n]);
        Ok(n)
    }

    fn interrupt_in(
        &mut self,
        ep: u8,
        buf: &mut [u8],
        timeout_us: u32,
    ) -> Result<Option<usize>, Self::Error> {
        self.log.push(Xfer::InterruptIn {
            ep,
            len: buf.len(),
            timeout_us,
        });
        match &self.interrupt_data {
            Some(data) => {
                let n = data.len().min(buf.len());
                buf[..n].copy_from_slice(&data[..n]);
                Ok(Some(n))
            }
            None => Ok(None),
        }
    }
}

/// A [`DelayNs`](embedded_hal::delay::DelayNs) that records what it was asked
/// to wait for instead of waiting.
#[derive(Default)]
pub struct MockDelay {
    pub waits_ns: Vec<u32>,
}

impl embedded_hal::delay::DelayNs for MockDelay {
    fn delay_ns(&mut self, ns: u32) {
        self.waits_ns.push(ns);
    }
}
