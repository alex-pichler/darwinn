//! DFU sequence tests, against `edgetpu_dfu_task.cc` and `usb_host_dfu.c`.

use std::vec;
use std::vec::Vec;

use super::{Mock, Xfer};
use crate::{Dfu, Error, DFU_BLOCK_SIZE};

/// Length of the real Apex image.
const REAL_FIRMWARE_LEN: usize = 10_783;

fn ramp(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i % 251) as u8).collect()
}

/// The `(bmRequestType, bRequest, wValue, wIndex, payload len)` of each control
/// transfer, which is the whole wire contract of the DFU stage.
fn requests(mock: &Mock) -> Vec<(u8, u8, u16, u16, usize)> {
    mock.log
        .iter()
        .map(|x| match x {
            Xfer::ControlIn {
                req_type,
                request,
                value,
                index,
                len,
                ..
            } => (*req_type, *request, *value, *index, *len),
            Xfer::ControlOut {
                req_type,
                request,
                value,
                index,
                data,
                ..
            } => (*req_type, *request, *value, *index, data.len()),
            other => panic!("unexpected non-control transfer in DFU: {other:?}"),
        })
        .collect()
}

#[test]
fn download_request_encodings_match_the_nxp_dfu_class() {
    let firmware = ramp(600);
    let mut mock = Mock::new();
    let mut readback = vec![0u8; firmware.len()];
    Dfu::new(&mut mock, 3)
        .download(&firmware, &mut readback)
        .unwrap();

    let reqs = requests(&mock);
    // SET_INTERFACE: a standard request to the interface, alternate setting 0.
    // No type bits, so bmRequestType is bare RECIPIENT_INTERFACE.
    assert_eq!(reqs[0], (0x01, 0x0B, 0, 3, 0));
    // DFU_GETSTATUS: class IN, wValue 0, six bytes.
    assert_eq!(reqs[1], (0xA1, 0x03, 0, 3, 6));
    // DFU_DNLOAD: class OUT, wValue = block number.
    assert_eq!(reqs[2], (0x21, 0x01, 0, 3, DFU_BLOCK_SIZE));
    // Every request carries wIndex = bInterfaceNumber.
    assert!(reqs.iter().all(|r| r.3 == 3));
}

#[test]
fn download_block_sizes_and_numbering() {
    // 600 bytes is two full 256-byte blocks plus an 88-byte tail.
    let firmware = ramp(600);
    let mut mock = Mock::new();
    let mut readback = vec![0u8; firmware.len()];
    let report = Dfu::new(&mut mock, 0)
        .download(&firmware, &mut readback)
        .unwrap();

    let dnloads: Vec<(u16, usize)> = requests(&mock)
        .iter()
        .filter(|(t, r, ..)| *t == 0x21 && *r == 0x01)
        .map(|(_, _, v, _, len)| (*v, *len))
        .collect();
    // Three data blocks numbered from zero, then the zero-length terminator
    // with the next block number.
    assert_eq!(dnloads, [(0, 256), (1, 256), (2, 88), (3, 0)]);
    assert_eq!(report.download_blocks, 3);
    assert_eq!(report.bytes, 600);
}

#[test]
fn download_status_polling_brackets_every_block() {
    let firmware = ramp(600);
    let mut mock = Mock::new();
    let mut readback = vec![0u8; firmware.len()];
    let report = Dfu::new(&mut mock, 0)
        .download(&firmware, &mut readback)
        .unwrap();

    // Download: GETSTATUS *before* each block, plus the one whose byte-count
    // test moves the state machine on to the zero-length terminator.
    // Read-back: GETSTATUS *after* each block. 3 + 1 + 3 = 7.
    assert_eq!(report.status_requests, 7);

    // The interleaving, as opcodes: SET_INTERFACE, then the download loop,
    // then the terminator, then the upload loop, then DETACH.
    let ops: Vec<(u8, u8)> = requests(&mock).iter().map(|r| (r.0, r.1)).collect();
    assert_eq!(
        ops,
        [
            (0x01, 0x0B), // SET_INTERFACE
            (0xA1, 0x03), // GETSTATUS
            (0x21, 0x01), // DNLOAD block 0
            (0xA1, 0x03),
            (0x21, 0x01), // DNLOAD block 1
            (0xA1, 0x03),
            (0x21, 0x01), // DNLOAD block 2
            (0xA1, 0x03),
            (0x21, 0x01), // zero-length DNLOAD
            (0xA1, 0x02), // UPLOAD block 0
            (0xA1, 0x03),
            (0xA1, 0x02), // UPLOAD block 1
            (0xA1, 0x03),
            (0xA1, 0x02), // UPLOAD block 2
            (0xA1, 0x03),
            (0x21, 0x00), // DETACH
        ]
    );
}

#[test]
fn detach_carries_the_full_sixteen_bit_timeout() {
    let firmware = ramp(16);
    let mut mock = Mock::new();
    let mut readback = vec![0u8; firmware.len()];
    Dfu::new(&mut mock, 0)
        .download(&firmware, &mut readback)
        .unwrap();

    let detach = requests(&mock)
        .into_iter()
        .find(|(t, r, ..)| *t == 0x21 && *r == 0x00)
        .unwrap();
    // The full 16-bit wDetachTimeOut, not truncated to 8 bits.
    assert_eq!(detach.2, 1000);
    assert_eq!(detach.4, 0);
}

#[test]
fn real_firmware_size_produces_forty_three_blocks() {
    // 10783 = 42 * 256 + 31, so 42 full blocks and a 31-byte tail.
    let firmware = ramp(REAL_FIRMWARE_LEN);
    let mut mock = Mock::new();
    let mut readback = vec![0u8; firmware.len()];
    let report = Dfu::new(&mut mock, 0)
        .download(&firmware, &mut readback)
        .unwrap();

    assert_eq!(report.download_blocks, 43);
    assert_eq!(report.upload_blocks, 43);
    assert_eq!(report.status_requests, 43 + 1 + 43);
    assert_eq!(report.bytes, REAL_FIRMWARE_LEN);

    let dnloads: Vec<(u16, usize)> = requests(&mock)
        .iter()
        .filter(|(t, r, ..)| *t == 0x21 && *r == 0x01)
        .map(|(_, _, v, _, len)| (*v, *len))
        .collect();
    assert_eq!(dnloads.len(), 44);
    assert_eq!(dnloads[42], (42, 31));
    assert_eq!(dnloads[43], (43, 0));
    // Block numbers stay inside the byte the NXP layer would truncate them to,
    // which is why its defect is invisible for this image.
    assert!(dnloads.iter().all(|(b, _)| *b < 256));
}

#[test]
fn downloaded_image_reaches_the_device_intact() {
    let firmware = ramp(1000);
    let mut mock = Mock::new();
    let mut readback = vec![0u8; firmware.len()];
    Dfu::new(&mut mock, 0)
        .download(&firmware, &mut readback)
        .unwrap();
    assert_eq!(mock.dfu_flash, firmware);
    assert_eq!(readback, firmware);
}

#[test]
fn readback_mismatch_is_rejected() {
    let firmware = ramp(600);
    let mut corrupt = firmware.clone();
    corrupt[300] ^= 0xFF;
    let mut mock = Mock::new();
    mock.dfu_upload_override = Some(corrupt);
    let mut readback = vec![0u8; firmware.len()];

    let err = Dfu::new(&mut mock, 0)
        .download(&firmware, &mut readback)
        .unwrap_err();
    // The vendor check is a whole-image memcmp; this reports where it first
    // differs.
    assert_eq!(err, Error::ReadbackMismatch { offset: 300 });
}

#[test]
fn nonzero_dfu_status_is_an_error() {
    let firmware = ramp(64);
    let mut mock = Mock::new();
    mock.dfu_status = 0x0F; // errUNKNOWN
    mock.dfu_state = 0x0A; // dfuERROR
    let mut readback = vec![0u8; firmware.len()];

    let err = Dfu::new(&mut mock, 0)
        .download(&firmware, &mut readback)
        .unwrap_err();
    // A device in dfuERROR must fail the download, not pass it.
    assert_eq!(
        err,
        Error::DfuStatus {
            status: 0x0F,
            state: 0x0A
        }
    );
}

#[test]
fn readback_buffer_must_be_large_enough() {
    let firmware = ramp(600);
    let mut mock = Mock::new();
    let mut readback = vec![0u8; 100];
    let err = Dfu::new(&mut mock, 0)
        .download(&firmware, &mut readback)
        .unwrap_err();
    assert_eq!(
        err,
        Error::BufferTooSmall {
            needed: 600,
            given: 100
        }
    );
}

#[test]
fn block_size_is_overridable_for_a_device_that_declares_one() {
    let firmware = ramp(300);
    let mut mock = Mock::new();
    mock.dfu_block_size = 64;
    let mut readback = vec![0u8; firmware.len()];
    // The override has to work for a caller that reads wTransferSize.
    let report = Dfu::new(&mut mock, 0)
        .with_block_size(64)
        .download(&firmware, &mut readback)
        .unwrap();
    assert_eq!(report.download_blocks, 5);
    assert_eq!(mock.dfu_flash, firmware);
    let dnloads: Vec<usize> = requests(&mock)
        .iter()
        .filter(|(t, r, ..)| *t == 0x21 && *r == 0x01)
        .map(|(_, _, _, _, len)| *len)
        .collect();
    assert_eq!(dnloads, [64, 64, 64, 64, 44, 0]);
}

#[test]
fn every_dfu_transfer_carries_a_timeout() {
    let firmware = ramp(600);
    let mut mock = Mock::new();
    let mut readback = vec![0u8; firmware.len()];
    Dfu::new(&mut mock, 0)
        .with_timeout_us(12_345)
        .download(&firmware, &mut readback)
        .unwrap();
    assert!(mock.log.iter().all(|x| match x {
        Xfer::ControlIn { timeout_us, .. } | Xfer::ControlOut { timeout_us, .. } =>
            *timeout_us == 12_345,
        _ => false,
    }));
}
