//! Runtime-driver tests, against `edgetpu_driver.cc`.

use std::vec;
use std::vec::Vec;

use super::{CsrOp, Mock, MockDelay, Xfer};
use crate::csr::{self, RunControl};
use crate::{DescriptorTag, Driver, Error, PerformanceMode, Timeouts};

fn driver(mock: Mock) -> Driver<Mock> {
    Driver::new(mock, Timeouts::DEFAULT)
}

// ---------------------------------------------------------------------------
// CSR encoding
// ---------------------------------------------------------------------------

#[test]
fn csr_read32_encoding() {
    let mut d = driver(Mock::new());
    d.csr_read32(csr::OMC0_00).unwrap();
    assert_eq!(
        d.transport().log[0],
        Xfer::ControlIn {
            // TYPE_VENDOR | RECIPIENT_DEVICE | DIR_IN.
            req_type: 0xC0,
            // bRequest 1 selects a 32-bit register.
            request: 1,
            value: 0xA000,
            index: 0x0001,
            len: 4,
            timeout_us: 200_000,
        }
    );
}

#[test]
fn csr_write32_encoding() {
    let mut d = driver(Mock::new());
    d.csr_write32(csr::SCU_CTRL_3, 0xDEAD_BEEF).unwrap();
    assert_eq!(
        d.transport().log[0],
        Xfer::ControlOut {
            // TYPE_VENDOR | RECIPIENT_DEVICE | DIR_OUT.
            req_type: 0x40,
            request: 1,
            value: 0xA318,
            index: 0x0001,
            data: vec![0xEF, 0xBE, 0xAD, 0xDE],
            timeout_us: 200_000,
        }
    );
}

#[test]
fn csr_read64_encoding() {
    let mut d = driver(Mock::new());
    d.csr_read64(csr::SCALAR_CORE_RUN_CONTROL).unwrap();
    assert_eq!(
        d.transport().log[0],
        Xfer::ControlIn {
            req_type: 0xC0,
            // bRequest 0 selects a 64-bit register.
            request: 0,
            value: 0x4018,
            index: 0x0004,
            len: 8,
            timeout_us: 200_000,
        }
    );
}

#[test]
fn csr_write64_encoding() {
    let mut d = driver(Mock::new());
    d.csr_write64(csr::TILECONFIG0, 0x7F).unwrap();
    assert_eq!(
        d.transport().log[0],
        Xfer::ControlOut {
            req_type: 0x40,
            request: 0,
            value: 0x8788,
            index: 0x0004,
            data: vec![0x7F, 0, 0, 0, 0, 0, 0, 0],
            timeout_us: 200_000,
        }
    );
}

#[test]
fn csr_address_splits_across_wvalue_and_windex() {
    // wValue = reg & 0xFFFF, wIndex = reg >> 16.
    // Bits 32-63 never reach the wire, which caps the CSR space at 32 bits.
    let mut d = driver(Mock::new());
    d.csr_read32(0xDEAD_1234_5678).unwrap();
    match &d.transport().log[0] {
        Xfer::ControlIn { value, index, .. } => {
            assert_eq!(*value, 0x5678);
            assert_eq!(*index, 0x1234);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn csr_round_trip_through_the_register_file() {
    let mut d = driver(Mock::new());
    d.csr_write64(csr::DEEP_SLEEP, csr::DEEP_SLEEP_INIT)
        .unwrap();
    assert_eq!(d.csr_read64(csr::DEEP_SLEEP).unwrap(), 0x1E02);
    d.csr_write32(csr::GCBB_CREDIT0, 0xF).unwrap();
    assert_eq!(d.csr_read32(csr::GCBB_CREDIT0).unwrap(), 0xF);
}

#[test]
fn short_csr_read_is_rejected() {
    // A truncated read must not silently yield a half-filled register.
    let mut mock = Mock::new();
    mock.control_in_short_by = 1;
    let mut d = driver(mock);
    assert_eq!(
        d.csr_read32(csr::OMC0_00).unwrap_err(),
        Error::ShortTransfer {
            expected: 4,
            actual: 3
        }
    );
}

// ---------------------------------------------------------------------------
// Bit-field helpers
// ---------------------------------------------------------------------------

#[test]
fn bitfield_set_preserves_other_bits_and_masks_the_value() {
    assert_eq!(csr::set_field(0x0000_0941, 8, 3, 0), 0x0000_0841);
    assert_eq!(csr::get_field(0x0000_0941, 8, 3), 1);
    // A value wider than the field is masked, not shifted out.
    assert_eq!(csr::set_field(0, 4, 2, 0xFF), 0x30);
}

#[test]
fn idle_register_write_value_matches_the_vendor_read_modify_write() {
    // Bring-up starts from the 0x9000 reset value, clears
    // disable_idle (already clear) and sets counter = 1. counter is
    // Bitfield<0, 31>, so its mask wipes the 0x9000: the word that reaches the
    // wire is 0x1.
    let raw = csr::set_field(0x9000, 31, 1, 0);
    assert_eq!(csr::set_field(raw, 0, 31, 1), csr::IDLE_REGISTER_RUN);
    assert_eq!(csr::IDLE_REGISTER_RUN, 0x1);
}

#[test]
fn deep_sleep_and_tileconfig_constants() {
    // to_sleep_delay = 2, to_wake_delay = 30.
    let raw = csr::set_field(
        0,
        csr::DEEP_SLEEP_TO_SLEEP_DELAY.0,
        csr::DEEP_SLEEP_TO_SLEEP_DELAY.1,
        2,
    );
    let raw = csr::set_field(
        raw,
        csr::DEEP_SLEEP_TO_WAKE_DELAY.0,
        csr::DEEP_SLEEP_TO_WAKE_DELAY.1,
        30,
    );
    assert_eq!(raw, csr::DEEP_SLEEP_INIT);
    // TileConfig<7>::set_broadcast() sets all seven tile bits.
    assert_eq!(csr::TILECONFIG_BROADCAST, 0x7F);
}

// ---------------------------------------------------------------------------
// init()
// ---------------------------------------------------------------------------

/// The complete CSR trace `Initialize` produces, in order, starting from the
/// power-on register values in [`Mock::default`].
fn expected_init_trace() -> Vec<CsrOp> {
    let mut e = vec![
        // 1-2. Chip ID, then the test_reg0 write/read-back self-test.
        CsrOp::R32(csr::OMC0_00),
        CsrOp::W32(csr::OMC0_00, 0x00AA_089A),
        CsrOp::R32(csr::OMC0_00),
        // 3. Disable inactive PHY mode, plus the vendor's discard read.
        CsrOp::R32(csr::SCU_CTRL_0),
        CsrOp::W32(csr::SCU_CTRL_0, 0x0000_0041),
        CsrOp::R32(csr::SCU_CTRL_0),
        // 4. Clock gating off for bring-up, plus its discard read.
        CsrOp::R32(csr::SCU_CTRL_2),
        CsrOp::W32(csr::SCU_CTRL_2, 0x0008_0000),
        CsrOp::R32(csr::SCU_CTRL_2),
        // 5. Into reset: force sleep, poll, pulse the bridge credit.
        CsrOp::R32(csr::SCU_CTRL_3),
        CsrOp::W32(csr::SCU_CTRL_3, 0x80C5_0410),
        CsrOp::R32(csr::SCU_CTRL_3),
        CsrOp::W32(csr::GCBB_CREDIT0, 0xF),
        CsrOp::W32(csr::GCBB_CREDIT0, 0x0),
        // 6. Clocks for PerformanceMode::Max, then out of reset.
        CsrOp::R32(csr::SCU_CTRL_3),
        CsrOp::W32(csr::SCU_CTRL_3, 0x0085_0610),
        CsrOp::R32(csr::SCU_CTRL_3),
        // 7. Reset exit confirmed on a known 64-bit register.
        CsrOp::R64(csr::SCALAR_CORE_RUN_CONTROL),
        // 8. Idle, tile broadcast, deep sleep.
        CsrOp::W64(csr::IDLE_REGISTER, csr::IDLE_REGISTER_RUN),
        CsrOp::W64(csr::TILECONFIG0, csr::TILECONFIG_BROADCAST),
        CsrOp::R64(csr::TILECONFIG0),
        CsrOp::W64(csr::DEEP_SLEEP, csr::DEEP_SLEEP_INIT),
        // 9. Clock gating back on.
        CsrOp::R32(csr::SCU_CTRL_2),
        CsrOp::W32(csr::SCU_CTRL_2, 0x0004_0000),
        // 10. USB HIB: single-endpoint framing.
        CsrOp::W64(csr::USB_DESCR_EP, 0xF0),
        CsrOp::W64(csr::USB_MULTI_BO_EP, 0),
        CsrOp::W64(csr::USB_OUTFEED_CHUNK_LENGTH, 0x20),
        // 11. Temperature sensor: clock, input ports, 100 us, flow.
        CsrOp::R32(csr::OMC0_D0),
        CsrOp::W32(csr::OMC0_D0, 0x0000_0C80),
        CsrOp::R32(csr::OMC0_D8),
        CsrOp::W32(csr::OMC0_D8, 0x0000_0007),
        CsrOp::R32(csr::OMC0_DC),
        CsrOp::W32(csr::OMC0_DC, 0x0000_0001),
    ];
    // 12. DoRunControl(kMoveToRun).
    e.extend(expected_run_control_trace(RunControl::MoveToRun));
    e
}

fn expected_run_control_trace(state: RunControl) -> Vec<CsrOp> {
    let v = state as u64;
    let mut e = Vec::new();
    for reg in csr::SCALAR_RUN_CONTROL_SEQUENCE {
        e.push(CsrOp::W64(reg, v));
    }
    e.push(CsrOp::W64(csr::TILECONFIG0, csr::TILECONFIG_BROADCAST));
    e.push(CsrOp::R64(csr::TILECONFIG0));
    for reg in csr::TILE_RUN_CONTROL_SEQUENCE {
        e.push(CsrOp::W64(reg, v));
    }
    e
}

#[test]
fn init_emits_the_exact_vendor_register_sequence() {
    let mut d = driver(Mock::new());
    let mut delay = MockDelay::default();
    d.init(PerformanceMode::Max, &mut delay).unwrap();
    assert_eq!(d.transport().csr_ops(), expected_init_trace());
}

#[test]
fn init_sequence_length() {
    let mut d = driver(Mock::new());
    let mut delay = MockDelay::default();
    d.init(PerformanceMode::Max, &mut delay).unwrap();
    // 33 CSR operations of bring-up plus 17 of DoRunControl.
    assert_eq!(d.transport().csr_ops().len(), 50);
    assert_eq!(expected_run_control_trace(RunControl::MoveToRun).len(), 17);
}

#[test]
fn init_waits_100_us_before_enabling_the_tempsense_flow() {
    // The only hardware settling delay in bring-up.
    let mut d = driver(Mock::new());
    let mut delay = MockDelay::default();
    d.init(PerformanceMode::Max, &mut delay).unwrap();
    assert_eq!(delay.waits_ns, vec![100_000]);
}

#[test]
fn init_programs_the_clock_triplet_for_each_performance_mode() {
    // gcb divider 0/1/2/3 selects 500/250/125/63 MHz; rg_axi_clk_125m is 1 for
    // 125 MHz; rg_8051_clk_250m is 1 for 250 MHz.
    for (mode, expected) in [
        (PerformanceMode::Max, 0x0085_0610u32),
        (PerformanceMode::High, 0x5085_0610),
        (PerformanceMode::Medium, 0x6085_0610),
        (PerformanceMode::Low, 0xF085_0610),
    ] {
        let mut d = driver(Mock::new());
        let mut delay = MockDelay::default();
        d.init(mode, &mut delay).unwrap();
        // The second scu_ctrl_3 write is the exit-reset/clock one.
        let writes: Vec<u32> = d
            .transport()
            .csr_ops()
            .into_iter()
            .filter_map(|op| match op {
                CsrOp::W32(r, v) if r == csr::SCU_CTRL_3 => Some(v),
                _ => None,
            })
            .collect();
        assert_eq!(writes[1], expected, "mode {mode:?}");
        assert_eq!(
            csr::get_field(
                u64::from(writes[1]),
                csr::SCU_CTRL_3_FORCE_SLEEP.0,
                csr::SCU_CTRL_3_FORCE_SLEEP.1
            ),
            csr::FORCE_SLEEP_RUN
        );
    }
}

#[test]
fn init_skips_the_force_sleep_block_when_already_in_reset() {
    // The whole entry-into-reset block is conditional on rg_force_sleep != 3.
    let mut mock = Mock::new();
    mock.regs.insert(csr::SCU_CTRL_3, 0x80C5_0610);
    let mut d = driver(mock);
    let mut delay = MockDelay::default();
    d.init(PerformanceMode::Max, &mut delay).unwrap();

    let ops = d.transport().csr_ops();
    assert!(!ops.contains(&CsrOp::W32(csr::GCBB_CREDIT0, 0xF)));
    assert!(!ops.contains(&CsrOp::W32(csr::SCU_CTRL_3, 0x80C5_0410)));
    // Four fewer operations than the full path: one write, one poll read and
    // the two credit writes.
    assert_eq!(ops.len(), 50 - 4);
}

#[test]
fn init_rejects_a_wrong_chip_id() {
    let mut mock = Mock::new();
    mock.regs.insert(csr::OMC0_00, 0x0123);
    let mut d = driver(mock);
    let mut delay = MockDelay::default();
    assert_eq!(
        d.init(PerformanceMode::Max, &mut delay).unwrap_err(),
        Error::ChipId { found: 0x123 }
    );
    // Nothing is written before the identity is confirmed.
    assert_eq!(d.transport().csr_ops().len(), 1);
}

#[test]
fn init_rejects_a_failed_control_path_self_test() {
    // omc0_00 accepts reads but drops writes: the read direction works and the
    // write direction does not.
    let mut mock = Mock::new();
    mock.readonly_regs.push(csr::OMC0_00);
    let mut d = driver(mock);
    let mut delay = MockDelay::default();
    assert_eq!(
        d.init(PerformanceMode::Max, &mut delay).unwrap_err(),
        Error::SelfTest { found: 0 }
    );
}

#[test]
fn init_polls_are_bounded() {
    // A chip that never wakes must fail, not spin forever.
    let mut mock = Mock::new();
    mock.power_state_follows = false;
    mock.regs.insert(csr::SCU_CTRL_3, 0x80C5_0610);
    let mut d = Driver::new(
        mock,
        Timeouts {
            poll_attempts: 5,
            ..Timeouts::DEFAULT
        },
    );
    let mut delay = MockDelay::default();
    assert_eq!(
        d.init(PerformanceMode::Max, &mut delay).unwrap_err(),
        Error::PollTimeout("scu_ctrl_3.cur_pwr_state == active")
    );
    // Exactly the bound, not one more: two non-poll reads of scu_ctrl_3 (the
    // force-sleep test at step 5 and the read-modify-write at step 6) plus the
    // five allowed poll reads.
    let reads = d
        .transport()
        .csr_ops()
        .into_iter()
        .filter(|op| *op == CsrOp::R32(csr::SCU_CTRL_3))
        .count();
    assert_eq!(reads, 2 + 5);
}

#[test]
fn run_control_write_order() {
    let mut d = driver(Mock::new());
    d.run_control(RunControl::MoveToIdle).unwrap();
    assert_eq!(
        d.transport().csr_ops(),
        expected_run_control_trace(RunControl::MoveToIdle)
    );
    // The re-broadcast of tileconfig0 sits between the scalar-core writes and
    // the tile writes because "hardware does not guarantee correct ordering".
    let ops = d.transport().csr_ops();
    assert_eq!(ops[5], CsrOp::W64(csr::TILECONFIG0, 0x7F));
    assert_eq!(ops[6], CsrOp::R64(csr::TILECONFIG0));
    assert_eq!(ops[7], CsrOp::W64(csr::OP_RUN_CONTROL, 0));
}

#[test]
fn temperature_uses_the_vendor_transfer_function() {
    // (662 - data) * 250 + 550 millidegrees.
    let mut mock = Mock::new();
    mock.regs.insert(
        csr::OMC0_DC,
        csr::set_field(0, csr::OMC0_DC_DATA.0, csr::OMC0_DC_DATA.1, 500),
    );
    let mut d = driver(mock);
    assert_eq!(
        d.temperature_millicelsius().unwrap(),
        (662 - 500) * 250 + 550
    );
}

// ---------------------------------------------------------------------------
// Bulk framing
// ---------------------------------------------------------------------------

#[test]
fn header_layout() {
    // The length as a little-endian u32 in bytes 0-3, the tag's low nibble in
    // byte 4, zero padding to eight bytes.
    let h = Driver::<Mock>::build_header(DescriptorTag::Parameters, 0x0102_0304);
    assert_eq!(h, [0x04, 0x03, 0x02, 0x01, 2, 0, 0, 0]);
    assert_eq!(
        Driver::<Mock>::build_header(DescriptorTag::Instructions, 1)[4],
        0
    );
    assert_eq!(
        Driver::<Mock>::build_header(DescriptorTag::InputActivations, 1)[4],
        1
    );
    // Only the low nibble is used, so the interrupt tags stay in range.
    assert_eq!(
        Driver::<Mock>::build_header(DescriptorTag::Interrupt3, 1)[4],
        7
    );
}

#[test]
fn send_data_is_a_header_transfer_then_a_payload_transfer() {
    // Two separate bulk transfers, not one.
    let mut d = driver(Mock::new());
    d.send_instructions(&[0xAA, 0xBB, 0xCC]).unwrap();
    assert_eq!(
        d.transport().bulk_writes(),
        vec![vec![3, 0, 0, 0, 0, 0, 0, 0], vec![0xAA, 0xBB, 0xCC]]
    );
    // Everything goes out on endpoint 1.
    assert!(d.transport().log.iter().all(|x| matches!(
        x,
        Xfer::BulkOut {
            ep: crate::BULK_OUT_ENDPOINT,
            ..
        }
    )));
}

#[test]
fn stream_tags() {
    let mut d = driver(Mock::new());
    d.send_instructions(&[1]).unwrap();
    d.send_inputs(&[2]).unwrap();
    d.send_parameters(&[3]).unwrap();
    let tags: Vec<u8> = d
        .transport()
        .bulk_writes()
        .iter()
        .step_by(2)
        .map(|h| h[4])
        .collect();
    assert_eq!(tags, [0, 1, 2]);
}

#[test]
fn bulk_out_chunks_at_thirty_two_kib() {
    // kMaxBulkBufferSize = 32 * 1024.
    let payload = vec![0x5Au8; 40_000];
    let mut d = driver(Mock::new());
    d.send_parameters(&payload).unwrap();
    let sizes: Vec<usize> = d.transport().bulk_writes().iter().map(Vec::len).collect();
    assert_eq!(sizes, [8, 32_768, 7_232]);
}

#[test]
fn get_outputs_sends_no_header_and_reassembles_short_reads() {
    // Outputs come back raw, and the device may answer a request in more than
    // one transfer.
    let mut mock = Mock::new();
    mock.bulk_in_data = vec![vec![1, 2, 3], vec![4, 5]];
    let mut d = driver(mock);
    let mut out = [0u8; 5];
    d.get_outputs(&mut out).unwrap();
    assert_eq!(out, [1, 2, 3, 4, 5]);
    assert!(!d
        .transport()
        .log
        .iter()
        .any(|x| matches!(x, Xfer::BulkOut { .. })));
}

#[test]
fn a_stalled_bulk_in_does_not_spin() {
    let mut mock = Mock::new();
    mock.bulk_in_data = vec![Vec::new()];
    let mut d = driver(mock);
    let mut out = [0u8; 4];
    assert_eq!(
        d.get_outputs(&mut out).unwrap_err(),
        Error::ShortTransfer {
            expected: 4,
            actual: 0
        }
    );
}

#[test]
fn read_event_parses_the_sixteen_byte_completion_payload() {
    // Eight address bytes, four length bytes, a tag nibble, and three bytes
    // the vendor parse does not account for.
    let mut mock = Mock::new();
    mock.event_data = vec![
        1, 2, 3, 4, 5, 6, 7, 8, // address
        0x10, 0x20, 0, 0,    // length
        0x33, // tag nibble
        0xFF, 0xFF, 0xFF,
    ];
    let mut d = driver(mock);
    let event = d.read_event().unwrap();
    assert_eq!(event.address, 0x0807_0605_0403_0201);
    assert_eq!(event.length, 0x2010);
    assert_eq!(event.tag, 3);
    // It is a bulk read on endpoint 2, not an interrupt transfer.
    assert_eq!(
        d.transport().log[0],
        Xfer::BulkIn {
            ep: crate::EVENT_ENDPOINT,
            len: 16,
            timeout_us: 200_000,
        }
    );
}

#[test]
fn read_event_rejects_a_short_payload() {
    let mut mock = Mock::new();
    mock.event_data = vec![0; 8];
    let mut d = driver(mock);
    assert_eq!(
        d.read_event().unwrap_err(),
        Error::ShortTransfer {
            expected: 16,
            actual: 8
        }
    );
}

#[test]
fn interrupt_endpoint_is_available_but_idle() {
    // The vendor driver never reads this pipe at all; nothing is known to
    // arrive on it.
    let mut d = driver(Mock::new());
    let mut buf = [0u8; 8];
    assert_eq!(d.poll_interrupt(&mut buf, 1_000).unwrap(), None);
    assert_eq!(
        d.transport().log[0],
        Xfer::InterruptIn {
            ep: crate::INTERRUPT_ENDPOINT,
            len: 8,
            timeout_us: 1_000,
        }
    );
}
