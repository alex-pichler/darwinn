# darwinn

A `no_std`, transport-agnostic driver for the Google Edge TPU (DarwiNN)
attached over USB: DFU firmware download, CSR access, DarwiNN executable
parsing and inference.

The Edge TPU is the coprocessor fitted to the Coral Dev Board Micro. This crate
is ported from the TPU half of Google's coralmicro C++ SDK (`libs/tpu`).

```toml
[dependencies]
darwinn = "0.1"
```

## Scope

| | |
|---|---|
| USB host controller | not here: the caller supplies one through `Transport` |
| Apex firmware blob | not here: it is Google's binary, and the application embeds it and passes the bytes to `Dfu::download` |
| DFU | 256-byte `DFU_DNLOAD` blocks, full read-back verification, detach |
| Register access | vendor control transfers, 32- and 64-bit |
| Bring-up | chip ID, self-test, reset, clocks, tiles, temperature sensor, run |
| Data path | bulk endpoint 1 both directions, 8-byte tag+length header outbound, raw inbound |
| Completion | 16-byte read on bulk endpoint 2 |
| Model format | zero-copy reader for the consumed subset of the DarwiNN FlatBuffer |
| Tensor plumbing | out of scope: TFLite integration, allocation, post-processing |

## The transport seam

```rust
pub trait Transport {
    type Error: core::fmt::Debug;
    fn control_in(&mut self, req_type: u8, request: u8, value: u16, index: u16, buf: &mut [u8], timeout_us: u32) -> Result<usize, Self::Error>;
    fn control_out(&mut self, req_type: u8, request: u8, value: u16, index: u16, data: &[u8], timeout_us: u32) -> Result<(), Self::Error>;
    fn bulk_out(&mut self, ep: u8, data: &[u8], timeout_us: u32) -> Result<(), Self::Error>;
    fn bulk_in(&mut self, ep: u8, buf: &mut [u8], timeout_us: u32) -> Result<usize, Self::Error>;
    fn interrupt_in(&mut self, ep: u8, buf: &mut [u8], timeout_us: u32) -> Result<Option<usize>, Self::Error>;
}
```

That is the crate's whole dependency on the outside world, which is what makes
it testable on a desktop. `interrupt_in` is part of the trait but is not on any
path the driver needs; completion arrives on the bulk event endpoint.

## Usage

```rust
use darwinn::{Dfu, Driver, Package, PerformanceMode, Timeouts};

// 1. Cold device (1a6e:089a): push the Apex firmware and detach.
Dfu::new(&mut transport, dfu_interface).download(firmware, &mut readback)?;
// ... the device re-enumerates as 18d1:9302 ...

// 2. Bring the chip out of reset.
let mut tpu = Driver::new(transport, Timeouts::DEFAULT);
tpu.init(PerformanceMode::Max, &mut delay)?;

// 3. Run the executable out of a TFLite custom-op payload.
let package = Package::from_custom_op(custom_options).unwrap();
let exe = package.inference_executable().unwrap();
tpu.invoke(&exe, &mut input, &mut [&mut staging])?;

// 4. De-tile the device's padded output into a flat tensor.
let layer = exe.output_layer(0).unwrap();
layer.relayout_into(&staging, &mut tensor)?;
layer.transform_signed_data_type(&mut tensor);
```

## Bounded waits

Every register poll is bounded by `Timeouts::poll_attempts` and fails with
`Error::PollTimeout` naming the register that stalled; every transfer carries an
explicit microsecond timeout; the bulk reassembly loop rejects a zero-length
read rather than spinning. No loop in this crate is bounded by anything other
than a constant or the length of a caller-provided buffer.

The crate is `#![forbid(unsafe_code)]`, and the FlatBuffer reader returns
`Option` rather than panicking on a malformed model.

## Tests

The host tests run against a mock transport that records every transfer and
answers out of a small behavioural model of the chip: a CSR register file whose
power-state bits follow what is written to its sleep bits, a DFU flash that
stores what is downloaded and serves it back, and queues for bulk and event
data. They pin the `bmRequestType`/`bRequest`/`wValue`/`wIndex` of every CSR
access, the DFU block sizes and status-polling interleave, the bring-up
sequence value by value, the bulk framing and chunking, and an end-to-end mock
inference.

The FlatBuffers walker is checked against a synthetic fixture built by a small
FlatBuffers writer in the test module, and, under `--ignored`, against a real
`*_edgetpu.tflite` model from a coralmicro checkout, walked from the TFLite
container through the FlexBuffer wrapper and both FlatBuffer layers down to
layer geometry and quantisation constants. That model is not vendored here.

```bash
cargo test
cargo test -- --ignored --nocapture   # needs a coralmicro model checkout
cargo clippy --all-targets -- -D warnings
cargo build --target thumbv7em-none-eabihf
```

## Licence

See [LICENSE](LICENSE) and [NOTICE](NOTICE).
