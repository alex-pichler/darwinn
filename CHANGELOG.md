# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Initial release: a `no_std`, transport-agnostic driver for the Google Edge TPU
(DarwiNN) over USB, ported from the TPU half of Google's coralmicro SDK.

### Added

- `Transport`, the five-call USB seam the crate depends on, and `Error`.
- `Dfu`: firmware download in `DFU_BLOCK_SIZE` blocks, read-back verification,
  detach, and `DFU_GETSTATUS` checking. `DfuStatus` and `DfuReport`.
- `Driver`: 32- and 64-bit CSR access, the bring-up sequence through to
  `RunControl::MoveToRun`, `PerformanceMode` clock selection, on-die
  temperature, bulk stream framing, completion events, and `invoke`.
- `Timeouts`, which bounds every blocking operation in the crate.
- `Package`, `MultiExecutable`, `Executable`, `Layer` and `Hint`: a zero-copy
  reader for the consumed subset of the DarwiNN executable FlatBuffer, plus the
  `relayout_into` and `transform_signed_data_type` output helpers.
- The `csr` module: register offsets, bit fields and the run-control sequences.
- Host tests against a mock transport with a behavioural model of the chip, and
  an `--ignored` test that walks a real `*_edgetpu.tflite` model.

### Notes

- MSRV is 1.81, set by `core::error::Error`.
- `#![forbid(unsafe_code)]`. No loop is bounded by anything other than a
  constant or a caller-provided buffer length.
- Not implemented: relocation of `InstructionBitstream.field_offsets`,
  signature and version checks, parameter-caching token bookkeeping, and
  `embedded-hal-async`.

[Unreleased]: https://github.com/alex-pichler/darwinn/commits/main
