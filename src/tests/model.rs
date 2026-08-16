//! Validates the FlatBuffers walker against a real Edge TPU model.
//!
//! The synthetic fixture in `super::executable` proves the reader agrees with
//! the writer in this same file, which would not catch a shared
//! misunderstanding of the format. This module parses a real
//! `*_edgetpu.tflite` produced by Google's Edge TPU compiler, all the way from
//! the TFLite container down to the DarwiNN layer metadata.
//!
//! The model is 6.9 MB, so it is not vendored into this crate: the test reads
//! it from its checked-out path and is `#[ignore]`d so CI stays hermetic.
//!
//! ```text
//! cargo test -- --ignored --nocapture
//! ```

use std::path::Path;
use std::string::String;
use std::vec::Vec;
use std::{eprintln, fs};

use crate::fb;
use crate::{DataType, ExecutableType, Hint, Package};

/// Fallback locations for the model, relative to the crate root. Override with
/// `DARWINN_TEST_MODEL`.
const MODEL_PATHS: [&str; 2] = [
    "../../coral/coralmicro/models/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite",
    "../../out/coral/coralmicro/models/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite",
];

/// The custom operator the Edge TPU compiler emits.
const CUSTOM_OP: &[u8] = b"edgetpu-custom-op";

// TFLite schema vtable offsets, from
// third_party/tflite-micro/tensorflow/lite/schema/schema_generated.h.
const MODEL_OPERATOR_CODES: u16 = 6;
const MODEL_SUBGRAPHS: u16 = 8;
const OPERATOR_CODE_CUSTOM_CODE: u16 = 6;
const SUBGRAPH_OPERATORS: u16 = 10;
const OPERATOR_OPCODE_INDEX: u16 = 4;
const OPERATOR_CUSTOM_OPTIONS: u16 = 14;

/// Pulls the `edgetpu-custom-op` payload out of a `.tflite` file.
///
/// The TFLite container is itself a FlatBuffer, so this reuses the crate's own
/// walker: `Model.operator_codes` is scanned for the custom code, then
/// `subgraphs[0].operators` for an operator using it, and that operator's
/// `custom_options` is the FlexBuffer map [`Package::from_custom_op`] expects.
fn extract_custom_op_payload(model: &[u8]) -> Option<&[u8]> {
    let root = fb::root(model)?;
    let codes = root.vector(MODEL_OPERATOR_CODES)?;
    let mut opcode_index = None;
    for i in 0..codes.len() {
        let code = codes.table(i)?;
        if code.str_bytes(OPERATOR_CODE_CUSTOM_CODE) == Some(CUSTOM_OP) {
            opcode_index = Some(i as i32);
            break;
        }
    }
    let opcode_index = opcode_index?;

    let subgraphs = root.vector(MODEL_SUBGRAPHS)?;
    for s in 0..subgraphs.len() {
        let ops = subgraphs.table(s)?.vector(SUBGRAPH_OPERATORS)?;
        for o in 0..ops.len() {
            let op = ops.table(o)?;
            if op.i32(OPERATOR_OPCODE_INDEX, 0) == opcode_index {
                return op.bytes(OPERATOR_CUSTOM_OPTIONS);
            }
        }
    }
    None
}

fn load_model() -> Option<Vec<u8>> {
    if let Ok(path) = std::env::var("DARWINN_TEST_MODEL") {
        return fs::read(path).ok();
    }
    MODEL_PATHS
        .iter()
        .map(Path::new)
        .find(|p| p.exists())
        .and_then(|p| fs::read(p).ok())
}

#[test]
#[ignore = "needs a real *_edgetpu.tflite model; see DARWINN_TEST_MODEL"]
fn real_model_parses_end_to_end() {
    let Some(model) = load_model() else {
        panic!("model not found; set DARWINN_TEST_MODEL or see {MODEL_PATHS:?}");
    };
    eprintln!("model: {} bytes", model.len());

    let name = |b: &[u8]| String::from_utf8_lossy(b).into_owned();
    let payload = extract_custom_op_payload(&model).expect("no edgetpu-custom-op in the model");
    eprintln!("custom-op payload: {} bytes", payload.len());
    assert!(payload.len() > 1000);

    // --- the FlexBuffer wrapper and the two FlatBuffer layers ---
    let package = Package::from_custom_op(payload).expect("flexbuffer map[\"4\"] -> Package");
    let multi = package.multi_executable().expect("MultiExecutable");
    eprintln!("executables in package: {}", multi.len());
    assert!(!multi.is_empty());

    let exe = package
        .inference_executable()
        .expect("package has no inference executable");
    assert!(matches!(
        exe.executable_type(),
        ExecutableType::ExecutionOnly | ExecutableType::StandAlone
    ));

    // --- instruction bitstreams ---
    let chunks = exe.instruction_bitstream_count();
    eprintln!("instruction chunks: {chunks}");
    assert!(chunks >= 1);
    let mut instruction_bytes = 0usize;
    let mut relocations = 0usize;
    for i in 0..chunks {
        let bs = exe
            .instruction_bitstream(i)
            .unwrap_or_else(|| panic!("chunk {i} missing"));
        assert!(!bs.is_empty(), "chunk {i} is empty");
        instruction_bytes += bs.len();
        relocations += exe.field_offset_count(i);
    }
    eprintln!("instruction bytes: {instruction_bytes}, relocation entries: {relocations}");
    // Real models do carry relocation entries, and the bitstream is sent
    // verbatim without applying any of them. The vendor stack runs this model
    // successfully, so the baked-in addresses must already suit this target.
    // That is inferred from one model working, not from anything in the format.
    assert_eq!(relocations, 10);
    for j in 0..relocations {
        let (bit, desc, n) = exe.field_offset(0, j).unwrap();
        eprintln!("  reloc[{j}] bit {bit} {desc:?} {:?}", name(n));
    }

    // --- parameters ---
    // This package splits into a PARAMETER_CACHING executable that owns the
    // weights and an EXECUTION_ONLY one that runs against them, which is the
    // two-executable case `EdgeTpuManager::Invoke` handles: the inference
    // executable's own parameter blob is empty.
    let params = exe.parameters();
    let caching = package.parameter_caching_executable();
    eprintln!(
        "parameter blob: inference {} bytes, caching {:?} bytes, token {:#x}",
        params.len(),
        caching.map(|c| c.parameters().len()),
        exe.parameter_caching_token()
    );
    if let Some(caching) = caching {
        assert_eq!(caching.executable_type(), ExecutableType::ParameterCaching);
        assert!(!caching.parameters().is_empty());
        // The token is what invalidates the on-device parameter cache.
        assert_ne!(caching.parameter_caching_token(), 0);
        // Its hints are the parameter uploads.
        let param_hints = caching
            .hints()
            .filter(|h| {
                matches!(
                    h,
                    Hint::Dma {
                        desc: crate::Description::Parameter,
                        ..
                    }
                )
            })
            .count();
        eprintln!("caching executable: {param_hints} parameter hints");
        assert!(param_hints > 0);
    } else {
        assert!(!params.is_empty());
    }

    // --- layers ---
    eprintln!("input layers: {}", exe.input_layer_count());
    assert!(exe.input_layer_count() >= 1);
    for i in 0..exe.input_layer_count() {
        let l = exe.input_layer(i).unwrap();
        eprintln!(
            "  in[{i}] {:?} {}x{}x{} {:?} size_bytes={} exec={}",
            name(l.name()),
            l.x_dim(),
            l.y_dim(),
            l.z_dim(),
            l.data_type(),
            l.size_bytes(),
            l.execution_count_per_inference()
        );
        assert!(!l.name().is_empty());
        assert_ne!(
            l.data_type(),
            DataType::Unknown(l.data_type().size_bytes() as i16)
        );
        assert!(l.data_type().size_bytes() > 0, "unknown input data type");
        assert!(l.size_bytes() > 0);
    }

    eprintln!("output layers: {}", exe.output_layer_count());
    assert!(exe.output_layer_count() >= 1);
    for i in 0..exe.output_layer_count() {
        let l = exe.output_layer(i).unwrap();
        eprintln!(
            "  out[{i}] {:?} {}x{}x{} {:?} size_bytes={} exec={} numerics={:?}",
            name(l.name()),
            l.x_dim(),
            l.y_dim(),
            l.z_dim(),
            l.data_type(),
            l.size_bytes(),
            l.execution_count_per_inference(),
            l.numerics()
        );
        assert!(!l.name().is_empty());
        assert!(l.data_type().size_bytes() > 0, "unknown output data type");
        assert!(l.padded_size_bytes() >= l.actual_size_bytes());
        // Every output layer the DMA hints will name must be findable by name.
        assert_eq!(exe.find_output_layer(l.name()), Some(i));
    }

    // --- DMA hints ---
    let mut counts = (0usize, 0usize, 0usize, 0usize, 0usize);
    for hint in exe.hints() {
        match hint {
            Hint::Instruction { chunk_index } => {
                assert!(
                    chunk_index < chunks,
                    "instruction hint indexes chunk {chunk_index} of {chunks}"
                );
                counts.0 += 1;
            }
            Hint::Dma {
                desc: crate::Description::Parameter,
                offset,
                size,
                ..
            } => {
                assert!(
                    offset + size <= params.len(),
                    "parameter hint {offset}+{size} exceeds the {} byte blob",
                    params.len()
                );
                counts.1 += 1;
            }
            Hint::Dma {
                desc: crate::Description::InputActivation,
                name: n,
                ..
            } => {
                assert!(
                    exe.find_input_layer(n).is_some(),
                    "input hint names {:?}, which is not an input layer",
                    name(n)
                );
                counts.2 += 1;
            }
            Hint::Dma {
                desc: crate::Description::OutputActivation,
                name: n,
                size,
                ..
            } => {
                let idx = exe
                    .find_output_layer(n)
                    .unwrap_or_else(|| panic!("output hint names unknown layer {:?}", name(n)));
                let layer = exe.output_layer(idx).unwrap();
                assert!(
                    size <= layer.padded_size_bytes(),
                    "output hint wants {size} bytes into a {} byte layer",
                    layer.padded_size_bytes()
                );
                counts.3 += 1;
            }
            _ => counts.4 += 1,
        }
    }
    eprintln!(
        "hints: {} total -- {} instruction, {} parameter, {} input, {} output, {} skipped",
        exe.hint_count(),
        counts.0,
        counts.1,
        counts.2,
        counts.3,
        counts.4
    );
    assert!(exe.hint_count() > 0);
    assert!(counts.0 > 0, "no instruction hints");
    assert!(counts.3 > 0, "no output hints");

    // A description of what an invoke would actually push, for the record.
    eprintln!(
        "one inference streams {instruction_bytes} instruction bytes over {} hints",
        exe.hint_count()
    );
}

#[test]
#[ignore = "needs a real *_edgetpu.tflite model; see DARWINN_TEST_MODEL"]
fn real_model_relayout_matches_layer_geometry() {
    let Some(model) = load_model() else {
        panic!("model not found; set DARWINN_TEST_MODEL or see {MODEL_PATHS:?}");
    };
    let payload = extract_custom_op_payload(&model).unwrap();
    let package = Package::from_custom_op(payload).unwrap();
    let exe = package.inference_executable().unwrap();

    // Relayout every output layer out of a buffer of the size the device would
    // actually deliver. This exercises the tile maps of a real executable,
    // which the synthetic fixture cannot: it checks that every index the six
    // OutputLayout vectors produce stays inside the padded buffer.
    for i in 0..exe.output_layer_count() {
        let layer = exe.output_layer(i).unwrap();
        let src = std::vec![0u8; layer.padded_size_bytes().max(1)];
        let mut dst = std::vec![0u8; layer.actual_size_bytes().max(1)];
        match layer.relayout_into(&src, &mut dst) {
            Ok(()) => eprintln!("out[{i}] relayout {} -> {} bytes ok", src.len(), dst.len()),
            Err(e) => panic!(
                "out[{i}] ({}) relayout failed: {e:?}",
                String::from_utf8_lossy(layer.name())
            ),
        }
    }
}
