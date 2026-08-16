//! Executable-reader tests against a synthetic FlatBuffer fixture, plus an
//! end-to-end mock inference.
//!
//! The fixture is built by [`Fb`], a ~100-line FlatBuffers writer that emits
//! the same vtable/offset encoding the real compiler does. Building the buffer
//! from the schema's `VT_*` constants and then reading it back with the crate's
//! own walker checks the two halves against each other; `super::model` checks
//! the walker against a real `.tflite` produced by Google's compiler, which is
//! what rules out a shared misunderstanding of the format.

use std::vec;
use std::vec::Vec;

use super::Mock;
use crate::{
    DataType, Description, Driver, Error, Executable, ExecutableType, Hint, Layer, LayoutError,
    Package, Timeouts,
};

// ---------------------------------------------------------------------------
// A minimal FlatBuffers writer
// ---------------------------------------------------------------------------

/// One table field value.
pub enum F {
    U8(u8),
    I16(i16),
    I32(i32),
    U64(u64),
    F32(f32),
    /// A `uoffset32` to another object, patched once its position is known.
    Off(usize),
}

impl F {
    fn size(&self) -> usize {
        match self {
            F::U8(_) => 1,
            F::I16(_) => 2,
            F::I32(_) | F::F32(_) | F::Off(_) => 4,
            F::U64(_) => 8,
        }
    }
}

/// A forward-writing FlatBuffers builder.
///
/// Real FlatBuffers builders write back to front so that every offset points
/// forward. This one writes front to back and patches offsets at the end, which
/// produces an equally valid buffer as long as every object is emitted before
/// the objects it points at, which is how the fixtures below are ordered.
#[derive(Default)]
pub struct Fb {
    buf: Vec<u8>,
    /// `(position of an offset field, id of its target)`.
    patches: Vec<(usize, usize)>,
    /// Absolute position of each reserved id.
    targets: Vec<usize>,
}

impl Fb {
    pub fn new() -> Self {
        Fb {
            // Reserve the root uoffset32.
            buf: vec![0; 4],
            ..Default::default()
        }
    }

    /// Reserves an id for an object that will be emitted later.
    pub fn id(&mut self) -> usize {
        self.targets.push(usize::MAX);
        self.targets.len() - 1
    }

    fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Emits a table with the given `(VT offset, value)` fields.
    pub fn table(&mut self, id: usize, fields: Vec<(u16, F)>) {
        let max_vt = fields.iter().map(|(o, _)| *o).max().unwrap_or(2);
        let vt_bytes = usize::from(max_vt) + 2;
        let mut slots = vec![0u16; (vt_bytes - 4) / 2];
        let mut inline = 4usize;
        for (vt, f) in &fields {
            slots[usize::from((*vt - 4) / 2)] = inline as u16;
            inline += f.size();
        }
        let vp = self.buf.len();
        self.u16(vt_bytes as u16);
        self.u16(inline as u16);
        for s in slots {
            self.u16(s);
        }
        let tp = self.buf.len();
        self.targets[id] = tp;
        self.u32((tp - vp) as u32);
        for (_, f) in fields {
            match f {
                F::U8(v) => self.buf.push(v),
                F::I16(v) => self.u16(v as u16),
                F::I32(v) => self.u32(v as u32),
                F::F32(v) => self.u32(v.to_bits()),
                F::U64(v) => self.buf.extend_from_slice(&v.to_le_bytes()),
                F::Off(target) => {
                    let pos = self.buf.len();
                    self.patches.push((pos, target));
                    self.u32(0);
                }
            }
        }
    }

    /// Emits a `[ubyte]` vector.
    pub fn vector_u8(&mut self, id: usize, data: &[u8]) {
        self.targets[id] = self.buf.len();
        self.u32(data.len() as u32);
        self.buf.extend_from_slice(data);
    }

    /// Emits a `string`, which is a `[ubyte]` vector with a NUL terminator.
    pub fn string(&mut self, id: usize, s: &[u8]) {
        self.vector_u8(id, s);
        self.buf.push(0);
    }

    /// Emits an `[int]` vector.
    pub fn vector_i32(&mut self, id: usize, data: &[i32]) {
        self.targets[id] = self.buf.len();
        self.u32(data.len() as u32);
        for v in data {
            self.u32(*v as u32);
        }
    }

    /// Emits a vector of offsets to tables or strings.
    pub fn vector_offsets(&mut self, id: usize, elements: &[usize]) {
        self.targets[id] = self.buf.len();
        self.u32(elements.len() as u32);
        for e in elements {
            let pos = self.buf.len();
            self.patches.push((pos, *e));
            self.u32(0);
        }
    }

    /// Resolves every offset and returns the finished buffer.
    pub fn finish(mut self, root: usize) -> Vec<u8> {
        let root_pos = self.targets[root];
        self.buf[..4].copy_from_slice(&(root_pos as u32).to_le_bytes());
        for (pos, target) in core::mem::take(&mut self.patches) {
            let t = self.targets[target];
            assert!(t != usize::MAX, "undefined target {target}");
            assert!(t >= pos, "offset must point forward: {pos} -> {t}");
            let rel = (t - pos) as u32;
            self.buf[pos..pos + 4].copy_from_slice(&rel.to_le_bytes());
        }
        self.buf
    }
}

/// Wraps `package` the way TFLite wraps an `edgetpu-custom-op`'s options: a
/// FlexBuffers map whose key `"4"` holds the serialised `Package`.
///
/// Every width is four bytes, which keeps the layout uniform and exercises the
/// same code path a real model does.
pub fn flex_wrap(package: &[u8]) -> Vec<u8> {
    let n = package.len();
    let mut b = Vec::new();
    b.extend_from_slice(b"4\0"); // key at 0, NUL-terminated
    b.extend_from_slice(&[0, 0]); // padding to a 4-byte boundary
    b.extend_from_slice(&(n as u32).to_le_bytes()); // string length at 4
    b.extend_from_slice(package); // string data at 8
    b.push(0); // string NUL terminator
    let keys_len_pos = b.len();
    b.extend_from_slice(&1u32.to_le_bytes()); // keys vector length
    let keys_data = b.len();
    // Key element: a *backward* offset from its own position to the key bytes.
    b.extend_from_slice(&(keys_data as u32).to_le_bytes());
    debug_assert_eq!(keys_len_pos + 4, keys_data);

    let values = b.len() + 12;
    b.extend_from_slice(&((values - 12 - keys_data) as u32).to_le_bytes()); // keys offset
    b.extend_from_slice(&4u32.to_le_bytes()); // keys byte width
    b.extend_from_slice(&1u32.to_le_bytes()); // map length
    b.extend_from_slice(&((values - 8) as u32).to_le_bytes()); // value 0 -> string
    b.push((5 << 2) | 2); // packed type: FBT_STRING, 4-byte width

    let root = b.len();
    b.extend_from_slice(&((root - values) as u32).to_le_bytes());
    b.push((9 << 2) | 2); // packed root type: FBT_MAP, 4-byte width
    b.push(4); // root byte width
    b
}

// ---------------------------------------------------------------------------
// The fixture
// ---------------------------------------------------------------------------

/// Byte pattern of the fixture's parameter blob.
pub const PARAMETERS: [u8; 8] = [0xB0, 0xB1, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7];
/// Instruction chunk 0 of the fixture.
pub const CHUNK0: [u8; 3] = [0xC0, 0xC1, 0xC2];
/// Instruction chunk 1 of the fixture.
pub const CHUNK1: [u8; 2] = [0xD0, 0xD1];

/// Builds an `Executable` covering every construct the crate reads.
///
/// * one signed input layer `in`, so the sign-bit fixup fires;
/// * two output layers, `flat` (a one-dimensional, per-execution-padded vector)
///   and `tiled` (a 2x2 tile-scattered plane);
/// * two instruction chunks;
/// * hints in a deliberately interleaved order, including a
///   `BASE_ADDRESS_SCRATCH` one that must be skipped.
pub fn build_executable() -> Vec<u8> {
    let mut fb = Fb::new();
    let exe = fb.id();
    let bitstreams = fb.id();
    let params = fb.id();
    let hints_table = fb.id();
    let inputs = fb.id();
    let outputs = fb.id();

    fb.table(
        exe,
        vec![
            (14, F::Off(bitstreams)),
            (16, F::Off(params)),
            (18, F::Off(hints_table)),
            (20, F::Off(inputs)),
            (22, F::Off(outputs)),
            (30, F::I16(2)), // ExecutableType::EXECUTION_ONLY
            (32, F::U64(0x0123_4567_89AB_CDEF)),
        ],
    );

    // --- instruction bitstreams ---
    let bs0 = fb.id();
    let bs1 = fb.id();
    fb.vector_offsets(bitstreams, &[bs0, bs1]);
    let bs0_data = fb.id();
    let bs1_data = fb.id();
    fb.table(bs0, vec![(4, F::Off(bs0_data))]);
    fb.table(bs1, vec![(4, F::Off(bs1_data))]);

    // --- DMA hints ---
    let hint_vec = fb.id();
    fb.table(hints_table, vec![(4, F::Off(hint_vec))]);
    let h: Vec<usize> = (0..6).map(|_| fb.id()).collect();
    fb.vector_offsets(hint_vec, &h);
    let hint_bodies: Vec<usize> = (0..6).map(|_| fb.id()).collect();
    // AnyHint: 1 = DmaDescriptorHint, 2 = InstructionHint.
    for (i, kind) in [2u8, 1, 1, 2, 1, 1].iter().enumerate() {
        fb.table(h[i], vec![(4, F::U8(*kind)), (6, F::Off(hint_bodies[i]))]);
    }
    let metas: Vec<usize> = (0..6).map(|_| fb.id()).collect();
    // 0: instruction chunk 0
    fb.table(hint_bodies[0], vec![(4, F::I32(0))]);
    // 1: parameters[2..5]
    fb.table(
        hint_bodies[1],
        vec![(4, F::Off(metas[1])), (6, F::I32(2)), (8, F::I32(3))],
    );
    // 2: input activations, whole tensor
    fb.table(
        hint_bodies[2],
        vec![(4, F::Off(metas[2])), (6, F::I32(0)), (8, F::I32(4))],
    );
    // 3: instruction chunk 1
    fb.table(hint_bodies[3], vec![(4, F::I32(1))]);
    // 4: output activations for "tiled"
    fb.table(
        hint_bodies[4],
        vec![(4, F::Off(metas[4])), (6, F::I32(0)), (8, F::I32(112))],
    );
    // 5: scratch, which has no branch in the vendor switch and must be skipped
    fb.table(
        hint_bodies[5],
        vec![(4, F::Off(metas[5])), (6, F::I32(0)), (8, F::I32(64))],
    );

    let name_in = fb.id();
    let name_tiled = fb.id();
    let name_scratch = fb.id();
    // Description: 0 output, 1 input, 2 parameter, 3 scratch.
    fb.table(metas[1], vec![(4, F::I16(2))]);
    fb.table(metas[2], vec![(4, F::I16(1)), (8, F::Off(name_in))]);
    fb.table(metas[4], vec![(4, F::I16(0)), (8, F::Off(name_tiled))]);
    fb.table(metas[5], vec![(4, F::I16(3)), (8, F::Off(name_scratch))]);
    // metas[0] and metas[3] belong to instruction hints and are never used;
    // define them anyway so the builder's forward-reference check passes.
    fb.table(metas[0], vec![(4, F::I16(0))]);
    fb.table(metas[3], vec![(4, F::I16(0))]);

    // --- layers ---
    let in_layer = fb.id();
    fb.vector_offsets(inputs, &[in_layer]);
    let flat_layer = fb.id();
    let tiled_layer = fb.id();
    fb.vector_offsets(outputs, &[flat_layer, tiled_layer]);

    let in_name = fb.id();
    let flat_name = fb.id();
    let tiled_name = fb.id();
    let numerics = fb.id();
    // DataType 8 = SIGNED_FIXED_POINT8, so the sign fixup applies.
    fb.table(
        in_layer,
        vec![
            (4, F::Off(in_name)),
            (6, F::I32(4)),  // size_bytes
            (8, F::I32(1)),  // y_dim
            (10, F::I32(2)), // x_dim
            (12, F::I32(2)), // z_dim
            (16, F::I16(8)), // data_type
        ],
    );
    // "flat": x = y = 1, so Relayout takes the padding-stripping branch.
    // size_bytes 6 against an actual 4 per execution, twice.
    fb.table(
        flat_layer,
        vec![
            (4, F::Off(flat_name)),
            (6, F::I32(6)),
            (8, F::I32(1)),
            (10, F::I32(1)),
            (12, F::I32(4)),
            (14, F::Off(numerics)),
            (16, F::I16(0)), // FIXED_POINT8
            (22, F::I32(2)), // execution_count_per_inference
        ],
    );
    // "tiled": 2x2x1, scattered over two x-tiles.
    let output_layer_union = fb.id();
    fb.table(
        tiled_layer,
        vec![
            (4, F::Off(tiled_name)),
            (6, F::I32(112)),
            (8, F::I32(2)),
            (10, F::I32(2)),
            (12, F::I32(1)),
            (16, F::I16(0)),
            (18, F::U8(1)), // AnyLayer::OutputLayer
            (20, F::Off(output_layer_union)),
        ],
    );
    let layout = fb.id();
    fb.table(output_layer_union, vec![(4, F::Off(layout))]);
    let (m0, m1, m2, m3, m4, m5) = (fb.id(), fb.id(), fb.id(), fb.id(), fb.id(), fb.id());
    fb.table(
        layout,
        vec![
            (4, F::Off(m0)),
            (6, F::Off(m1)),
            (8, F::Off(m2)),
            (10, F::Off(m3)),
            (12, F::Off(m4)),
            (14, F::Off(m5)),
        ],
    );
    fb.table(numerics, vec![(4, F::I32(-128)), (6, F::F32(0.25))]);

    // --- leaf data ---
    fb.vector_u8(params, &PARAMETERS);
    fb.vector_u8(bs0_data, &CHUNK0);
    fb.vector_u8(bs1_data, &CHUNK1);
    fb.string(name_in, b"in");
    fb.string(name_tiled, b"tiled");
    fb.string(name_scratch, b"scratch");
    fb.string(in_name, b"in");
    fb.string(flat_name, b"flat");
    fb.string(tiled_name, b"tiled");
    fb.vector_i32(m0, &[0, 0]); // y -> linear tile id
    fb.vector_i32(m1, &[0, 1]); // x -> linear tile id
    fb.vector_i32(m2, &[0, 100]); // linearized tile byte offset
    fb.vector_i32(m3, &[0, 0]); // x -> local byte offset
    fb.vector_i32(m4, &[0, 1]); // y -> local y offset
    fb.vector_i32(m5, &[8, 8]); // x -> local y row size

    fb.finish(exe)
}

/// Wraps [`build_executable`] in a `MultiExecutable` and a `Package`.
pub fn build_package() -> Vec<u8> {
    let exe_bytes = build_executable();

    let mut mfb = Fb::new();
    let multi = mfb.id();
    let list = mfb.id();
    let entry = mfb.id();
    mfb.table(multi, vec![(4, F::Off(list))]);
    mfb.vector_offsets(list, &[entry]);
    mfb.string(entry, &exe_bytes);
    let multi_bytes = mfb.finish(multi);

    let mut pfb = Fb::new();
    let package = pfb.id();
    let payload = pfb.id();
    pfb.table(package, vec![(6, F::Off(payload))]);
    pfb.vector_u8(payload, &multi_bytes);
    pfb.finish(package)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn executable_scalar_and_vector_fields() {
    let bytes = build_executable();
    let exe = Executable::from_bytes(&bytes).unwrap();
    assert_eq!(exe.executable_type(), ExecutableType::ExecutionOnly);
    assert_eq!(exe.parameter_caching_token(), 0x0123_4567_89AB_CDEF);
    assert_eq!(exe.parameters(), &PARAMETERS);
    assert_eq!(exe.instruction_bitstream_count(), 2);
    assert_eq!(exe.instruction_bitstream(0).unwrap(), &CHUNK0);
    assert_eq!(exe.instruction_bitstream(1).unwrap(), &CHUNK1);
    assert_eq!(exe.instruction_bitstream(2), None);
    // The fixture carries no field_offsets.
    assert_eq!(exe.field_offset_count(0), 0);
}

#[test]
fn executable_layers() {
    let bytes = build_executable();
    let exe = Executable::from_bytes(&bytes).unwrap();
    assert_eq!(exe.input_layer_count(), 1);
    assert_eq!(exe.output_layer_count(), 2);
    assert_eq!(exe.find_input_layer(b"in"), Some(0));
    assert_eq!(exe.find_output_layer(b"flat"), Some(0));
    assert_eq!(exe.find_output_layer(b"tiled"), Some(1));
    assert_eq!(exe.find_output_layer(b"absent"), None);

    let flat = exe.output_layer(0).unwrap();
    assert_eq!(flat.name(), b"flat");
    assert_eq!(flat.size_bytes(), 6);
    assert_eq!((flat.x_dim(), flat.y_dim(), flat.z_dim()), (1, 1, 4));
    assert_eq!(flat.data_type(), DataType::FixedPoint8);
    assert_eq!(flat.execution_count_per_inference(), 2);
    assert_eq!(flat.actual_size_bytes(), 8);
    assert_eq!(flat.padded_size_bytes(), 12);
    let numerics = flat.numerics().unwrap();
    assert_eq!(numerics.zero_point, -128);
    assert_eq!(numerics.dequantization_factor, 0.25);

    // execution_count_per_inference defaults to 1 when absent.
    let tiled = exe.output_layer(1).unwrap();
    assert_eq!(tiled.execution_count_per_inference(), 1);
    assert_eq!(tiled.numerics(), None);
}

#[test]
fn executable_hints_in_array_order() {
    let bytes = build_executable();
    let exe = Executable::from_bytes(&bytes).unwrap();
    let hints: Vec<Hint> = exe.hints().collect();
    assert_eq!(
        hints,
        [
            Hint::Instruction { chunk_index: 0 },
            Hint::Dma {
                desc: Description::Parameter,
                name: b"",
                offset: 2,
                size: 3
            },
            Hint::Dma {
                desc: Description::InputActivation,
                name: b"in",
                offset: 0,
                size: 4
            },
            Hint::Instruction { chunk_index: 1 },
            Hint::Dma {
                desc: Description::OutputActivation,
                name: b"tiled",
                offset: 0,
                size: 112
            },
            Hint::Dma {
                desc: Description::Scratch,
                name: b"scratch",
                offset: 0,
                size: 64
            },
        ]
    );
}

#[test]
fn package_unwraps_flexbuffer_then_two_flatbuffer_layers() {
    let package_bytes = build_package();
    let custom_options = flex_wrap(&package_bytes);
    let package = Package::from_custom_op(&custom_options).unwrap();
    assert_eq!(package.multi_executable().unwrap().len(), 1);
    let exe = package.inference_executable().unwrap();
    assert_eq!(exe.executable_type(), ExecutableType::ExecutionOnly);
    assert_eq!(exe.parameters(), &PARAMETERS);
    // The fixture has no parameter-caching executable, which is legal: only the
    // inference executable is required.
    assert!(package.parameter_caching_executable().is_none());
}

#[test]
fn malformed_buffers_are_rejected_rather_than_panicking() {
    assert!(Package::from_custom_op(&[]).is_none());
    assert!(Package::from_custom_op(&[0; 3]).is_none());
    assert!(Executable::from_bytes(&[0xFF; 8]).is_none());
    // A truncated but structurally plausible buffer must not read out of range.
    let bytes = build_executable();
    for cut in [4, 16, 32, bytes.len() / 2] {
        let exe = Executable::from_bytes(&bytes[..cut]);
        if let Some(exe) = exe {
            let _ = exe.parameters();
            let _ = exe.hint_count();
            let _ = exe.output_layer_count();
        }
    }
}

#[test]
fn data_type_sizes_and_signedness() {
    for (t, size, signed) in [
        (DataType::FixedPoint8, 1, false),
        (DataType::FixedPoint16, 2, false),
        (DataType::SignedFixedPoint32, 4, false),
        (DataType::BFloat, 2, false),
        (DataType::Half, 2, false),
        (DataType::Single, 4, false),
        (DataType::SignedFixedPoint8, 1, true),
        (DataType::SignedFixedPoint16, 2, true),
    ] {
        assert_eq!(t.size_bytes(), size, "{t:?}");
        assert_eq!(t.is_signed(), signed, "{t:?}");
    }
    // SIGNED_FIXED_POINT32 is not treated as signed despite its name; the
    // vendor source flags this as a suspected bug and this port keeps it.
    assert!(!DataType::SignedFixedPoint32.is_signed());
    assert_eq!(DataType::Unknown(42).size_bytes(), 0);
}

#[test]
fn sign_transform_flips_the_msb_of_each_element() {
    // XOR 128 into the last byte of each little-endian element.
    let mut buf = [0x00, 0x01, 0x02, 0x03];
    Layer::transform_signed_data_type_raw(&mut buf, 2, 2, 1, 1);
    assert_eq!(buf, [0x00, 0x81, 0x02, 0x83]);
    // Applying it twice restores the original, which is why the same helper
    // serves both directions.
    Layer::transform_signed_data_type_raw(&mut buf, 2, 2, 1, 1);
    assert_eq!(buf, [0x00, 0x01, 0x02, 0x03]);
}

#[test]
fn sign_transform_is_skipped_for_unsigned_layers_and_short_buffers() {
    let bytes = build_executable();
    let exe = Executable::from_bytes(&bytes).unwrap();
    let unsigned = exe.output_layer(0).unwrap();
    let mut buf = [0x11u8; 8];
    unsigned.transform_signed_data_type(&mut buf);
    assert_eq!(buf, [0x11; 8]);

    // A signed layer with a buffer shorter than actual_size_bytes is a silent
    // no-op in the vendor code.
    let signed = exe.input_layer(0).unwrap();
    let mut short = [0x11u8; 2];
    signed.transform_signed_data_type(&mut short);
    assert_eq!(short, [0x11; 2]);
    let mut full = [0x11u8; 4];
    signed.transform_signed_data_type(&mut full);
    assert_eq!(full, [0x91; 4]);
}

#[test]
fn relayout_strips_per_execution_padding_for_one_dimensional_outputs() {
    // "flat" is 4 bytes of real data per execution inside a 6-byte slot, run
    // twice.
    let bytes = build_executable();
    let exe = Executable::from_bytes(&bytes).unwrap();
    let flat = exe.output_layer(0).unwrap();
    let src: Vec<u8> = (0..12).collect();
    let mut dst = [0u8; 8];
    flat.relayout_into(&src, &mut dst).unwrap();
    assert_eq!(dst, [0, 1, 2, 3, 6, 7, 8, 9]);
}

#[test]
fn relayout_de_interleaves_a_tiled_output() {
    // The fixture's maps put (y,x) at source indices 0, 100, 8, 108, and
    // z_bytes == 1 selects the vendor's hardcoded padded stride of 4.
    let bytes = build_executable();
    let exe = Executable::from_bytes(&bytes).unwrap();
    let tiled = exe.output_layer(1).unwrap();
    let src: Vec<u8> = (0..200).map(|i| i as u8).collect();
    let mut dst = [0u8; 4];
    tiled.relayout_into(&src, &mut dst).unwrap();
    assert_eq!(dst, [0, 100, 8, 108]);
}

#[test]
fn relayout_reports_a_too_small_destination() {
    let bytes = build_executable();
    let exe = Executable::from_bytes(&bytes).unwrap();
    let tiled = exe.output_layer(1).unwrap();
    let src = vec![0u8; 200];
    let mut dst = [0u8; 2];
    assert_eq!(
        tiled.relayout_into(&src, &mut dst),
        Err(LayoutError::BufferTooSmall {
            needed: 4,
            given: 2
        })
    );
    // And converts into the crate error type for use with `?`.
    let converted: Error<super::MockError> = LayoutError::BufferTooSmall {
        needed: 4,
        given: 2,
    }
    .into();
    assert_eq!(
        converted,
        Error::BufferTooSmall {
            needed: 4,
            given: 2
        }
    );
}

// ---------------------------------------------------------------------------
// End-to-end inference
// ---------------------------------------------------------------------------

#[test]
fn invoke_streams_everything_in_hint_order_then_waits_for_completion() {
    let bytes = build_executable();
    let exe = Executable::from_bytes(&bytes).unwrap();

    let mut mock = Mock::new();
    mock.event_data = vec![0u8; 16];
    // 112 bytes of ramp for the "tiled" output.
    mock.bulk_in_data = vec![(0..112).map(|i| i as u8).collect()];
    let mut d = Driver::new(mock, Timeouts::DEFAULT);

    let mut input = [0x00, 0x10, 0x20, 0x30];
    let mut tiled_staging = vec![0u8; 112];
    let mut flat_staging = vec![0u8; 12];
    {
        let mut outputs: [&mut [u8]; 2] = [&mut flat_staging, &mut tiled_staging];
        d.invoke(&exe, &mut input, &mut outputs).unwrap();
    }

    // The input tensor is sign-flipped in place before it goes out, because
    // the "in" layer is SIGNED_FIXED_POINT8.
    assert_eq!(input, [0x80, 0x90, 0xA0, 0xB0]);

    let writes = d.transport().bulk_writes();
    assert_eq!(
        writes,
        vec![
            // chunk 0, tagged kInstructions
            vec![3, 0, 0, 0, 0, 0, 0, 0],
            CHUNK0.to_vec(),
            // parameters[2..5], tagged kParameters
            vec![3, 0, 0, 0, 2, 0, 0, 0],
            PARAMETERS[2..5].to_vec(),
            // the whole input, tagged kInputActivations
            vec![4, 0, 0, 0, 1, 0, 0, 0],
            vec![0x80, 0x90, 0xA0, 0xB0],
            // chunk 1
            vec![2, 0, 0, 0, 0, 0, 0, 0],
            CHUNK1.to_vec(),
        ]
    );
    // The scratch hint produced no traffic at all.
    assert_eq!(writes.len(), 8);

    // The output landed in the staging buffer of the layer the hint named.
    assert_eq!(tiled_staging[..4], [0, 1, 2, 3]);
    assert_eq!(flat_staging, vec![0u8; 12]);

    // Completion is the last thing that happens, on the event endpoint.
    let last = d.transport().log.last().unwrap().clone();
    assert!(matches!(
        last,
        super::Xfer::BulkIn {
            ep: crate::EVENT_ENDPOINT,
            len: 16,
            ..
        }
    ));
}

#[test]
fn invoke_rejects_a_staging_buffer_that_is_too_small() {
    let bytes = build_executable();
    let exe = Executable::from_bytes(&bytes).unwrap();
    let mut mock = Mock::new();
    mock.event_data = vec![0u8; 16];
    let mut d = Driver::new(mock, Timeouts::DEFAULT);

    let mut input = [0u8; 4];
    let mut flat = vec![0u8; 12];
    let mut tiled = vec![0u8; 8];
    let mut outputs: [&mut [u8]; 2] = [&mut flat, &mut tiled];
    assert_eq!(
        d.invoke(&exe, &mut input, &mut outputs).unwrap_err(),
        Error::BufferTooSmall {
            needed: 112,
            given: 8
        }
    );
}
