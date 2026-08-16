//! Zero-copy reader for the DarwiNN executable FlatBuffer.
//!
//! Nothing is copied out of the model buffer: every type here is a borrowed
//! view, so a model can live in flash and be run from there.
//!
//! # Container chain
//!
//! What TFLite hands the custom op is a FlexBuffer map, not a FlatBuffer.
//! Unwrapping it takes three steps:
//!
//! ```text
//! flexbuffer map["4"]
//!   -> Package.serialized_multi_executable     -- VT 6
//!     -> MultiExecutable.serialized_executables -- VT 4, vector<string>
//!       -> Executable                           -- one per string
//! ```
//!
//! There is no verifier pass. Every accessor is bounds-checked against the
//! backing slice and returns `None` rather than reading out of range.
//!
//! # What is deliberately not implemented
//!
//! `InstructionBitstream.field_offsets`, the relocation entries for addresses
//! baked into instructions. Bitstreams are sent verbatim;
//! [`Executable::field_offset_count`] exists so a caller can detect an
//! executable that would need relocation.
//!
//! `InterruptHint`, `FenceHint`, `TensorShape`, `TensorLayout` and
//! `OutputShapeInfo` are unused. `NumericsConstants` is exposed, because
//! quantisation parameters are cheap to read and a caller that wants to
//! dequantise otherwise has to re-parse the model.

use crate::fb::{self, FlexMap, Table, Vector};

// --- vtable offsets, from executable_generated.h ---------------------------

// Package.
const PACKAGE_SERIALIZED_MULTI_EXECUTABLE: u16 = 6;
// MultiExecutable.
const MULTI_SERIALIZED_EXECUTABLES: u16 = 4;
// Executable.
const EXE_INSTRUCTION_BITSTREAMS: u16 = 14;
const EXE_PARAMETERS: u16 = 16;
const EXE_DMA_HINTS: u16 = 18;
const EXE_INPUT_LAYERS: u16 = 20;
const EXE_OUTPUT_LAYERS: u16 = 22;
const EXE_TYPE: u16 = 30;
const EXE_PARAMETER_CACHING_TOKEN: u16 = 32;
// InstructionBitstream.
const BITSTREAM_BITSTREAM: u16 = 4;
const BITSTREAM_FIELD_OFFSETS: u16 = 6;
// FieldOffset.
const FIELD_OFFSET_META: u16 = 4;
const FIELD_OFFSET_OFFSET_BIT: u16 = 6;
// DmaHints.
const DMA_HINTS_HINTS: u16 = 4;
// DmaHint.
const HINT_ANY_HINT_TYPE: u16 = 4;
const HINT_ANY_HINT: u16 = 6;
// DmaDescriptorHint.
const DESC_HINT_META: u16 = 4;
const DESC_HINT_OFFSET_IN_BYTES: u16 = 6;
const DESC_HINT_SIZE_IN_BYTES: u16 = 8;
// InstructionHint.
const INSTR_HINT_CHUNK_INDEX: u16 = 4;
// Meta.
const META_DESC: u16 = 4;
const META_NAME: u16 = 8;
// Layer.
const LAYER_NAME: u16 = 4;
const LAYER_SIZE_BYTES: u16 = 6;
const LAYER_Y_DIM: u16 = 8;
const LAYER_X_DIM: u16 = 10;
const LAYER_Z_DIM: u16 = 12;
const LAYER_NUMERICS: u16 = 14;
const LAYER_DATA_TYPE: u16 = 16;
const LAYER_ANY_LAYER_TYPE: u16 = 18;
const LAYER_ANY_LAYER: u16 = 20;
const LAYER_EXECUTION_COUNT_PER_INFERENCE: u16 = 22;
// OutputLayer.
const OUTPUT_LAYER_LAYOUT: u16 = 4;
// OutputLayout.
const LAYOUT_Y_TO_TILE_ID: u16 = 4;
const LAYOUT_X_TO_TILE_ID: u16 = 6;
const LAYOUT_TILE_BYTE_OFFSET: u16 = 8;
const LAYOUT_X_TO_LOCAL_BYTE_OFFSET: u16 = 10;
const LAYOUT_Y_TO_LOCAL_Y_OFFSET: u16 = 12;
const LAYOUT_X_TO_LOCAL_Y_ROW_SIZE: u16 = 14;
// NumericsConstants.
const NUMERICS_ZERO_POINT: u16 = 4;
const NUMERICS_DEQUANTIZATION_FACTOR: u16 = 6;

// AnyLayer.
const ANY_LAYER_OUTPUT_LAYER: u8 = 1;
// AnyHint.
const ANY_HINT_DMA_DESCRIPTOR: u8 = 1;
const ANY_HINT_INSTRUCTION: u8 = 2;

/// FlexBuffer map key under which the serialised `Package` lives.
const CUSTOM_OP_EXECUTABLE_KEY: &[u8] = b"4";

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// What a `DmaDescriptorHint` refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Description {
    /// Output activations, read back from the device.
    OutputActivation,
    /// Input activations, sent to the device.
    InputActivation,
    /// A slice of the executable's parameter blob.
    Parameter,
    /// Scratch memory. Skipped.
    Scratch,
    /// A value the schema does not define.
    Unknown(i16),
}

impl Description {
    fn from_raw(v: i16) -> Self {
        match v {
            0 => Description::OutputActivation,
            1 => Description::InputActivation,
            2 => Description::Parameter,
            3 => Description::Scratch,
            other => Description::Unknown(other),
        }
    }
}

/// Element type of a layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    /// Unsigned 8-bit fixed point.
    FixedPoint8,
    /// Unsigned 16-bit fixed point.
    FixedPoint16,
    /// 32-bit fixed point. Named "signed", but see [`DataType::is_signed`].
    SignedFixedPoint32,
    /// bfloat16.
    BFloat,
    /// IEEE half precision.
    Half,
    /// IEEE single precision.
    Single,
    /// Signed 8-bit fixed point.
    SignedFixedPoint8,
    /// Signed 16-bit fixed point.
    SignedFixedPoint16,
    /// A value the schema does not define.
    Unknown(i16),
}

impl DataType {
    fn from_raw(v: i16) -> Self {
        match v {
            0 => DataType::FixedPoint8,
            1 => DataType::FixedPoint16,
            2 => DataType::SignedFixedPoint32,
            3 => DataType::BFloat,
            4 => DataType::Half,
            5 => DataType::Single,
            8 => DataType::SignedFixedPoint8,
            9 => DataType::SignedFixedPoint16,
            other => DataType::Unknown(other),
        }
    }

    /// Size of one element in bytes.
    ///
    /// `0` for an unrecognised type.
    #[must_use]
    pub fn size_bytes(self) -> usize {
        match self {
            DataType::FixedPoint8 | DataType::SignedFixedPoint8 => 1,
            DataType::FixedPoint16
            | DataType::SignedFixedPoint16
            | DataType::BFloat
            | DataType::Half => 2,
            DataType::SignedFixedPoint32 | DataType::Single => 4,
            DataType::Unknown(_) => 0,
        }
    }

    /// Whether the sign-bit fixup applies to this type.
    ///
    /// Only the two explicitly-signed fixed-point types return `true`.
    /// `SignedFixedPoint32` returns `false` despite its name: that is the
    /// vendor behaviour, and any model compiled against it depends on it.
    #[must_use]
    pub fn is_signed(self) -> bool {
        matches!(
            self,
            DataType::SignedFixedPoint8 | DataType::SignedFixedPoint16
        )
    }
}

/// Role of an executable inside a package.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutableType {
    /// Runs on its own; treated as the inference executable.
    StandAlone,
    /// Uploads parameters only, run when the caching token changes.
    ParameterCaching,
    /// Runs against already-cached parameters; treated as the inference
    /// executable.
    ExecutionOnly,
    /// A value the schema does not define.
    Unknown(i16),
}

impl ExecutableType {
    fn from_raw(v: i16) -> Self {
        match v {
            0 => ExecutableType::StandAlone,
            1 => ExecutableType::ParameterCaching,
            2 => ExecutableType::ExecutionOnly,
            other => ExecutableType::Unknown(other),
        }
    }
}

/// Quantisation constants for a layer.
///
/// Surfaced so a caller that wants to dequantise an output does not have to
/// re-parse the model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Numerics {
    /// Quantised value that represents real zero.
    pub zero_point: i32,
    /// Scale factor from quantised units to real units.
    pub dequantization_factor: f32,
}

/// One entry of the executable's DMA hint list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hint<'a> {
    /// A DMA descriptor: move `size` bytes at `offset` in the buffer named by
    /// `desc`.
    Dma {
        /// Which buffer the descriptor refers to.
        desc: Description,
        /// The layer name from `Meta.name`, empty when absent.
        name: &'a [u8],
        /// Byte offset into that buffer.
        offset: usize,
        /// Byte length to move.
        size: usize,
    },
    /// Send instruction bitstream chunk `chunk_index`.
    Instruction {
        /// Index into [`Executable::instruction_bitstream`].
        chunk_index: usize,
    },
    /// An `InterruptHint`, `FenceHint`, `NONE`, or a malformed entry.
    Ignored,
}

/// Errors from the layout helpers, which do not touch the USB transport and so
/// cannot produce a [`crate::Error::Transport`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum LayoutError {
    /// The layer's `OutputLayout` maps were missing or indexed out of range.
    Malformed(&'static str),
    /// A caller-provided buffer was too small.
    BufferTooSmall {
        /// Bytes required.
        needed: usize,
        /// Bytes provided.
        given: usize,
    },
}

impl<E> From<LayoutError> for crate::Error<E> {
    fn from(e: LayoutError) -> Self {
        match e {
            LayoutError::Malformed(m) => crate::Error::Malformed(m),
            LayoutError::BufferTooSmall { needed, given } => {
                crate::Error::BufferTooSmall { needed, given }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Package / MultiExecutable
// ---------------------------------------------------------------------------

/// The outermost DarwiNN container.
///
/// `min_runtime_version`, `signature`, `keypair_version`, `compiler_version`,
/// `virtual_chip_id`, `multi_chip_package` and `model_identifier` are in the
/// schema but unread: no signature or version check is performed.
#[derive(Clone, Copy, Debug)]
pub struct Package<'a> {
    table: Table<'a>,
}

impl<'a> Package<'a> {
    /// Parses the FlexBuffer payload TFLite hands an `edgetpu-custom-op`.
    ///
    /// Reads the root FlexBuffer map, takes key `"4"`, and treats its bytes
    /// as a `Package` FlatBuffer.
    #[must_use]
    pub fn from_custom_op(custom_options: &'a [u8]) -> Option<Self> {
        let map = FlexMap::root(custom_options)?;
        Self::from_bytes(map.bytes(CUSTOM_OP_EXECUTABLE_KEY)?)
    }

    /// Parses an already-unwrapped `Package` FlatBuffer.
    #[must_use]
    pub fn from_bytes(bytes: &'a [u8]) -> Option<Self> {
        Some(Package {
            table: fb::root(bytes)?,
        })
    }

    /// The nested `MultiExecutable`.
    #[must_use]
    pub fn multi_executable(&self) -> Option<MultiExecutable<'a>> {
        let bytes = self.table.bytes(PACKAGE_SERIALIZED_MULTI_EXECUTABLE)?;
        Some(MultiExecutable {
            table: fb::root(bytes)?,
        })
    }

    /// The executable that performs inference.
    ///
    /// `EXECUTION_ONLY` and `STAND_ALONE` both qualify; the last matching
    /// entry wins.
    #[must_use]
    pub fn inference_executable(&self) -> Option<Executable<'a>> {
        let multi = self.multi_executable()?;
        let mut found = None;
        for i in 0..multi.len() {
            let exe = multi.get(i)?;
            if matches!(
                exe.executable_type(),
                ExecutableType::ExecutionOnly | ExecutableType::StandAlone
            ) {
                found = Some(exe);
            }
        }
        found
    }

    /// The optional parameter-caching executable.
    #[must_use]
    pub fn parameter_caching_executable(&self) -> Option<Executable<'a>> {
        let multi = self.multi_executable()?;
        let mut found = None;
        for i in 0..multi.len() {
            let exe = multi.get(i)?;
            if exe.executable_type() == ExecutableType::ParameterCaching {
                found = Some(exe);
            }
        }
        found
    }
}

/// A list of serialised executables.
#[derive(Clone, Copy, Debug)]
pub struct MultiExecutable<'a> {
    table: Table<'a>,
}

impl<'a> MultiExecutable<'a> {
    /// Number of executables.
    #[must_use]
    pub fn len(&self) -> usize {
        self.table
            .vector(MULTI_SERIALIZED_EXECUTABLES)
            .map_or(0, |v| v.len())
    }

    /// Whether the list is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Executable `i`.
    #[must_use]
    pub fn get(&self, i: usize) -> Option<Executable<'a>> {
        let bytes = self
            .table
            .vector(MULTI_SERIALIZED_EXECUTABLES)?
            .str_bytes(i)?;
        Some(Executable {
            table: fb::root(bytes)?,
        })
    }
}

// ---------------------------------------------------------------------------
// Executable
// ---------------------------------------------------------------------------

/// One DarwiNN executable.
#[derive(Clone, Copy, Debug)]
pub struct Executable<'a> {
    table: Table<'a>,
}

impl<'a> Executable<'a> {
    /// Parses an executable FlatBuffer directly, bypassing the package wrapper.
    #[must_use]
    pub fn from_bytes(bytes: &'a [u8]) -> Option<Self> {
        Some(Executable {
            table: fb::root(bytes)?,
        })
    }

    /// `type`, the executable's role (VT 30).
    #[must_use]
    pub fn executable_type(&self) -> ExecutableType {
        ExecutableType::from_raw(self.table.i16(EXE_TYPE, 0))
    }

    /// `parameter_caching_token` (VT 32).
    ///
    /// The manager re-runs the parameter-caching executable whenever this
    /// changes.
    #[must_use]
    pub fn parameter_caching_token(&self) -> u64 {
        self.table.u64(EXE_PARAMETER_CACHING_TOKEN, 0)
    }

    /// `parameters`, the whole parameter blob (VT 16).
    ///
    /// DMA hints address slices of it by byte offset.
    #[must_use]
    pub fn parameters(&self) -> &'a [u8] {
        self.table.bytes(EXE_PARAMETERS).unwrap_or(&[])
    }

    /// Number of instruction bitstream chunks (VT 14).
    #[must_use]
    pub fn instruction_bitstream_count(&self) -> usize {
        self.table
            .vector(EXE_INSTRUCTION_BITSTREAMS)
            .map_or(0, |v| v.len())
    }

    /// Instruction bitstream chunk `i`, sent verbatim to the device.
    #[must_use]
    pub fn instruction_bitstream(&self, i: usize) -> Option<&'a [u8]> {
        self.table
            .vector(EXE_INSTRUCTION_BITSTREAMS)?
            .table(i)?
            .bytes(BITSTREAM_BITSTREAM)
    }

    /// Number of relocation entries attached to bitstream chunk `i`.
    ///
    /// Nothing here acts on them; bitstreams are sent unmodified. Real models
    /// do carry them, so a non-zero count means the model expects relocation
    /// that will not happen. See [`Self::field_offset`].
    #[must_use]
    pub fn field_offset_count(&self, i: usize) -> usize {
        self.field_offsets(i).map_or(0, |v| v.len())
    }

    /// Relocation entry `j` of instruction chunk `i`, as
    /// `(bit offset into the bitstream, what it refers to, its name)`.
    ///
    /// Exposed so a caller can check what a model expects to be patched;
    /// nothing in this crate acts on it.
    #[must_use]
    pub fn field_offset(&self, i: usize, j: usize) -> Option<(i32, Description, &'a [u8])> {
        let entry = self.field_offsets(i)?.table(j)?;
        let meta = entry.table(FIELD_OFFSET_META);
        Some((
            entry.i32(FIELD_OFFSET_OFFSET_BIT, 0),
            Description::from_raw(meta.map_or(0, |m| m.i16(META_DESC, 0))),
            meta.and_then(|m| m.str_bytes(META_NAME))
                .unwrap_or(&[] as &[u8]),
        ))
    }

    fn field_offsets(&self, i: usize) -> Option<Vector<'a>> {
        self.table
            .vector(EXE_INSTRUCTION_BITSTREAMS)?
            .table(i)?
            .vector(BITSTREAM_FIELD_OFFSETS)
    }

    /// Number of input layers (VT 20).
    #[must_use]
    pub fn input_layer_count(&self) -> usize {
        self.table.vector(EXE_INPUT_LAYERS).map_or(0, |v| v.len())
    }

    /// Input layer `i`.
    #[must_use]
    pub fn input_layer(&self, i: usize) -> Option<Layer<'a>> {
        Some(Layer {
            table: self.table.vector(EXE_INPUT_LAYERS)?.table(i)?,
        })
    }

    /// Number of output layers (VT 22).
    ///
    /// TFLite output tensor `i` corresponds to output layer `i`.
    #[must_use]
    pub fn output_layer_count(&self) -> usize {
        self.table.vector(EXE_OUTPUT_LAYERS).map_or(0, |v| v.len())
    }

    /// Output layer `i`.
    #[must_use]
    pub fn output_layer(&self, i: usize) -> Option<Layer<'a>> {
        Some(Layer {
            table: self.table.vector(EXE_OUTPUT_LAYERS)?.table(i)?,
        })
    }

    /// Index of the input layer called `name`.
    #[must_use]
    pub fn find_input_layer(&self, name: &[u8]) -> Option<usize> {
        (0..self.input_layer_count())
            .find(|i| self.input_layer(*i).is_some_and(|l| l.name() == name))
    }

    /// Index of the output layer called `name`.
    #[must_use]
    pub fn find_output_layer(&self, name: &[u8]) -> Option<usize> {
        (0..self.output_layer_count())
            .find(|i| self.output_layer(*i).is_some_and(|l| l.name() == name))
    }

    /// Number of DMA hints (VT 18 -> `DmaHints.hints`).
    #[must_use]
    pub fn hint_count(&self) -> usize {
        self.hints_vector().map_or(0, |v| v.len())
    }

    /// DMA hint `i`.
    ///
    /// Hints are consumed strictly in array order: the compiler interleaved
    /// the instruction, parameter and activation chunks in the order the
    /// hardware expects, and nothing here reorders them.
    #[must_use]
    pub fn hint(&self, i: usize) -> Hint<'a> {
        let Some(hint) = self.hints_vector().and_then(|v| v.table(i)) else {
            return Hint::Ignored;
        };
        match hint.u8(HINT_ANY_HINT_TYPE, 0) {
            ANY_HINT_DMA_DESCRIPTOR => {
                let Some(d) = hint.union(HINT_ANY_HINT) else {
                    return Hint::Ignored;
                };
                let meta = d.table(DESC_HINT_META);
                let desc = Description::from_raw(meta.map_or(0, |m| m.i16(META_DESC, 0)));
                let name = meta
                    .and_then(|m| m.str_bytes(META_NAME))
                    .unwrap_or(&[] as &[u8]);
                let offset = d.i32(DESC_HINT_OFFSET_IN_BYTES, 0);
                let size = d.i32(DESC_HINT_SIZE_IN_BYTES, 0);
                if offset < 0 || size < 0 {
                    return Hint::Ignored;
                }
                Hint::Dma {
                    desc,
                    name,
                    offset: offset as usize,
                    size: size as usize,
                }
            }
            ANY_HINT_INSTRUCTION => {
                let Some(d) = hint.union(HINT_ANY_HINT) else {
                    return Hint::Ignored;
                };
                let idx = d.i32(INSTR_HINT_CHUNK_INDEX, 0);
                if idx < 0 {
                    return Hint::Ignored;
                }
                Hint::Instruction {
                    chunk_index: idx as usize,
                }
            }
            _ => Hint::Ignored,
        }
    }

    /// Iterates the DMA hints in order.
    pub fn hints(&self) -> impl Iterator<Item = Hint<'a>> + '_ {
        (0..self.hint_count()).map(|i| self.hint(i))
    }

    fn hints_vector(&self) -> Option<Vector<'a>> {
        self.table.table(EXE_DMA_HINTS)?.vector(DMA_HINTS_HINTS)
    }
}

// ---------------------------------------------------------------------------
// Layer
// ---------------------------------------------------------------------------

/// One input or output layer.
#[derive(Clone, Copy, Debug)]
pub struct Layer<'a> {
    table: Table<'a>,
}

impl<'a> Layer<'a> {
    /// `name`, as raw bytes. Empty when the field is absent.
    #[must_use]
    pub fn name(&self) -> &'a [u8] {
        self.table.str_bytes(LAYER_NAME).unwrap_or(&[])
    }

    /// `size_bytes`: the *padded* size of one execution's worth of data, which
    /// is how much scratch space the output staging buffer needs.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.table.i32(LAYER_SIZE_BYTES, 0).max(0) as usize
    }

    /// `x_dim`.
    #[must_use]
    pub fn x_dim(&self) -> usize {
        self.table.i32(LAYER_X_DIM, 0).max(0) as usize
    }

    /// `y_dim`.
    #[must_use]
    pub fn y_dim(&self) -> usize {
        self.table.i32(LAYER_Y_DIM, 0).max(0) as usize
    }

    /// `z_dim`.
    #[must_use]
    pub fn z_dim(&self) -> usize {
        self.table.i32(LAYER_Z_DIM, 0).max(0) as usize
    }

    /// `data_type`.
    #[must_use]
    pub fn data_type(&self) -> DataType {
        DataType::from_raw(self.table.i16(LAYER_DATA_TYPE, 0))
    }

    /// `execution_count_per_inference`, whose schema default is `1`.
    #[must_use]
    pub fn execution_count_per_inference(&self) -> usize {
        self.table
            .i32(LAYER_EXECUTION_COUNT_PER_INFERENCE, 1)
            .max(0) as usize
    }

    /// `numerics`, the quantisation constants, when present.
    #[must_use]
    pub fn numerics(&self) -> Option<Numerics> {
        let t = self.table.table(LAYER_NUMERICS)?;
        Some(Numerics {
            zero_point: t.i32(NUMERICS_ZERO_POINT, 0),
            dequantization_factor: t.f32(NUMERICS_DEQUANTIZATION_FACTOR, 0.0),
        })
    }

    /// Unpadded size of the whole layer, in bytes.
    #[must_use]
    pub fn actual_size_bytes(&self) -> usize {
        self.x_dim()
            .saturating_mul(self.y_dim())
            .saturating_mul(self.z_dim())
            .saturating_mul(self.data_type().size_bytes())
            .saturating_mul(self.execution_count_per_inference())
    }

    /// Padded size of the whole layer, in bytes: what the device actually
    /// sends, and the minimum size of the staging buffer.
    #[must_use]
    pub fn padded_size_bytes(&self) -> usize {
        self.size_bytes()
            .saturating_mul(self.execution_count_per_inference())
    }

    /// Flips the MSB of every element of `buffer`, in place.
    ///
    /// The device represents signed fixed-point values with the sign bit
    /// inverted, so the same XOR converts in both directions: apply it to input
    /// tensors before sending and to output tensors after relayout.
    ///
    /// Elements are `data_type_size` bytes little-endian, so the MSB is the
    /// last byte of each. Exactly `x_dim * y_dim * z_dim` elements are
    /// transformed, one execution's worth, even when
    /// `execution_count_per_inference > 1`.
    pub fn transform_signed_data_type_raw(
        buffer: &mut [u8],
        data_type_size: usize,
        x_dim: usize,
        y_dim: usize,
        z_dim: usize,
    ) {
        if data_type_size == 0 {
            return;
        }
        let elements = x_dim.saturating_mul(y_dim).saturating_mul(z_dim);
        let mut index = 0usize;
        for _ in 0..elements {
            let msb = index + data_type_size - 1;
            match buffer.get_mut(msb) {
                Some(b) => *b ^= 128,
                None => return,
            }
            index += data_type_size;
        }
    }

    /// Applies [`Self::transform_signed_data_type_raw`] if this layer's type is
    /// signed and `buffer` is large enough.
    ///
    /// Silently does nothing when `buffer` is shorter than
    /// [`Self::actual_size_bytes`].
    pub fn transform_signed_data_type(&self, buffer: &mut [u8]) {
        if !self.data_type().is_signed() {
            return;
        }
        if buffer.len() < self.actual_size_bytes() {
            return;
        }
        Self::transform_signed_data_type_raw(
            buffer,
            self.data_type().size_bytes(),
            self.x_dim(),
            self.y_dim(),
            self.z_dim(),
        );
    }

    /// The `OutputLayout` tile maps, when this layer carries an `OutputLayer`
    /// union variant.
    fn layout(&self) -> Option<OutputLayout<'a>> {
        if self.table.u8(LAYER_ANY_LAYER_TYPE, 0) != ANY_LAYER_OUTPUT_LAYER {
            return None;
        }
        let layout = self
            .table
            .union(LAYER_ANY_LAYER)?
            .table(OUTPUT_LAYER_LAYOUT)?;
        Some(OutputLayout {
            y_to_tile_id: layout.vector(LAYOUT_Y_TO_TILE_ID)?,
            x_to_tile_id: layout.vector(LAYOUT_X_TO_TILE_ID)?,
            tile_byte_offset: layout.vector(LAYOUT_TILE_BYTE_OFFSET)?,
            x_to_local_byte_offset: layout.vector(LAYOUT_X_TO_LOCAL_BYTE_OFFSET)?,
            y_to_local_y_offset: layout.vector(LAYOUT_Y_TO_LOCAL_Y_OFFSET)?,
            x_to_local_y_row_size: layout.vector(LAYOUT_X_TO_LOCAL_Y_ROW_SIZE)?,
        })
    }

    /// De-interleaves the device's tile-scattered output into the flat
    /// `(y, x, z)` row-major buffer a caller expects.
    ///
    /// `src` is the staging buffer that [`crate::Driver::invoke`] filled;
    /// `dst` receives [`Self::actual_size_bytes`] bytes.
    ///
    /// Two shapes are handled:
    ///
    /// * `x_dim == 1 && y_dim == 1`: a plain vector. The only work is
    ///   stripping per-execution padding when
    ///   `execution_count_per_inference > 1`.
    /// * anything else: walk `(y, x)`, resolve each coordinate to a tile and
    ///   a local byte offset through the six `OutputLayout` maps, and copy
    ///   `z_dim * element_size` bytes per column.
    ///
    /// The padded z stride is *derived*, not read from the executable: the
    /// difference between the buffer index of `(0, 1, 0)` and `(0, 0, 0)`, or
    /// the y equivalent when `x_dim == 1`. A `z_bytes` of 1 or 3 overrides that
    /// with a hardcoded stride of 4, which covers the grayscale and RGB cases.
    /// The override changes which bytes are read, so it is not a fast path.
    pub fn relayout_into(&self, src: &[u8], dst: &mut [u8]) -> Result<(), LayoutError> {
        let dts = self.data_type().size_bytes();
        if dts == 0 {
            return Err(LayoutError::Malformed("unknown layer data type"));
        }
        let (x_dim, y_dim, z_dim) = (self.x_dim(), self.y_dim(), self.z_dim());
        let z_bytes = z_dim.saturating_mul(dts);
        let executions = self.execution_count_per_inference();

        if y_dim == 1 && x_dim == 1 {
            return self.relayout_vector(src, dst, z_bytes, executions);
        }

        let layout = self
            .layout()
            .ok_or(LayoutError::Malformed("output layer has no layout maps"))?;

        // The 1- and 3-byte cases override the derivation entirely.
        let z_bytes_padded = match z_bytes {
            1 | 3 => 4,
            _ => {
                let base = layout.buffer_index(&layout.y_index(0)?, 0, 0)?;
                let next = if x_dim > 1 {
                    layout.buffer_index(&layout.y_index(0)?, 1, 0)?
                } else {
                    layout.buffer_index(&layout.y_index(1)?, 0, 0)?
                };
                next.checked_sub(base)
                    .ok_or(LayoutError::Malformed("negative z stride"))?
                    .saturating_mul(dts)
            }
        };

        let needed = self.actual_size_bytes();
        let given = dst.len();
        if given < needed {
            return Err(LayoutError::BufferTooSmall { needed, given });
        }

        let mut out = 0usize;
        for y in 0..y_dim {
            let y_index = layout.y_index(y)?;
            let mut tile_start_x = 0usize;
            while tile_start_x < x_dim {
                // Run length of the tile that column tile_start_x belongs to.
                let tile_id = layout.x_tile_id(tile_start_x)?;
                let mut tile_x_size = 1usize;
                while tile_start_x + tile_x_size < x_dim
                    && layout.x_tile_id(tile_start_x + tile_x_size)? == tile_id
                {
                    tile_x_size += 1;
                }
                let mut source = layout
                    .buffer_index(&y_index, tile_start_x, 0)?
                    .saturating_mul(dts);
                for _ in 0..tile_x_size {
                    let s = src
                        .get(source..source.saturating_add(z_bytes))
                        .ok_or(LayoutError::Malformed("relayout source out of range"))?;
                    let d = dst
                        .get_mut(out..out.saturating_add(z_bytes))
                        .ok_or(LayoutError::BufferTooSmall { needed, given })?;
                    d.copy_from_slice(s);
                    out += z_bytes;
                    source = source.saturating_add(z_bytes_padded);
                }
                tile_start_x += tile_x_size;
            }
        }
        Ok(())
    }

    /// The `x_dim == 1 && y_dim == 1` branch of `Relayout`.
    fn relayout_vector(
        &self,
        src: &[u8],
        dst: &mut [u8],
        z_bytes: usize,
        executions: usize,
    ) -> Result<(), LayoutError> {
        let padded = self.padded_size_bytes();
        let actual = self.actual_size_bytes();
        let total = z_bytes.saturating_mul(executions);
        if dst.len() < total {
            return Err(LayoutError::BufferTooSmall {
                needed: total,
                given: dst.len(),
            });
        }
        if executions == 1 || padded == actual {
            let s = src
                .get(..total)
                .ok_or(LayoutError::Malformed("relayout source too short"))?;
            dst[..total].copy_from_slice(s);
            return Ok(());
        }
        // Strip the padding at the end of each execution.
        let pad_per_execution = padded.saturating_sub(actual) / executions;
        let mut si = 0usize;
        let mut di = 0usize;
        for _ in 0..executions {
            let s = src
                .get(si..si.saturating_add(z_bytes))
                .ok_or(LayoutError::Malformed("relayout source too short"))?;
            dst[di..di + z_bytes].copy_from_slice(s);
            di += z_bytes;
            si = si.saturating_add(z_bytes).saturating_add(pad_per_execution);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// OutputLayout
// ---------------------------------------------------------------------------

/// The six parallel `[int32]` tile maps of an `OutputLayout`.
#[derive(Clone, Copy, Debug)]
struct OutputLayout<'a> {
    y_to_tile_id: Vector<'a>,
    x_to_tile_id: Vector<'a>,
    tile_byte_offset: Vector<'a>,
    x_to_local_byte_offset: Vector<'a>,
    y_to_local_y_offset: Vector<'a>,
    x_to_local_y_row_size: Vector<'a>,
}

/// The `(tile id, local y)` pair `GetYBufferIndex` returns.
struct YBufferIndex {
    linearized_tile_id: i64,
    local_y_coordinate: i64,
}

impl OutputLayout<'_> {
    fn y_index(&self, y: usize) -> Result<YBufferIndex, LayoutError> {
        Ok(YBufferIndex {
            linearized_tile_id: i64::from(
                self.y_to_tile_id
                    .i32(y)
                    .ok_or(LayoutError::Malformed("y_coordinate_to_linear_tile_id_map"))?,
            ),
            local_y_coordinate: i64::from(
                self.y_to_local_y_offset
                    .i32(y)
                    .ok_or(LayoutError::Malformed("y_coordinate_to_local_y_offset"))?,
            ),
        })
    }

    fn x_tile_id(&self, x: usize) -> Result<i32, LayoutError> {
        self.x_to_tile_id
            .i32(x)
            .ok_or(LayoutError::Malformed("x_coordinate_to_linear_tile_id_map"))
    }

    /// The `OutputLayout` fields are named "byte offset", but every caller
    /// multiplies the result by the element size, so the units are elements.
    fn buffer_index(&self, y: &YBufferIndex, x: usize, z: i64) -> Result<usize, LayoutError> {
        let tile = y.linearized_tile_id + i64::from(self.x_tile_id(x)?);
        let tile_index =
            usize::try_from(tile).map_err(|_| LayoutError::Malformed("negative tile id"))?;
        let global = i64::from(
            self.tile_byte_offset
                .i32(tile_index)
                .ok_or(LayoutError::Malformed("linearized_tile_byte_offset"))?,
        );
        let local_x = i64::from(
            self.x_to_local_byte_offset
                .i32(x)
                .ok_or(LayoutError::Malformed("x_coordinate_to_local_byte_offset"))?,
        );
        let row_size = i64::from(
            self.x_to_local_y_row_size
                .i32(x)
                .ok_or(LayoutError::Malformed("x_coordinate_to_local_y_row_size"))?,
        );
        let local_y = y.local_y_coordinate * row_size;
        usize::try_from(global + local_y + local_x + z)
            .map_err(|_| LayoutError::Malformed("negative buffer index"))
    }
}
