//! A minimal, allocation-free, panic-free FlatBuffers and FlexBuffers reader.
//!
//! The DarwiNN executable format is a FlatBuffer and the TFLite custom-op
//! payload wrapping it is a FlexBuffer map. The `flatbuffers` crate would cost
//! a `std`/`alloc` dependency and a code-generation step for a schema only
//! four tables are read out of, so the offset arithmetic is done by hand.
//!
//! Everything in this module returns [`Option`] rather than panicking: the
//! bytes come off a model file that this crate does not produce, and a
//! malformed buffer must surface as an error, not as a fault on the target.
//!
//! # FlatBuffers layout, for reference
//!
//! * The first 4 bytes of a buffer are a `uoffset32` from position 0 to the
//!   root table.
//! * A table starts with an `soffset32`; the vtable lives at
//!   `table_pos - soffset`.
//! * A vtable is `u16 vtable_bytes`, `u16 table_bytes`, then one `u16` per
//!   field. A field's vtable slot is at `vtable_pos + vt_offset`, where
//!   `vt_offset` is the `VT_*` constant from the generated header. A slot value
//!   of `0`, or a `vt_offset` past `vtable_bytes`, means "field absent, use the
//!   default".
//! * `uoffset32` values stored inside tables and vectors are relative to the
//!   position of the offset field itself, and always point forward.
//! * A vector is `u32 len` followed by `len` elements. A string is a vector of
//!   bytes with an extra NUL terminator that is not counted in `len`.

// ---------------------------------------------------------------------------
// Scalar reads
// ---------------------------------------------------------------------------

fn u8_at(buf: &[u8], pos: usize) -> Option<u8> {
    buf.get(pos).copied()
}

fn u16_at(buf: &[u8], pos: usize) -> Option<u16> {
    let b = buf.get(pos..pos.checked_add(2)?)?;
    Some(u16::from_le_bytes([b[0], b[1]]))
}

fn u32_at(buf: &[u8], pos: usize) -> Option<u32> {
    let b = buf.get(pos..pos.checked_add(4)?)?;
    Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn u64_at(buf: &[u8], pos: usize) -> Option<u64> {
    let b = buf.get(pos..pos.checked_add(8)?)?;
    Some(u64::from_le_bytes([
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
    ]))
}

/// Reads a little-endian unsigned integer of `width` bytes (1, 2, 4 or 8).
///
/// Used by the FlexBuffers reader, where the width of every scalar is carried
/// out-of-band instead of being implied by the schema.
fn uint_at(buf: &[u8], pos: usize, width: usize) -> Option<u64> {
    let b = buf.get(pos..pos.checked_add(width)?)?;
    let mut v: u64 = 0;
    for (i, byte) in b.iter().enumerate() {
        v |= u64::from(*byte) << (8 * i);
    }
    Some(v)
}

// ---------------------------------------------------------------------------
// FlatBuffers
// ---------------------------------------------------------------------------

/// A view of one FlatBuffers table inside a buffer.
#[derive(Clone, Copy, Debug)]
pub struct Table<'a> {
    buf: &'a [u8],
    pos: usize,
    vtable: usize,
}

/// Returns the root table of a FlatBuffer.
pub fn root(buf: &[u8]) -> Option<Table<'_>> {
    let off = u32_at(buf, 0)? as usize;
    Table::at(buf, off)
}

impl<'a> Table<'a> {
    /// Builds a table view for the table whose header starts at `pos`.
    fn at(buf: &'a [u8], pos: usize) -> Option<Self> {
        // The leading soffset32 is signed and is *subtracted* from the table
        // position, so vtables may live either before or after the table.
        let soff = u32_at(buf, pos)? as i32;
        let vtable = (pos as i64).checked_sub(i64::from(soff))?;
        if vtable < 0 {
            return None;
        }
        let vtable = usize::try_from(vtable).ok()?;
        // Require at least the two-u16 vtable header to be in bounds.
        u16_at(buf, vtable.checked_add(2)?)?;
        Some(Table { buf, pos, vtable })
    }

    /// Absolute position of the field's data, or `None` when the field is
    /// absent and the caller should use the schema default.
    fn field(&self, vt_offset: u16) -> Option<usize> {
        let vtable_bytes = u16_at(self.buf, self.vtable)?;
        if vt_offset >= vtable_bytes {
            return None;
        }
        let slot = u16_at(self.buf, self.vtable.checked_add(usize::from(vt_offset))?)?;
        if slot == 0 {
            return None;
        }
        self.pos.checked_add(usize::from(slot))
    }

    /// Follows the `uoffset32` stored in `vt_offset` to its absolute target.
    fn indirect(&self, vt_offset: u16) -> Option<usize> {
        let p = self.field(vt_offset)?;
        let rel = u32_at(self.buf, p)? as usize;
        p.checked_add(rel)
    }

    /// Reads a `u8`/`bool` field, or `default` when absent.
    pub fn u8(&self, vt_offset: u16, default: u8) -> u8 {
        self.field(vt_offset)
            .and_then(|p| u8_at(self.buf, p))
            .unwrap_or(default)
    }

    /// Reads an `i16` field (FlatBuffers enums default to `short`), or
    /// `default` when absent.
    pub fn i16(&self, vt_offset: u16, default: i16) -> i16 {
        self.field(vt_offset)
            .and_then(|p| u16_at(self.buf, p))
            .map(|v| v as i16)
            .unwrap_or(default)
    }

    /// Reads an `i32` field, or `default` when absent.
    pub fn i32(&self, vt_offset: u16, default: i32) -> i32 {
        self.field(vt_offset)
            .and_then(|p| u32_at(self.buf, p))
            .map(|v| v as i32)
            .unwrap_or(default)
    }

    /// Reads a `u64` field, or `default` when absent.
    pub fn u64(&self, vt_offset: u16, default: u64) -> u64 {
        self.field(vt_offset)
            .and_then(|p| u64_at(self.buf, p))
            .unwrap_or(default)
    }

    /// Reads an `f32` field, or `default` when absent.
    pub fn f32(&self, vt_offset: u16, default: f32) -> f32 {
        self.field(vt_offset)
            .and_then(|p| u32_at(self.buf, p))
            .map(f32::from_bits)
            .unwrap_or(default)
    }

    /// Reads a nested-table field.
    pub fn table(&self, vt_offset: u16) -> Option<Table<'a>> {
        Table::at(self.buf, self.indirect(vt_offset)?)
    }

    /// Reads a union value field. The union's discriminant lives in a separate
    /// `u8` field, which the caller checks first.
    pub fn union(&self, vt_offset: u16) -> Option<Table<'a>> {
        self.table(vt_offset)
    }

    /// Reads a vector field.
    pub fn vector(&self, vt_offset: u16) -> Option<Vector<'a>> {
        Vector::at(self.buf, self.indirect(vt_offset)?)
    }

    /// Reads a `[ubyte]` field as a borrowed slice.
    pub fn bytes(&self, vt_offset: u16) -> Option<&'a [u8]> {
        let v = self.vector(vt_offset)?;
        v.as_bytes()
    }

    /// Reads a `string` field as a borrowed slice, without the NUL terminator.
    ///
    /// Returned as bytes rather than `&str`: DarwiNN layer names are ASCII in
    /// practice, but nothing in the format guarantees UTF-8 and a validation
    /// failure here must not cost the caller its layer lookup.
    pub fn str_bytes(&self, vt_offset: u16) -> Option<&'a [u8]> {
        let v = Vector::at(self.buf, self.indirect(vt_offset)?)?;
        v.as_bytes()
    }
}

/// A view of one FlatBuffers vector.
#[derive(Clone, Copy, Debug)]
pub struct Vector<'a> {
    buf: &'a [u8],
    /// Position of element 0 (the `u32` length prefix sits just before it).
    data: usize,
    len: usize,
}

impl<'a> Vector<'a> {
    fn at(buf: &'a [u8], pos: usize) -> Option<Self> {
        let len = u32_at(buf, pos)? as usize;
        Some(Vector {
            buf,
            data: pos.checked_add(4)?,
            len,
        })
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// The whole vector as a byte slice, for `[ubyte]` vectors and strings.
    pub fn as_bytes(&self) -> Option<&'a [u8]> {
        self.buf.get(self.data..self.data.checked_add(self.len)?)
    }

    /// Element `i` of an `[int]`/`[uint]` vector.
    pub fn i32(&self, i: usize) -> Option<i32> {
        if i >= self.len {
            return None;
        }
        let p = self.data.checked_add(i.checked_mul(4)?)?;
        Some(u32_at(self.buf, p)? as i32)
    }

    /// Element `i` of a vector of tables.
    pub fn table(&self, i: usize) -> Option<Table<'a>> {
        Table::at(self.buf, self.offset_element(i)?)
    }

    /// Element `i` of a vector of strings, without the NUL terminator.
    pub fn str_bytes(&self, i: usize) -> Option<&'a [u8]> {
        Vector::at(self.buf, self.offset_element(i)?)?.as_bytes()
    }

    /// Resolves element `i` of a vector of `uoffset32`.
    fn offset_element(&self, i: usize) -> Option<usize> {
        if i >= self.len {
            return None;
        }
        let p = self.data.checked_add(i.checked_mul(4)?)?;
        let rel = u32_at(self.buf, p)? as usize;
        p.checked_add(rel)
    }
}

// ---------------------------------------------------------------------------
// FlexBuffers
// ---------------------------------------------------------------------------

/// FlexBuffers type tag for a map.
const FBT_MAP: u8 = 9;
/// FlexBuffers type tag for a string.
const FBT_STRING: u8 = 5;
/// FlexBuffers type tag for a blob.
const FBT_BLOB: u8 = 25;

/// A view of the root FlexBuffers map of a buffer.
///
/// TFLite hands a custom op its options as a FlexBuffer map; key `"4"` holds
/// the serialised DarwiNN `Package`.
#[derive(Clone, Copy, Debug)]
pub struct FlexMap<'a> {
    buf: &'a [u8],
    /// Position of value 0 of the values vector.
    values: usize,
    /// Position of key 0 of the keys vector.
    keys: usize,
    len: usize,
    byte_width: usize,
    keys_byte_width: usize,
}

/// A FlexBuffers offset counts *backwards* from the position of the offset
/// field, unlike a FlatBuffers `uoffset32`.
fn flex_indirect(buf: &[u8], pos: usize, width: usize) -> Option<usize> {
    let rel = uint_at(buf, pos, width)? as usize;
    pos.checked_sub(rel)
}

impl<'a> FlexMap<'a> {
    /// Parses the root of a FlexBuffer and returns it if it is a map.
    ///
    /// The root of a FlexBuffer is stored at the *end* of the buffer: the last
    /// byte is the width of the root *slot*, the byte before it is the root's
    /// packed type, and the root value occupies that many bytes before it.
    ///
    /// Two different widths are in play and conflating them is the easy
    /// mistake here. The root slot width is what the offset itself is read at.
    /// The map's own width, covering its length, its key-vector fields and its
    /// element stride, comes from the low two bits of the packed type. In the
    /// models this crate loads the first is 1 and the second is 4.
    pub fn root(buf: &'a [u8]) -> Option<Self> {
        let n = buf.len();
        if n < 3 {
            return None;
        }
        let parent_width = usize::from(buf[n - 1]);
        if !matches!(parent_width, 1 | 2 | 4 | 8) {
            return None;
        }
        let packed_type = buf[n - 2];
        if packed_type >> 2 != FBT_MAP {
            return None;
        }
        let byte_width = 1usize << (packed_type & 3);
        let value_pos = n.checked_sub(2)?.checked_sub(parent_width)?;
        let values = flex_indirect(buf, value_pos, parent_width)?;

        // A map is a vector of values with three extra fields in front of it:
        // [keys_offset][keys_byte_width][length][value 0]...
        let len = uint_at(buf, values.checked_sub(byte_width)?, byte_width)? as usize;
        let keys_byte_width = uint_at(
            buf,
            values.checked_sub(byte_width.checked_mul(2)?)?,
            byte_width,
        )? as usize;
        if !matches!(keys_byte_width, 1 | 2 | 4 | 8) {
            return None;
        }
        let keys_offset_pos = values.checked_sub(byte_width.checked_mul(3)?)?;
        let keys = flex_indirect(buf, keys_offset_pos, byte_width)?;

        Some(FlexMap {
            buf,
            values,
            keys,
            len,
            byte_width,
            keys_byte_width,
        })
    }

    /// The NUL-terminated key `i`, without its terminator.
    fn key(&self, i: usize) -> Option<&'a [u8]> {
        let p = self
            .keys
            .checked_add(i.checked_mul(self.keys_byte_width)?)?;
        let start = flex_indirect(self.buf, p, self.keys_byte_width)?;
        let rest = self.buf.get(start..)?;
        let end = rest.iter().position(|b| *b == 0)?;
        rest.get(..end)
    }

    /// The string or blob stored under `key`, as raw bytes.
    ///
    /// Keys are sorted in a FlexBuffers map, but the maps this crate reads have
    /// a handful of entries, so this scans them linearly instead of binary
    /// searching. A linear scan does not depend on that ordering and is bounded
    /// by the map length.
    pub fn bytes(&self, key: &[u8]) -> Option<&'a [u8]> {
        for i in 0..self.len {
            if self.key(i)? != key {
                continue;
            }
            let value_pos = self.values.checked_add(i.checked_mul(self.byte_width)?)?;
            let type_pos = self
                .values
                .checked_add(self.len.checked_mul(self.byte_width)?)?
                .checked_add(i)?;
            let packed = u8_at(self.buf, type_pos)?;
            if !matches!(packed >> 2, FBT_STRING | FBT_BLOB) {
                return None;
            }
            // The value's own width comes from the low two bits of its packed
            // type; the *offset* is read at the parent vector's width.
            let elem_width = 1usize << (packed & 3);
            let data = flex_indirect(self.buf, value_pos, self.byte_width)?;
            let len = uint_at(self.buf, data.checked_sub(elem_width)?, elem_width)? as usize;
            return self.buf.get(data..data.checked_add(len)?);
        }
        None
    }
}
