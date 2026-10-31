//! Portable, append-only KV frame store with exact BF16/F16/F32 bytes.
//!
//! Separate from the audited legacy BF16 cold store. Frames have explicit dtype,
//! geometry, SHA-256 and a commit footer. A torn last append is not published;
//! corruption of a fully committed record is rejected. There is NO quantization
//! and NO claim of entropy compression. Reading a frame verifies its digest.
//!
//! Format: FILE_MAGIC(8) + version(u32) + q_heads(u64) + kv_heads(u64)
//!       + head_dim(u64) [36-byte file header]
//! Each frame: FRAME_MAGIC(8) + tokens(u64) + k_dtype(u8) + v_dtype(u8)
//!       + reserved(6) + k_len(u64) + v_len(u64) + sha256(32)
//!       + k_payload + v_payload + COMMIT_MAGIC(8).
//! All integers and tensors are little-endian.

use crate::portable::{AttentionGeometry, ExactKvFrames, FloatFormat, RawTensor};
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{self, ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::Path;

const FILE_MAGIC: &[u8; 8] = b"TSKVN001";
const RECORD_MAGIC: &[u8; 8] = b"KVFRAME1";
const COMMIT_MAGIC: &[u8; 8] = b"COMMIT01";
const VERSION: u32 = 1;
const FILE_HEADER_BYTES: u64 = 36;
const FRAME_HEADER_BYTES: u64 = 72;
const COMMIT_BYTES: u64 = 8;
const MAX_FRAME_PAYLOAD: u64 = 512 * 1024 * 1024;
const HASH_DOMAIN: &[u8] = b"mslim-portable-kv-record-v1";

fn corrupt(msg: &'static str) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, msg)
}

fn format_byte(dtype: FloatFormat) -> u8 {
    match dtype {
        FloatFormat::Bf16 => 1,
        FloatFormat::F16 => 2,
        FloatFormat::F32 => 3,
    }
}

fn decode_format(byte: u8) -> io::Result<FloatFormat> {
    match byte {
        1 => Ok(FloatFormat::Bf16),
        2 => Ok(FloatFormat::F16),
        3 => Ok(FloatFormat::F32),
        _ => Err(corrupt("unknown floating-point format")),
    }
}

fn get_u64(slice: &[u8]) -> u64 {
    u64::from_le_bytes(slice.try_into().expect("fixed-size internal slice"))
}

fn bounded_size(dtype: FloatFormat, geometry: AttentionGeometry, tokens: u64) -> io::Result<u64> {
    if tokens == 0 {
        return Err(corrupt("frame must have at least one token"));
    }
    tokens
        .checked_mul(geometry.kv_heads as u64)
        .and_then(|v| v.checked_mul(geometry.head_dim as u64))
        .and_then(|v| v.checked_mul(dtype.bytes_per_element() as u64))
        .filter(|&v| v > 0 && v <= MAX_FRAME_PAYLOAD)
        .ok_or_else(|| corrupt("frame length overflow or exceeds 512 MiB limit"))
}

fn digest_start(
    geometry: AttentionGeometry,
    tokens: u64,
    kformat: FloatFormat,
    vformat: FloatFormat,
    kbytes: u64,
    vbytes: u64,
) -> Sha256 {
    let mut hash = Sha256::new();
    hash.update(HASH_DOMAIN);
    // Bind the logical attention shape into every persisted record digest.
    for value in [geometry.query_heads, geometry.kv_heads, geometry.head_dim] {
        hash.update((value as u64).to_le_bytes());
    }
    hash.update(tokens.to_le_bytes());
    hash.update([format_byte(kformat), format_byte(vformat)]);
    hash.update(kbytes.to_le_bytes());
    hash.update(vbytes.to_le_bytes());
    hash
}

fn digest_frame(
    geometry: AttentionGeometry,
    tokens: u64,
    key: &RawTensor,
    value: &RawTensor,
) -> [u8; 32] {
    let mut digest = digest_start(
        geometry,
        tokens,
        key.dtype(),
        value.dtype(),
        key.len_bytes() as u64,
        value.len_bytes() as u64,
    );
    digest.update(key.as_le_bytes());
    digest.update(value.as_le_bytes());
    digest.finalize().into()
}

#[derive(Clone, Debug)]
pub struct FrameIndex {
    pub tokens: u64,
    pub key_dtype: FloatFormat,
    pub value_dtype: FloatFormat,
    pub key_bytes: u64,
    pub value_bytes: u64,
    record_offset: u64,
    digest: [u8; 32],
}

impl FrameIndex {
    pub fn total_payload_bytes(&self) -> u64 {
        self.key_bytes + self.value_bytes
    }
}

/// File-backed exact KV records. Read frame-by-frame to avoid resident full KV.
pub struct PortableKvStore {
    file: File,
    geometry: AttentionGeometry,
    frames: Vec<FrameIndex>,
    next_append: u64,
    read_only: bool,
    trailing_uncommitted: u64,
}

impl PortableKvStore {
    /// Never truncate an existing user store.
    pub fn create(path: impl AsRef<Path>, geometry: AttentionGeometry) -> io::Result<Self> {
        let mut file = OpenOptions::new()
            .write(true)
            .read(true)
            .create_new(true)
            .open(path)?;
        file.write_all(FILE_MAGIC)?;
        file.write_all(&VERSION.to_le_bytes())?;
        for value in [geometry.query_heads, geometry.kv_heads, geometry.head_dim] {
            file.write_all(&(value as u64).to_le_bytes())?;
        }
        file.sync_data()?;
        Ok(Self {
            file,
            geometry,
            frames: Vec::new(),
            next_append: FILE_HEADER_BYTES,
            read_only: false,
            trailing_uncommitted: 0,
        })
    }

    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        Self::open_mode(path, false)
    }

    pub fn open_read_only(path: impl AsRef<Path>) -> io::Result<Self> {
        Self::open_mode(path, true)
    }

    fn open_mode(path: impl AsRef<Path>, read_only: bool) -> io::Result<Self> {
        let mut file = OpenOptions::new().read(true).write(!read_only).open(path)?;
        let file_len = file.metadata()?.len();
        if file_len < FILE_HEADER_BYTES {
            return Err(corrupt("file header truncated"));
        }
        let mut prelude = [0u8; FILE_HEADER_BYTES as usize];
        file.read_exact(&mut prelude)?;
        if &prelude[..8] != FILE_MAGIC {
            return Err(corrupt("not a M-SLIM portable KV store"));
        }
        if u32::from_le_bytes(prelude[8..12].try_into().unwrap()) != VERSION {
            return Err(corrupt("unsupported portable KV store version"));
        }
        let raw = [
            get_u64(&prelude[12..20]),
            get_u64(&prelude[20..28]),
            get_u64(&prelude[28..36]),
        ];
        let geometry = AttentionGeometry::new(
            usize::try_from(raw[0]).map_err(|_| corrupt("q_heads overflow"))?,
            usize::try_from(raw[1]).map_err(|_| corrupt("kv_heads overflow"))?,
            usize::try_from(raw[2]).map_err(|_| corrupt("head_dim overflow"))?,
        )
        .map_err(|_| corrupt("invalid portable KV file geometry"))?;
        let mut store = Self {
            file,
            geometry,
            frames: Vec::new(),
            next_append: FILE_HEADER_BYTES,
            read_only,
            trailing_uncommitted: 0,
        };
        store.rebuild_index(file_len)?;
        Ok(store)
    }

    fn rebuild_index(&mut self, file_len: u64) -> io::Result<()> {
        let mut position = FILE_HEADER_BYTES;
        while position < file_len {
            // A partial final record is uncommitted, never exposed as KV.
            if file_len - position < FRAME_HEADER_BYTES {
                break;
            }
            self.file.seek(SeekFrom::Start(position))?;
            let mut header = [0u8; FRAME_HEADER_BYTES as usize];
            self.file.read_exact(&mut header)?;
            if &header[..8] != RECORD_MAGIC {
                return Err(corrupt("bad frame magic in committed area"));
            }
            if header[18..24] != [0u8; 6] {
                return Err(corrupt("reserved frame header bytes are not zero"));
            }
            let tokens = get_u64(&header[8..16]);
            let key_dtype = decode_format(header[16])?;
            let value_dtype = decode_format(header[17])?;
            let key_bytes = get_u64(&header[24..32]);
            let value_bytes = get_u64(&header[32..40]);
            let expected_key = bounded_size(key_dtype, self.geometry, tokens)?;
            let expected_value = bounded_size(value_dtype, self.geometry, tokens)?;
            if key_bytes != expected_key || value_bytes != expected_value {
                return Err(corrupt("KV frame lengths contradict header geometry"));
            }
            let end = position
                .checked_add(FRAME_HEADER_BYTES)
                .and_then(|v| v.checked_add(key_bytes))
                .and_then(|v| v.checked_add(value_bytes))
                .and_then(|v| v.checked_add(COMMIT_BYTES))
                .ok_or_else(|| corrupt("record offset overflow"))?;
            if end > file_len {
                break;
            }
            self.file.seek(SeekFrom::Start(end - COMMIT_BYTES))?;
            let mut footer = [0u8; 8];
            self.file.read_exact(&mut footer)?;
            if &footer != COMMIT_MAGIC {
                return Err(corrupt("fully written frame has invalid commit footer"));
            }
            let mut hash_bytes = [0u8; 32];
            hash_bytes.copy_from_slice(&header[40..72]);
            let index = FrameIndex {
                tokens,
                key_dtype,
                value_dtype,
                key_bytes,
                value_bytes,
                record_offset: position,
                digest: hash_bytes,
            };
            self.verify_record(&index)?;
            self.frames.push(index);
            position = end;
        }
        self.next_append = position;
        self.trailing_uncommitted = file_len - position;
        if self.trailing_uncommitted > 0 && !self.read_only {
            // Any partial trailing record is not committed. This is the only
            // automatic truncation and is performed only on reopening writable.
            self.file.set_len(position)?;
            self.file.sync_data()?;
        }
        Ok(())
    }

    /// Verify every byte of one record before accepting its reconstruction.
    fn verify_record(&mut self, record: &FrameIndex) -> io::Result<()> {
        let mut hash = digest_start(
            self.geometry,
            record.tokens,
            record.key_dtype,
            record.value_dtype,
            record.key_bytes,
            record.value_bytes,
        );
        self.file
            .seek(SeekFrom::Start(record.record_offset + FRAME_HEADER_BYTES))?;
        let mut remaining = record.key_bytes + record.value_bytes;
        let mut buffer = [0u8; 65536];
        while remaining > 0 {
            let take = (remaining as usize).min(buffer.len());
            self.file.read_exact(&mut buffer[..take])?;
            hash.update(&buffer[..take]);
            remaining -= take as u64;
        }
        if hash.finalize().as_slice() != record.digest {
            return Err(corrupt("committed KV record SHA-256 mismatch"));
        }
        Ok(())
    }

    pub fn geometry(&self) -> AttentionGeometry {
        self.geometry
    }

    pub fn frames(&self) -> &[FrameIndex] {
        &self.frames
    }

    pub fn trailing_uncommitted_bytes(&self) -> u64 {
        self.trailing_uncommitted
    }

    pub fn is_read_only(&self) -> bool {
        self.read_only
    }

    pub fn append(&mut self, key: &RawTensor, value: &RawTensor) -> io::Result<usize> {
        if self.read_only {
            return Err(io::Error::new(
                ErrorKind::PermissionDenied,
                "store is read-only",
            ));
        }
        let shape = key.shape();
        if shape.len() != 3
            || shape != value.shape()
            || shape[1] != self.geometry.kv_heads
            || shape[2] != self.geometry.head_dim
        {
            return Err(corrupt("KV payloads must be [tokens,kv_heads,head_dim]"));
        }
        let tokens = shape[0] as u64;
        let kbytes = bounded_size(key.dtype(), self.geometry, tokens)?;
        let vbytes = bounded_size(value.dtype(), self.geometry, tokens)?;
        if kbytes != key.len_bytes() as u64 || vbytes != value.len_bytes() as u64 {
            return Err(corrupt("KV byte length is incompatible with geometry"));
        }
        let digest = digest_frame(self.geometry, tokens, key, value);
        let mut header = Vec::with_capacity(FRAME_HEADER_BYTES as usize);
        header.extend_from_slice(RECORD_MAGIC);
        header.extend_from_slice(&tokens.to_le_bytes());
        header.extend_from_slice(&[format_byte(key.dtype()), format_byte(value.dtype())]);
        header.extend_from_slice(&[0; 6]);
        header.extend_from_slice(&kbytes.to_le_bytes());
        header.extend_from_slice(&vbytes.to_le_bytes());
        header.extend_from_slice(&digest);
        debug_assert_eq!(header.len(), FRAME_HEADER_BYTES as usize);

        let start = self.next_append;
        self.file.seek(SeekFrom::Start(start))?;
        let operation = (|| -> io::Result<()> {
            self.file.write_all(&header)?;
            self.file.write_all(key.as_le_bytes())?;
            self.file.write_all(value.as_le_bytes())?;
            self.file.write_all(COMMIT_MAGIC)?;
            self.file.sync_data()?;
            Ok(())
        })();
        if let Err(error) = operation {
            // Failed append may have produced a torn record; never publish it.
            let _ = self.file.set_len(start);
            let _ = self.file.sync_data();
            return Err(error);
        }
        self.next_append = start + FRAME_HEADER_BYTES + kbytes + vbytes + COMMIT_BYTES;
        self.frames.push(FrameIndex {
            tokens,
            key_dtype: key.dtype(),
            value_dtype: value.dtype(),
            key_bytes: kbytes,
            value_bytes: vbytes,
            record_offset: start,
            digest,
        });
        Ok(self.frames.len() - 1)
    }

    pub fn read_frame(&mut self, index: usize) -> io::Result<(RawTensor, RawTensor)> {
        let record = self
            .frames
            .get(index)
            .ok_or_else(|| io::Error::new(ErrorKind::NotFound, "frame index out of range"))?
            .clone();
        self.verify_record(&record)?;
        self.file
            .seek(SeekFrom::Start(record.record_offset + FRAME_HEADER_BYTES))?;
        let key_size = usize::try_from(record.key_bytes)
            .map_err(|_| corrupt("key length unsupported on this platform"))?;
        let value_size = usize::try_from(record.value_bytes)
            .map_err(|_| corrupt("value length unsupported on this platform"))?;
        let mut key_bits = vec![0u8; key_size];
        let mut value_bits = vec![0u8; value_size];
        self.file.read_exact(&mut key_bits)?;
        self.file.read_exact(&mut value_bits)?;
        let shape = vec![
            usize::try_from(record.tokens).map_err(|_| corrupt("token count unsupported"))?,
            self.geometry.kv_heads,
            self.geometry.head_dim,
        ];
        let key = RawTensor::from_le_bytes(record.key_dtype, shape.clone(), key_bits)
            .map_err(|_| corrupt("decoded key shape incompatible"))?;
        let value = RawTensor::from_le_bytes(record.value_dtype, shape, value_bits)
            .map_err(|_| corrupt("decoded value shape incompatible"))?;
        Ok((key, value))
    }

    /// Explicit opt-in: hydrates *all* pages into RAM, so may be expensive.
    /// Prefer read_frame for bounded memory when working with large context.
    pub fn load_all_into_memory(&mut self) -> io::Result<ExactKvFrames> {
        let mut cache = ExactKvFrames::new(self.geometry);
        for index in 0..self.frames.len() {
            let (key, value) = self.read_frame(index)?;
            cache
                .append(key, value)
                .map_err(|_| corrupt("decoded frame geometry mismatch"))?;
        }
        Ok(cache)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, OpenOptions};
    use std::io::Write;

    fn geometry() -> AttentionGeometry {
        AttentionGeometry::new(4, 2, 2).unwrap()
    }

    fn raw(dtype: FloatFormat, b: &[u8]) -> RawTensor {
        RawTensor::from_le_bytes(dtype, vec![1, 2, 2], b.to_vec()).unwrap()
    }

    #[test]
    fn exact_bf16_fp16_fp32_roundtrip_and_readonly() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("kv.portable");
        let mut store = PortableKvStore::create(&path, geometry()).unwrap();
        let key = raw(FloatFormat::Bf16, &[0, 0, 0x80, 0x3f, 0, 0x40, 0xc0, 0x7f]);
        let value = raw(FloatFormat::F16, &[0, 0x3c, 0, 0xc0, 1, 0x7e, 0, 0x40]);
        store.append(&key, &value).unwrap();
        let full = RawTensor::from_f32(vec![1, 2, 2], &[-0.0, 1.5, f32::NAN, 42.0]).unwrap();
        store.append(&full, &full).unwrap();
        assert_eq!(store.frames().len(), 2);
        assert_eq!(
            store.read_frame(0).unwrap().0.as_le_bytes(),
            key.as_le_bytes()
        );
        drop(store);

        let mut reopened = PortableKvStore::open_read_only(&path).unwrap();
        assert_eq!(reopened.geometry(), geometry());
        assert!(reopened.is_read_only());
        assert_eq!(
            reopened.read_frame(0).unwrap(),
            (key.clone(), value.clone())
        );
        assert_eq!(
            reopened.read_frame(1).unwrap(),
            (full.clone(), full.clone())
        );
        assert_eq!(reopened.load_all_into_memory().unwrap().tokens(), 2);
        assert_eq!(
            reopened.append(&key, &value).unwrap_err().kind(),
            ErrorKind::PermissionDenied
        );
        assert!(reopened.read_frame(2).is_err());
    }

    #[test]
    fn torn_final_append_ignored_then_repaired_on_rw_open() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("torn.kv");
        let key = raw(FloatFormat::Bf16, &[0; 8]);
        let mut store = PortableKvStore::create(&path, geometry()).unwrap();
        store.append(&key, &key).unwrap();
        drop(store);
        let committed_len = fs::metadata(&path).unwrap().len();
        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        file.write_all(b"KVFRAME1").unwrap();
        file.write_all(&[1, 2, 3]).unwrap();
        file.sync_data().unwrap();
        drop(file);
        let readonly = PortableKvStore::open_read_only(&path).unwrap();
        assert_eq!(readonly.frames().len(), 1);
        assert_eq!(readonly.trailing_uncommitted_bytes(), 11);
        assert_eq!(fs::metadata(&path).unwrap().len(), committed_len + 11);
        drop(readonly);
        let mut repaired = PortableKvStore::open(&path).unwrap();
        assert_eq!(repaired.frames().len(), 1);
        assert_eq!(fs::metadata(&path).unwrap().len(), committed_len);
        repaired.append(&key, &key).unwrap();
        assert_eq!(repaired.frames().len(), 2);
    }

    #[test]
    fn committed_payload_corruption_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("corrupt.kv");
        let key = raw(FloatFormat::Bf16, &[0; 8]);
        let mut store = PortableKvStore::create(&path, geometry()).unwrap();
        store.append(&key, &key).unwrap();
        let offset = store.frames()[0].record_offset + FRAME_HEADER_BYTES;
        drop(store);
        let mut file = OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(offset)).unwrap();
        file.write_all(&[99]).unwrap();
        file.sync_data().unwrap();
        drop(file);
        assert!(PortableKvStore::open_read_only(&path).is_err());
    }

    #[test]
    fn committed_geometry_tampering_rejected_by_sha256() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("header_modified.kv");
        let key = raw(FloatFormat::Bf16, &[0; 8]);
        let mut store = PortableKvStore::create(&path, geometry()).unwrap();
        store.append(&key, &key).unwrap();
        drop(store);
        // Change q_heads from 4 to 6. GQA geometry remains mathematically
        // valid and payload lengths are unchanged: only the digest can catch it.
        let mut file = OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(12)).unwrap();
        file.write_all(&6u64.to_le_bytes()).unwrap();
        file.sync_data().unwrap();
        drop(file);
        assert!(PortableKvStore::open_read_only(&path).is_err());
    }

    #[test]
    fn rejects_overflow_and_zero_length_frames() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("shape.kv");
        let mut store = PortableKvStore::create(&path, geometry()).unwrap();
        let invalid = RawTensor::from_f32(vec![1, 1, 2], &[0.0; 2]).unwrap();
        let valid = RawTensor::from_f32(vec![1, 2, 2], &[0.0; 4]).unwrap();
        assert!(store.append(&invalid, &valid).is_err());
        assert_eq!(store.frames().len(), 0);
    }

    #[test]
    fn existing_store_never_truncated_by_create() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("existing.kv");
        fs::write(&path, b"existing").unwrap();
        assert!(PortableKvStore::create(&path, geometry()).is_err());
        assert_eq!(fs::read(&path).unwrap(), b"existing");
    }
}
