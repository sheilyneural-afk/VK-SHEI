//! Auditable append-only cold storage for Terrsvusion encoded KV pages.
//!
//! The caller supplies exact BF16 byte payloads.
//! This module never transforms model weights or changes KV semantics. Each
//! record is self-describing and committed by a footer; a torn final append is
//! ignored on reopen, while corruption of a committed record is rejected.

use std::collections::{HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{self, ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::MmapOptions;
use rayon::prelude::*;
use sha2::{Digest, Sha256};

const HEADER_MAGIC: &[u8; 8] = b"TSKVPG04";
const FOOTER_MAGIC: &[u8; 8] = b"TSKVCMIT";
const VERSION: u32 = 3;
const HEADER_LEN: usize = 104;
const FOOTER_LEN: usize = 56;
const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PageGeometry {
    pub layer: u32,
    pub token_start: u64,
    pub tokens: u32,
    pub heads: u32,
    pub head_dim: u32,
    pub tile: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ColdPagePayload {
    pub key: Vec<u8>,
    pub value: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PageMeta {
    pub page_id: u64,
    pub geometry: PageGeometry,
    pub record_offset: u64,
    pub payload_offset: u64,
    pub key_hashes_offset: u64,
    pub value_hashes_offset: u64,
    pub key_len: u64,
    pub value_len: u64,
    pub key_checksum: u64,
    pub value_checksum: u64,
    pub key_hashes_checksum: u64,
    pub value_hashes_checksum: u64,
    pub record_checksum: u64,
    pub record_digest: [u8; 32],
    pub record_len: u64,
}

/// Complete immutable record assembled away from the inference thread. Its
/// offsets are relative to the beginning of the record until append.
pub struct PreparedPage {
    meta: PageMeta,
    record: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ColdStoreStats {
    pub pages: usize,
    pub committed_bytes: u64,
    pub reads: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub evictions: u64,
    pub hot_pages: usize,
    pub hot_bytes: usize,
    pub hot_capacity: usize,
    pub bytes_read: u64,
    pub partial_value_reads: u64,
    pub partial_value_payload_bytes: u64,
    pub partial_value_bytes_avoided: u64,
    pub partial_key_reads: u64,
    pub partial_key_payload_bytes: u64,
    pub partial_key_bytes_avoided: u64,
    pub cryptographic_digests_verified: u64,
}

/// A single-file append-only page store with bounded decoded-byte residency.
pub struct ColdStore {
    path: PathBuf,
    file: File,
    pages: Vec<PageMeta>,
    hot_capacity: usize,
    hot: HashMap<u64, Arc<ColdPagePayload>>,
    lru: VecDeque<u64>,
    reads: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
    committed_bytes: u64,
    bytes_read: u64,
    partial_value_reads: u64,
    partial_value_payload_bytes: u64,
    partial_value_bytes_avoided: u64,
    partial_key_reads: u64,
    partial_key_payload_bytes: u64,
    partial_key_bytes_avoided: u64,
    cryptographic_digests_verified: u64,
}

impl ColdStore {
    pub fn create(path: impl AsRef<Path>, hot_capacity: usize) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .append(true)
            .open(&path)?;
        Ok(Self::empty(path, file, hot_capacity))
    }

    pub fn open(path: impl AsRef<Path>, hot_capacity: usize) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new().read(true).append(true).open(&path)?;
        let mut store = Self::empty(path, file, hot_capacity);
        store.rebuild_catalogue()?;
        Ok(store)
    }

    fn empty(path: PathBuf, file: File, hot_capacity: usize) -> Self {
        Self {
            path,
            file,
            pages: Vec::new(),
            hot_capacity,
            hot: HashMap::new(),
            lru: VecDeque::new(),
            reads: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
            committed_bytes: 0,
            bytes_read: 0,
            partial_value_reads: 0,
            partial_value_payload_bytes: 0,
            partial_value_bytes_avoided: 0,
            partial_key_reads: 0,
            partial_key_payload_bytes: 0,
            partial_key_bytes_avoided: 0,
            cryptographic_digests_verified: 0,
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn metadata(&self) -> &[PageMeta] {
        &self.pages
    }

    pub fn stats(&self) -> ColdStoreStats {
        ColdStoreStats {
            pages: self.pages.len(),
            committed_bytes: self.committed_bytes,
            reads: self.reads,
            cache_hits: self.hits,
            cache_misses: self.misses,
            evictions: self.evictions,
            hot_pages: self.hot.len(),
            hot_bytes: self
                .hot
                .values()
                .map(|page| page.key.len() + page.value.len())
                .sum(),
            hot_capacity: self.hot_capacity,
            bytes_read: self.bytes_read,
            partial_value_reads: self.partial_value_reads,
            partial_value_payload_bytes: self.partial_value_payload_bytes,
            partial_value_bytes_avoided: self.partial_value_bytes_avoided,
            partial_key_reads: self.partial_key_reads,
            partial_key_payload_bytes: self.partial_key_payload_bytes,
            partial_key_bytes_avoided: self.partial_key_bytes_avoided,
            cryptographic_digests_verified: self.cryptographic_digests_verified,
        }
    }

    /// Verify all committed records in parallel through a temporary immutable
    /// mapping. The mapping is dropped after the audit; inference reads only
    /// the explicitly selected BF16 ranges through the canonical partial-I/O
    /// path.
    pub fn audit_mapped(&mut self) -> io::Result<()> {
        self.file.sync_data()?;
        if self.file.metadata()?.len() == 0 {
            return Err(invalid("almacén KV vacío no se puede mapear"));
        }
        let mapping = unsafe { MmapOptions::new().map(&self.file)? };
        self.pages
            .par_iter()
            .try_for_each(|meta| verify_mapped_record(&mapping, meta))?;
        self.cryptographic_digests_verified += self.pages.len() as u64;
        Ok(())
    }

    /// Append and commit one converted K/V page. IDs are monotonic and stable.
    pub fn append_page(
        &mut self,
        geometry: PageGeometry,
        key: &[u8],
        value: &[u8],
        durable: bool,
    ) -> io::Result<PageMeta> {
        let page_id = self.pages.len() as u64;
        let prepared = Self::prepare_page(page_id, geometry, key, value)?;
        self.append_prepared(prepared, durable)
    }

    /// Performs validation, token hashing and record assembly without access
    /// to store state, so bounded background workers can do the CPU work.
    pub fn prepare_page(
        page_id: u64,
        geometry: PageGeometry,
        key: &[u8],
        value: &[u8],
    ) -> io::Result<PreparedPage> {
        validate_geometry(geometry, key, value)?;
        let key_checksum = checksum(key);
        let value_checksum = checksum(value);
        let key_hashes = token_hashes(key, geometry)?;
        let value_hashes = token_hashes(value, geometry)?;
        let key_hash_bytes = u64_slice_bytes(&key_hashes);
        let value_hash_bytes = u64_slice_bytes(&value_hashes);
        let key_hashes_checksum = checksum(&key_hash_bytes);
        let value_hashes_checksum = checksum(&value_hash_bytes);
        let mut header = Vec::with_capacity(HEADER_LEN);
        header.extend_from_slice(HEADER_MAGIC);
        put_u32(&mut header, VERSION);
        put_u32(&mut header, HEADER_LEN as u32);
        put_u64(&mut header, page_id);
        put_u32(&mut header, geometry.layer);
        put_u32(&mut header, 0);
        put_u64(&mut header, geometry.token_start);
        put_u32(&mut header, geometry.tokens);
        put_u32(&mut header, geometry.heads);
        put_u32(&mut header, geometry.head_dim);
        put_u32(&mut header, geometry.tile);
        put_u64(&mut header, key.len() as u64);
        put_u64(&mut header, value.len() as u64);
        put_u64(&mut header, key_checksum);
        put_u64(&mut header, value_checksum);
        put_u64(&mut header, key_hashes_checksum);
        put_u64(&mut header, value_hashes_checksum);
        debug_assert_eq!(header.len(), HEADER_LEN);

        let record_checksum =
            checksum_many(&[&header, &key_hash_bytes, &value_hash_bytes, key, value]);
        let record_digest =
            cryptographic_digest(&[&header, &key_hash_bytes, &value_hash_bytes, key, value]);
        let record_len = (HEADER_LEN
            + key_hash_bytes.len()
            + value_hash_bytes.len()
            + key.len()
            + value.len()
            + FOOTER_LEN) as u64;
        let mut footer = Vec::with_capacity(FOOTER_LEN);
        footer.extend_from_slice(FOOTER_MAGIC);
        put_u64(&mut footer, record_checksum);
        put_u64(&mut footer, record_len);
        footer.extend_from_slice(&record_digest);

        let mut record = Vec::with_capacity(record_len as usize);
        record.extend_from_slice(&header);
        record.extend_from_slice(&key_hash_bytes);
        record.extend_from_slice(&value_hash_bytes);
        record.extend_from_slice(key);
        record.extend_from_slice(value);
        record.extend_from_slice(&footer);
        debug_assert_eq!(record.len(), record_len as usize);

        let key_hashes_offset = HEADER_LEN as u64;
        let value_hashes_offset = key_hashes_offset + key_hash_bytes.len() as u64;
        let meta = PageMeta {
            page_id,
            geometry,
            record_offset: 0,
            payload_offset: value_hashes_offset + value_hash_bytes.len() as u64,
            key_hashes_offset,
            value_hashes_offset,
            key_len: key.len() as u64,
            value_len: value.len() as u64,
            key_checksum,
            value_checksum,
            key_hashes_checksum,
            value_hashes_checksum,
            record_checksum,
            record_digest,
            record_len,
        };
        Ok(PreparedPage { meta, record })
    }

    /// Appends one already assembled record in monotonic order and converts
    /// its relative offsets into authoritative file offsets.
    pub fn append_prepared(
        &mut self,
        mut prepared: PreparedPage,
        durable: bool,
    ) -> io::Result<PageMeta> {
        if prepared.meta.page_id != self.pages.len() as u64 {
            return Err(invalid("secuencia preparada KV no monótona"));
        }
        if prepared.record.len() as u64 != prepared.meta.record_len {
            return Err(invalid("registro KV preparado con longitud inválida"));
        }
        let record_offset = self.file.metadata()?.len();
        self.file.write_all(&prepared.record)?;
        self.file.flush()?;
        if durable {
            self.file.sync_data()?;
        }
        prepared.meta.record_offset = record_offset;
        prepared.meta.key_hashes_offset += record_offset;
        prepared.meta.value_hashes_offset += record_offset;
        prepared.meta.payload_offset += record_offset;
        self.committed_bytes += prepared.meta.record_len;
        let meta = prepared.meta;
        self.pages.push(meta.clone());
        Ok(meta)
    }

    /// Read exactly one page, verify all checksums, and place it in the LRU.
    pub fn read_page(&mut self, page_id: u64) -> io::Result<Arc<ColdPagePayload>> {
        if let Some(page) = self.hot.get(&page_id).cloned() {
            self.hits += 1;
            self.touch(page_id);
            return Ok(page);
        }
        let meta = self
            .pages
            .get(page_id as usize)
            .filter(|meta| meta.page_id == page_id)
            .cloned()
            .ok_or_else(|| {
                io::Error::new(ErrorKind::NotFound, format!("página KV {page_id} ausente"))
            })?;
        self.misses += 1;
        self.reads += 1;
        let payload = Arc::new(read_and_verify(&self.file, &meta)?);
        self.cryptographic_digests_verified += 1;
        self.bytes_read += meta.record_len - FOOTER_LEN as u64;
        if self.hot_capacity > 0 {
            while self.hot.len() >= self.hot_capacity {
                if let Some(oldest) = self.lru.pop_front()
                    && self.hot.remove(&oldest).is_some()
                {
                    self.evictions += 1;
                }
            }
            self.hot.insert(page_id, payload.clone());
            self.touch(page_id);
        }
        Ok(payload)
    }

    /// Read and verify only K. Used by exact-mass planning so V is not pulled
    /// from cold storage before the dimensional selector asks for it.
    pub fn read_key(&mut self, page_id: u64) -> io::Result<Vec<u8>> {
        let meta = self.page_meta(page_id)?;
        let mut key = vec![0_u8; meta.key_len as usize];
        read_exact_at(&self.file, meta.payload_offset, &mut key)?;
        if checksum(&key) != meta.key_checksum {
            return Err(invalid("checksum K frío inválido"));
        }
        self.reads += 1;
        self.misses += 1;
        self.bytes_read += meta.key_len;
        Ok(key)
    }

    /// Read and verify only V after selection.
    pub fn read_value(&mut self, page_id: u64) -> io::Result<Vec<u8>> {
        let meta = self.page_meta(page_id)?;
        let mut value = vec![0_u8; meta.value_len as usize];
        read_exact_at(&self.file, meta.payload_offset + meta.key_len, &mut value)?;
        if checksum(&value) != meta.value_checksum {
            return Err(invalid("checksum V frío inválido"));
        }
        self.reads += 1;
        self.misses += 1;
        self.bytes_read += meta.value_len;
        Ok(value)
    }

    /// Read only requested token coordinates from V and verify each one
    /// against the committed token-hash table. Output layout remains
    /// `[head][requested_token][bytes_per_head_token]`, so it can be turned
    /// directly into an exact BF16 tensor without materializing the page.
    pub fn read_value_tokens(
        &mut self,
        page_id: u64,
        token_indices: &[usize],
    ) -> io::Result<Vec<u8>> {
        self.read_token_subset(page_id, token_indices, false)
    }

    /// K counterpart of `read_value_tokens`, with the same exact head-major
    /// output layout and per-token integrity proof.
    pub fn read_key_tokens(
        &mut self,
        page_id: u64,
        token_indices: &[usize],
    ) -> io::Result<Vec<u8>> {
        self.read_token_subset(page_id, token_indices, true)
    }

    fn read_token_subset(
        &mut self,
        page_id: u64,
        token_indices: &[usize],
        key_payload: bool,
    ) -> io::Result<Vec<u8>> {
        let meta = self.page_meta(page_id)?;
        let tokens = meta.geometry.tokens as usize;
        if token_indices.is_empty() || token_indices.iter().any(|token| *token >= tokens) {
            return Err(invalid("coordenadas KV parciales inválidas"));
        }
        if token_indices.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(invalid(
                "coordenadas KV parciales deben ser estrictamente crecientes",
            ));
        }
        if token_indices.len() == tokens && token_indices.iter().copied().eq(0..tokens) {
            return if key_payload {
                self.read_key(page_id)
            } else {
                self.read_value(page_id)
            };
        }
        let (hashes_offset, hashes_checksum, payload_offset, payload_len) = if key_payload {
            (
                meta.key_hashes_offset,
                meta.key_hashes_checksum,
                meta.payload_offset,
                meta.key_len,
            )
        } else {
            (
                meta.value_hashes_offset,
                meta.value_hashes_checksum,
                meta.payload_offset + meta.key_len,
                meta.value_len,
            )
        };
        let hashes = self.read_hash_table(hashes_offset, tokens, hashes_checksum)?;
        let heads = meta.geometry.heads as usize;
        let per_head_token = payload_bytes_per_head_token(payload_len, meta.geometry)?;
        let mut output = Vec::with_capacity(heads * token_indices.len() * per_head_token);
        for head in 0..heads {
            let mut run_start = 0_usize;
            while run_start < token_indices.len() {
                let mut run_end = run_start + 1;
                while run_end < token_indices.len()
                    && token_indices[run_end] == token_indices[run_end - 1] + 1
                {
                    run_end += 1;
                }
                let source = payload_offset
                    + ((head * tokens + token_indices[run_start]) * per_head_token) as u64;
                let mut bytes = vec![0_u8; (run_end - run_start) * per_head_token];
                read_exact_at(&self.file, source, &mut bytes)?;
                output.extend_from_slice(&bytes);
                run_start = run_end;
            }
        }
        // Hashes are defined over all heads for a token, while output is
        // head-major. Reassemble each selected token only for verification.
        for (selected_index, token) in token_indices.iter().copied().enumerate() {
            let mut token_bytes = Vec::with_capacity(heads * per_head_token);
            for head in 0..heads {
                let start = (head * token_indices.len() + selected_index) * per_head_token;
                token_bytes.extend_from_slice(&output[start..start + per_head_token]);
            }
            if checksum(&token_bytes) != hashes[token] {
                return Err(invalid(format!(
                    "checksum {} token {token} inválido en página {page_id}",
                    if key_payload { "K" } else { "V" }
                )));
            }
        }
        self.reads += 1;
        self.misses += 1;
        self.bytes_read += (tokens * 8 + output.len()) as u64;
        if key_payload {
            self.partial_key_reads += 1;
            self.partial_key_payload_bytes += output.len() as u64;
            self.partial_key_bytes_avoided += meta.key_len - output.len() as u64;
        } else {
            self.partial_value_reads += 1;
            self.partial_value_payload_bytes += output.len() as u64;
            self.partial_value_bytes_avoided += meta.value_len - output.len() as u64;
        }
        Ok(output)
    }

    fn read_hash_table(
        &self,
        offset: u64,
        tokens: usize,
        expected_checksum: u64,
    ) -> io::Result<Vec<u64>> {
        let mut bytes = vec![0_u8; tokens * 8];
        read_exact_at(&self.file, offset, &mut bytes)?;
        if checksum(&bytes) != expected_checksum {
            return Err(invalid("tabla de hashes por token inválida"));
        }
        Ok(bytes
            .chunks_exact(8)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
            .collect())
    }

    fn page_meta(&self, page_id: u64) -> io::Result<PageMeta> {
        self.pages
            .get(page_id as usize)
            .filter(|meta| meta.page_id == page_id)
            .cloned()
            .ok_or_else(|| {
                io::Error::new(ErrorKind::NotFound, format!("página KV {page_id} ausente"))
            })
    }

    /// Re-read every committed page and fail on the first integrity mismatch.
    pub fn audit(&self) -> io::Result<()> {
        for meta in &self.pages {
            read_and_verify(&self.file, meta)?;
        }
        Ok(())
    }

    fn touch(&mut self, page_id: u64) {
        if let Some(position) = self.lru.iter().position(|&id| id == page_id) {
            self.lru.remove(position);
        }
        self.lru.push_back(page_id);
    }

    fn rebuild_catalogue(&mut self) -> io::Result<()> {
        let file_len = self.file.metadata()?.len();
        let mut offset = 0_u64;
        while offset < file_len {
            let remaining = file_len - offset;
            if remaining < HEADER_LEN as u64 {
                break; // torn uncommitted tail
            }
            let mut header = vec![0_u8; HEADER_LEN];
            read_exact_at(&self.file, offset, &mut header)?;
            let parsed = parse_header(&header, offset)?;
            let hash_tables_len = u64::from(parsed.geometry.tokens) * 8 * 2;
            let record_len = HEADER_LEN as u64
                + hash_tables_len
                + parsed.key_len
                + parsed.value_len
                + FOOTER_LEN as u64;
            if record_len > remaining {
                break; // footer cannot exist, so the append was never committed
            }
            let mut footer = [0_u8; FOOTER_LEN];
            read_exact_at(
                &self.file,
                offset + record_len - FOOTER_LEN as u64,
                &mut footer,
            )?;
            if &footer[..8] != FOOTER_MAGIC {
                return Err(invalid("pie de commit KV inválido"));
            }
            let record_checksum = get_u64(&footer, 8)?;
            if get_u64(&footer, 16)? != record_len {
                return Err(invalid("longitud de registro KV inconsistente"));
            }
            if parsed.page_id != self.pages.len() as u64 {
                return Err(invalid("secuencia de page_id KV no monótona"));
            }
            let meta = PageMeta {
                record_checksum,
                record_digest: footer[24..56].try_into().unwrap(),
                record_len,
                ..parsed
            };
            read_and_verify(&self.file, &meta)?;
            self.pages.push(meta);
            self.committed_bytes += record_len;
            offset += record_len;
        }
        // Only bytes after the last commit footer are discarded. Without this
        // recovery step a later append would sit behind an unreadable torn
        // tail and become unreachable on the next restart.
        if offset < file_len {
            self.file.set_len(offset)?;
            self.file.sync_data()?;
        }
        Ok(())
    }
}

fn validate_geometry(geometry: PageGeometry, key: &[u8], value: &[u8]) -> io::Result<()> {
    if geometry.tokens == 0 || geometry.heads == 0 || geometry.head_dim == 0 || geometry.tile == 0 {
        return Err(invalid("geometría KV contiene cero"));
    }
    if !geometry.head_dim.is_multiple_of(geometry.tile) {
        return Err(invalid("head_dim no es múltiplo de tile"));
    }
    if key.is_empty() || value.is_empty() {
        return Err(invalid("la página KV convertida no puede estar vacía"));
    }
    let _ = payload_bytes_per_head_token(key.len() as u64, geometry)?;
    let _ = payload_bytes_per_head_token(value.len() as u64, geometry)?;
    Ok(())
}

fn payload_bytes_per_head_token(length: u64, geometry: PageGeometry) -> io::Result<usize> {
    let elements = u64::from(geometry.heads) * u64::from(geometry.tokens);
    if elements == 0 || !length.is_multiple_of(elements) {
        return Err(invalid("payload KV no divisible por cabezales y tokens"));
    }
    usize::try_from(length / elements).map_err(|_| invalid("token KV excede usize"))
}

fn token_hashes(payload: &[u8], geometry: PageGeometry) -> io::Result<Vec<u64>> {
    let heads = geometry.heads as usize;
    let tokens = geometry.tokens as usize;
    let per_head_token = payload_bytes_per_head_token(payload.len() as u64, geometry)?;
    let mut hashes = Vec::with_capacity(tokens);
    for token in 0..tokens {
        let mut state = FNV_OFFSET;
        for head in 0..heads {
            let start = (head * tokens + token) * per_head_token;
            for byte in &payload[start..start + per_head_token] {
                state ^= u64::from(*byte);
                state = state.wrapping_mul(FNV_PRIME);
            }
        }
        hashes.push(state);
    }
    Ok(hashes)
}

fn u64_slice_bytes(values: &[u64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn parse_header(header: &[u8], record_offset: u64) -> io::Result<PageMeta> {
    if header.len() != HEADER_LEN || &header[..8] != HEADER_MAGIC {
        return Err(invalid("cabecera de página KV inválida"));
    }
    if get_u32(header, 8)? != VERSION || get_u32(header, 12)? as usize != HEADER_LEN {
        return Err(invalid("versión de página KV no soportada"));
    }
    let geometry = PageGeometry {
        layer: get_u32(header, 24)?,
        token_start: get_u64(header, 32)?,
        tokens: get_u32(header, 40)?,
        heads: get_u32(header, 44)?,
        head_dim: get_u32(header, 48)?,
        tile: get_u32(header, 52)?,
    };
    let key_len = get_u64(header, 56)?;
    let value_len = get_u64(header, 64)?;
    if key_len == 0 || value_len == 0 {
        return Err(invalid("longitud KV fría inválida"));
    }
    let key_hashes_offset = record_offset + HEADER_LEN as u64;
    let value_hashes_offset = key_hashes_offset + u64::from(geometry.tokens) * 8;
    Ok(PageMeta {
        page_id: get_u64(header, 16)?,
        geometry,
        record_offset,
        payload_offset: value_hashes_offset + u64::from(geometry.tokens) * 8,
        key_hashes_offset,
        value_hashes_offset,
        key_len,
        value_len,
        key_checksum: get_u64(header, 72)?,
        value_checksum: get_u64(header, 80)?,
        key_hashes_checksum: get_u64(header, 88)?,
        value_hashes_checksum: get_u64(header, 96)?,
        record_checksum: 0,
        record_digest: [0; 32],
        record_len: 0,
    })
}

fn read_and_verify(file: &File, meta: &PageMeta) -> io::Result<ColdPagePayload> {
    let key_len = usize::try_from(meta.key_len).map_err(|_| invalid("K excede usize"))?;
    let value_len = usize::try_from(meta.value_len).map_err(|_| invalid("V excede usize"))?;
    let mut header = vec![0_u8; HEADER_LEN];
    let mut key_hashes = vec![0_u8; meta.geometry.tokens as usize * 8];
    let mut value_hashes = vec![0_u8; meta.geometry.tokens as usize * 8];
    let mut key = vec![0_u8; key_len];
    let mut value = vec![0_u8; value_len];
    read_exact_at(file, meta.record_offset, &mut header)?;
    read_exact_at(file, meta.key_hashes_offset, &mut key_hashes)?;
    read_exact_at(file, meta.value_hashes_offset, &mut value_hashes)?;
    read_exact_at(file, meta.payload_offset, &mut key)?;
    read_exact_at(file, meta.payload_offset + meta.key_len, &mut value)?;
    if checksum(&key) != meta.key_checksum
        || checksum(&value) != meta.value_checksum
        || checksum(&key_hashes) != meta.key_hashes_checksum
        || checksum(&value_hashes) != meta.value_hashes_checksum
    {
        return Err(invalid(format!(
            "checksum K/V inválido en página {}",
            meta.page_id
        )));
    }
    let expected_key_hashes = u64_slice_bytes(&token_hashes(&key, meta.geometry)?);
    let expected_value_hashes = u64_slice_bytes(&token_hashes(&value, meta.geometry)?);
    if key_hashes != expected_key_hashes || value_hashes != expected_value_hashes {
        return Err(invalid(format!(
            "tabla de hashes por token inconsistente en página {}",
            meta.page_id
        )));
    }
    if checksum_many(&[&header, &key_hashes, &value_hashes, &key, &value]) != meta.record_checksum {
        return Err(invalid(format!(
            "checksum de registro inválido en página {}",
            meta.page_id
        )));
    }
    if cryptographic_digest(&[&header, &key_hashes, &value_hashes, &key, &value])
        != meta.record_digest
    {
        return Err(invalid(format!(
            "digest criptográfico inválido en página {}",
            meta.page_id
        )));
    }
    Ok(ColdPagePayload { key, value })
}

fn verify_mapped_record(mapping: &[u8], meta: &PageMeta) -> io::Result<()> {
    let range = |offset: u64, length: usize| -> io::Result<&[u8]> {
        let start = usize::try_from(offset).map_err(|_| invalid("offset KV excede usize"))?;
        mapping
            .get(start..start.saturating_add(length))
            .ok_or_else(|| invalid("registro KV mapeado truncado"))
    };
    let hash_len = meta.geometry.tokens as usize * 8;
    let header = range(meta.record_offset, HEADER_LEN)?;
    let key_hashes = range(meta.key_hashes_offset, hash_len)?;
    let value_hashes = range(meta.value_hashes_offset, hash_len)?;
    let key = range(meta.payload_offset, meta.key_len as usize)?;
    let value = range(meta.payload_offset + meta.key_len, meta.value_len as usize)?;
    let footer = range(
        meta.record_offset + meta.record_len - FOOTER_LEN as u64,
        FOOTER_LEN,
    )?;
    if &footer[..8] != FOOTER_MAGIC
        || get_u64(footer, 8)? != meta.record_checksum
        || get_u64(footer, 16)? != meta.record_len
        || footer[24..56] != meta.record_digest
    {
        return Err(invalid(format!(
            "commit mapeado inválido en página {}",
            meta.page_id
        )));
    }
    if checksum(key) != meta.key_checksum
        || checksum(value) != meta.value_checksum
        || checksum(key_hashes) != meta.key_hashes_checksum
        || checksum(value_hashes) != meta.value_hashes_checksum
        || checksum_many(&[header, key_hashes, value_hashes, key, value]) != meta.record_checksum
    {
        return Err(invalid(format!(
            "checksum mapeado inválido en página {}",
            meta.page_id
        )));
    }
    if cryptographic_digest(&[header, key_hashes, value_hashes, key, value]) != meta.record_digest {
        return Err(invalid(format!(
            "digest criptográfico mapeado inválido en página {}",
            meta.page_id
        )));
    }
    let expected_key_hashes = u64_slice_bytes(&token_hashes(key, meta.geometry)?);
    let expected_value_hashes = u64_slice_bytes(&token_hashes(value, meta.geometry)?);
    if key_hashes != expected_key_hashes || value_hashes != expected_value_hashes {
        return Err(invalid(format!(
            "hash por token mapeado inválido en página {}",
            meta.page_id
        )));
    }
    Ok(())
}

fn read_exact_at(file: &File, offset: u64, output: &mut [u8]) -> io::Result<()> {
    let mut reader = file.try_clone()?;
    reader.seek(SeekFrom::Start(offset))?;
    reader.read_exact(output)
}

fn checksum(bytes: &[u8]) -> u64 {
    checksum_many(&[bytes])
}

fn checksum_many(parts: &[&[u8]]) -> u64 {
    let mut state = FNV_OFFSET;
    for part in parts {
        for &byte in *part {
            state ^= byte as u64;
            state = state.wrapping_mul(FNV_PRIME);
        }
    }
    state
}

fn cryptographic_digest(parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part);
    }
    hasher.finalize().into()
}

fn put_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}
fn put_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_le_bytes());
}
fn get_u32(input: &[u8], offset: usize) -> io::Result<u32> {
    let bytes: [u8; 4] = input
        .get(offset..offset + 4)
        .ok_or_else(|| invalid("u32 truncado"))?
        .try_into()
        .unwrap();
    Ok(u32::from_le_bytes(bytes))
}
fn get_u64(input: &[u8], offset: usize) -> io::Result<u64> {
    let bytes: [u8; 8] = input
        .get(offset..offset + 8)
        .ok_or_else(|| invalid("u64 truncado"))?
        .try_into()
        .unwrap();
    Ok(u64::from_le_bytes(bytes))
}
fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "terrsvusion-cold-{name}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn geometry(page: u64) -> PageGeometry {
        PageGeometry {
            layer: 3,
            token_start: page * 4,
            tokens: 4,
            heads: 2,
            head_dim: 4,
            tile: 4,
        }
    }

    #[test]
    fn mapped_audit_verifies_committed_payload_without_resident_cache() {
        let path = temp_path("mapped");
        let mut store = ColdStore::create(&path, 0).unwrap();
        let key = vec![7_u8; 64];
        let value = vec![11_u8; 64];
        store.append_page(geometry(0), &key, &value, false).unwrap();
        store.audit_mapped().unwrap();
        let stats = store.stats();
        assert_eq!(stats.reads, 0);
        assert_eq!(stats.hot_bytes, 0);
        assert_eq!(stats.cryptographic_digests_verified, 1);
        drop(store);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn append_reopen_selective_read_and_audit() {
        let path = temp_path("roundtrip");
        let mut store = ColdStore::create(&path, 2).unwrap();
        for page in 0..4 {
            let key = vec![page as u8; 32];
            let value = vec![100 + page as u8; 32];
            let meta = store
                .append_page(geometry(page), &key, &value, false)
                .unwrap();
            assert_eq!(meta.page_id, page);
        }
        drop(store);
        let mut reopened = ColdStore::open(&path, 2).unwrap();
        assert_eq!(reopened.metadata().len(), 4);
        assert_eq!(reopened.read_page(2).unwrap().key, vec![2; 32]);
        assert_eq!(reopened.read_page(2).unwrap().value, vec![102; 32]);
        reopened.audit().unwrap();
        assert_eq!(reopened.stats().reads, 1);
        assert_eq!(reopened.stats().cache_hits, 1);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn selected_value_tokens_keep_head_major_layout_and_detect_corruption() {
        let path = temp_path("selected-tokens");
        let mut store = ColdStore::create(&path, 0).unwrap();
        let mut key = Vec::new();
        let mut value = Vec::new();
        for head in 0..2_u8 {
            for token in 0..4_u8 {
                key.extend_from_slice(&[100 + head * 10 + token; 4]);
                value.extend_from_slice(&[head * 10 + token; 4]);
            }
        }
        let meta = store.append_page(geometry(0), &key, &value, true).unwrap();
        assert_eq!(
            store.read_value_tokens(0, &[1, 3]).unwrap(),
            [vec![1; 4], vec![3; 4], vec![11; 4], vec![13; 4]].concat()
        );
        let stats = store.stats();
        assert_eq!(stats.partial_value_reads, 1);
        assert_eq!(stats.partial_value_payload_bytes, 16);
        assert_eq!(stats.partial_value_bytes_avoided, 16);
        assert_eq!(stats.bytes_read, 48);
        assert_eq!(
            store.read_key_tokens(0, &[0, 2]).unwrap(),
            [vec![100; 4], vec![102; 4], vec![110; 4], vec![112; 4]].concat()
        );
        let stats = store.stats();
        assert_eq!(stats.partial_key_reads, 1);
        assert_eq!(stats.partial_key_payload_bytes, 16);
        assert_eq!(stats.partial_key_bytes_avoided, 16);

        let corrupt_key_offset = meta.payload_offset + 2 * 4;
        let mut key_file = OpenOptions::new().write(true).open(&path).unwrap();
        key_file.seek(SeekFrom::Start(corrupt_key_offset)).unwrap();
        key_file.write_all(&[99]).unwrap();
        assert_eq!(
            store.read_key_tokens(0, &[2]).unwrap_err().kind(),
            ErrorKind::InvalidData
        );
        drop(key_file);

        let corrupt_offset = meta.payload_offset + meta.key_len + 3 * 4;
        let mut file = OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(corrupt_offset)).unwrap();
        file.write_all(&[99]).unwrap();
        assert_eq!(
            store.read_value_tokens(0, &[3]).unwrap_err().kind(),
            ErrorKind::InvalidData
        );
        drop(file);
        drop(store);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn lru_is_strictly_bounded() {
        let path = temp_path("lru");
        let mut store = ColdStore::create(&path, 2).unwrap();
        for page in 0..3 {
            store
                .append_page(geometry(page), &[page as u8; 32], &[page as u8; 32], false)
                .unwrap();
        }
        store.read_page(0).unwrap();
        store.read_page(1).unwrap();
        store.read_page(0).unwrap();
        store.read_page(2).unwrap();
        assert!(store.hot.contains_key(&0));
        assert!(!store.hot.contains_key(&1));
        assert!(store.hot.contains_key(&2));
        assert_eq!(store.stats().hot_pages, 2);
        assert_eq!(store.stats().evictions, 1);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn torn_final_append_is_not_catalogued() {
        let path = temp_path("torn");
        let mut store = ColdStore::create(&path, 0).unwrap();
        store
            .append_page(geometry(0), &[1; 32], &[2; 32], true)
            .unwrap();
        drop(store);
        OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(HEADER_MAGIC)
            .unwrap();
        let mut reopened = ColdStore::open(&path, 0).unwrap();
        assert_eq!(reopened.metadata().len(), 1);
        assert_eq!(
            reopened.stats().committed_bytes,
            (HEADER_LEN + 4 * 8 * 2 + 64 + FOOTER_LEN) as u64
        );
        reopened
            .append_page(geometry(1), &[3; 32], &[4; 32], true)
            .unwrap();
        drop(reopened);
        let reopened_again = ColdStore::open(&path, 0).unwrap();
        assert_eq!(reopened_again.metadata().len(), 2);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn committed_corruption_is_rejected() {
        let path = temp_path("corrupt");
        let mut store = ColdStore::create(&path, 0).unwrap();
        let meta = store
            .append_page(geometry(0), &[7; 32], &[9; 32], true)
            .unwrap();
        drop(store);
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        file.seek(SeekFrom::Start(meta.payload_offset + 4)).unwrap();
        file.write_all(&[8]).unwrap();
        let error = match ColdStore::open(&path, 0) {
            Ok(_) => panic!("la corrupción comprometida fue aceptada"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), ErrorKind::InvalidData);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn cryptographic_digest_corruption_is_rejected_even_when_fast_checksums_match() {
        let path = temp_path("digest-corrupt");
        let mut store = ColdStore::create(&path, 0).unwrap();
        let meta = store
            .append_page(geometry(0), &[11; 32], &[13; 32], true)
            .unwrap();
        drop(store);

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let digest_byte = meta.record_offset + meta.record_len - 1;
        file.seek(SeekFrom::Start(digest_byte)).unwrap();
        let mut original = [0_u8; 1];
        file.read_exact(&mut original).unwrap();
        file.seek(SeekFrom::Start(digest_byte)).unwrap();
        file.write_all(&[original[0] ^ 0xff]).unwrap();
        file.sync_data().unwrap();

        let error = match ColdStore::open(&path, 0) {
            Ok(_) => panic!("la corrupción del digest criptográfico fue aceptada"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), ErrorKind::InvalidData);
        std::fs::remove_file(path).unwrap();
    }
}
