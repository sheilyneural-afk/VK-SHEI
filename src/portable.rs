//! Backend-neutral, bit-preserving floating-point KV frames.
//!
//! This is a portable CPU *reference*, not a replacement for the audited
//! BF16 configured-model inference path, and not a claim of GPU or LLM compatibility.
//! Stored BF16/F16/F32 bits remain untouched; calculations use f64 accumulation
//! and return f32. FP16/BF16/F32 output precision is not emulated bit-for-bit.

use half::{bf16, f16};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{error::Error, fmt, sync::Arc};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FloatFormat {
    Bf16,
    F16,
    F32,
}

impl FloatFormat {
    pub const fn bytes_per_element(self) -> usize {
        match self {
            Self::Bf16 | Self::F16 => 2,
            Self::F32 => 4,
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::F16 => "FP16",
            Self::F32 => "FP32",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PortableError {
    InvalidGeometry(&'static str),
    ShapeMismatch(&'static str),
    NonFinite(&'static str),
    UnsupportedDevice(String),
    BackendFailure(String),
}

impl fmt::Display for PortableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGeometry(msg) => write!(f, "invalid geometry: {msg}"),
            Self::ShapeMismatch(msg) => write!(f, "shape mismatch: {msg}"),
            Self::NonFinite(msg) => write!(f, "non-finite computation: {msg}"),
            Self::UnsupportedDevice(name) => write!(f, "no native kernel registered for {name}"),
            Self::BackendFailure(message) => write!(f, "backend failure: {message}"),
        }
    }
}

impl Error for PortableError {}

/// Owned immutable, little-endian tensor bits. No quantization or conversion
/// happens when constructing or sharing this value. NaN payloads are preserved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawTensor {
    dtype: FloatFormat,
    shape: Vec<usize>,
    bytes: Arc<[u8]>,
}

impl RawTensor {
    pub fn from_le_bytes(
        dtype: FloatFormat,
        shape: Vec<usize>,
        bytes: impl Into<Arc<[u8]>>,
    ) -> Result<Self, PortableError> {
        let elements = shape
            .iter()
            .try_fold(1_usize, |total, &dim| {
                (dim > 0).then_some(total)?.checked_mul(dim)
            })
            .filter(|_| !shape.is_empty())
            .ok_or(PortableError::InvalidGeometry(
                "shape empty, zero-sized or overflowing",
            ))?;
        let expected = elements.checked_mul(dtype.bytes_per_element()).ok_or(
            PortableError::InvalidGeometry("tensor byte count overflows"),
        )?;
        let bytes = bytes.into();
        if bytes.len() != expected {
            return Err(PortableError::ShapeMismatch(
                "payload length differs from dtype*shape",
            ));
        }
        Ok(Self {
            dtype,
            shape,
            bytes,
        })
    }

    pub fn from_f32(shape: Vec<usize>, data: &[f32]) -> Result<Self, PortableError> {
        let mut bytes = Vec::with_capacity(data.len().saturating_mul(4));
        for value in data {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        Self::from_le_bytes(FloatFormat::F32, shape, bytes)
    }

    pub fn dtype(&self) -> FloatFormat {
        self.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn as_le_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn len_bytes(&self) -> usize {
        self.bytes.len()
    }

    /// Zero-allocation scalar decode for the reference streaming attention.
    /// The internal caller only supplies indices checked by the tensor shape.
    fn element_f32(&self, index: usize) -> f32 {
        match self.dtype {
            FloatFormat::Bf16 => {
                let i = index * 2;
                bf16::from_bits(u16::from_le_bytes([self.bytes[i], self.bytes[i + 1]])).to_f32()
            }
            FloatFormat::F16 => {
                let i = index * 2;
                f16::from_bits(u16::from_le_bytes([self.bytes[i], self.bytes[i + 1]])).to_f32()
            }
            FloatFormat::F32 => {
                let i = index * 4;
                f32::from_bits(u32::from_le_bytes([
                    self.bytes[i],
                    self.bytes[i + 1],
                    self.bytes[i + 2],
                    self.bytes[i + 3],
                ]))
            }
        }
    }

    pub fn decode_f32(&self) -> Vec<f32> {
        match self.dtype {
            FloatFormat::Bf16 => self
                .bytes
                .chunks_exact(2)
                .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
                .collect(),
            FloatFormat::F16 => self
                .bytes
                .chunks_exact(2)
                .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
                .collect(),
            FloatFormat::F32 => self
                .bytes
                .chunks_exact(4)
                .map(|b| f32::from_bits(u32::from_le_bytes([b[0], b[1], b[2], b[3]])))
                .collect(),
        }
    }
}

/// Physical encoding for immutable model weights. Quantized variants are
/// symmetric block formats with one finite positive FP32 scale per block.
/// This type describes storage only; a backend must still advertise and
/// implement matching kernels before selecting it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightEncoding {
    Bf16,
    Q8Symmetric,
    Q4Symmetric,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PackedWeight {
    encoding: WeightEncoding,
    shape: Vec<usize>,
    block_size: usize,
    data: Arc<[u8]>,
    scales: Arc<[f32]>,
}

impl PackedWeight {
    pub fn from_bf16(
        shape: Vec<usize>,
        bytes: impl Into<Arc<[u8]>>,
    ) -> Result<Self, PortableError> {
        let elements = checked_elements(&shape)?;
        let data = bytes.into();
        if data.len()
            != elements
                .checked_mul(2)
                .ok_or(PortableError::InvalidGeometry(
                    "BF16 weight byte count overflows",
                ))?
        {
            return Err(PortableError::ShapeMismatch(
                "BF16 weight payload length differs from shape",
            ));
        }
        Ok(Self {
            encoding: WeightEncoding::Bf16,
            shape,
            block_size: 0,
            data,
            scales: Arc::from([]),
        })
    }

    pub fn from_quantized_parts(
        encoding: WeightEncoding,
        shape: Vec<usize>,
        block_size: usize,
        data: impl Into<Arc<[u8]>>,
        scales: impl Into<Arc<[f32]>>,
    ) -> Result<Self, PortableError> {
        if encoding == WeightEncoding::Bf16 {
            return Err(PortableError::InvalidGeometry(
                "use from_bf16 for unquantized weights",
            ));
        }
        if block_size == 0 {
            return Err(PortableError::InvalidGeometry(
                "quantized weight block size must be positive",
            ));
        }
        let elements = checked_elements(&shape)?;
        let expected_data = match encoding {
            WeightEncoding::Q8Symmetric => elements,
            WeightEncoding::Q4Symmetric => elements.div_ceil(2),
            WeightEncoding::Bf16 => unreachable!(),
        };
        let data = data.into();
        if data.len() != expected_data {
            return Err(PortableError::ShapeMismatch(
                "quantized payload length differs from encoding and shape",
            ));
        }
        let scales = scales.into();
        if scales.len() != elements.div_ceil(block_size) {
            return Err(PortableError::ShapeMismatch(
                "scale count differs from quantized block count",
            ));
        }
        if scales
            .iter()
            .any(|scale| !scale.is_finite() || *scale <= 0.0)
        {
            return Err(PortableError::NonFinite(
                "quantized scales must be finite and positive",
            ));
        }
        if encoding == WeightEncoding::Q4Symmetric
            && elements % 2 == 1
            && data.last().is_some_and(|byte| byte & 0xf0 != 0)
        {
            return Err(PortableError::ShapeMismatch(
                "unused high Q4 nibble must be zero",
            ));
        }
        Ok(Self {
            encoding,
            shape,
            block_size,
            data,
            scales,
        })
    }

    pub fn quantize_symmetric(
        encoding: WeightEncoding,
        shape: Vec<usize>,
        block_size: usize,
        values: &[f32],
    ) -> Result<Self, PortableError> {
        if encoding == WeightEncoding::Bf16 {
            let bytes = values
                .iter()
                .flat_map(|value| bf16::from_f32(*value).to_bits().to_le_bytes())
                .collect::<Vec<_>>();
            return Self::from_bf16(shape, bytes);
        }
        let elements = checked_elements(&shape)?;
        if values.len() != elements {
            return Err(PortableError::ShapeMismatch(
                "source values differ from weight shape",
            ));
        }
        if block_size == 0 {
            return Err(PortableError::InvalidGeometry(
                "quantized weight block size must be positive",
            ));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(PortableError::NonFinite(
                "cannot quantize non-finite model weights",
            ));
        }
        let limit = match encoding {
            WeightEncoding::Q8Symmetric => 127.0,
            WeightEncoding::Q4Symmetric => 7.0,
            WeightEncoding::Bf16 => unreachable!(),
        };
        let mut scales = Vec::with_capacity(elements.div_ceil(block_size));
        let mut quantized = Vec::with_capacity(elements);
        for block in values.chunks(block_size) {
            let maximum = block.iter().copied().map(f32::abs).fold(0.0, f32::max);
            let scale = if maximum == 0.0 { 1.0 } else { maximum / limit };
            scales.push(scale);
            quantized.extend(
                block
                    .iter()
                    .map(|value| (value / scale).round_ties_even().clamp(-limit, limit) as i8),
            );
        }
        let data = match encoding {
            WeightEncoding::Q8Symmetric => quantized
                .into_iter()
                .map(|value| value as u8)
                .collect::<Vec<_>>(),
            WeightEncoding::Q4Symmetric => {
                let mut packed = vec![0u8; elements.div_ceil(2)];
                for (index, value) in quantized.into_iter().enumerate() {
                    packed[index / 2] |= ((value as u8) & 0x0f) << ((index & 1) * 4);
                }
                packed
            }
            WeightEncoding::Bf16 => unreachable!(),
        };
        Self::from_quantized_parts(encoding, shape, block_size, data, scales)
    }

    pub fn encoding(&self) -> WeightEncoding {
        self.encoding
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn packed_bytes(&self) -> &[u8] {
        &self.data
    }

    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    pub fn decode_f32(&self) -> Vec<f32> {
        let elements = self.shape.iter().product();
        (0..elements).map(|index| self.element_f32(index)).collect()
    }

    pub fn element_f32(&self, index: usize) -> f32 {
        match self.encoding {
            WeightEncoding::Bf16 => {
                let offset = index * 2;
                bf16::from_bits(u16::from_le_bytes([
                    self.data[offset],
                    self.data[offset + 1],
                ]))
                .to_f32()
            }
            WeightEncoding::Q8Symmetric => {
                (self.data[index] as i8) as f32 * self.scales[index / self.block_size]
            }
            WeightEncoding::Q4Symmetric => {
                let nibble = (self.data[index / 2] >> ((index & 1) * 4)) & 0x0f;
                let signed = if nibble & 0x08 == 0 {
                    nibble as i8
                } else {
                    nibble as i8 - 16
                };
                signed as f32 * self.scales[index / self.block_size]
            }
        }
    }
}

fn checked_elements(shape: &[usize]) -> Result<usize, PortableError> {
    shape
        .iter()
        .try_fold(1usize, |total, &dimension| {
            (dimension > 0).then_some(total)?.checked_mul(dimension)
        })
        .filter(|_| !shape.is_empty())
        .ok_or(PortableError::InvalidGeometry(
            "shape empty, zero-sized or overflowing",
        ))
}

/// GQA/MQA mapping with arbitrary attention and KV head counts.
/// This geometry does *not* assume that hidden_size == q_heads * head_dim.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttentionGeometry {
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

impl AttentionGeometry {
    pub fn new(
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, PortableError> {
        if query_heads == 0
            || kv_heads == 0
            || head_dim == 0
            || !query_heads.is_multiple_of(kv_heads)
            || query_heads.checked_mul(head_dim).is_none()
            || kv_heads.checked_mul(head_dim).is_none()
        {
            return Err(PortableError::InvalidGeometry(
                "heads and dimension must be positive, q_heads divisible by kv_heads, products finite",
            ));
        }
        Ok(Self {
            query_heads,
            kv_heads,
            head_dim,
        })
    }

    pub fn kv_head_for_query(self, query_head: usize) -> Result<usize, PortableError> {
        if query_head >= self.query_heads {
            return Err(PortableError::ShapeMismatch("query head out of range"));
        }
        Ok(query_head / (self.query_heads / self.kv_heads))
    }

    pub fn query_elements(self) -> usize {
        self.query_heads * self.head_dim
    }

    pub fn kv_elements_per_token(self) -> usize {
        self.kv_heads * self.head_dim
    }
}

/// Key/value frames are in [tokens, kv_heads, head_dim] order. This is a
/// bitwise-exact byte store. No tensor compression is implied.
#[derive(Debug, Clone)]
pub struct ExactKvFrames {
    geometry: AttentionGeometry,
    frames: Vec<(RawTensor, RawTensor)>,
    tokens: usize,
}

impl ExactKvFrames {
    pub fn new(geometry: AttentionGeometry) -> Self {
        Self {
            geometry,
            frames: Vec::new(),
            tokens: 0,
        }
    }

    pub fn append(&mut self, key: RawTensor, value: RawTensor) -> Result<(), PortableError> {
        let matches_shape = |shape: &[usize]| {
            shape.len() == 3
                && shape[1] == self.geometry.kv_heads
                && shape[2] == self.geometry.head_dim
        };
        if !matches_shape(key.shape()) || key.shape() != value.shape() {
            return Err(PortableError::ShapeMismatch(
                "KV frame needs [tokens, kv_heads, head_dim]",
            ));
        }
        let tokens = self
            .tokens
            .checked_add(key.shape()[0])
            .ok_or(PortableError::InvalidGeometry("token count overflow"))?;
        self.frames.push((key, value));
        self.tokens = tokens;
        Ok(())
    }

    pub fn geometry(&self) -> AttentionGeometry {
        self.geometry
    }

    pub fn tokens(&self) -> usize {
        self.tokens
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    pub fn payload_bytes(&self) -> usize {
        self.frames
            .iter()
            .map(|(k, v)| k.len_bytes() + v.len_bytes())
            .sum()
    }

    pub fn frames(&self) -> &[(RawTensor, RawTensor)] {
        &self.frames
    }

    /// Logical digest of dtype, shapes and original bytes, independent of
    /// the runtime allocator. This is not the legacy M-SLIM KV SHA.
    pub fn digest_sha256(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"mslim-portable-exact-kv-v1");
        for n in [
            self.geometry.query_heads,
            self.geometry.kv_heads,
            self.geometry.head_dim,
            self.tokens,
        ] {
            hasher.update((n as u64).to_le_bytes());
        }
        for (key, value) in &self.frames {
            for tensor in [key, value] {
                hasher.update([match tensor.dtype {
                    FloatFormat::Bf16 => 1,
                    FloatFormat::F16 => 2,
                    FloatFormat::F32 => 3,
                }]);
                for &dim in tensor.shape() {
                    hasher.update((dim as u64).to_le_bytes());
                }
                hasher.update(tensor.as_le_bytes());
            }
        }
        format!("{:x}", hasher.finalize())
    }

    /// Single-step causal decode (query attends to all tokens already in KV).
    /// Masking, RoPE, tokenizer, model projections and prefill are the caller's
    /// responsibility; they are not implemented by this interface.
    pub fn decode<B: AttentionBackend + ?Sized>(
        &self,
        backend: &B,
        query: &[f32],
    ) -> Result<Vec<f32>, PortableError> {
        backend.decode(self, query)
    }
}

/// Pluggable kernel contract. Only CpuReferenceBackend is currently implemented.
pub trait AttentionBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn decode(&self, kv: &ExactKvFrames, query: &[f32]) -> Result<Vec<f32>, PortableError>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct CpuReferenceBackend;

impl AttentionBackend for CpuReferenceBackend {
    fn name(&self) -> &'static str {
        "rust-cpu-reference"
    }

    fn decode(&self, kv: &ExactKvFrames, query: &[f32]) -> Result<Vec<f32>, PortableError> {
        let geometry = kv.geometry();
        if query.len() != geometry.query_elements() {
            return Err(PortableError::ShapeMismatch(
                "Q must be [query_heads, head_dim]",
            ));
        }
        if query.iter().any(|x| !x.is_finite()) {
            return Err(PortableError::NonFinite("query"));
        }
        if kv.is_empty() {
            return Ok(vec![0.0; query.len()]);
        }
        // Online log-sum-exp softmax: no scores matrix and no full KV copy.
        // f64 is deliberate for reference numerical stability; not native BF16
        // arithmetic and not a promise of bitwise parity with accelerated GPUs.
        let mut output = vec![0.0; geometry.query_elements()];
        let scale = 1.0_f64 / (geometry.head_dim as f64).sqrt();
        for q_head in 0..geometry.query_heads {
            let kv_head = geometry.kv_head_for_query(q_head)?;
            let q = &query[q_head * geometry.head_dim..(q_head + 1) * geometry.head_dim];
            let mut max_score = f64::NEG_INFINITY;
            let mut denominator = 0.0_f64;
            let mut weighted = vec![0.0_f64; geometry.head_dim];
            for (key, value) in kv.frames() {
                let frame_tokens = key.shape()[0];
                for token in 0..frame_tokens {
                    let offset = (token * geometry.kv_heads + kv_head) * geometry.head_dim;
                    let mut dot = 0.0_f64;
                    for (i, &qvalue) in q.iter().enumerate() {
                        let kvalue = key.element_f32(offset + i);
                        if !kvalue.is_finite() {
                            return Err(PortableError::NonFinite("key"));
                        }
                        dot += (qvalue as f64) * (kvalue as f64);
                    }
                    let score = dot * scale;
                    if !score.is_finite() {
                        return Err(PortableError::NonFinite("Q·K score"));
                    }
                    let next_max = max_score.max(score);
                    let prior = (max_score - next_max).exp();
                    let current = (score - next_max).exp();
                    denominator = denominator * prior + current;
                    for (i, dest) in weighted.iter_mut().enumerate() {
                        let item = value.element_f32(offset + i);
                        if !item.is_finite() {
                            return Err(PortableError::NonFinite("value"));
                        }
                        *dest = *dest * prior + current * item as f64;
                    }
                    max_score = next_max;
                }
            }
            if !denominator.is_finite() || denominator <= 0.0 {
                return Err(PortableError::NonFinite("softmax denominator"));
            }
            let base = q_head * geometry.head_dim;
            for (dest, weighted) in output[base..base + geometry.head_dim]
                .iter_mut()
                .zip(weighted)
            {
                let value = weighted / denominator;
                if !value.is_finite() || value.abs() > f32::MAX as f64 {
                    return Err(PortableError::NonFinite("attention output"));
                }
                *dest = value as f32;
            }
        }
        Ok(output)
    }
}

/// Intentional fail-closed selection: no fake GPU execution or CPU silently
/// masquerading as CUDA/ROCm/Metal/XPU.
pub fn require_backend(device: &str) -> Result<CpuReferenceBackend, PortableError> {
    match device.to_ascii_lowercase().as_str() {
        "cpu" | "auto" => Ok(CpuReferenceBackend),
        _ => Err(PortableError::UnsupportedDevice(device.to_owned())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_bits_for_all_float_formats() {
        let cases = [
            (FloatFormat::Bf16, vec![0x80, 0x3f, 0xc1, 0x7f]), // 1.0, NaN payload
            (FloatFormat::F16, vec![0x00, 0x3c, 0x01, 0x7e]),
            (FloatFormat::F32, vec![0, 0, 0x80, 0x3f, 1, 0, 0xc0, 0x7f]),
        ];
        for (dtype, original) in cases {
            let tensor = RawTensor::from_le_bytes(dtype, vec![2], original.clone()).unwrap();
            assert_eq!(tensor.as_le_bytes(), original.as_slice());
            assert_eq!(tensor.decode_f32()[0], 1.0);
            assert!(tensor.decode_f32()[1].is_nan());
        }
        let subnormal = RawTensor::from_le_bytes(FloatFormat::F16, vec![1], vec![1, 0]).unwrap();
        assert_eq!(subnormal.decode_f32()[0], 2.0_f32.powi(-24));
        let signed_zero = RawTensor::from_f32(vec![1], &[-0.0]).unwrap();
        assert_eq!(
            signed_zero.as_le_bytes(),
            (-0.0_f32).to_bits().to_le_bytes()
        );
    }

    #[test]
    fn mha_gqa_mqa_map_heads() {
        for (qheads, kvheads) in [(2, 2), (4, 2), (8, 1)] {
            let geom = AttentionGeometry::new(qheads, kvheads, 2).unwrap();
            let keys = vec![0.0; geom.kv_elements_per_token()];
            let values: Vec<f32> = (0..kvheads)
                .flat_map(|h| [h as f32 + 1.0, -(h as f32 + 1.0)])
                .collect();
            let mut cache = ExactKvFrames::new(geom);
            cache
                .append(
                    RawTensor::from_f32(vec![1, kvheads, 2], &keys).unwrap(),
                    RawTensor::from_f32(vec![1, kvheads, 2], &values).unwrap(),
                )
                .unwrap();
            let output = cache
                .decode(&CpuReferenceBackend, &vec![0.0; geom.query_elements()])
                .unwrap();
            for head in 0..qheads {
                let kv = geom.kv_head_for_query(head).unwrap();
                assert_eq!(
                    &output[head * 2..head * 2 + 2],
                    &[values[kv * 2], values[kv * 2 + 1]]
                );
            }
            assert_eq!(cache.tokens(), 1);
        }
    }

    #[test]
    fn streaming_softmax_is_stable_for_extreme_scores() {
        let geom = AttentionGeometry::new(1, 1, 1).unwrap();
        let mut cache = ExactKvFrames::new(geom);
        cache
            .append(
                RawTensor::from_f32(vec![2, 1, 1], &[10_000.0, -10_000.0]).unwrap(),
                RawTensor::from_f32(vec![2, 1, 1], &[3.0, 9.0]).unwrap(),
            )
            .unwrap();
        assert_eq!(
            cache.decode(&CpuReferenceBackend, &[10_000.0]).unwrap(),
            vec![3.0]
        );
    }

    #[test]
    fn chunks_keep_exact_payload_and_digests() {
        let geom = AttentionGeometry::new(2, 1, 2).unwrap();
        let mut cache = ExactKvFrames::new(geom);
        let key =
            RawTensor::from_le_bytes(FloatFormat::Bf16, vec![1, 1, 2], vec![0, 0, 0x80, 0x3f])
                .unwrap();
        let value =
            RawTensor::from_le_bytes(FloatFormat::F16, vec![1, 1, 2], vec![0, 0x3c, 0, 0x40])
                .unwrap();
        cache.append(key.clone(), value.clone()).unwrap();
        assert_eq!(cache.payload_bytes(), 8);
        assert_eq!(cache.frames()[0].0.as_le_bytes(), key.as_le_bytes());
        assert_eq!(cache.frames()[0].1.as_le_bytes(), value.as_le_bytes());
        let previous = cache.digest_sha256();
        cache.append(key, value).unwrap();
        assert_ne!(previous, cache.digest_sha256());
        assert_eq!(cache.tokens(), 2);
    }

    #[test]
    fn packed_q8_and_q4_have_verifiable_geometry_and_error() {
        let values = [-1.0, -0.5, 0.0, 0.5, 1.0, 0.25, -0.25];
        for encoding in [WeightEncoding::Q8Symmetric, WeightEncoding::Q4Symmetric] {
            let weight =
                PackedWeight::quantize_symmetric(encoding, vec![1, values.len()], 4, &values)
                    .unwrap();
            assert_eq!(weight.shape(), &[1, values.len()]);
            assert_eq!(weight.scales().len(), 2);
            assert!(weight.scales().iter().all(|scale| *scale > 0.0));
            let decoded = weight.decode_f32();
            let tolerance = if encoding == WeightEncoding::Q8Symmetric {
                0.01
            } else {
                0.08
            };
            assert!(
                decoded
                    .iter()
                    .zip(values)
                    .all(|(actual, expected)| (actual - expected).abs() <= tolerance)
            );
        }
    }

    #[test]
    fn packed_weights_reject_malformed_or_nonfinite_parts() {
        assert!(
            PackedWeight::from_quantized_parts(
                WeightEncoding::Q8Symmetric,
                vec![4],
                2,
                vec![0; 3],
                vec![1.0, 1.0],
            )
            .is_err()
        );
        assert!(
            PackedWeight::from_quantized_parts(
                WeightEncoding::Q4Symmetric,
                vec![3],
                2,
                vec![0, 0xf0],
                vec![1.0, 1.0],
            )
            .is_err()
        );
        assert!(
            PackedWeight::quantize_symmetric(
                WeightEncoding::Q8Symmetric,
                vec![1],
                32,
                &[f32::NAN],
            )
            .is_err()
        );
    }

    #[test]
    fn reject_incorrect_shapes_devices_and_nonfinite_math() {
        assert!(AttentionGeometry::new(3, 2, 32).is_err());
        assert!(RawTensor::from_le_bytes(FloatFormat::Bf16, vec![2, 3], vec![0; 6]).is_err());
        assert!(require_backend("cuda:0").is_err());
        assert!(require_backend("auto").is_ok());
        let geom = AttentionGeometry::new(1, 1, 2).unwrap();
        let mut cache = ExactKvFrames::new(geom);
        let good = RawTensor::from_f32(vec![1, 1, 2], &[1.0, 2.0]).unwrap();
        let bad = RawTensor::from_f32(vec![1, 1, 3], &[1.0, 2.0, 3.0]).unwrap();
        assert!(cache.append(good.clone(), bad).is_err());
        assert_eq!(cache.tokens(), 0);
        cache.append(good.clone(), good).unwrap();
        assert!(cache.decode(&CpuReferenceBackend, &[0.0]).is_err());
        assert!(
            cache
                .decode(&CpuReferenceBackend, &[f32::NAN, 1.0])
                .is_err()
        );
    }
}
