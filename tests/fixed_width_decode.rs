//! Regression coverage for fixed-width, little-endian decoding.
//! All 16-bit encodings are covered, including zeros, subnormals and NaNs.
use half::{bf16, f16};
use mslim::portable::{FloatFormat, RawTensor};

fn every_u16_encoding() -> Vec<u8> {
    (0..=u16::MAX).flat_map(u16::to_le_bytes).collect()
}

#[test]
fn bf16_array_chunks_preserve_every_encoding() {
    let bytes = every_u16_encoding();
    let tensor = RawTensor::from_le_bytes(FloatFormat::Bf16, vec![65536], bytes.clone()).unwrap();
    let decoded = tensor.decode_f32();
    assert_eq!(decoded.len(), 65536);
    for (bits, actual) in (0..=u16::MAX).zip(decoded) {
        assert_eq!(actual.to_bits(), bf16::from_bits(bits).to_f32().to_bits());
    }
    assert_eq!(tensor.as_le_bytes(), bytes);
}

#[test]
fn f16_array_chunks_preserve_every_encoding() {
    let bytes = every_u16_encoding();
    let tensor = RawTensor::from_le_bytes(FloatFormat::F16, vec![65536], bytes.clone()).unwrap();
    let decoded = tensor.decode_f32();
    assert_eq!(decoded.len(), 65536);
    for (bits, actual) in (0..=u16::MAX).zip(decoded) {
        assert_eq!(actual.to_bits(), f16::from_bits(bits).to_f32().to_bits());
    }
    assert_eq!(tensor.as_le_bytes(), bytes);
}

#[test]
fn f32_array_chunks_preserve_edge_case_bits_and_order() {
    let patterns = [
        0_u32,
        0x8000_0000,
        1,
        0x007f_ffff,
        0x0080_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x7f7f_ffff,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_1234,
        0x7f80_0001,
        0xffc0_5678,
    ];
    let bytes: Vec<u8> = patterns
        .iter()
        .flat_map(|bits| bits.to_le_bytes())
        .collect();
    let tensor =
        RawTensor::from_le_bytes(FloatFormat::F32, vec![patterns.len()], bytes.clone()).unwrap();
    let actual: Vec<u32> = tensor.decode_f32().into_iter().map(f32::to_bits).collect();
    assert_eq!(actual, patterns);
    assert_eq!(tensor.as_le_bytes(), bytes);
}

#[test]
fn malformed_tensor_width_is_rejected_instead_of_truncated() {
    for dtype in [FloatFormat::Bf16, FloatFormat::F16, FloatFormat::F32] {
        let width = dtype.bytes_per_element();
        for length in [width - 1, width + 1] {
            assert!(RawTensor::from_le_bytes(dtype, vec![1], vec![0; length]).is_err());
        }
    }
}
