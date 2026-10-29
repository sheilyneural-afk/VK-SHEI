//! Terrsvusion native long-context engine.
//!
//! The canonical architecture keeps model weights and authoritative K/V in
//! BF16. A four-bit parallel dimension index locates reversible token
//! coordinates; exact BF16 state fragments are then read from the append-only cold
//! store. There is no alternate low-bit K/V authority.

pub mod adaptive_extent;
pub mod attention_certificate;
pub mod cold_store;
pub mod conversion_worker;
pub mod inference_worker;
#[cfg(feature = "torch-backend")]
pub mod prefill_tch;
pub mod shadow_metrics;
#[cfg(feature = "torch-backend")]
pub mod torch_runtime;

// Backend-neutral types and Hugging Face metadata inspection.
pub mod model_config;
pub mod portable;
pub mod portable_store;

#[cfg(feature = "opencl-backend")]
pub mod opencl_backend;

#[cfg(feature = "opencl-backend")]
pub mod opencl_llm;

#[cfg(feature = "opencl-backend")]
pub mod gpu_canonical_lod;

#[cfg(feature = "hip-backend")]
pub mod native_hip;

#[cfg(feature = "opencl-backend")]
pub mod gpu_micro_regions;
