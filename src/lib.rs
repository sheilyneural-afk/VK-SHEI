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
pub mod prefill_tch;
pub mod shadow_metrics;
pub mod torch_runtime;
