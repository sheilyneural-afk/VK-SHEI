//! Universal context/KV inference engine.
//!
//! The canonical architecture keeps model weights and authoritative K/V in
//! BF16. A four-bit parallel dimension index locates reversible token
//! coordinates; exact BF16 state fragments are then read from the append-only cold
//! store. There is no alternate low-bit K/V authority.

pub mod adapter_catalog;
pub mod adaptive_extent;
pub mod attention_certificate;
pub mod cold_store;
pub mod conformance;
pub mod conversion_worker;
pub mod document_graph;
pub mod engine;
pub mod inference_worker;
#[cfg(feature = "torch-backend")]
pub mod prefill_tch;
pub mod shadow_metrics;
#[cfg(feature = "torch-backend")]
pub mod torch_runtime;

#[cfg(feature = "torch-backend")]
pub mod torch_adapter;

// Backend-neutral types and Hugging Face metadata inspection.
pub use kvslim_adapter_api::{
    AdapterAvailability, AdapterCapabilities, AdapterDescriptor, AdapterSelection, AdapterStatus,
    ComputeKind, ModelAdapter, select_adapter, select_available_adapter,
};
#[cfg(feature = "model-bundle")]
pub mod model_bundle;
pub mod model_catalog;
pub mod model_config;
pub mod model_resolver;
pub mod portable;
pub mod portable_store;
pub mod worker_registry;

#[cfg(feature = "accelerator-backend")]
pub mod opencl_backend;

#[cfg(feature = "accelerator-backend")]
pub mod opencl_adapter;

#[cfg(feature = "accelerator-backend")]
pub mod opencl_llm;

#[cfg(feature = "accelerator-backend")]
pub mod gpu_canonical_lod;

#[cfg(feature = "hip-backend")]
pub mod native_hip;

#[cfg(feature = "accelerator-backend")]
pub mod gpu_micro_regions;
