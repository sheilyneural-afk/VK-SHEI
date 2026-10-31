//! Backend-neutral context/KV inference engine.
//!
//! The canonical architecture keeps model weights and authoritative K/V in
//! BF16. A four-bit parallel dimension index locates reversible token
//! coordinates; exact BF16 state fragments are then read from the append-only cold
//! store. There is no alternate low-bit K/V authority.

pub mod adapter_catalog;
pub mod adaptive_extent;
pub mod attention_certificate;
pub mod calibration_contract;
pub mod calibration_predictor;
pub mod calibration_profile;
pub mod calibration_transaction;
pub mod cold_store;
pub mod conformance;
pub mod continuity_contract;
pub mod conversion_worker;
#[cfg(feature = "model-bundle")]
pub mod decoder_layout;
pub mod document_graph;
pub mod engine;
pub mod evidence_stats;
pub mod fused_retrieval;
pub mod inference_worker;
pub mod ipc_transport;
pub mod page_fusion;
#[cfg(feature = "accelerator-backend")]
pub mod prefill_calibration;
#[cfg(feature = "torch-backend")]
pub mod prefill_tch;
pub mod runtime_host;
pub mod runtime_registry;
pub mod session_coordinator;
pub mod session_store;
pub mod shadow_metrics;
#[cfg(feature = "torch-backend")]
pub mod torch_runtime;

#[cfg(feature = "torch-backend")]
pub mod torch_adapter;

#[cfg(feature = "native-cpu-backend")]
pub mod native_cpu_adapter;

// Backend-neutral types and Hugging Face metadata inspection.
pub use mslim_adapter_api::{
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
pub mod gpu_backend;

#[cfg(feature = "accelerator-backend")]
pub mod gpu_adapter;

#[cfg(feature = "accelerator-backend")]
pub mod gpu_llm;

pub mod gpu_region_store;

#[cfg(feature = "hip-backend")]
pub mod native_hip;

pub mod generic_runtime;
pub mod gpu_micro_regions;
