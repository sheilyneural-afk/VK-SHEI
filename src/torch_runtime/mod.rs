//! LibTorch inference backend.
//!
//! Splits into three focused submodules:
//! - `weight_io`: safetensors shard loading (no device selection)
//! - `kv_cache`: KV memory model, pages, certificates, speculation
//! - `runtime`: TorchModelRuntime, DecoderLayer forward pass, generation types
//!
//! Hardware detection and adapter selection live in `adapter_catalog` and
//! `runtime_registry`; this module only runs if the `torch-backend` feature
//! was explicitly selected.

pub mod kv_cache;
pub mod runtime;
pub mod weight_io;

// Re-export the public surface so call sites keep working unchanged.
pub use kv_cache::{
    AttentionCache, DenseKvCache, DecoderGeometry, DecoderLayer, DecoderModel, LayerOutputTrace,
    MslimKvCache,
};
pub use runtime::{
    AppliedOmissionReport, DenseGeneration, MultiTokenVerificationOracle,
    NativeInferenceParity, NativeStepDivergence, ReferenceSpeculativeGeneration,
    TorchChatMessage, TorchGeneration, TorchModelRuntime,
};
pub use weight_io::{WeightArchive, WeightFile, model_weight_file};
