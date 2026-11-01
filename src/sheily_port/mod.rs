//! Test-only Rust ports used as design inspiration, not product integration.
//!
//! Each submodule is a self-contained port of the corresponding Sheily
//! implementation (`SHEILY-COGNITIVA/cognition/native/...`). Functions take an
//! explicit clock where the original read the system clock, so behaviour is
//! testable without sleeping. This module is compiled only under `cfg(test)`;
//! no production engine, worker, or gateway calls it.

pub mod budget;
pub mod circuit_breaker;
pub mod compression;
pub mod forget;
pub mod glymphatic;
pub mod integrity;
pub mod surprise;
pub mod workload;
