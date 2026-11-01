//! Test-only reference algorithms; not product integration.
//!
//! Each submodule is isolated from product entrypoints. Functions take an
//! explicit clock where their reference algorithm used the system clock, so
//! behaviour is testable without sleeping. This module is compiled only under
//! `cfg(test)`; no runtime engine, worker, provider, or gateway calls it.

pub mod budget;
pub mod circuit_breaker;
pub mod compression;
pub mod forget;
pub mod integrity;
pub mod retention_pressure;
pub mod surprise;
pub mod workload;
