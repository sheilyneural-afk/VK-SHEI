//! Capability-based registry for KVSLIM inference workers.
//!
//! The registry is deliberately independent of HTTP, model names and
//! accelerator vendors. A worker advertises measured capabilities; routing
//! code must select from this registry instead of assuming a fixed CPU/GPU
//! pair.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkerCapabilities {
    pub compute: String,
    pub supports_prefill: bool,
    pub supports_decode: bool,
    pub supports_kv_snapshot: bool,
    pub supports_kv_restore: bool,
    pub model_digests: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkerDescriptor {
    pub id: String,
    pub endpoint: String,
    pub capabilities: WorkerCapabilities,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkerRegistry {
    workers: Vec<WorkerDescriptor>,
}

impl WorkerRegistry {
    /// Builds a registry from `id=endpoint` entries supplied by the launcher.
    /// No worker kind or model name is inferred from the identifier.
    pub fn from_assignments(assignments: &[String]) -> anyhow::Result<Self> {
        let mut workers = Vec::with_capacity(assignments.len());
        for assignment in assignments {
            let (id, endpoint) = assignment
                .split_once('=')
                .ok_or_else(|| anyhow::anyhow!("worker assignment must be id=endpoint"))?;
            workers.push(WorkerDescriptor {
                id: id.trim().to_owned(),
                endpoint: endpoint.trim().trim_end_matches('/').to_owned(),
                capabilities: WorkerCapabilities {
                    compute: "unknown-until-probed".to_owned(),
                    supports_prefill: false,
                    supports_decode: false,
                    supports_kv_snapshot: false,
                    supports_kv_restore: false,
                    model_digests: Vec::new(),
                },
            });
        }
        Self::new(workers)
    }

    pub fn new(workers: Vec<WorkerDescriptor>) -> anyhow::Result<Self> {
        let mut ids = std::collections::BTreeSet::new();
        for worker in &workers {
            anyhow::ensure!(!worker.id.trim().is_empty(), "worker id cannot be empty");
            anyhow::ensure!(
                worker.endpoint.starts_with("http://"),
                "worker endpoint must use http"
            );
            anyhow::ensure!(
                ids.insert(worker.id.clone()),
                "duplicate worker id: {}",
                worker.id
            );
        }
        Ok(Self { workers })
    }

    pub fn workers(&self) -> &[WorkerDescriptor] {
        &self.workers
    }

    pub fn compatible_with(&self, model_digest: Option<&str>) -> Vec<&WorkerDescriptor> {
        self.workers
            .iter()
            .filter(|worker| {
                worker.capabilities.supports_prefill
                    && worker.capabilities.supports_decode
                    && model_digest.is_none_or(|digest| {
                        worker.capabilities.model_digests.is_empty()
                            || worker
                                .capabilities
                                .model_digests
                                .iter()
                                .any(|candidate| candidate == digest)
                    })
            })
            .collect()
    }

    pub fn get(&self, id: &str) -> Option<&WorkerDescriptor> {
        self.workers.iter().find(|worker| worker.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn worker(id: &str, endpoint: &str) -> WorkerDescriptor {
        WorkerDescriptor {
            id: id.to_owned(),
            endpoint: endpoint.to_owned(),
            capabilities: WorkerCapabilities {
                compute: "detected".to_owned(),
                supports_prefill: true,
                supports_decode: true,
                supports_kv_snapshot: true,
                supports_kv_restore: true,
                model_digests: Vec::new(),
            },
        }
    }

    #[test]
    fn registry_rejects_duplicate_ids() {
        assert!(
            WorkerRegistry::new(vec![
                worker("one", "http://127.0.0.1:1"),
                worker("one", "http://127.0.0.1:2"),
            ])
            .is_err()
        );
    }

    #[test]
    fn selection_uses_capabilities_not_worker_kind_names() {
        let registry =
            WorkerRegistry::new(vec![worker("any-runtime", "http://127.0.0.1:1")]).unwrap();
        assert_eq!(registry.compatible_with(None).len(), 1);
        assert!(registry.get("any-runtime").is_some());
    }

    #[test]
    fn assignments_do_not_infer_compute_kind() {
        let registry =
            WorkerRegistry::from_assignments(&["worker-a=http://127.0.0.1:9000".to_owned()])
                .unwrap();
        assert_eq!(
            registry.workers()[0].capabilities.compute,
            "unknown-until-probed"
        );
        assert!(registry.compatible_with(None).is_empty());
    }
}
