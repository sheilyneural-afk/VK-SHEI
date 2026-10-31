//! Provider-neutral runtime discovery and capability manifests.
//!
//! This boundary intentionally contains no accelerator/vendor concepts. A runtime
//! is selected only by the operations it proves it can execute.
use anyhow::{Result, bail};
use kvslim_adapter_api::RuntimeSession;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    TensorLoad,
    TensorStore,
    MatrixMultiply,
    Attention,
    Prefill,
    Decode,
    SnapshotKv,
    RestoreKv,
    Cancellation,
    ExactArithmetic,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RuntimeManifest {
    pub schema: String,
    pub id: String,
    pub capabilities: BTreeSet<Capability>,
    pub priority: u32,
    pub protocol: String,
}

impl RuntimeManifest {
    pub const SCHEMA: &'static str = "kvslim.runtime.manifest/v1";

    pub fn validate(&self) -> Result<()> {
        if self.schema != Self::SCHEMA {
            bail!("unsupported runtime manifest schema");
        }
        if self.id.is_empty()
            || self.id.len() > 128
            || self.id.contains('/')
            || self.id.contains('\\')
        {
            bail!("invalid runtime id");
        }
        if self.protocol.is_empty() {
            bail!("runtime protocol is required");
        }
        Ok(())
    }

    pub fn supports_all<'a>(&self, required: impl IntoIterator<Item = &'a Capability>) -> bool {
        required
            .into_iter()
            .all(|cap| self.capabilities.contains(cap))
    }
}

pub trait RuntimeProbe: Send + Sync {
    fn manifest(&self) -> &RuntimeManifest;
    fn probe(&self, model_dir: &Path) -> Result<bool>;

    /// Opens an executable session only when the probe has demonstrated the
    /// requested capabilities. Providers may reject this default explicitly.
    fn open_session(&self, _model_dir: &Path) -> Result<Box<dyn RuntimeSession>> {
        bail!("runtime probe does not expose an executable session")
    }
}

#[derive(Default)]
pub struct GenericRuntimeRegistry {
    probes: Vec<Box<dyn RuntimeProbe>>,
}

impl GenericRuntimeRegistry {
    pub fn register(&mut self, probe: Box<dyn RuntimeProbe>) -> Result<()> {
        probe.manifest().validate()?;
        self.probes.push(probe);
        Ok(())
    }

    pub fn discover(&self, model_dir: &Path, required: &[Capability]) -> Result<&dyn RuntimeProbe> {
        self.probes
            .iter()
            .filter(|probe| probe.manifest().supports_all(required.iter()))
            .filter(|probe| probe.probe(model_dir).unwrap_or(false))
            .max_by_key(|probe| probe.manifest().priority)
            .map(|probe| probe.as_ref())
            .ok_or_else(|| anyhow::anyhow!("no runtime satisfies required capabilities"))
    }

    pub fn manifests(&self) -> impl Iterator<Item = &RuntimeManifest> {
        self.probes.iter().map(|probe| probe.manifest())
    }

    pub fn open_session(
        &self,
        model_dir: &Path,
        required: &[Capability],
    ) -> Result<Box<dyn RuntimeSession>> {
        self.discover(model_dir, required)?.open_session(model_dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Probe(RuntimeManifest);
    impl RuntimeProbe for Probe {
        fn manifest(&self) -> &RuntimeManifest {
            &self.0
        }
        fn probe(&self, _: &Path) -> Result<bool> {
            Ok(true)
        }
    }

    #[test]
    fn discovers_only_by_capabilities() {
        let mut caps = BTreeSet::new();
        caps.insert(Capability::Prefill);
        caps.insert(Capability::Decode);
        let mut registry = GenericRuntimeRegistry::default();
        registry
            .register(Box::new(Probe(RuntimeManifest {
                schema: RuntimeManifest::SCHEMA.into(),
                id: "runtime-a".into(),
                capabilities: caps,
                priority: 1,
                protocol: "v1".into(),
            })))
            .unwrap();
        assert_eq!(
            registry
                .discover(Path::new("."), &[Capability::Prefill])
                .unwrap()
                .manifest()
                .id,
            "runtime-a"
        );
    }
}
