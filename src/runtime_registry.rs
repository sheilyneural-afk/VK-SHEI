//! Backend-neutral runtime registry.
//! Concrete implementations register capabilities; selection never matches
//! vendor/backend names.
use anyhow::{Result, bail};
use kvslim_adapter_api::{
    AdapterAvailability, AdapterSelection, AdapterStatus, ComputeKind, ModelAdapter,
    select_available_adapter,
};
use std::path::Path;

pub trait RuntimeFactory: Send + Sync {
    fn status(&self) -> AdapterStatus;
    fn create(&self) -> Result<Box<dyn ModelAdapter>>;
}

pub struct RuntimeRegistry {
    factories: Vec<Box<dyn RuntimeFactory>>,
}

impl RuntimeRegistry {
    pub fn new() -> Self {
        Self {
            factories: Vec::new(),
        }
    }
    pub fn register(&mut self, factory: Box<dyn RuntimeFactory>) {
        self.factories.push(factory);
    }
    pub fn statuses(&self) -> Vec<AdapterStatus> {
        self.factories.iter().map(|f| f.status()).collect()
    }
    pub fn select(&self, requested_compute: Option<ComputeKind>) -> Result<AdapterSelection> {
        select_available_adapter(&self.statuses(), None, requested_compute)
            .ok_or_else(|| anyhow::anyhow!("no runtime satisfies requested capabilities"))
    }
    pub fn create_selected(&self, selection: &AdapterSelection) -> Result<Box<dyn ModelAdapter>> {
        for factory in &self.factories {
            let status = factory.status();
            if status.descriptor.id == selection.adapter_id
                && status.availability == AdapterAvailability::RuntimeAvailable
            {
                return factory.create();
            }
        }
        bail!("selected runtime is unavailable")
    }
    pub fn inspect_selected(
        &self,
        selection: &AdapterSelection,
        model_dir: &Path,
        config: &serde_json::Value,
    ) -> Result<Box<dyn ModelAdapter>> {
        let adapter = self.create_selected(selection)?;
        adapter.inspect(model_dir, config)?;
        Ok(adapter)
    }
}
impl Default for RuntimeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct F;
    impl RuntimeFactory for F {
        fn status(&self) -> AdapterStatus {
            AdapterStatus {
                descriptor: kvslim_adapter_api::AdapterDescriptor {
                    id: "test-runtime".into(),
                    backend: "generic".into(),
                    compute: ComputeKind::Any,
                    priority: 1,
                    activation: kvslim_adapter_api::ActivationPolicy::Default,
                    supported_model_types: Vec::new(),
                },
                availability: AdapterAvailability::RuntimeAvailable,
                reason: "probe ok".into(),
            }
        }
        fn create(&self) -> Result<Box<dyn ModelAdapter>> {
            bail!("test factory")
        }
    }
    #[test]
    fn selects_only_runtime_available_capabilities() {
        let mut r = RuntimeRegistry::new();
        r.register(Box::new(F));
        assert_eq!(r.select(None).unwrap().adapter_id, "test-runtime");
    }
}
