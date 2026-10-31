//! Runtime adapter catalog for the local deployment profile.
//!
//! The catalog reports only adapters compiled into this binary. Hardware and
//! model compatibility are still checked by the selected adapter at inspect
//! and load time.

use std::path::Path;

use anyhow::{Result, bail};
use mslim_adapter_api::{
    ActivationPolicy, AdapterDescriptor, AdapterSelection, ComputeKind, select_adapter,
    select_available_adapter,
};
use mslim_adapter_api::{AdapterAvailability, AdapterStatus, ModelAdapter};

#[derive(Debug, Clone, serde::Serialize)]
pub struct HostCapabilities {
    pub architecture: &'static str,
    pub cpu_available: bool,
    pub cpu_parallelism: usize,
    pub memory_total_bytes: Option<u64>,
    pub memory_available_bytes: Option<u64>,
    pub accelerators: Vec<AcceleratorCapabilities>,
    pub simd: Vec<&'static str>,
    pub context_kv_core_available: bool,
    pub inference_adapter_available: bool,
    pub adapters: Vec<AdapterStatus>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct AcceleratorCapabilities {
    pub backend: String,
    pub name: String,
    pub vendor: Option<String>,
    pub memory_bytes: Option<u64>,
    pub max_allocation_bytes: Option<u64>,
    pub available: bool,
}

fn memory_capabilities() -> (Option<u64>, Option<u64>) {
    #[cfg(target_os = "linux")]
    {
        let contents = std::fs::read_to_string("/proc/meminfo").ok();
        let value = |key: &str| -> Option<u64> {
            contents.as_deref()?.lines().find_map(|line| {
                let mut fields = line.split_whitespace();
                if fields.next()? != key {
                    return None;
                }
                fields
                    .next()?
                    .parse::<u64>()
                    .ok()
                    .map(|kilobytes| kilobytes * 1024)
            })
        };
        return (value("MemTotal:"), value("MemAvailable:"));
    }
    #[allow(unreachable_code)]
    (None, None)
}

fn runtime_simd() -> Vec<&'static str> {
    let mut features = Vec::new();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("sse2") {
            features.push("sse2");
        }
        if std::is_x86_feature_detected!("avx") {
            features.push("avx");
        }
        if std::is_x86_feature_detected!("avx2") {
            features.push("avx2");
        }
        if std::is_x86_feature_detected!("fma") {
            features.push("fma");
        }
        if std::is_x86_feature_detected!("avx512f") {
            features.push("avx512f");
        }
    }
    features
}

fn accelerator_capabilities() -> Vec<AcceleratorCapabilities> {
    #[allow(unused_mut)]
    let mut accelerators = Vec::new();
    #[cfg(feature = "accelerator-backend")]
    if let Ok(runtime) = crate::gpu_backend::GpuBackend::new() {
        let info = runtime.device_info();
        accelerators.push(AcceleratorCapabilities {
            backend: "accelerator".into(),
            name: info.name.clone(),
            vendor: Some(info.vendor.clone()),
            memory_bytes: Some(info.reported_global_memory_bytes),
            max_allocation_bytes: Some(info.max_allocation_bytes),
            available: true,
        });
    }
    accelerators
}

/// Inventario efímero del host. No lee ni persiste nombres de usuario,
/// rutas personales, identificadores de máquina ni datos de identidad.
pub fn host_capabilities() -> HostCapabilities {
    let (memory_total_bytes, memory_available_bytes) = memory_capabilities();
    HostCapabilities {
        architecture: std::env::consts::ARCH,
        cpu_available: true,
        cpu_parallelism: std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1),
        memory_total_bytes,
        memory_available_bytes,
        accelerators: accelerator_capabilities(),
        simd: runtime_simd(),
        context_kv_core_available: true,
        inference_adapter_available: adapter_status()
            .iter()
            .any(|status| status.availability == AdapterAvailability::RuntimeAvailable),
        adapters: adapter_status(),
    }
}

#[allow(unused_mut, clippy::vec_init_then_push)]
pub fn compiled_adapters() -> Vec<AdapterDescriptor> {
    let mut adapters = Vec::new();

    #[cfg(feature = "native-cpu-backend")]
    adapters.push(AdapterDescriptor {
        id: "native-cpu-runtime".to_owned(),
        backend: "cpu".to_owned(),
        compute: ComputeKind::Cpu,
        priority: 100,
        activation: ActivationPolicy::Default,
        // This is an architecture registry, not a claim that arbitrary
        // operators are compatible. inspect() remains the final authority.
        supported_model_types: Vec::new(),
    });

    #[cfg(feature = "torch-backend")]
    adapters.push(AdapterDescriptor {
        id: "libtorch-cpu-runtime".to_owned(),
        backend: "cpu".to_owned(),
        compute: ComputeKind::Cpu,
        priority: 200,
        activation: ActivationPolicy::Default,
        supported_model_types: Vec::new(),
    });

    #[cfg(feature = "accelerator-backend")]
    adapters.push(AdapterDescriptor {
        id: "accelerator-runtime".to_owned(),
        backend: "accelerator".to_owned(),
        compute: ComputeKind::Any,
        priority: 300,
        activation: ActivationPolicy::Default,
        // Do not authorize a family from a hard-coded label. ModelAdapter::inspect
        // is the runtime authority for the actual configuration and operators.
        supported_model_types: Vec::new(),
    });

    adapters
}

/// Compiled support is not the same as a usable runtime on this machine.
pub fn adapter_status() -> Vec<AdapterStatus> {
    compiled_adapters()
        .into_iter()
        .map(|descriptor| {
            let (availability, reason) = match descriptor.id.as_str() {
                "accelerator-runtime" => {
                    #[cfg(feature = "accelerator-backend")]
                    {
                        match crate::gpu_backend::GpuBackend::new() {
                            Ok(_) => (
                                AdapterAvailability::RuntimeAvailable,
                                "acelerador detectado".to_owned(),
                            ),
                            Err(error) => (
                                AdapterAvailability::RuntimeUnavailable,
                                format!("acelerador no disponible: {error}"),
                            ),
                        }
                    }
                    #[cfg(not(feature = "accelerator-backend"))]
                    {
                        (
                            AdapterAvailability::Compiled,
                            "backend no compilado".to_owned(),
                        )
                    }
                }
                "native-cpu-runtime" => (
                    AdapterAvailability::RuntimeAvailable,
                    "runtime CPU nativo Rust disponible".to_owned(),
                ),
                "libtorch-cpu-runtime" => (
                    AdapterAvailability::RuntimeAvailable,
                    "runtime CPU LibTorch disponible".to_owned(),
                ),
                _ => (
                    AdapterAvailability::RequiresProbe,
                    "requiere prueba del runtime".to_owned(),
                ),
            };
            AdapterStatus {
                descriptor,
                availability,
                reason,
            }
        })
        .collect()
}

/// Select a compiled adapter only after inspecting the actual model metadata.
/// The model directory is never accepted solely because a backend feature is
/// enabled.
pub fn select_for_model(
    model_dir: impl AsRef<Path>,
    requested_backend: Option<&str>,
    requested_compute: Option<ComputeKind>,
) -> Result<AdapterSelection> {
    let model_dir = model_dir.as_ref();
    let available = compiled_adapters().into_iter().collect::<Vec<_>>();
    let selection = select_adapter(&available, requested_backend, requested_compute)
        .ok_or_else(|| anyhow::anyhow!("no runtime adapter is available by default; select an installed adapter explicitly after capability probing"))?;

    // Validate the concrete selected adapter, not its broad backend label.
    // Multiple CPU adapters may coexist and the native Rust adapter must not
    // acquire an accidental LibTorch feature dependency.
    let adapter = create_adapter(&selection.adapter_id)?;

    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(model_dir.join("config.json"))?)?;
    adapter.inspect(model_dir, &config)?;
    Ok(selection)
}

/// Selects only from adapters whose local runtime probe succeeded.
#[allow(unused_variables)]
pub fn select_available_for_model(
    model_dir: impl AsRef<Path>,
    requested_backend: Option<&str>,
    requested_compute: Option<ComputeKind>,
) -> Result<AdapterSelection> {
    let model_dir = model_dir.as_ref();
    let statuses = adapter_status().into_iter().collect::<Vec<_>>();
    let selection = select_available_adapter(&statuses, requested_backend, requested_compute)
        .ok_or_else(|| {
            anyhow::anyhow!("no hay un runtime validado para este equipo y esta solicitud")
        })?;
    match selection.backend.as_str() {
        "cpu" => {
            #[cfg(feature = "torch-backend")]
            let _ = create_adapter(&selection.adapter_id)?;
        }
        "accelerator" => {
            #[cfg(feature = "accelerator-backend")]
            let _ = create_adapter(&selection.adapter_id)?;
        }
        backend => bail!("adapter has no validation policy: {backend}"),
    }
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(model_dir.join("config.json"))?)?;
    let adapter = create_adapter(&selection.adapter_id)?;
    adapter.inspect(model_dir, &config)?;
    Ok(selection)
}

/// Builds the backend-neutral runtime registry. Concrete constructors stay
/// confined to this compatibility boundary; callers select only by capabilities.
pub fn runtime_registry() -> crate::runtime_registry::RuntimeRegistry {
    let mut registry = crate::runtime_registry::RuntimeRegistry::new();
    for descriptor in compiled_adapters() {
        let status = adapter_status()
            .into_iter()
            .find(|candidate| candidate.descriptor.id == descriptor.id)
            .unwrap_or(AdapterStatus {
                descriptor: descriptor.clone(),
                availability: AdapterAvailability::RequiresProbe,
                reason: "runtime probe required".to_owned(),
            });
        let id = descriptor.id.clone();
        registry.register(Box::new(CatalogFactory { status, id }));
    }
    registry
}

struct CatalogFactory {
    status: AdapterStatus,
    id: String,
}

impl crate::runtime_registry::RuntimeFactory for CatalogFactory {
    fn status(&self) -> AdapterStatus {
        self.status.clone()
    }

    fn create(&self) -> Result<Box<dyn ModelAdapter>> {
        create_adapter(&self.id)
    }
}

/// Construct only an adapter whose implementation is compiled in this binary.
pub fn create_adapter(adapter_id: &str) -> Result<Box<dyn ModelAdapter>> {
    match adapter_id {
        "accelerator-runtime" => {
            #[cfg(feature = "accelerator-backend")]
            return Ok(Box::new(crate::gpu_adapter::GpuModelAdapter::default()));
            #[cfg(not(feature = "accelerator-backend"))]
            bail!("accelerator adapter is not compiled");
        }
        "native-cpu-runtime" => {
            #[cfg(feature = "native-cpu-backend")]
            return Ok(Box::new(
                crate::native_cpu_adapter::NativeCpuModelAdapter::default(),
            ));
            #[cfg(not(feature = "native-cpu-backend"))]
            bail!("native CPU adapter is not compiled");
        }
        "libtorch-cpu-runtime" => {
            #[cfg(feature = "torch-backend")]
            return Ok(Box::new(crate::torch_adapter::TorchModelAdapter::default()));
            #[cfg(not(feature = "torch-backend"))]
            bail!("LibTorch CPU adapter is not compiled");
        }
        other => bail!("unknown adapter: {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn write_config(dir: &Path, config: &str) {
        std::fs::write(dir.join("config.json"), config).unwrap();
    }

    #[test]
    fn compiled_adapters_have_unique_ids_and_default_activation() {
        let adapters = compiled_adapters();
        let ids: HashSet<_> = adapters.iter().map(|a| a.id.as_str()).collect();
        assert_eq!(ids.len(), adapters.len(), "ids de adaptador duplicados");
        for adapter in &adapters {
            assert_eq!(adapter.activation, ActivationPolicy::Default);
            assert!(
                adapter.supported_model_types.is_empty(),
                "{}: la familia del modelo la decide inspect(), no una etiqueta fija",
                adapter.id
            );
        }
    }

    #[test]
    fn compiled_adapters_follow_enabled_features() {
        let ids: Vec<_> = compiled_adapters().into_iter().map(|a| a.id).collect();
        let has = |id: &str| ids.iter().any(|candidate| candidate == id);
        assert_eq!(
            has("native-cpu-runtime"),
            cfg!(feature = "native-cpu-backend")
        );
        assert_eq!(has("libtorch-cpu-runtime"), cfg!(feature = "torch-backend"));
        assert_eq!(
            has("accelerator-runtime"),
            cfg!(feature = "accelerator-backend")
        );
    }

    #[test]
    fn adapter_priorities_prefer_accelerator_then_torch_then_native() {
        let priority = |id: &str| {
            compiled_adapters()
                .into_iter()
                .find(|a| a.id == id)
                .map(|a| a.priority)
        };
        if let (Some(native), Some(torch)) = (
            priority("native-cpu-runtime"),
            priority("libtorch-cpu-runtime"),
        ) {
            assert!(torch > native);
        }
        if let (Some(cpu), Some(accel)) = (
            priority("native-cpu-runtime").or(priority("libtorch-cpu-runtime")),
            priority("accelerator-runtime"),
        ) {
            assert!(accel > cpu);
        }
    }

    #[test]
    fn adapter_status_covers_every_compiled_adapter_in_order() {
        let descriptors = compiled_adapters();
        let statuses = adapter_status();
        assert_eq!(statuses.len(), descriptors.len());
        for (status, descriptor) in statuses.iter().zip(&descriptors) {
            assert_eq!(&status.descriptor, descriptor);
            assert!(!status.reason.is_empty());
            assert_ne!(
                status.availability,
                AdapterAvailability::RequiresProbe,
                "{} no tiene política de disponibilidad",
                descriptor.id
            );
        }
    }

    #[cfg(feature = "native-cpu-backend")]
    #[test]
    fn native_cpu_runtime_is_reported_available() {
        let status = adapter_status()
            .into_iter()
            .find(|s| s.descriptor.id == "native-cpu-runtime")
            .expect("native CPU compilado");
        assert_eq!(status.availability, AdapterAvailability::RuntimeAvailable);
    }

    #[test]
    fn host_capabilities_are_consistent() {
        let host = host_capabilities();
        assert_eq!(host.architecture, std::env::consts::ARCH);
        assert!(host.cpu_available);
        assert!(host.cpu_parallelism >= 1);
        assert!(host.context_kv_core_available);
        if let (Some(total), Some(available)) =
            (host.memory_total_bytes, host.memory_available_bytes)
        {
            assert!(available <= total);
        }
        let any_available = host
            .adapters
            .iter()
            .any(|s| s.availability == AdapterAvailability::RuntimeAvailable);
        assert_eq!(host.inference_adapter_available, any_available);
        assert_eq!(host.adapters, adapter_status());
        // Sin el backend acelerador no se debe anunciar ningún dispositivo.
        if !cfg!(feature = "accelerator-backend") {
            assert!(host.accelerators.is_empty());
        }
    }

    #[test]
    fn host_capabilities_serialize_without_identity_fields() {
        let json = serde_json::to_value(host_capabilities()).unwrap();
        let keys: HashSet<_> = json.as_object().unwrap().keys().cloned().collect();
        for forbidden in ["hostname", "user", "username", "home", "machine_id"] {
            assert!(!keys.contains(forbidden), "campo de identidad: {forbidden}");
        }
    }

    #[test]
    fn create_adapter_rejects_unknown_id() {
        let error = create_adapter("does-not-exist").err().unwrap();
        assert!(error.to_string().contains("unknown adapter"));
    }

    #[test]
    fn create_adapter_matches_compiled_features() {
        assert_eq!(
            create_adapter("native-cpu-runtime").is_ok(),
            cfg!(feature = "native-cpu-backend")
        );
        assert_eq!(
            create_adapter("libtorch-cpu-runtime").is_ok(),
            cfg!(feature = "torch-backend")
        );
        if !cfg!(feature = "accelerator-backend") {
            assert!(create_adapter("accelerator-runtime").is_err());
        }
    }

    #[test]
    fn runtime_registry_mirrors_adapter_status() {
        let registry = runtime_registry();
        assert_eq!(registry.statuses(), adapter_status());
    }

    #[test]
    fn runtime_registry_select_matches_available_selection() {
        let registry = runtime_registry();
        let expected = select_available_adapter(&adapter_status(), None, None);
        match (registry.select(None), expected) {
            (Ok(selected), Some(expected)) => {
                assert_eq!(selected, expected);
                assert!(registry.create_selected(&selected).is_ok());
            }
            (Err(_), None) => {}
            (got, expected) => panic!("registro {got:?} distinto de la selección {expected:?}"),
        }
    }

    #[test]
    fn runtime_registry_refuses_unregistered_selection() {
        let selection = AdapterSelection {
            adapter_id: "does-not-exist".into(),
            backend: "cpu".into(),
            compute: ComputeKind::Cpu,
        };
        assert!(runtime_registry().create_selected(&selection).is_err());
    }

    #[test]
    fn select_for_model_rejects_unknown_backend() {
        let dir = tempfile::tempdir().unwrap();
        write_config(dir.path(), "{}");
        let error = select_for_model(dir.path(), Some("no-such-backend"), None)
            .err()
            .unwrap();
        assert!(error.to_string().contains("no runtime adapter"));
    }

    #[test]
    fn select_available_for_model_rejects_unknown_backend() {
        let dir = tempfile::tempdir().unwrap();
        write_config(dir.path(), "{}");
        let error = select_available_for_model(dir.path(), Some("no-such-backend"), None)
            .err()
            .unwrap();
        assert!(error.to_string().contains("no hay un runtime validado"));
    }

    #[cfg(feature = "native-cpu-backend")]
    #[test]
    fn select_for_model_requires_config_json() {
        let dir = tempfile::tempdir().unwrap();
        assert!(select_for_model(dir.path(), Some("cpu"), None).is_err());
        assert!(select_available_for_model(dir.path(), Some("cpu"), None).is_err());
    }

    #[cfg(feature = "native-cpu-backend")]
    #[test]
    fn select_for_model_rejects_malformed_config() {
        let dir = tempfile::tempdir().unwrap();
        write_config(dir.path(), "{not json");
        assert!(select_for_model(dir.path(), Some("cpu"), None).is_err());
        assert!(select_available_for_model(dir.path(), Some("cpu"), None).is_err());
    }

    /// Un config.json válido sin pesos no basta: inspect() es la autoridad final
    /// y el feature compilado no autoriza el directorio por sí solo.
    #[cfg(feature = "native-cpu-backend")]
    #[test]
    fn select_for_model_rejects_config_without_weights() {
        let dir = tempfile::tempdir().unwrap();
        write_config(
            dir.path(),
            r#"{"model_type":"llama","hidden_size":64,"num_hidden_layers":1}"#,
        );
        assert!(select_for_model(dir.path(), Some("cpu"), None).is_err());
        assert!(select_available_for_model(dir.path(), Some("cpu"), None).is_err());
    }
}
