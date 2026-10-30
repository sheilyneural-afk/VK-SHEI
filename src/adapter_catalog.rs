//! Runtime adapter catalog for the local deployment profile.
//!
//! The catalog reports only adapters compiled into this binary. Hardware and
//! model compatibility are still checked by the selected adapter at inspect
//! and load time.

use std::path::Path;

use anyhow::{Result, bail};
#[allow(unused_imports)]
use kvslim_adapter_api::{
    ActivationPolicy, AdapterDescriptor, AdapterSelection, ComputeKind, select_adapter,
    select_available_adapter,
};
use kvslim_adapter_api::{AdapterAvailability, AdapterStatus, ModelAdapter};

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
    if let Ok(runtime) = crate::opencl_backend::OpenClBackend::new() {
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
        priority: 200,
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
        priority: 100,
        activation: ActivationPolicy::Explicit,
        supported_model_types: Vec::new(),
    });

    #[cfg(feature = "accelerator-backend")]
    adapters.push(AdapterDescriptor {
        id: "accelerator-runtime".to_owned(),
        backend: "accelerator".to_owned(),
        compute: ComputeKind::Any,
        priority: 100,
        activation: ActivationPolicy::Explicit,
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
                        match crate::opencl_backend::OpenClBackend::new() {
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
#[allow(unreachable_code, unused_variables)]
pub fn select_for_model(
    model_dir: impl AsRef<Path>,
    requested_backend: Option<&str>,
    requested_compute: Option<ComputeKind>,
) -> Result<AdapterSelection> {
    let model_dir = model_dir.as_ref();
    let available = compiled_adapters().into_iter().collect::<Vec<_>>();
    let selection = select_adapter(&available, requested_backend, requested_compute)
        .ok_or_else(|| anyhow::anyhow!("no runtime adapter is available by default; select an installed adapter explicitly after capability probing"))?;

    match selection.backend.as_str() {
        "cpu" => {
            #[cfg(feature = "torch-backend")]
            let _ = create_adapter(&selection.adapter_id)?;
            #[cfg(not(feature = "torch-backend"))]
            bail!("torch adapter is not compiled");
        }
        "accelerator" => {
            #[cfg(feature = "accelerator-backend")]
            let _ = create_adapter(&selection.adapter_id)?;
            #[cfg(not(feature = "accelerator-backend"))]
            bail!("accelerator adapter is not compiled");
        }
        backend => bail!("adapter has no validation policy: {backend}"),
    }

    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(model_dir.join("config.json"))?)?;
    let adapter = create_adapter(&selection.adapter_id)?;
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

/// Construct only an adapter whose implementation is compiled in this binary.
pub fn create_adapter(adapter_id: &str) -> Result<Box<dyn ModelAdapter>> {
    match adapter_id {
        "accelerator-runtime" => {
            #[cfg(feature = "accelerator-backend")]
            return Ok(Box::new(
                crate::opencl_adapter::OpenClModelAdapter::default(),
            ));
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
