use mslim_adapter_api::{
    ActivationPolicy, AdapterAvailability, AdapterDescriptor, AdapterStatus, ComputeKind,
    select_adapter, select_available_adapter,
};

#[test]
fn selection_is_capability_based_and_does_not_fallback_across_backend() {
    let adapters = vec![AdapterDescriptor {
        id: "cpu-adapter".into(),
        backend: "cpu".into(),
        compute: ComputeKind::Cpu,
        priority: 10,
        activation: ActivationPolicy::Default,
        supported_model_types: vec!["generic".into()],
    }];

    assert!(select_adapter(&adapters, Some("gpu"), None).is_none());
    assert_eq!(
        select_adapter(&adapters, Some("cpu"), Some(ComputeKind::Cpu))
            .expect("registered CPU adapter")
            .adapter_id,
        "cpu-adapter"
    );
}

#[test]
fn automatic_selection_rejects_compiled_but_unprobed_runtimes() {
    let status = vec![AdapterStatus {
        descriptor: AdapterDescriptor {
            id: "optional-runtime".into(),
            backend: "optional".into(),
            compute: ComputeKind::Gpu,
            priority: 1000,
            activation: ActivationPolicy::Explicit,
            supported_model_types: vec!["generic".into()],
        },
        availability: AdapterAvailability::RequiresProbe,
        reason: "not probed".into(),
    }];
    assert!(select_available_adapter(&status, None, None).is_none());
}
