use mslim::{
    attention_kernel::{AttentionKernelConfig, AttentionKernelMode},
    calibration_profile::PrefillSettings,
};
#[test]
fn legacy_profile_retains_reference_and_id() {
    let settings: PrefillSettings =
        serde_json::from_str(r#"{"batch_tokens":64,"weight_reuse_tokens":8}"#).unwrap();
    assert_eq!(settings.attention, AttentionKernelConfig::default());
    assert_eq!(settings.id(), "batch-64-reuse-8");
    settings.validate().unwrap();
}
#[test]
fn exact_calibration_cannot_promote_online_contract() {
    let mut s = PrefillSettings {
        batch_tokens: Some(64),
        weight_reuse_tokens: 8,
        attention: AttentionKernelConfig::default(),
    };
    s.attention.mode = AttentionKernelMode::OnlineExperimental;
    assert!(s.validate().is_err());
    s.attention.mode = AttentionKernelMode::StrictTiled;
    assert!(s.validate().is_ok());
    assert_ne!(s.id(), "batch-64-reuse-8");
}

#[test]
fn snapshots_cannot_cross_numeric_contracts() {
    let reference = AttentionKernelConfig::default();
    reference.check_snapshot(None).unwrap();
    let strict = AttentionKernelConfig {
        mode: AttentionKernelMode::StrictTiled,
        ..reference
    };
    let online = AttentionKernelConfig {
        mode: AttentionKernelMode::OnlineExperimental,
        ..reference
    };
    assert!(strict.check_snapshot(None).is_err());
    assert!(
        reference
            .check_snapshot(Some(&serde_json::to_value(online).unwrap()))
            .is_err()
    );
    strict
        .check_snapshot(Some(&serde_json::to_value(strict).unwrap()))
        .unwrap();
}
