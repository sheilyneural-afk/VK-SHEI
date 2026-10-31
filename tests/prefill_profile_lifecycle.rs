//! Filesystem integration tests. These do not claim physical GPU performance.
use mslim::calibration_profile::{
    PrefillEnvironment, PrefillProfileParameters, PrefillSettings, read_promoted_profile, sha256,
};
use mslim::calibration_transaction::{
    CalibrationGates, CalibrationObservation, CalibrationProposal, CalibrationState,
    CalibrationTransaction,
};
use std::fs;

fn environment() -> PrefillEnvironment {
    PrefillEnvironment {
        model_digest: "a".repeat(64),
        tokenizer_digest: "b".repeat(64),
        backend: "test-contract".into(),
        hardware_digest: "c".repeat(64),
        execution_digest: "d".repeat(64),
        implementation_digest: "e".repeat(64),
    }
}
fn settings(batch: usize) -> PrefillSettings {
    PrefillSettings {
        batch_tokens: Some(batch),
        weight_reuse_tokens: 8,
    }
}
fn profile(batch: usize) -> mslim::calibration_transaction::CalibrationProfile {
    PrefillProfileParameters {
        environment: environment(),
        settings: settings(batch),
        validated_context_tokens: vec![16, 32, 64],
        prompt_suite_sha256: "f".repeat(64),
    }
    .into_profile()
    .unwrap()
}
fn observation(batch: usize, throughput: f64) -> CalibrationObservation {
    CalibrationObservation {
        profile_id: settings(batch).id(),
        workload_id: "contract-workload".into(),
        seed: 0,
        sample_count: 9,
        parity_ok: true,
        maximum_exactness_error: 0.0,
        maximum_certificate_error: 0.0,
        throughput_tokens_per_second: throughput,
        latency_p95_ms: 10.0,
        peak_memory_bytes: 1000,
        bytes_read_per_token: 0,
    }
}
fn setup() -> (
    tempfile::TempDir,
    std::path::PathBuf,
    CalibrationTransaction,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("active.json");
    fs::write(&path, serde_json::to_vec_pretty(&profile(128)).unwrap()).unwrap();
    let proposal = CalibrationProposal {
        proposal_id: "test-lifecycle".into(),
        baseline_profile_digest: sha256(&fs::read(&path).unwrap()),
        candidate: profile(64),
        gates: CalibrationGates::default(),
    };
    let tx = CalibrationTransaction::new(dir.path(), &path, proposal).unwrap();
    (dir, path, tx)
}
#[test]
fn only_explicitly_promoted_exact_profile_can_be_consumed() {
    let (_dir, path, mut tx) = setup();
    tx.propose().unwrap();
    tx.begin_shadow().unwrap();
    assert!(
        tx.complete_shadow(&observation(128, 10.0), &observation(64, 12.0))
            .unwrap()
            .accepted
    );
    tx.begin_canary().unwrap();
    assert!(read_promoted_profile(&path, &environment()).is_err());
    assert!(
        tx.complete_canary(&observation(128, 10.0), &observation(64, 12.0))
            .unwrap()
            .accepted
    );
    assert!(read_promoted_profile(&path, &environment()).is_err());
    tx.promote("test-explicit-authority").unwrap();
    assert!(read_promoted_profile(&path, &environment()).is_err()); // writer still owns its lease
    drop(tx);
    assert_eq!(
        read_promoted_profile(&path, &environment())
            .unwrap()
            .settings,
        settings(64)
    );
}
#[test]
fn corruption_after_promotion_is_detected() {
    let (_dir, path, mut tx) = setup();
    tx.propose().unwrap();
    tx.begin_shadow().unwrap();
    tx.complete_shadow(&observation(128, 10.0), &observation(64, 12.0))
        .unwrap();
    tx.begin_canary().unwrap();
    tx.complete_canary(&observation(128, 10.0), &observation(64, 12.0))
        .unwrap();
    tx.promote("test").unwrap();
    drop(tx);
    fs::write(&path, serde_json::to_vec_pretty(&profile(32)).unwrap()).unwrap();
    assert!(read_promoted_profile(&path, &environment()).is_err());
}
#[test]
fn regression_rolls_back_and_cannot_leave_a_loadable_candidate() {
    let (_dir, path, mut tx) = setup();
    let original = fs::read(&path).unwrap();
    tx.propose().unwrap();
    tx.begin_shadow().unwrap();
    tx.complete_shadow(&observation(128, 10.0), &observation(64, 12.0))
        .unwrap();
    tx.begin_canary().unwrap();
    let mut wrong = observation(64, 100.0);
    wrong.parity_ok = false;
    assert!(
        !tx.complete_canary(&observation(128, 10.0), &wrong)
            .unwrap()
            .accepted
    );
    assert_eq!(tx.state(), CalibrationState::RolledBack);
    drop(tx);
    assert_eq!(fs::read(&path).unwrap(), original);
    assert!(read_promoted_profile(&path, &environment()).is_err());
}
#[test]
fn rolled_back_proposal_keeps_the_previously_promoted_profile_loadable() {
    let (dir, path, mut tx) = setup();
    tx.propose().unwrap();
    tx.begin_shadow().unwrap();
    tx.complete_shadow(&observation(128, 10.0), &observation(64, 12.0))
        .unwrap();
    tx.begin_canary().unwrap();
    tx.complete_canary(&observation(128, 10.0), &observation(64, 12.0))
        .unwrap();
    tx.promote("test-explicit-authority").unwrap();
    drop(tx);
    assert_eq!(
        read_promoted_profile(&path, &environment())
            .unwrap()
            .settings,
        settings(64)
    );

    // A later proposal that fails its canary rolls back to the promoted profile.
    let proposal = CalibrationProposal {
        proposal_id: "second-proposal".into(),
        baseline_profile_digest: sha256(&fs::read(&path).unwrap()),
        candidate: profile(32),
        gates: CalibrationGates::default(),
    };
    let mut second = CalibrationTransaction::new(dir.path(), &path, proposal).unwrap();
    second.propose().unwrap();
    second.begin_shadow().unwrap();
    second
        .complete_shadow(&observation(64, 12.0), &observation(32, 14.0))
        .unwrap();
    second.begin_canary().unwrap();
    let mut wrong = observation(32, 100.0);
    wrong.parity_ok = false;
    assert!(
        !second
            .complete_canary(&observation(64, 12.0), &wrong)
            .unwrap()
            .accepted
    );
    assert_eq!(second.state(), CalibrationState::RolledBack);
    drop(second);
    assert_eq!(
        read_promoted_profile(&path, &environment())
            .unwrap()
            .settings,
        settings(64)
    );
}
#[test]
fn missing_measurements_are_not_replaced_by_predictions() {
    let (_dir, path, mut tx) = setup();
    let original = fs::read(&path).unwrap();
    tx.propose().unwrap();
    tx.begin_shadow().unwrap();
    let mut predicted = observation(64, 100.0);
    predicted.sample_count = 0;
    assert!(
        tx.complete_shadow(&observation(128, 10.0), &predicted)
            .is_err()
    );
    assert!(tx.begin_canary().is_err());
    assert_eq!(fs::read(path).unwrap(), original);
}
