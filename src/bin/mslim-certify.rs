//! Project-local physical conformance runner.
//!
//! It invokes `mslim-validate-model` once and records inspect, load, prefill,
//! decode, snapshot and restore results for the selected adapter. The receipt
//! does not replace the four-route validation matrix, calibration, gateway E2E
//! or release installation checks.

use anyhow::{Context, Result, bail};
use serde_json::json;
use std::{
    env, fs,
    path::PathBuf,
    process::{Command, Stdio},
    time::{SystemTime, UNIX_EPOCH},
};

fn main() -> Result<()> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut args = env::args().skip(1);
    let mut model = None;
    let mut timeout_seconds = None;
    let mut features = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model = Some(PathBuf::from(args.next().context("missing --model path")?)),
            "--timeout-secs" => {
                timeout_seconds = Some(
                    args.next()
                        .context("missing --timeout-secs value")?
                        .parse::<u64>()?,
                );
            }
            "--features" => features = Some(args.next().context("missing --features value")?),
            "--help" | "-h" => {
                println!("mslim-certify --model DIR [--features F1,F2] [--timeout-secs N]");
                return Ok(());
            }
            other => bail!("unrecognized option {other}"),
        }
    }

    if let Some(path) = &model
        && !path.is_dir()
    {
        bail!("model path is not a directory: {}", path.display());
    }

    let host = mslim::adapter_catalog::host_capabilities();
    let physical_available = host
        .adapters
        .iter()
        .any(|status| status.availability == mslim::AdapterAvailability::RuntimeAvailable);
    let selected_features = features
        .clone()
        .or_else(|| physical_available.then(|| "accelerator-backend".to_owned()));
    let started = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let (status, checks_run, exit_code, stdout, stderr) = if !physical_available {
        ("not_applicable", 0, None, String::new(), String::new())
    } else if model.is_none() {
        (
            "blocked",
            0,
            None,
            String::new(),
            "physical adapter detected but no model was supplied".to_owned(),
        )
    } else {
        let mut command = Command::new("cargo");
        command
            .current_dir(&root)
            .args(["run", "--locked", "--bin", "mslim-validate-model"]);
        if let Some(value) = &selected_features {
            command.args(["--features", value]);
        }
        command
            .arg("--")
            .arg(model.as_ref().expect("model checked above"));
        command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(path) = &model {
            command.env("MSLIM_TEST_MODEL", path);
        }
        if let Some(seconds) = timeout_seconds {
            command.env("MSLIM_CERTIFICATION_TIMEOUT_SECS", seconds.to_string());
        }
        let output = command.output().context("run physical test harness")?;
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        let status = if output.status.success() {
            "passed"
        } else {
            "failed"
        };
        (status, 6, output.status.code(), stdout, stderr)
    };
    let receipt = json!({
        "project": mslim::project_identity::PROJECT,
        "product": mslim::project_identity::PRODUCT,
        "technical_namespace": mslim::project_identity::TECHNICAL_NAMESPACE,
        "schema": "mslim.certification/v1",
        "source_revision": mslim::build_info::source_revision(),
        "started_unix_seconds": started,
        "finished_unix_seconds": SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        "status": status,
        "checks_run": checks_run,
        "exit_code": exit_code,
        "features": selected_features,
        "model": model.as_ref().map(|path| path.display().to_string()),
        "host": host,
        "harness": "mslim-validate-model",
        "stdout": stdout,
        "stderr": stderr,
    });
    let destination =
        root.join("deploy/local-production/runtime/validation/physical-certification.json");
    fs::create_dir_all(destination.parent().context("receipt parent missing")?)?;
    fs::write(&destination, serde_json::to_vec_pretty(&receipt)?)?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    if status == "failed" {
        bail!(
            "physical certification failed; receipt: {}",
            destination.display()
        );
    }
    Ok(())
}
