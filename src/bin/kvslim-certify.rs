//! Project-local physical certification runner.
//!
//! The runner deliberately does not contain a list of model, vendor, device,
//! or test names. Cargo's test harness remains the source of truth for the
//! physical test set; this binary only discovers, executes, and records it.

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
    let mut extra = Vec::new();
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
                println!(
                    "kvslim-certify [--model DIR] [--features F1,F2] [--timeout-secs N] [-- cargo test arguments]"
                );
                return Ok(());
            }
            "--" => extra.extend(&mut args),
            other => extra.push(other.to_owned()),
        }
    }

    if let Some(path) = &model {
        if !path.is_dir() {
            bail!("model path is not a directory: {}", path.display());
        }
    }

    let started = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let mut command = Command::new("cargo");
    command
        .current_dir(&root)
        .args(["test", "--locked", "--lib"]);
    if let Some(value) = &features {
        command.args(["--features", value]);
    }
    command
        .args(["--", "--ignored", "--nocapture"])
        .args(extra)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(path) = &model {
        command.env("KVSLIM_TEST_MODEL", path);
    }
    if let Some(seconds) = timeout_seconds {
        command.env("KVSLIM_CERTIFICATION_TIMEOUT_SECS", seconds.to_string());
    }

    let output = command.output().context("run physical test harness")?;
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let tests_run = stdout
        .lines()
        .find_map(|line| {
            let rest = line.strip_prefix("running ")?;
            let count = rest
                .strip_suffix(" tests")
                .or_else(|| rest.strip_suffix(" test"))?;
            count.parse::<usize>().ok()
        })
        .unwrap_or(0);
    let status = if output.status.success() && tests_run > 0 {
        "passed"
    } else if output.status.success() {
        "blocked"
    } else if model.is_none() {
        "blocked"
    } else {
        "failed"
    };
    let receipt = json!({
        "schema": "kvslim.physical_certification/v1",
        "started_unix_seconds": started,
        "finished_unix_seconds": SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        "status": status,
        "tests_run": tests_run,
        "exit_code": output.status.code(),
        "features": features,
        "model": model.as_ref().map(|path| path.display().to_string()),
        "host": kvslim::adapter_catalog::host_capabilities(),
        "harness": "cargo-test-ignored",
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
