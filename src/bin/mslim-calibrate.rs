//! Measure sustained execution and persist observations, never inferred claims.

use anyhow::{Context, Result, bail, ensure};
use mslim::calibration_contract::{
    CALIBRATION_RECEIPT_SCHEMA, CALIBRATION_RECEIPT_STATUS, LOGICAL_REPLAY_SNAPSHOT_KIND,
    PHYSICAL_COLDSTORE_SNAPSHOT_KIND, validate_calibration_receipt,
};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const DEFAULT_CONTEXTS: &[usize] = &[1, 128, 512, 2_048];

#[derive(Debug, Serialize)]
struct CalibrationReceipt {
    schema: &'static str,
    bundle: String,
    adapter: String,
    identity_digest: String,
    started_unix_seconds: u64,
    load_micros: u128,
    warmup_runs: usize,
    measured_runs: usize,
    decode_tokens_per_run: usize,
    context_samples: Vec<ContextSample>,
    snapshot_contract: SnapshotContract,
    validation_matrix_gate: &'static str,
    peak_rss_bytes: Option<u64>,
    checks: Vec<&'static str>,
    status: &'static str,
}

#[derive(Debug, Serialize)]
struct SnapshotContract {
    kind: &'static str,
    physical_kv_persisted: bool,
    restore_latency_measured_as_physical_kv_hydration: bool,
    note: &'static str,
}

#[derive(Debug, Serialize)]
struct ContextSample {
    prompt_tokens: usize,
    prefill_micros: Distribution,
    prefill_tokens_per_second_median: f64,
    decode_tokens: usize,
    decode_micros: Distribution,
    decode_tokens_per_second_median: f64,
    snapshot_handle_bytes: usize,
    peak_rss_bytes: Option<u64>,
    status: &'static str,
}

#[derive(Debug, Serialize)]
struct Distribution {
    samples: Vec<u128>,
    minimum: u128,
    median: u128,
    p95: u128,
    maximum: u128,
}

#[derive(Debug)]
struct Options {
    bundle: PathBuf,
    force: bool,
    warmups: usize,
    repeats: usize,
    decode_tokens: usize,
    contexts: Vec<usize>,
}

fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.iter().any(|arg| arg == "--tune-prefill") {
        #[cfg(feature = "accelerator-backend")]
        {
            let report = mslim::prefill_calibration::run_cli(args)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            return Ok(());
        }
        #[cfg(not(feature = "accelerator-backend"))]
        bail!(
            "--tune-prefill requires the physical accelerator-backend; no CPU simulation is substituted"
        );
    }
    let options = parse_options(args.into_iter())?;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let bundle = if options.bundle.is_absolute() {
        options.bundle.clone()
    } else {
        root.join(&options.bundle)
    };
    let digest = bundle
        .file_name()
        .and_then(|name| name.to_str())
        .context("digest inválido")?;
    ensure!(
        digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "el bundle debe estar identificado por su SHA-256"
    );

    let destination = root.join("deploy/local-production/runtime/calibration");
    let path = destination.join(format!("{digest}.json"));
    let mut config: Value = serde_json::from_slice(&fs::read(bundle.join("config.json"))?)?;
    let selection = mslim::adapter_catalog::select_available_for_model(&bundle, None, None)?;
    let validation_matrix_gate =
        require_passing_validation_matrix_if_accelerated(&root, digest, &selection.adapter_id)?;
    let calibration_coldstore = if selection.adapter_id == "accelerator-runtime" {
        let temporary_root = root.join("deploy/local-production/runtime/tmp");
        fs::create_dir_all(&temporary_root)?;
        let coldstore = tempfile::Builder::new()
            .prefix("calibration-coldstore-")
            .tempdir_in(&temporary_root)?;
        config["mslim_kv_mode"] = Value::String("exact-coldstore".into());
        config["mslim_lod_epsilon"] = Value::from(0.0);
        config["mslim_cold_directory"] =
            Value::String(coldstore.path().to_string_lossy().into_owned());
        Some(coldstore)
    } else {
        None
    };
    let host = mslim::adapter_catalog::host_capabilities();
    let mut identity_config = config.clone();
    if calibration_coldstore.is_some() {
        identity_config["mslim_cold_directory"] =
            Value::String("temporary-calibration-namespace".into());
    }
    let identity_input = serde_json::json!({
        "model_digest": digest,
        "config": identity_config,
        "adapter": selection,
        "host": host,
        "engine_schema": "mslim.engine/v1",
        "benchmark_schema": CALIBRATION_RECEIPT_SCHEMA,
        "warmups": options.warmups,
        "repeats": options.repeats,
        "decode_tokens": options.decode_tokens,
        "contexts": options.contexts,
    });
    let identity_digest = hex::encode(Sha256::digest(serde_json::to_vec(&identity_input)?));
    if !options.force
        && let Ok(bytes) = fs::read(&path)
    {
        let existing: Value = serde_json::from_slice(&bytes)
            .with_context(|| format!("calibración existente inválida: {}", path.display()))?;
        if validate_calibration_receipt(&existing, digest).is_ok()
            && existing.get("identity_digest").and_then(Value::as_str) == Some(&identity_digest)
        {
            println!("{}", serde_json::to_string_pretty(&existing)?);
            return Ok(());
        }
    }

    let adapter = mslim::adapter_catalog::create_adapter(&selection.adapter_id)?;
    let mut engine = mslim::engine::MslimEngine::new(adapter);
    engine.inspect(&bundle, &config)?;
    let started = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let load_start = Instant::now();
    engine.load(&bundle, &config)?;
    let load_micros = load_start.elapsed().as_micros();
    let empty_snapshot = engine.snapshot_kv()?;
    ensure!(!empty_snapshot.is_empty(), "snapshot lógico inicial vacío");

    let max_context = config
        .get("max_context_tokens")
        .or_else(|| config.get("max_position_embeddings"))
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .context("configuración sin límite de contexto")?;
    let contexts = options
        .contexts
        .iter()
        .copied()
        .filter(|&tokens| tokens + options.decode_tokens <= max_context)
        .collect::<Vec<_>>();
    ensure!(
        !contexts.is_empty(),
        "ningún contexto medible cabe junto al decode solicitado"
    );

    let mut context_samples = Vec::with_capacity(contexts.len());
    for prompt_tokens in contexts {
        for _ in 0..options.warmups {
            run_once(
                &mut engine,
                &empty_snapshot,
                prompt_tokens,
                options.decode_tokens,
            )?;
        }
        let mut prefill_samples = Vec::with_capacity(options.repeats);
        let mut decode_samples = Vec::with_capacity(options.repeats);
        let mut snapshot_bytes = 0usize;
        for _ in 0..options.repeats {
            let measurement = run_once(
                &mut engine,
                &empty_snapshot,
                prompt_tokens,
                options.decode_tokens,
            )?;
            prefill_samples.push(measurement.0);
            decode_samples.push(measurement.1);
            snapshot_bytes = measurement.2;
        }
        let prefill = distribution(prefill_samples)?;
        let decode = distribution(decode_samples)?;
        context_samples.push(ContextSample {
            prompt_tokens,
            prefill_tokens_per_second_median: rate(prompt_tokens, prefill.median),
            decode_tokens: options.decode_tokens,
            decode_tokens_per_second_median: rate(options.decode_tokens, decode.median),
            prefill_micros: prefill,
            decode_micros: decode,
            snapshot_handle_bytes: snapshot_bytes,
            peak_rss_bytes: peak_rss_bytes(),
            status: "measured",
        });
    }

    let receipt = CalibrationReceipt {
        schema: CALIBRATION_RECEIPT_SCHEMA,
        bundle: digest.to_owned(),
        adapter: selection.adapter_id,
        identity_digest,
        started_unix_seconds: started,
        load_micros,
        warmup_runs: options.warmups,
        measured_runs: options.repeats,
        decode_tokens_per_run: options.decode_tokens,
        context_samples,
        snapshot_contract: SnapshotContract {
            kind: if calibration_coldstore.is_some() {
                PHYSICAL_COLDSTORE_SNAPSHOT_KIND
            } else {
                LOGICAL_REPLAY_SNAPSHOT_KIND
            },
            physical_kv_persisted: calibration_coldstore.is_some(),
            restore_latency_measured_as_physical_kv_hydration: false,
            note: if calibration_coldstore.is_some() {
                "La calibración usa un namespace ColdStore temporal exacto; no mide la latencia de restore por separado."
            } else {
                "La calibración conserva una receta lógica de replay; no afirma persistencia física de tensores K/V."
            },
        },
        validation_matrix_gate,
        peak_rss_bytes: peak_rss_bytes(),
        checks: vec![
            "inspect",
            "load",
            "same_backend_validation_matrix_gate",
            "warmup",
            "repeated_prefill",
            "sustained_decode",
            if calibration_coldstore.is_some() {
                "physical_coldstore_snapshot_restore"
            } else {
                "logical_replay_snapshot_restore"
            },
        ],
        status: CALIBRATION_RECEIPT_STATUS,
    };
    fs::create_dir_all(&destination)?;
    let temporary = tempfile::NamedTempFile::new_in(&destination)?;
    fs::write(temporary.path(), serde_json::to_vec_pretty(&receipt)?)?;
    temporary.persist(&path).map_err(|error| error.error)?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}

fn run_once(
    engine: &mut mslim::engine::MslimEngine,
    empty_snapshot: &[u8],
    prompt_tokens: usize,
    decode_tokens: usize,
) -> Result<(u128, u128, usize)> {
    engine.restore_kv(empty_snapshot)?;
    let input = vec![0u32; prompt_tokens];
    let prefill_start = Instant::now();
    let result = engine.prefill(&input)?;
    let prefill_micros = prefill_start.elapsed().as_micros();
    ensure!(result.is_object(), "prefill no estructurado");

    let decode_start = Instant::now();
    let mut token = 0u32;
    for _ in 0..decode_tokens {
        token = engine.decode(token)?;
    }
    let decode_micros = decode_start.elapsed().as_micros();
    let logical_snapshot = engine.snapshot_kv()?;
    ensure!(!logical_snapshot.is_empty(), "snapshot lógico vacío");
    Ok((prefill_micros, decode_micros, logical_snapshot.len()))
}

fn require_passing_validation_matrix_if_accelerated(
    root: &Path,
    digest: &str,
    adapter_id: &str,
) -> Result<&'static str> {
    if adapter_id != "accelerator-runtime" {
        return Ok("not-required-for-non-accelerator");
    }
    let path = root
        .join("deploy/local-production/runtime/validation")
        .join(format!("validation-matrix-{digest}.json"));
    let receipt: Value = serde_json::from_slice(&fs::read(&path).with_context(|| {
        format!(
            "falta la matriz obligatoria de validación del acelerador: {}",
            path.display()
        )
    })?)
    .context("matriz de validación inválida")?;
    ensure!(
        receipt.get("schema").and_then(Value::as_str) == Some("mslim.validation-matrix/v1"),
        "la validación del acelerador no usa el contrato de matriz v1"
    );
    ensure!(
        receipt.get("status").and_then(Value::as_str) == Some("passed"),
        "la matriz del acelerador falló; se prohíbe calibrarlo para activación"
    );
    ensure!(
        receipt.get("bundle_sha256").and_then(Value::as_str) == Some(digest),
        "la matriz de validación pertenece a otro bundle"
    );
    ensure!(
        receipt
            .pointer("/determinism/all_passed")
            .and_then(Value::as_bool)
            == Some(true),
        "la matriz no demuestra determinismo de las cuatro rutas"
    );
    ensure!(
        receipt
            .pointer("/same_backend/all_passed")
            .and_then(Value::as_bool)
            == Some(true),
        "la matriz no demuestra exactitud densa contra ColdStore dentro de cada backend"
    );
    ensure!(
        receipt
            .pointer("/snapshot_contract/accelerator_exact_physical_snapshot")
            .and_then(Value::as_bool)
            == Some(true),
        "la matriz no demuestra snapshot físico del ColdStore acelerado"
    );
    Ok("same-backend-and-kv-integrity-passed")
}

fn distribution(mut samples: Vec<u128>) -> Result<Distribution> {
    ensure!(!samples.is_empty(), "distribución sin muestras");
    samples.sort_unstable();
    let len = samples.len();
    let p95_index = ((len * 95).saturating_add(99) / 100)
        .saturating_sub(1)
        .min(len - 1);
    Ok(Distribution {
        minimum: samples[0],
        median: samples[len / 2],
        p95: samples[p95_index],
        maximum: samples[len - 1],
        samples,
    })
}

fn rate(tokens: usize, micros: u128) -> f64 {
    if micros == 0 {
        f64::INFINITY
    } else {
        tokens as f64 * 1_000_000.0 / micros as f64
    }
}

fn parse_options(args: impl Iterator<Item = String>) -> Result<Options> {
    let mut args = args.peekable();
    let bundle = PathBuf::from(args.next().context(
        "uso: mslim-calibrate <bundle> [--force] [--warmups N] [--repeats N] [--decode-tokens N] [--contexts 1,128,512,2048]",
    )?);
    let mut options = Options {
        bundle,
        force: false,
        warmups: 1,
        repeats: 5,
        decode_tokens: 64,
        contexts: DEFAULT_CONTEXTS.to_vec(),
    };
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--force" => options.force = true,
            "--warmups" => options.warmups = args.next().context("falta --warmups")?.parse()?,
            "--repeats" => options.repeats = args.next().context("falta --repeats")?.parse()?,
            "--decode-tokens" => {
                options.decode_tokens = args.next().context("falta --decode-tokens")?.parse()?
            }
            "--contexts" => {
                options.contexts = parse_contexts(&args.next().context("falta --contexts")?)?
            }
            "--skip-context-sweep" => {
                bail!("--skip-context-sweep fue retirado: use --contexts con valores explícitos")
            }
            other => bail!("argumento desconocido: {other}"),
        }
    }
    ensure!(
        (3..=20).contains(&options.repeats),
        "--repeats debe ser 3..=20"
    );
    ensure!(
        (1..=10).contains(&options.warmups),
        "--warmups debe ser 1..=10"
    );
    ensure!(
        (8..=512).contains(&options.decode_tokens),
        "--decode-tokens debe ser 8..=512"
    );
    ensure!(
        options.contexts.len() >= 2,
        "--contexts debe incluir al menos dos longitudes distintas"
    );
    Ok(options)
}

fn parse_contexts(value: &str) -> Result<Vec<usize>> {
    let mut contexts = value
        .split(',')
        .map(|part| part.trim().parse::<usize>().context("contexto inválido"))
        .collect::<Result<Vec<_>>>()?;
    ensure!(!contexts.is_empty(), "lista de contextos vacía");
    ensure!(
        contexts.iter().all(|&tokens| tokens > 0),
        "los contextos deben ser positivos"
    );
    contexts.sort_unstable();
    contexts.dedup();
    Ok(contexts)
}

#[cfg(target_os = "linux")]
fn peak_rss_bytes() -> Option<u64> {
    let text = fs::read_to_string("/proc/self/status").ok()?;
    text.lines()
        .find(|line| line.starts_with("VmHWM:"))
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|value| value.parse::<u64>().ok())
        .map(|kib| kib * 1024)
}

#[cfg(not(target_os = "linux"))]
fn peak_rss_bytes() -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::{distribution, parse_contexts, require_passing_validation_matrix_if_accelerated};
    use serde_json::json;
    use std::fs;

    #[test]
    fn contexts_are_sorted_and_deduplicated() {
        assert_eq!(parse_contexts("512,1,128,512").unwrap(), vec![1, 128, 512]);
    }

    #[test]
    fn distribution_reports_upper_tail() {
        let values = distribution(vec![50, 10, 40, 20, 30]).unwrap();
        assert_eq!(values.minimum, 10);
        assert_eq!(values.median, 30);
        assert_eq!(values.p95, 50);
        assert_eq!(values.maximum, 50);
    }

    #[test]
    fn accelerator_calibration_requires_same_backend_integrity_not_cross_backend_bits() {
        let root = tempfile::tempdir().unwrap();
        let digest = "a".repeat(64);
        let directory = root
            .path()
            .join("deploy/local-production/runtime/validation");
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join(format!("validation-matrix-{digest}.json"));
        let mut receipt = json!({
            "schema": "mslim.validation-matrix/v1",
            "status": "passed",
            "bundle_sha256": digest,
            "determinism": {"all_passed": true},
            "same_backend": {"all_passed": true},
            "snapshot_contract": {"accelerator_exact_physical_snapshot": true},
            "cross_backend_diagnostic": {"all_logits_exact": false},
        });
        fs::write(&path, serde_json::to_vec(&receipt).unwrap()).unwrap();
        assert_eq!(
            require_passing_validation_matrix_if_accelerated(
                root.path(),
                &digest,
                "accelerator-runtime"
            )
            .unwrap(),
            "same-backend-and-kv-integrity-passed"
        );

        receipt["same_backend"]["all_passed"] = json!(false);
        fs::write(&path, serde_json::to_vec(&receipt).unwrap()).unwrap();
        assert!(
            require_passing_validation_matrix_if_accelerated(
                root.path(),
                &digest,
                "accelerator-runtime"
            )
            .is_err()
        );
    }
}
