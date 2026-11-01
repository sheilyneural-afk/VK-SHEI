//! Measure sustained execution and persist observations, never inferred claims.

use anyhow::{Context, Result, bail, ensure};
use mslim::calibration_contract::{
    ACCELERATOR_TOKEN_ONLY_PATH, ACCELERATOR_TOKEN_ONLY_PREFILL_PATH, ADAPTER_TOKEN_ONLY_PATH,
    ADAPTER_TOKEN_ONLY_PREFILL_PATH, CALIBRATION_RECEIPT_SCHEMA, CALIBRATION_RECEIPT_STATUS,
    LOGICAL_REPLAY_SNAPSHOT_KIND, PHYSICAL_COLDSTORE_SNAPSHOT_KIND, TOKEN_ONLY_DECODE_KIND,
    TOKEN_ONLY_PREFILL_KIND, validate_accelerator_runtime_configuration,
    validate_calibration_receipt,
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
struct SustainedCalibrationReceipt {
    project: &'static str,
    product: &'static str,
    technical_namespace: &'static str,
    schema: &'static str,
    source_revision: Value,
    bundle: String,
    adapter: String,
    identity_digest: String,
    started_unix_seconds: u64,
    load_micros: u128,
    warmup_runs: usize,
    measured_runs: usize,
    decode_tokens_per_run: usize,
    calibration_environment: Value,
    frozen_runtime_configuration: Value,
    prefill_execution_contract: TokenOnlyExecutionContract,
    decode_execution_contract: TokenOnlyExecutionContract,
    context_samples: Vec<ContextSample>,
    snapshot_contract: SnapshotContract,
    validation_matrix_gate: &'static str,
    validation_matrix_sha256: Option<String>,
    peak_rss_bytes: Option<u64>,
    checks: Vec<&'static str>,
    status: &'static str,
}

#[derive(Debug, Serialize)]
struct TokenOnlyExecutionContract {
    kind: &'static str,
    adapter_path: &'static str,
    diagnostic_full_vocabulary_logits_captured: bool,
    note: &'static str,
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
    max_context_tokens: Option<usize>,
    weight_mode: &'static str,
}

#[derive(Debug)]
struct ValidationMatrixGate {
    status: &'static str,
    frozen_runtime_configuration: Option<Value>,
    receipt_sha256: Option<String>,
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
    let calibration_environment = calibration_environment(&selection.adapter_id)?;
    let validation_matrix = require_passing_validation_matrix_if_accelerated(
        &root,
        digest,
        &selection.adapter_id,
        &calibration_environment,
    )?;
    let configured_max_context = options.max_context_tokens.unwrap_or_else(|| {
        config
            .get("max_context_tokens")
            .or_else(|| config.get("max_position_embeddings"))
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(0)
    });
    ensure!(
        (1..=32_768).contains(&configured_max_context),
        "configuración sin límite de contexto válido"
    );
    let frozen_runtime_configuration = if selection.adapter_id == "accelerator-runtime" {
        let matrix_configuration = validation_matrix
            .frozen_runtime_configuration
            .as_ref()
            .context("la matriz no contiene configuración congelada del acelerador")?;
        let effective = accelerator_runtime_configuration(
            matrix_configuration,
            configured_max_context,
            options.weight_mode,
        )?;
        apply_accelerator_runtime_configuration(&mut config, &effective)?;
        effective
    } else {
        config["max_context_tokens"] = Value::from(configured_max_context as u64);
        serde_json::json!({
            "adapter": selection.adapter_id,
            "max_context_tokens": configured_max_context,
            "weight_mode": "adapter-default",
        })
    };
    let validation_matrix_gate = validation_matrix.status;
    let validation_matrix_sha256 = validation_matrix.receipt_sha256.clone();
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
    let mut identity_config = config.clone();
    if calibration_coldstore.is_some() {
        identity_config["mslim_cold_directory"] =
            Value::String("temporary-calibration-namespace".into());
    }
    let identity_input = serde_json::json!({
        "source_revision": mslim::build_info::source_revision(),
        "model_digest": digest,
        "config": identity_config,
        "adapter": selection,
        "calibration_environment": calibration_environment,
        "engine_schema": "mslim.engine/v1",
        "benchmark_schema": CALIBRATION_RECEIPT_SCHEMA,
        "warmups": options.warmups,
        "repeats": options.repeats,
        "decode_tokens": options.decode_tokens,
        "prefill_execution_kind": TOKEN_ONLY_PREFILL_KIND,
        "decode_execution_kind": TOKEN_ONLY_DECODE_KIND,
        "contexts": options.contexts,
        "frozen_runtime_configuration": frozen_runtime_configuration,
        "validation_matrix_sha256": validation_matrix_sha256,
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
    if selection.adapter_id == "accelerator-runtime" {
        let loaded_environment = engine
            .calibration_environment()?
            .context("el adapter acelerado cargado no publica su entorno físico")?;
        ensure!(
            loaded_environment == calibration_environment,
            "el dispositivo acelerador cargado cambió después del sondeo de calibración"
        );
    }
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

    let decode_adapter_path = if selection.adapter_id == "accelerator-runtime" {
        ACCELERATOR_TOKEN_ONLY_PATH
    } else {
        ADAPTER_TOKEN_ONLY_PATH
    };
    let prefill_adapter_path = if selection.adapter_id == "accelerator-runtime" {
        ACCELERATOR_TOKEN_ONLY_PREFILL_PATH
    } else {
        ADAPTER_TOKEN_ONLY_PREFILL_PATH
    };
    let receipt = SustainedCalibrationReceipt {
        project: mslim::project_identity::PROJECT,
        product: mslim::project_identity::PRODUCT,
        technical_namespace: mslim::project_identity::TECHNICAL_NAMESPACE,
        schema: CALIBRATION_RECEIPT_SCHEMA,
        source_revision: mslim::build_info::source_revision(),
        bundle: digest.to_owned(),
        adapter: selection.adapter_id,
        identity_digest,
        started_unix_seconds: started,
        load_micros,
        warmup_runs: options.warmups,
        measured_runs: options.repeats,
        decode_tokens_per_run: options.decode_tokens,
        calibration_environment,
        frozen_runtime_configuration,
        prefill_execution_contract: TokenOnlyExecutionContract {
            kind: TOKEN_ONLY_PREFILL_KIND,
            adapter_path: prefill_adapter_path,
            diagnostic_full_vocabulary_logits_captured: false,
            note: "Mide el prompt completo y obtiene solo el argmax siguiente; ningún backend solicita la copia diagnóstica del vocabulario.",
        },
        decode_execution_contract: TokenOnlyExecutionContract {
            kind: TOKEN_ONLY_DECODE_KIND,
            adapter_path: decode_adapter_path,
            diagnostic_full_vocabulary_logits_captured: false,
            note: "Mide avance causal y argmax del siguiente token; no solicita logits completos. En el acelerador evita su lectura al host.",
        },
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
        validation_matrix_sha256,
        peak_rss_bytes: peak_rss_bytes(),
        checks: vec![
            "inspect",
            "load",
            "same_backend_validation_matrix_gate",
            "warmup",
            "repeated_token_only_prefill",
            "sustained_token_only_decode",
            if calibration_coldstore.is_some() {
                "physical_coldstore_snapshot_restore"
            } else {
                "logical_replay_snapshot_restore"
            },
        ],
        status: CALIBRATION_RECEIPT_STATUS,
    };
    let receipt_value = serde_json::to_value(&receipt)?;
    validate_calibration_receipt(&receipt_value, digest)?;
    fs::create_dir_all(&destination)?;
    let temporary = tempfile::NamedTempFile::new_in(&destination)?;
    fs::write(temporary.path(), serde_json::to_vec_pretty(&receipt)?)?;
    temporary.persist(&path).map_err(|error| error.error)?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}

fn calibration_environment(adapter_id: &str) -> Result<Value> {
    if adapter_id == "accelerator-runtime" {
        #[cfg(feature = "accelerator-backend")]
        {
            return mslim::gpu_llm::probed_calibration_environment();
        }
        #[cfg(not(feature = "accelerator-backend"))]
        bail!("accelerator calibration requires the accelerator-backend feature");
    }
    Ok(serde_json::json!({
        "schema": mslim::calibration_contract::CALIBRATION_ENVIRONMENT_SCHEMA,
        "backend": adapter_id,
        "architecture": std::env::consts::ARCH,
    }))
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
    let (result, mut token) = engine.prefill_token_only(&input)?;
    let prefill_micros = prefill_start.elapsed().as_micros();
    ensure!(result.is_object(), "prefill no estructurado");

    let decode_start = Instant::now();
    for _ in 0..decode_tokens {
        token = engine.decode_token_only(token)?;
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
    expected_environment: &Value,
) -> Result<ValidationMatrixGate> {
    if adapter_id != "accelerator-runtime" {
        return Ok(ValidationMatrixGate {
            status: "not-required-for-non-accelerator",
            frozen_runtime_configuration: None,
            receipt_sha256: None,
        });
    }
    let path = root
        .join("deploy/local-production/runtime/validation")
        .join(format!("validation-matrix-{digest}.json"));
    let bytes = fs::read(&path).with_context(|| {
        format!(
            "falta la matriz obligatoria de validación del acelerador: {}",
            path.display()
        )
    })?;
    let receipt_sha256 = hex::encode(Sha256::digest(&bytes));
    let receipt: Value = serde_json::from_slice(&bytes).context("matriz de validación inválida")?;
    ensure!(
        receipt.get("schema").and_then(Value::as_str)
            == Some(mslim::project_identity::VALIDATION_MATRIX_SCHEMA),
        "la validación del acelerador no usa el contrato canónico de matriz"
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
        mslim::build_info::is_current_clean_source_revision(receipt.get("source_revision")),
        "la matriz de validación pertenece a otra revisión o a un build sucio"
    );
    ensure!(
        receipt.get("calibration_environment") == Some(expected_environment),
        "la matriz de validación pertenece a otro dispositivo o driver"
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
    ensure!(
        receipt
            .pointer("/policy/approximation_epsilon")
            .and_then(Value::as_f64)
            == Some(0.0),
        "la matriz no certifica la ruta exacta epsilon cero"
    );
    let frozen_runtime_configuration = receipt
        .get("frozen_runtime_configuration")
        .cloned()
        .context("la matriz no declara su configuración congelada")?;
    Ok(ValidationMatrixGate {
        status: "same-backend-and-kv-integrity-passed",
        frozen_runtime_configuration: Some(frozen_runtime_configuration),
        receipt_sha256: Some(receipt_sha256),
    })
}

fn accelerator_runtime_configuration(
    matrix: &Value,
    max_context_tokens: usize,
    weight_mode: &str,
) -> Result<Value> {
    let field = |name: &str| {
        matrix
            .get(name)
            .cloned()
            .with_context(|| format!("la matriz no declara {name}"))
    };
    let configuration = serde_json::json!({
        "kv_mode": "exact-coldstore",
        "weight_mode": weight_mode,
        "max_context_tokens": max_context_tokens,
        "lod_epsilon": 0.0,
        "stage_tokens": field("stage_tokens")?,
        "prefill_batch_tokens": field("prefill_batch_tokens")?,
        "prefill_weight_reuse_tokens": field("prefill_weight_reuse_tokens")?,
        "hot_v_tokens": field("hot_v_tokens")?,
        "prompt_replay_tokens": field("prompt_replay_tokens")?,
        "shared_prefix": field("shared_prefix")?,
        "micro_hot_v_tokens": 0,
        "deferred_cold_commit": field("deferred_cold_commit")?,
        "exact_hot_fused_attention": field("exact_hot_fused_attention")?,
        "decode_unroll4": field("decode_unroll4")?,
        "decode_fused_kernels": field("decode_fused_kernels")?,
        "prefill_fuse_projections": field("prefill_fuse_projections")?,
        "prefill_profile_id": Value::Null,
    });
    validate_accelerator_runtime_configuration(&configuration)?;
    Ok(configuration)
}

fn apply_accelerator_runtime_configuration(config: &mut Value, frozen: &Value) -> Result<()> {
    validate_accelerator_runtime_configuration(frozen)?;
    let config = config
        .as_object_mut()
        .context("model config must be a JSON object")?;
    let frozen = frozen
        .as_object()
        .context("accelerator runtime configuration must be an object")?;
    let mappings = [
        ("max_context_tokens", "max_context_tokens"),
        ("weight_mode", "mslim_weight_mode"),
        ("lod_epsilon", "mslim_lod_epsilon"),
        ("stage_tokens", "mslim_stage_tokens"),
        ("prefill_batch_tokens", "mslim_prefill_batch_tokens"),
        (
            "prefill_weight_reuse_tokens",
            "mslim_prefill_weight_reuse_tokens",
        ),
        ("hot_v_tokens", "mslim_hot_v_tokens"),
        ("prompt_replay_tokens", "mslim_prompt_replay_tokens"),
        ("shared_prefix", "mslim_shared_prefix"),
        ("deferred_cold_commit", "mslim_deferred_cold_commit"),
        (
            "exact_hot_fused_attention",
            "mslim_exact_hot_fused_attention",
        ),
        ("decode_unroll4", "mslim_decode_unroll4"),
        ("decode_fused_kernels", "mslim_decode_fused_kernels"),
        ("prefill_fuse_projections", "mslim_prefill_fuse_projections"),
    ];
    for (source, destination) in mappings {
        config.insert(destination.into(), frozen[source].clone());
    }
    // The worker CLI has no independent attention-kernel switch and a
    // promoted prefill profile has its own evidence lifecycle.  A model
    // bundle must not smuggle either route into a calibration whose canonical
    // configuration explicitly says `prefill_profile_id: null`.
    config.remove("mslim_prefill_profile");
    config.remove("mslim_attention_kernel");
    config.remove("mslim_allow_experimental_attention");
    config.insert(
        "mslim_kv_mode".into(),
        Value::String("exact-coldstore".into()),
    );
    Ok(())
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
        "uso: mslim-calibrate <bundle> [--force] [--warmups N] [--repeats N] [--decode-tokens N] [--contexts 1,128,512,2048] [--max-context-tokens N] [--resident|--stream-weights]",
    )?);
    let mut options = Options {
        bundle,
        force: false,
        warmups: 1,
        repeats: 5,
        decode_tokens: 64,
        contexts: DEFAULT_CONTEXTS.to_vec(),
        max_context_tokens: None,
        weight_mode: "resident",
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
            "--max-context-tokens" => {
                options.max_context_tokens =
                    Some(args.next().context("falta --max-context-tokens")?.parse()?)
            }
            "--resident" => options.weight_mode = "resident",
            "--stream-weights" => options.weight_mode = "stream",
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
    use super::{
        accelerator_runtime_configuration, apply_accelerator_runtime_configuration, distribution,
        parse_contexts, require_passing_validation_matrix_if_accelerated,
    };
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
    fn canonical_calibration_removes_unrepresented_profiles_and_attention_kernels() {
        let matrix = json!({
            "stage_tokens": 64,
            "prefill_batch_tokens": 128,
            "prefill_weight_reuse_tokens": 8,
            "hot_v_tokens": 512,
            "prompt_replay_tokens": 0,
            "shared_prefix": false,
            "deferred_cold_commit": false,
            "exact_hot_fused_attention": true,
            "decode_unroll4": false,
            "decode_fused_kernels": true,
            "prefill_fuse_projections": true
        });
        let frozen = accelerator_runtime_configuration(&matrix, 8192, "resident").unwrap();
        let mut config = json!({
            "max_position_embeddings": 8192,
            "mslim_prefill_profile": "foreign-profile.json",
            "mslim_attention_kernel": {"mode": "experimental"},
            "mslim_allow_experimental_attention": true
        });
        apply_accelerator_runtime_configuration(&mut config, &frozen).unwrap();
        assert!(config.get("mslim_prefill_profile").is_none());
        assert!(config.get("mslim_attention_kernel").is_none());
        assert!(config.get("mslim_allow_experimental_attention").is_none());
        assert_eq!(config["mslim_weight_mode"], "resident");
        assert_eq!(config["max_context_tokens"], 8192);
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
        let environment = json!({
            "schema": mslim::calibration_contract::CALIBRATION_ENVIRONMENT_SCHEMA,
            "backend": "accelerator-runtime",
            "device_key": "d".repeat(64)
        });
        let mut receipt = json!({
            "project": mslim::project_identity::PROJECT,
            "product": mslim::project_identity::PRODUCT,
            "technical_namespace": mslim::project_identity::TECHNICAL_NAMESPACE,
            "schema": mslim::project_identity::VALIDATION_MATRIX_SCHEMA,
            "status": "passed",
            "source_revision": {
                "commit": mslim::build_info::SOURCE_COMMIT,
                "tracked_files_modified_at_build": false
            },
            "bundle_sha256": digest,
            "calibration_environment": environment,
            "determinism": {"all_passed": true},
            "same_backend": {"all_passed": true},
            "snapshot_contract": {"accelerator_exact_physical_snapshot": true},
            "policy": {"approximation_epsilon": 0.0},
            "frozen_runtime_configuration": {
                "stage_tokens": 64,
                "prefill_batch_tokens": 128,
                "prefill_weight_reuse_tokens": 8,
                "hot_v_tokens": 512,
                "prompt_replay_tokens": 0,
                "shared_prefix": false,
                "deferred_cold_commit": false,
                "exact_hot_fused_attention": true,
                "decode_unroll4": false,
                "decode_fused_kernels": true,
                "prefill_fuse_projections": true
            },
            "cross_backend_diagnostic": {"all_logits_exact": false},
        });
        fs::write(&path, serde_json::to_vec(&receipt).unwrap()).unwrap();
        let canonical = require_passing_validation_matrix_if_accelerated(
            root.path(),
            &digest,
            "accelerator-runtime",
            &environment,
        );
        if mslim::build_info::current_build_is_clean() {
            let canonical = canonical.unwrap();
            assert_eq!(canonical.status, "same-backend-and-kv-integrity-passed");
            let effective = accelerator_runtime_configuration(
                canonical.frozen_runtime_configuration.as_ref().unwrap(),
                32_768,
                "resident",
            )
            .unwrap();
            assert_eq!(effective["max_context_tokens"], 32_768);
            assert_eq!(effective["prefill_weight_reuse_tokens"], 8);
        } else {
            assert!(canonical.is_err());
        }

        receipt["source_revision"]["tracked_files_modified_at_build"] = json!(true);
        fs::write(&path, serde_json::to_vec(&receipt).unwrap()).unwrap();
        assert!(
            require_passing_validation_matrix_if_accelerated(
                root.path(),
                &digest,
                "accelerator-runtime",
                &environment
            )
            .is_err()
        );
        receipt["source_revision"]["tracked_files_modified_at_build"] = json!(false);
        receipt["same_backend"]["all_passed"] = json!(false);
        fs::write(&path, serde_json::to_vec(&receipt).unwrap()).unwrap();
        assert!(
            require_passing_validation_matrix_if_accelerated(
                root.path(),
                &digest,
                "accelerator-runtime",
                &environment
            )
            .is_err()
        );
    }
}
