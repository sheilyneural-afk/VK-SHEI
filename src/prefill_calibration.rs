//! Physical prefill calibration using the existing adapter and transaction.
//! The predictor only changes measurement order. Every candidate is measured;
//! every promotion requires fresh exact logits/KV checks and an explicit flag.

use crate::{
    calibration_predictor::PrefillCostPredictor,
    calibration_profile::{PrefillEnvironment, PrefillProfileParameters, PrefillSettings, sha256},
    calibration_transaction::{
        CalibrationGates, CalibrationObservation, CalibrationProposal, CalibrationState,
        CalibrationTransaction, atomic_write, evaluate_candidate,
    },
    evidence_stats::PairedTimingEvidence,
    gpu_adapter::GpuModelAdapter,
    gpu_llm::GpuLlm,
};
use anyhow::{Context, Result, bail, ensure};
use mslim_adapter_api::ModelAdapter;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::AtomicBool,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

pub const REPORT_SCHEMA: &str = "mslim.prefill_calibration/v1";

#[derive(Debug)]
struct Options {
    model: PathBuf,
    contexts: Vec<usize>,
    candidates: Vec<PrefillSettings>,
    repeats: usize,
    warmups: usize,
    decode_probes: usize,
    max_context: usize,
    settings: Option<PathBuf>,
    history: Option<PathBuf>,
    output: Option<PathBuf>,
    promote: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Fingerprint {
    logits_sha256: String,
    prefill_kv_sha256: String,
    final_kv_sha256: String,
    greedy_tokens: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Sample {
    tokens: usize,
    prefill_micros: f64,
    fingerprint: Fingerprint,
    exact_match: bool,
    maximum_logit_error: f64,
}

struct Reference {
    fingerprint: Fingerprint,
    logits: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Measurement {
    settings: PrefillSettings,
    observation: CalibrationObservation,
    samples: Vec<Sample>,
}

fn parse_options(args: Vec<String>) -> Result<Options> {
    let mut args = args.into_iter().filter(|s| s != "--tune-prefill");
    let model = PathBuf::from(args.next().context("usage: mslim-calibrate MODEL --tune-prefill [--contexts 128,256,512] [--candidates 32:8,64:8,256:8] [--settings JSON] [--history REPORT] [--output NEW_DIRECTORY] [--promote AUTHORITY]")?);
    let mut options = Options {
        model,
        contexts: vec![128, 256, 512],
        candidates: vec![
            PrefillSettings {
                batch_tokens: Some(32),
                weight_reuse_tokens: 8,
            },
            PrefillSettings {
                batch_tokens: Some(64),
                weight_reuse_tokens: 8,
            },
            PrefillSettings {
                batch_tokens: Some(256),
                weight_reuse_tokens: 8,
            },
        ],
        repeats: 3,
        warmups: 1,
        decode_probes: 2,
        max_context: 2048,
        settings: None,
        history: None,
        output: None,
        promote: None,
    };
    while let Some(arg) = args.next() {
        let value = args
            .next()
            .with_context(|| format!("missing value for {arg}"))?;
        match arg.as_str() {
            "--contexts" => {
                options.contexts = value
                    .split(',')
                    .map(str::parse)
                    .collect::<std::result::Result<_, _>>()?
            }
            "--candidates" => {
                options.candidates = value
                    .split(',')
                    .map(|s| {
                        let (batch, reuse) =
                            s.split_once(':').context("candidate must be batch:reuse")?;
                        Ok(PrefillSettings {
                            batch_tokens: Some(batch.parse()?),
                            weight_reuse_tokens: reuse.parse()?,
                        })
                    })
                    .collect::<Result<_>>()?;
            }
            "--repeats" => options.repeats = value.parse()?,
            "--warmups" => options.warmups = value.parse()?,
            "--decode-probes" => options.decode_probes = value.parse()?,
            "--max-context-tokens" => options.max_context = value.parse()?,
            "--settings" => options.settings = Some(value.into()),
            "--history" => options.history = Some(value.into()),
            "--output" => options.output = Some(value.into()),
            "--promote" => {
                ensure!(!value.trim().is_empty(), "promotion authority is empty");
                options.promote = Some(value);
            }
            _ => bail!("unknown prefill calibration option: {arg}"),
        }
    }
    options.contexts.sort_unstable();
    options.contexts.dedup();
    ensure!(
        (3..=16).contains(&options.contexts.len()),
        "prefill learning requires 3..=16 distinct context lengths"
    );
    ensure!(
        (3..=10).contains(&options.repeats) && (1..=3).contains(&options.warmups),
        "repeats must be 3..=10; warmups 1..=3"
    );
    ensure!(
        (2..=8).contains(&options.decode_probes) && (8..=32768).contains(&options.max_context),
        "invalid decode or context budget"
    );
    ensure!(
        options.contexts.iter().all(|&n| n > 0
            && n.checked_add(options.decode_probes)
                .is_some_and(|v| v < options.max_context)),
        "context probes exceed runtime capacity"
    );
    ensure!(
        (1..=16).contains(&options.candidates.len()),
        "candidate budget must be 1..=16"
    );
    let mut seen = std::collections::BTreeSet::new();
    for setting in &options.candidates {
        setting.validate()?;
        ensure!(seen.insert(setting.id()), "duplicate candidate");
    }
    let work: usize = options.contexts.iter().sum::<usize>()
        * (options.repeats + options.warmups)
        * (options.candidates.len() + 4);
    ensure!(work <= 2_000_000, "calibration token budget exceeded");
    Ok(options)
}

fn runtime_config(model: &Path, options: &Options, cold: &Path) -> Result<Value> {
    let mut config: Value = serde_json::from_slice(&fs::read(model.join("config.json"))?)?;
    if let Some(path) = &options.settings {
        ensure!(
            fs::metadata(path)?.len() <= 65536,
            "settings file too large"
        );
        let patch: Value = serde_json::from_slice(&fs::read(path)?)?;
        let allowed = [
            "mslim_weight_mode",
            "mslim_prefill_batch_tokens",
            "mslim_prefill_weight_reuse_tokens",
            "mslim_hot_v_tokens",
            "mslim_prompt_replay_tokens",
            "mslim_shared_prefix",
            "mslim_deferred_cold_commit",
            "mslim_exact_hot_fused_attention",
            "mslim_decode_unroll4",
            "mslim_decode_fused_kernels",
            "mslim_prefill_fuse_projections",
            "mslim_stage_tokens",
        ];
        for (key, value) in patch
            .as_object()
            .context("settings must be a JSON object")?
        {
            ensure!(
                allowed.contains(&key.as_str()),
                "setting outside calibration allowlist: {key}"
            );
            match key.as_str() {
                "mslim_weight_mode" => ensure!(
                    matches!(value.as_str(), Some("resident" | "stream")),
                    "invalid weight mode"
                ),
                "mslim_prefill_batch_tokens"
                | "mslim_prefill_weight_reuse_tokens"
                | "mslim_hot_v_tokens"
                | "mslim_prompt_replay_tokens"
                | "mslim_stage_tokens" => ensure!(
                    value.as_u64().is_some(),
                    "{key} must be a nonnegative integer"
                ),
                _ => ensure!(value.is_boolean(), "{key} must be a boolean"),
            }
            config[key] = value.clone();
        }
    }
    // Calibration owns the tested settings: never let an old attached profile
    // override a candidate while measuring it.
    config
        .as_object_mut()
        .context("model config must be an object")?
        .remove("mslim_prefill_profile");
    config["max_context_tokens"] = json!(options.max_context);
    config["mslim_kv_mode"] = json!("exact-coldstore");
    config["mslim_lod_epsilon"] = json!(0.0);
    config["mslim_cold_directory"] = json!(cold);
    Ok(config)
}

fn peak_rss_bytes() -> Result<u64> {
    let text = fs::read_to_string("/proc/self/status")
        .context("this physical calibrator requires Linux process-RSS telemetry")?;
    let value = text
        .lines()
        .find(|l| l.starts_with("VmHWM:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .context("VmHWM is unavailable")?
        .parse::<u64>()?;
    value.checked_mul(1024).context("RSS overflow")
}

fn prepare_probe(runtime: &mut GpuLlm, scratch: &Path, settings: PrefillSettings) -> Result<()> {
    let old = runtime.active_cold_directory();
    let directory = tempfile::Builder::new()
        .prefix("probe-")
        .tempdir_in(scratch)?
        .keep();
    runtime.reset_session_at(Some(directory))?;
    // Only remove namespaces owned by this isolated calibration, never production data.
    if let Some(old) = old
        && old.starts_with(scratch)
        && old != scratch
    {
        fs::remove_dir_all(old)?;
    }
    runtime.set_calibration_prefill_settings(settings)
}

fn probe(runtime: &mut GpuLlm, prompt: &[u32], decode_probes: usize) -> Result<(f64, Reference)> {
    let started = Instant::now();
    let (step, _) = runtime.prefill_only(prompt, true, &AtomicBool::new(false))?;
    let micros = started.elapsed().as_secs_f64() * 1_000_000.0;
    let mut logits = step.logits.context("prefill logits were not captured")?;
    let mut greedy_tokens = vec![step.argmax_token];
    let prefill_kv_sha256 = runtime.canonical_kv_sha256()?;
    // Teacher-forced continuation makes the comparison use identical inputs even on failure.
    for index in 0..decode_probes {
        let token = prompt[index % prompt.len()];
        let step = runtime.forward_token(token, true)?;
        logits.extend(step.logits.context("decode logits were not captured")?);
        greedy_tokens.push(step.argmax_token);
    }
    ensure!(
        logits.iter().all(|v| v.is_finite()),
        "non-finite measured logits"
    );
    let bytes: Vec<u8> = logits
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect();
    let fingerprint = Fingerprint {
        logits_sha256: sha256(&bytes),
        prefill_kv_sha256,
        final_kv_sha256: runtime.canonical_kv_sha256()?,
        greedy_tokens,
    };
    ensure!(
        micros.is_finite() && micros > 0.0,
        "invalid physical timing"
    );
    Ok((
        micros,
        Reference {
            fingerprint,
            logits,
        },
    ))
}

fn measure(
    runtime: &mut GpuLlm,
    scratch: &Path,
    settings: PrefillSettings,
    prompts: &[Vec<u32>],
    options: &Options,
    workload: &str,
    reference: Option<&[Reference]>,
) -> Result<(Measurement, Vec<Reference>)> {
    let mut own_references = Vec::<Reference>::new();
    let mut samples = Vec::new();
    let mut maximum_error = 0.0f64;
    let mut all_match = true;
    for (context_index, prompt) in prompts.iter().enumerate() {
        for _ in 0..options.warmups {
            prepare_probe(runtime, scratch, settings)?;
            probe(runtime, prompt, options.decode_probes)?;
        }
        for repeat in 0..options.repeats {
            prepare_probe(runtime, scratch, settings)?;
            let (micros, captured) = probe(runtime, prompt, options.decode_probes)?;
            let expected = reference
                .map(|r| &r[context_index])
                .or_else(|| own_references.get(context_index));
            let (exact_match, error) = if let Some(expected) = expected {
                ensure!(
                    captured.logits.len() == expected.logits.len(),
                    "logit dimensions changed"
                );
                let error = captured
                    .logits
                    .iter()
                    .zip(&expected.logits)
                    .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
                    .fold(0.0f64, f64::max);
                (captured.fingerprint == expected.fingerprint, error)
            } else {
                (true, 0.0)
            };
            all_match &= exact_match;
            maximum_error = maximum_error.max(error);
            samples.push(Sample {
                tokens: prompt.len(),
                prefill_micros: micros,
                fingerprint: captured.fingerprint.clone(),
                exact_match,
                maximum_logit_error: error,
            });
            if repeat == 0 {
                own_references.push(captured);
            }
        }
        eprintln!(
            "prefill calibration: {} context={} measured={} exact={}",
            settings.id(),
            prompt.len(),
            options.repeats,
            all_match
        );
    }
    let mut times = samples.iter().map(|s| s.prefill_micros).collect::<Vec<_>>();
    times.sort_by(f64::total_cmp);
    let p95 = times[(times.len() * 95)
        .div_ceil(100)
        .saturating_sub(1)
        .min(times.len() - 1)];
    let total_tokens = samples.iter().map(|s| s.tokens).sum::<usize>();
    let observation = CalibrationObservation {
        profile_id: settings.id(),
        workload_id: workload.into(),
        seed: 0,
        sample_count: samples.len() as u64,
        parity_ok: all_match,
        maximum_exactness_error: maximum_error,
        maximum_certificate_error: runtime
            .gpu_region_stats()
            .map_or(0.0, |s| s.maximum_certified_error_bound),
        throughput_tokens_per_second: total_tokens as f64 * 1_000_000.0 / times.iter().sum::<f64>(),
        latency_p95_ms: p95 / 1000.0,
        peak_memory_bytes: peak_rss_bytes()?,
        bytes_read_per_token: 0,
    };
    observation.validate()?;
    Ok((
        Measurement {
            settings,
            observation,
            samples,
        },
        own_references,
    ))
}

/// Paired timing evidence for one candidate against the baseline. Samples are
/// produced in the same context/repeat order by `measure`, so pairing by index
/// is valid only when the token counts line up; otherwise nothing is claimed.
/// The result is advisory report data and never feeds a gate decision.
fn timing_evidence(baseline: &Measurement, candidate: &Measurement) -> Value {
    let aligned = baseline.samples.len() == candidate.samples.len()
        && baseline
            .samples
            .iter()
            .zip(&candidate.samples)
            .all(|(b, c)| b.tokens == c.tokens);
    if !aligned {
        return json!({"status": "samples_not_paired"});
    }
    let pairs = baseline
        .samples
        .iter()
        .zip(&candidate.samples)
        .map(|(b, c)| (b.prefill_micros, c.prefill_micros))
        .collect::<Vec<_>>();
    match PairedTimingEvidence::from_pairs(&pairs) {
        Ok(evidence) => json!(evidence),
        Err(error) => json!({"status": "invalid_timings", "error": error.to_string()}),
    }
}

fn learn(
    predictor: &mut PrefillCostPredictor,
    environment: &str,
    evidence_prefix: &str,
    measurement: &Measurement,
) -> Result<()> {
    if !measurement.observation.parity_ok {
        return Ok(());
    }
    for (i, sample) in measurement.samples.iter().enumerate() {
        ensure!(
            sample.exact_match && sample.maximum_logit_error == 0.0,
            "inconsistent exactness evidence"
        );
        ensure!(
            sample.prefill_micros.is_finite() && sample.prefill_micros > 0.0,
            "history contains invalid timings"
        );
        if let Err(error) = predictor.observe(
            environment,
            &measurement.settings.id(),
            &format!("{evidence_prefix}/{i}"),
            sample.tokens,
            sample.prefill_micros,
        ) {
            eprintln!(
                "advisory predictor declined an update; physical calibration continues: {error}"
            );
        }
    }
    Ok(())
}

fn write_report(path: &Path, report: &mut Value) -> Result<()> {
    report
        .as_object_mut()
        .context("report must be an object")?
        .remove("report_sha256");
    let digest = sha256(&serde_json::to_vec(report)?);
    report["report_sha256"] = json!(digest);
    atomic_write(path, &serde_json::to_vec_pretty(report)?)?;
    Ok(())
}

fn load_history(
    path: &Path,
    environment: &PrefillEnvironment,
    predictor: &mut PrefillCostPredictor,
) -> Result<String> {
    ensure!(
        fs::metadata(path)?.len() <= 8 * 1024 * 1024,
        "history exceeds 8 MiB budget"
    );
    let mut report: Value = serde_json::from_slice(&fs::read(path)?)?;
    let digest = report
        .as_object_mut()
        .context("invalid history object")?
        .remove("report_sha256")
        .and_then(|v| v.as_str().map(str::to_owned))
        .context("history lacks integrity digest")?;
    ensure!(
        sha256(&serde_json::to_vec(&report)?) == digest,
        "history digest mismatch"
    );
    ensure!(
        report["schema"] == REPORT_SCHEMA,
        "unsupported prefill history schema"
    );
    let prior: PrefillEnvironment = serde_json::from_value(report["environment"].clone())?;
    if &prior != environment {
        return Ok("ignored_incompatible_environment".into());
    }
    let measurements: Vec<Measurement> = serde_json::from_value(report["measurements"].clone())?;
    ensure!(measurements.len() <= 17, "history has too many profiles");
    for (i, m) in measurements.iter().enumerate() {
        m.settings.validate()?;
        m.observation.validate()?;
        ensure!(
            m.observation.profile_id == m.settings.id()
                && m.samples.len() <= 160
                && m.observation.sample_count == m.samples.len() as u64,
            "inconsistent history measurement"
        );
        learn(
            predictor,
            &environment.digest()?,
            &format!("history-{digest}-{i}"),
            m,
        )?;
    }
    Ok("replayed_measured_evidence".into())
}

pub fn run_cli(args: Vec<String>) -> Result<Value> {
    let options = parse_options(args)?;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let model = if options.model.is_absolute() {
        options.model.clone()
    } else {
        root.join(&options.model)
    };
    let runtime_root = root.join("deploy/local-production/runtime");
    fs::create_dir_all(runtime_root.join("tmp"))?;
    let scratch = tempfile::Builder::new()
        .prefix("prefill-calibration-")
        .tempdir_in(runtime_root.join("tmp"))?;
    let cold = scratch.path().join("initial");
    fs::create_dir(&cold)?;
    let config = runtime_config(&model, &options, &cold)?;
    let run_id = format!(
        "{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    );
    let output = options
        .output
        .clone()
        .unwrap_or_else(|| runtime_root.join("calibration/prefill").join(&run_id));
    let output = if output.is_absolute() {
        output
    } else {
        root.join(output)
    };
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(&output).context(
        "calibration output must be a new directory; existing evidence is never overwritten",
    )?;
    let output = output.canonicalize()?;
    let report_path = output.join("report.json");
    eprintln!("prefill calibration evidence: {}", report_path.display());
    let mut adapter = GpuModelAdapter::default();
    adapter.load(&model, &config)?;
    let runtime = adapter.calibration_runtime_mut()?;
    let original = runtime.prefill_settings();
    let environment = runtime.prefill_environment()?;
    let environment_digest = environment.digest()?;
    let candidates = options
        .candidates
        .iter()
        .copied()
        .filter(|s| *s != original)
        .collect::<Vec<_>>();
    ensure!(
        !candidates.is_empty(),
        "all candidates are identical to the active settings"
    );
    for &candidate in &candidates {
        runtime.set_calibration_prefill_settings(candidate)?;
    }
    runtime.set_calibration_prefill_settings(original)?;

    let tokenizer = tokenizers::Tokenizer::from_file(model.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let seed = tokenizer.encode("Analiza este documento y comprueba sus datos antes de responder. Conserva las evidencias originales y diferencia los hechos de las hipótesis. ", true)
        .map_err(|e| anyhow::anyhow!("tokenize calibration prompt: {e}"))?;
    ensure!(!seed.get_ids().is_empty(), "empty calibration seed");
    let prompts = options
        .contexts
        .iter()
        .map(|&n| {
            seed.get_ids()
                .iter()
                .copied()
                .cycle()
                .take(n)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let workload = sha256(&serde_json::to_vec(
        &json!({"prompts": prompts,"decode_probes": options.decode_probes}),
    )?);
    let mut predictor = PrefillCostPredictor::new(environment_digest.clone())?;
    let history_status = if let Some(path) = &options.history {
        load_history(path, &environment, &mut predictor)?
    } else {
        "not_requested".into()
    };
    let names = candidates
        .iter()
        .map(PrefillSettings::id)
        .collect::<Vec<_>>();
    let prediction_tokens = options.contexts[options.contexts.len() / 2];
    let order = predictor.order_candidates(&environment_digest, &names, prediction_tokens);
    let predictions = names.iter().map(|id| json!({"profile": id,"prediction": predictor.predict(&environment_digest,id,prediction_tokens)})).collect::<Vec<_>>();
    let gates = CalibrationGates::default();
    let result = (|| -> Result<Value> {
        let (baseline, reference) = measure(
            runtime,
            scratch.path(),
            original,
            &prompts,
            &options,
            &workload,
            None,
        )?;
        ensure!(
            baseline.observation.parity_ok,
            "baseline is not deterministic; tuning is blocked"
        );
        learn(
            &mut predictor,
            &environment_digest,
            &format!("{run_id}/baseline"),
            &baseline,
        )?;
        let mut measurements = vec![baseline.clone()];
        let mut best: Option<Measurement> = None;
        let mut reports = Vec::new();
        for &index in &order {
            let settings = candidates[index];
            let (measurement, _) = measure(
                runtime,
                scratch.path(),
                settings,
                &prompts,
                &options,
                &workload,
                Some(&reference),
            )?;
            let gate = evaluate_candidate(&baseline.observation, &measurement.observation, &gates)?;
            learn(
                &mut predictor,
                &environment_digest,
                &format!("{run_id}/{}", settings.id()),
                &measurement,
            )?;
            if gate.accepted
                && best.as_ref().is_none_or(|m| {
                    measurement.observation.throughput_tokens_per_second
                        > m.observation.throughput_tokens_per_second
                })
            {
                best = Some(measurement.clone());
            }
            reports.push(json!({
                "profile": settings.id(),
                "gates": gate,
                "timing_evidence": timing_evidence(&baseline, &measurement),
            }));
            measurements.push(measurement);
        }
        let profile = |settings| -> Result<_> {
            PrefillProfileParameters {
                environment: environment.clone(),
                settings,
                validated_context_tokens: options.contexts.clone(),
                prompt_suite_sha256: workload.clone(),
            }
            .into_profile()
        };
        let active_path = output.join("active.json");
        atomic_write(
            &active_path,
            &serde_json::to_vec_pretty(&profile(original)?)?,
        )?;
        let mut report = json!({"schema": REPORT_SCHEMA, "run_id": run_id, "environment": environment,
            "source_revision": crate::build_info::source_revision(),
            "execution_configuration": runtime.prefill_execution_configuration(),
            "runtime_config": config.as_object().map(|o| o.iter().filter(|(k,_)| k.starts_with("mslim_") && !k.contains("directory")).collect::<std::collections::BTreeMap<_,_>>()),
            "workload_sha256": workload, "prompt_token_ids": prompts, "decode_probes": options.decode_probes,
            "measurement_kind": "physical_model_execution_with_deterministic_text_load_probes_not_a_language_quality_benchmark",
            "memory_metric": "Linux_process_VmHWM_cumulative_not_VRAM; no_memory_saving_claim",
            "bytes_read_per_token": "not_measured; zero_in_legacy_observation_field_is_not_a_measurement",
            "history_status": history_status, "predictions_before_measurement": predictions,
            "measurement_order": order.iter().map(|&i| &names[i]).collect::<Vec<_>>(),
            "all_candidates_measured": true, "measurements": measurements, "candidate_gates": reports,
            "predictor": predictor, "status": "no_candidate_passed", "active_profile_path": active_path});
        write_report(&report_path, &mut report)?;
        if let Some(best) = best {
            let proposal = CalibrationProposal {
                proposal_id: run_id.clone(),
                baseline_profile_digest: sha256(&fs::read(&active_path)?),
                candidate: profile(best.settings)?,
                gates,
            };
            let mut transaction = CalibrationTransaction::new(&output, &active_path, proposal)?;
            transaction.propose()?;
            transaction.begin_shadow()?;
            ensure!(
                transaction
                    .complete_shadow(&baseline.observation, &best.observation)?
                    .accepted,
                "winning profile failed shadow gate"
            );
            report["status"] = json!("shadow_passed_not_activated");
            write_report(&report_path, &mut report)?;
            if let Some(authority) = &options.promote {
                transaction.begin_canary()?;
                let canary = (|| -> Result<_> {
                    // Fresh measurements, not the shadow samples reused as canary evidence.
                    let (fresh_baseline, _) = measure(
                        runtime,
                        scratch.path(),
                        original,
                        &prompts,
                        &options,
                        &workload,
                        Some(&reference),
                    )?;
                    ensure!(
                        fresh_baseline.observation.parity_ok,
                        "baseline drifted during canary"
                    );
                    let (fresh_candidate, _) = measure(
                        runtime,
                        scratch.path(),
                        best.settings,
                        &prompts,
                        &options,
                        &workload,
                        Some(&reference),
                    )?;
                    let gate = transaction.complete_canary(
                        &fresh_baseline.observation,
                        &fresh_candidate.observation,
                    )?;
                    Ok((fresh_baseline, fresh_candidate, gate))
                })();
                match canary {
                    Ok((fresh_baseline, fresh_candidate, gate)) => {
                        report["canary"] = json!({
                            "timing_evidence": timing_evidence(&fresh_baseline, &fresh_candidate),
                            "baseline": fresh_baseline,
                            "candidate": fresh_candidate,
                            "gates": gate,
                        });
                        if gate.accepted {
                            transaction.promote(authority)?;
                            report["status"] = json!("promoted");
                        } else {
                            report["status"] = json!("canary_failed_rolled_back");
                        }
                    }
                    Err(error) => {
                        if transaction.state() == CalibrationState::Canary {
                            transaction.rollback("physical_canary_execution_failed")?;
                        }
                        report["status"] = json!("canary_error_rolled_back");
                        report["error"] = json!(format!("{error:#}"));
                    }
                }
            }
            drop(transaction);
        }
        prepare_probe(runtime, scratch.path(), original)?;
        ensure!(
            runtime.prefill_settings() == original,
            "physical runtime restoration failed"
        );
        report["runtime_restored_to_original"] = json!(true);
        write_report(&report_path, &mut report)?;
        Ok(report)
    })();
    // This runtime is isolated, never a production worker. Even error paths restore
    // settings when possible and then drop it; no live session is hot-reconfigured.
    if result.is_err() {
        let restored = prepare_probe(runtime, scratch.path(), original).is_ok();
        let mut failure = json!({"schema":REPORT_SCHEMA,"status":"failed","runtime_restored_to_original":restored,
            "error":result.as_ref().err().map(|e|format!("{e:#}")),"environment":environment});
        write_report(&output.join("failure.json"), &mut failure)?;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    fn args(extra: &[&str]) -> Vec<String> {
        ["model", "--tune-prefill"]
            .into_iter()
            .chain(extra.iter().copied())
            .map(str::to_owned)
            .collect()
    }
    #[test]
    fn rejects_insufficient_contexts_and_mutating_flags() {
        assert!(parse_options(args(&["--contexts", "16,32"])).is_err());
        assert!(parse_options(args(&["--lod-epsilon", "0.1"])).is_err());
        assert!(parse_options(args(&["--repeats", "1"])).is_err());
        assert!(parse_options(args(&["--candidates", "128:7"])).is_err());
    }
    #[test]
    fn promotion_is_explicit_and_candidates_are_bounded() {
        let options = parse_options(args(&[
            "--contexts",
            "16,32,64",
            "--candidates",
            "64:8,128:16",
        ]))
        .unwrap();
        assert!(options.promote.is_none());
        assert_eq!(options.candidates.len(), 2);
        assert!(parse_options(args(&["--candidates", "64:8,64:8"])).is_err());
    }
    fn measurement(timings: &[(usize, f64)]) -> Measurement {
        let fingerprint = Fingerprint {
            logits_sha256: String::new(),
            prefill_kv_sha256: String::new(),
            final_kv_sha256: String::new(),
            greedy_tokens: Vec::new(),
        };
        Measurement {
            settings: PrefillSettings {
                batch_tokens: Some(32),
                weight_reuse_tokens: 8,
            },
            observation: CalibrationObservation {
                profile_id: "test".into(),
                workload_id: "test".into(),
                seed: 0,
                sample_count: timings.len() as u64,
                parity_ok: true,
                maximum_exactness_error: 0.0,
                maximum_certificate_error: 0.0,
                throughput_tokens_per_second: 1.0,
                latency_p95_ms: 1.0,
                peak_memory_bytes: 1,
                bytes_read_per_token: 0,
            },
            samples: timings
                .iter()
                .map(|&(tokens, prefill_micros)| Sample {
                    tokens,
                    prefill_micros,
                    fingerprint: fingerprint.clone(),
                    exact_match: true,
                    maximum_logit_error: 0.0,
                })
                .collect(),
        }
    }
    #[test]
    fn timing_evidence_pairs_only_aligned_samples_and_is_advisory() {
        let baseline = measurement(&[(128, 1000.0), (128, 1010.0), (256, 2000.0), (256, 1990.0)]);
        let faster = measurement(&[(128, 800.0), (128, 805.0), (256, 1600.0), (256, 1590.0)]);
        let evidence = timing_evidence(&baseline, &faster);
        assert_eq!(evidence["claim"], "candidate_faster_with_confidence");
        assert_eq!(
            evidence["authority"],
            "advisory_never_changes_a_gate_decision"
        );
        assert_eq!(evidence["pairs"], 4);

        let misaligned = measurement(&[(128, 800.0), (64, 805.0), (256, 1600.0), (256, 1590.0)]);
        assert_eq!(
            timing_evidence(&baseline, &misaligned)["status"],
            "samples_not_paired"
        );
        let short = measurement(&[(128, 800.0)]);
        assert_eq!(
            timing_evidence(&baseline, &short)["status"],
            "samples_not_paired"
        );
        let invalid = measurement(&[(128, 0.0), (128, 805.0), (256, 1600.0), (256, 1590.0)]);
        assert_eq!(
            timing_evidence(&baseline, &invalid)["status"],
            "invalid_timings"
        );
    }
    #[test]
    fn report_integrity_detects_edited_measurements() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("report.json");
        let mut value = json!({"schema":REPORT_SCHEMA,"status":"test-only"});
        write_report(&path, &mut value).unwrap();
        let hash = value
            .as_object_mut()
            .unwrap()
            .remove("report_sha256")
            .unwrap();
        assert_eq!(
            hash.as_str().unwrap(),
            sha256(&serde_json::to_vec(&value).unwrap())
        );
        value["status"] = json!("forged");
        assert_ne!(
            hash.as_str().unwrap(),
            sha256(&serde_json::to_vec(&value).unwrap())
        );
    }
    #[test]
    fn settings_cannot_silently_fall_back_on_wrong_types() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("config.json"), b"{}").unwrap();
        let settings = dir.path().join("settings.json");
        let mut options = parse_options(args(&[])).unwrap();
        options.settings = Some(settings.clone());
        for bad in [
            json!({"mslim_prefill_batch_tokens": "128"}),
            json!({"mslim_hot_v_tokens": -1}),
            json!({"mslim_shared_prefix": 1}),
            json!({"mslim_lod_epsilon": 0.1}),
            json!({"mslim_weight_mode": "q4"}),
        ] {
            fs::write(&settings, serde_json::to_vec(&bad).unwrap()).unwrap();
            assert!(runtime_config(dir.path(), &options, dir.path()).is_err());
        }
    }
    #[test]
    fn calibration_does_not_load_an_old_profile_over_its_candidates() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("config.json"),
            br#"{"mslim_prefill_profile":"old.json"}"#,
        )
        .unwrap();
        let options = parse_options(args(&[])).unwrap();
        let config = runtime_config(dir.path(), &options, dir.path()).unwrap();
        assert!(config.get("mslim_prefill_profile").is_none());
        assert_eq!(config["mslim_lod_epsilon"], json!(0.0));
    }
}
