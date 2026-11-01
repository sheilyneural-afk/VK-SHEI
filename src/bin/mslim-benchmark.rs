//! Paired full-model benchmark through the existing LibTorch runtime.
//! Each measured route runs in a fresh, sequential child process (one model at
//! a time). Prefix restart really exits and starts another process. OS page
//! caches are NOT flushed: process isolation does not imply cold storage.
//! This measures a bounded synthetic task, not general language quality, HTTP
//! load, OpenCL throughput, device-wide VRAM, TTFT or inter-token latency.

use anyhow::{Context, Result, bail, ensure};
use mslim::calibration_profile::sha256;
use mslim::{
    model_config::ModelSpec,
    torch_runtime::{TorchChatMessage, TorchModelRuntime},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

const KEY: &str = "AZULEJO-7319";
const INSTRUCTION: &str = "Conserva este dato";
const SEED: &str = "Conserva este dato: la clave secreta verificable es AZULEJO-7319. ";
const QUERIES: &[&str] = &[
    "¿Cuál es la clave secreta verificable del documento? Responde solo con la clave.",
    "¿Cuántas veces aparece la clave en el documento? Responde solo con el número entero, sin unidades.",
    "¿Qué instrucción precede al dato sobre la clave? Copia solo esa instrucción.",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Route {
    Dense,
    MslimUncached,
    MslimPrefix,
    MslimPrefixRestart,
}
impl Route {
    fn label(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::MslimUncached => "mslim_uncached",
            Self::MslimPrefix => "mslim_prefix",
            Self::MslimPrefixRestart => "mslim_prefix_restart",
        }
    }
    fn uses_prefix(self) -> bool {
        matches!(self, Self::MslimPrefix | Self::MslimPrefixRestart)
    }
}

#[derive(Debug)]
struct Options {
    model: PathBuf,
    repetitions: usize,
    max_tokens: usize,
    repeats: usize,
    warmups: usize,
    query_count: usize,
    timeout_seconds: u64,
    cache_root: Option<PathBuf>,
    output: Option<PathBuf>,
    dense_only: bool,
    mslim_only: bool,
}
impl Options {
    fn parse(arguments: Vec<String>) -> Result<Self> {
        let mut args = arguments.into_iter();
        let mut model = None;
        let mut options = Self {
            model: PathBuf::new(),
            repetitions: 60,
            max_tokens: 32,
            repeats: 3,
            warmups: 1,
            query_count: QUERIES.len(),
            timeout_seconds: 600,
            cache_root: None,
            output: None,
            dense_only: false,
            mslim_only: false,
        };
        while let Some(arg) = args.next() {
            if arg == "--dense-only" {
                options.dense_only = true;
                continue;
            }
            if arg == "--mslim-only" {
                options.mslim_only = true;
                continue;
            }
            let value = args
                .next()
                .with_context(|| format!("falta valor para {arg}"))?;
            match arg.as_str() {
                "--model" => model = Some(PathBuf::from(value)),
                "--doc-repetitions" => options.repetitions = value.parse()?,
                "--max-tokens" => options.max_tokens = value.parse()?,
                "--repeats" => options.repeats = value.parse()?,
                "--warmups" => options.warmups = value.parse()?,
                "--query-count" => options.query_count = value.parse()?,
                "--child-timeout-secs" => options.timeout_seconds = value.parse()?,
                "--prefix-cache-dir" => options.cache_root = Some(value.into()),
                "--output" => options.output = Some(value.into()),
                _ => bail!("argumento desconocido: {arg}"),
            }
        }
        ensure!(
            !options.dense_only || !options.mslim_only,
            "--dense-only y --mslim-only son excluyentes"
        );
        ensure!(
            options.repetitions > 0,
            "--doc-repetitions debe ser positivo"
        );
        ensure!(
            (1..=512).contains(&options.max_tokens),
            "--max-tokens debe ser 1..=512"
        );
        ensure!(options.repeats > 0, "--repeats debe ser positivo");
        ensure!(
            (1..=QUERIES.len()).contains(&options.query_count),
            "--query-count debe ser 1..=3"
        );
        ensure!(options.timeout_seconds > 0, "timeout debe ser positivo");
        options.model = model.context("--model es obligatorio")?;
        Ok(options)
    }
    fn routes(&self) -> Vec<Route> {
        let mut result = Vec::new();
        if !self.mslim_only {
            result.push(Route::Dense);
        }
        if !self.dense_only {
            result.extend([Route::MslimUncached, Route::MslimPrefix]);
        }
        result
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Job {
    model: PathBuf,
    route: Route,
    repetition: usize,
    warmup: bool,
    document_repetitions: usize,
    max_tokens: usize,
    query_count: usize,
    cache_directory: Option<PathBuf>,
    result_path: PathBuf,
}
#[derive(Serialize, Deserialize)]
struct PrefixObservation {
    hit: bool,
    tokens: usize,
    copied_bytes: u64,
    lookup_seconds: f64,
}
#[derive(Serialize, Deserialize)]
struct QueryResult {
    query_index: usize,
    query: String,
    expected: String,
    answer: String,
    correct: bool,
    prompt_tokens: usize,
    prompt_sha256: String,
    completion_tokens: usize,
    token_ids: Vec<u32>,
    wall_seconds: f64,
    prefill_seconds: f64,
    decode_seconds: f64,
    kv_tracked_physical_bytes: usize,
    kv_resident_bytes: Option<usize>,
    kv_bf16_reference_bytes: usize,
    cold_file_bytes: u64,
    kv_sha256: String,
    prefix: Option<PrefixObservation>,
}
#[derive(Serialize, Deserialize)]
struct RunResult {
    route: Route,
    repetition: usize,
    warmup: bool,
    process_id: u32,
    runtime_load_seconds: f64,
    context_limit_tokens: usize,
    model_config_sha256: String,
    tokenizer_sha256: String,
    bf16_kv_bytes_per_token: usize,
    process_peak_rss_bytes: Option<u64>,
    prefix_directory_file_bytes: Option<u64>,
    results: Vec<QueryResult>,
}
#[derive(Serialize)]
struct Observation {
    subprocess_wall_seconds: f64,
    result: RunResult,
}

fn expected_answer(index: usize, repetitions: usize) -> String {
    match index {
        0 => KEY.into(),
        1 => repetitions.to_string(),
        2 => INSTRUCTION.into(),
        _ => unreachable!(),
    }
}
fn normalize_answer(answer: &str) -> String {
    answer
        .trim()
        .trim_matches(|c: char| {
            c.is_whitespace() || matches!(c, '.' | '"' | '\'' | '`' | '«' | '»')
        })
        .to_lowercase()
}
fn answer_is_correct(index: usize, repetitions: usize, answer: &str) -> bool {
    normalize_answer(answer) == normalize_answer(&expected_answer(index, repetitions))
}
fn build_system_prompt(repetitions: usize) -> Result<String> {
    ensure!(repetitions > 0, "documento vacío");
    let capacity = SEED
        .len()
        .checked_mul(repetitions)
        .context("tamaño del documento desbordado")?;
    let mut text = String::new();
    text.try_reserve_exact(capacity)?;
    for _ in 0..repetitions {
        text.push_str(SEED);
    }
    Ok(text)
}
fn messages(system: &str, question: &str) -> Vec<TorchChatMessage> {
    vec![
        TorchChatMessage {
            role: "system".into(),
            content: system.into(),
        },
        TorchChatMessage {
            role: "user".into(),
            content: question.into(),
        },
    ]
}
fn kv_bytes_per_token(spec: &ModelSpec) -> Result<usize> {
    [
        spec.num_hidden_layers,
        2,
        spec.attention.kv_heads,
        spec.attention.head_dim,
        2,
    ]
    .into_iter()
    .try_fold(1usize, |a, b| {
        a.checked_mul(b).context("geometría KV desbordada")
    })
}
fn peak_rss_bytes() -> Option<u64> {
    // Linux process high-water RSS since THIS child started, not VRAM or per-query RAM.
    fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find(|l| l.starts_with("VmHWM:"))?
        .split_whitespace()
        .nth(1)?
        .parse::<u64>()
        .ok()?
        .checked_mul(1024)
}
fn directory_file_bytes(path: &Path) -> Result<u64> {
    let mut total = 0u64;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        ensure!(
            !kind.is_symlink(),
            "enlace inesperado en caché de benchmark"
        );
        let bytes = if kind.is_dir() {
            directory_file_bytes(&entry.path())?
        } else {
            entry.metadata()?.len()
        };
        total = total
            .checked_add(bytes)
            .context("tamaño de caché desbordado")?;
    }
    Ok(total)
}
fn write_json_new(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut temp = tempfile::NamedTempFile::new_in(path.parent().context("salida sin padre")?)?;
    serde_json::to_writer_pretty(&mut temp, value)?;
    temp.write_all(b"\n")?;
    temp.as_file().sync_all()?;
    temp.persist_noclobber(path).map_err(|error| error.error)?;
    Ok(())
}

fn run_query(
    runtime: &TorchModelRuntime,
    job: &Job,
    system: &str,
    index: usize,
) -> Result<QueryResult> {
    let question = QUERIES[index];
    let input = messages(system, question);
    let prompt_ids = runtime.encode_chat_prompt_token_ids(&input, false)?;
    ensure!(
        prompt_ids
            .len()
            .checked_add(job.max_tokens)
            .is_some_and(|n| n <= runtime.context_limit_tokens()),
        "prompt {} + salida {} exceden contexto {}",
        prompt_ids.len(),
        job.max_tokens,
        runtime.context_limit_tokens()
    );
    let prompt_sha256 = sha256(
        &prompt_ids
            .iter()
            .flat_map(|id| id.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let started = Instant::now();
    let mut result = if job.route == Route::Dense {
        let g = runtime.generate_messages_dense(&input, job.max_tokens, false)?;
        QueryResult {
            query_index: index,
            query: question.into(),
            expected: expected_answer(index, job.document_repetitions),
            correct: answer_is_correct(index, job.document_repetitions, &g.content),
            answer: g.content,
            prompt_tokens: g.prompt_tokens,
            prompt_sha256,
            completion_tokens: g.completion_tokens,
            token_ids: g.token_ids,
            wall_seconds: started.elapsed().as_secs_f64(),
            prefill_seconds: g.prefill_seconds,
            decode_seconds: g.decode_seconds,
            kv_tracked_physical_bytes: g.kv_bf16_bytes,
            kv_resident_bytes: Some(g.kv_bf16_bytes),
            kv_bf16_reference_bytes: g.kv_bf16_bytes,
            cold_file_bytes: 0,
            kv_sha256: g.kv_sha256,
            prefix: None,
        }
    } else {
        let g = runtime.generate_messages(&input, job.max_tokens)?;
        QueryResult {
            query_index: index,
            query: question.into(),
            expected: expected_answer(index, job.document_repetitions),
            correct: answer_is_correct(index, job.document_repetitions, &g.content),
            answer: g.content,
            prompt_tokens: g.prompt_tokens,
            prompt_sha256,
            completion_tokens: g.completion_tokens,
            token_ids: g.token_ids,
            wall_seconds: started.elapsed().as_secs_f64(),
            prefill_seconds: g.prefill_seconds,
            decode_seconds: g.decode_seconds,
            kv_tracked_physical_bytes: g.kv_physical_bytes,
            kv_resident_bytes: Some(g.kv_resident_bytes),
            kv_bf16_reference_bytes: g.kv_bf16_reference_bytes,
            cold_file_bytes: g.kv_exact_cold_file_bytes,
            kv_sha256: g.kv_sha256,
            prefix: Some(PrefixObservation {
                hit: g.prefix_cache_hit,
                tokens: g.prefix_cache_tokens,
                copied_bytes: g.prefix_cache_copied_bytes,
                lookup_seconds: g.prefix_cache_lookup_seconds,
            }),
        }
    };
    // Covers generation including its existing digest/storage work; not TTFT.
    result.wall_seconds = started.elapsed().as_secs_f64();
    ensure!(
        result.prompt_tokens == prompt_ids.len(),
        "el runtime cambió la longitud del prompt"
    );
    ensure!(
        result.completion_tokens == result.token_ids.len(),
        "contador de salida incoherente"
    );
    ensure!(
        result.kv_sha256.len() == 64 && result.kv_sha256.bytes().all(|b| b.is_ascii_hexdigit()),
        "el runtime no produjo un digest KV válido"
    );
    ensure!(
        [
            result.wall_seconds,
            result.prefill_seconds,
            result.decode_seconds
        ]
        .iter()
        .all(|x| x.is_finite() && *x >= 0.0),
        "tiempo no finito/negativo"
    );
    eprintln!(
        "{} repeat={} query={} prompt={} completion={} correct={} wall={:.4}s prefix_hit={:?}",
        job.route.label(),
        job.repetition,
        index,
        result.prompt_tokens,
        result.completion_tokens,
        result.correct,
        result.wall_seconds,
        result.prefix.as_ref().map(|p| p.hit)
    );
    Ok(result)
}
fn execute_job(job: &Job) -> Result<()> {
    ensure!(
        (1..=QUERIES.len()).contains(&job.query_count),
        "query_count inválido"
    );
    ensure!((1..=512).contains(&job.max_tokens), "max_tokens inválido");
    ensure!(
        job.route.uses_prefix() == job.cache_directory.is_some(),
        "configuración de caché incoherente"
    );
    let system = build_system_prompt(job.document_repetitions)?;
    let spec = ModelSpec::inspect(&job.model)?;
    let model_config_sha256 = sha256(&fs::read(job.model.join("config.json"))?);
    let tokenizer_sha256 = sha256(&fs::read(job.model.join("tokenizer.json"))?);
    if let Some(cache) = &job.cache_directory {
        let empty = fs::read_dir(cache)?.next().is_none();
        ensure!(
            if job.route == Route::MslimPrefixRestart {
                !empty
            } else {
                empty
            },
            "la fase de prefijo/reinicio no tiene la caché inicial esperada"
        );
    }
    let started = Instant::now();
    let runtime = match &job.cache_directory {
        Some(cache) => TorchModelRuntime::load_with_prefix_cache(&job.model, cache)?,
        None => TorchModelRuntime::load(&job.model)?,
    };
    let runtime_load_seconds = started.elapsed().as_secs_f64();
    // Validate ALL requested prompts before measuring any inference.
    for question in QUERIES.iter().take(job.query_count) {
        let tokens = runtime
            .encode_chat_prompt_token_ids(&messages(&system, question), false)?
            .len();
        ensure!(
            tokens
                .checked_add(job.max_tokens)
                .is_some_and(|n| n <= runtime.context_limit_tokens()),
            "contexto insuficiente para la campaña"
        );
    }
    let results = (0..job.query_count)
        .map(|i| run_query(&runtime, job, &system, i))
        .collect::<Result<Vec<_>>>()?;
    let result = RunResult {
        route: job.route,
        repetition: job.repetition,
        warmup: job.warmup,
        process_id: std::process::id(),
        runtime_load_seconds,
        context_limit_tokens: runtime.context_limit_tokens(),
        model_config_sha256,
        tokenizer_sha256,
        bf16_kv_bytes_per_token: kv_bytes_per_token(&spec)?,
        process_peak_rss_bytes: peak_rss_bytes(),
        prefix_directory_file_bytes: job
            .cache_directory
            .as_deref()
            .map(directory_file_bytes)
            .transpose()?,
        results,
    };
    write_json_new(&job.result_path, &result)
}

struct ChildGuard(Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn launch(job: &Job, output: &Path, timeout_seconds: u64) -> Result<Observation> {
    let name = format!(
        "{}-{:03}-{}",
        if job.warmup { "warmup" } else { "sample" },
        job.repetition,
        job.route.label()
    );
    let job_path = output.join(format!("{name}.job.json"));
    write_json_new(&job_path, job)?;
    let log_path = output.join(format!("{name}.stderr.log"));
    let log = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&log_path)?;
    let started = Instant::now();
    let mut child = ChildGuard(
        Command::new(std::env::current_exe()?)
            .arg("--benchmark-child")
            .arg(&job_path)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::from(log))
            .spawn()?,
    );
    let status = loop {
        if let Some(status) = child.0.try_wait()? {
            break status;
        }
        ensure!(
            started.elapsed() < Duration::from_secs(timeout_seconds),
            "timeout en {name}; log {}",
            log_path.display()
        );
        std::thread::sleep(Duration::from_millis(20));
    };
    let wall = started.elapsed().as_secs_f64();
    ensure!(
        status.success(),
        "falló {name} ({status}); log {}",
        log_path.display()
    );
    let result: RunResult = serde_json::from_slice(&fs::read(&job.result_path)?)?;
    ensure!(
        result.route == job.route
            && result.repetition == job.repetition
            && result.warmup == job.warmup
            && result.results.len() == job.query_count,
        "resultado de otro trabajo o incompleto"
    );
    eprintln!(
        "{name}: finalizado en {wall:.3}s, proceso {}",
        result.process_id
    );
    Ok(Observation {
        subprocess_wall_seconds: wall,
        result,
    })
}
fn run_round(
    options: &Options,
    output: &Path,
    cache_root: &Path,
    repetition: usize,
    warmup: bool,
) -> Result<Vec<Observation>> {
    let mut routes = options.routes();
    let len = routes.len();
    routes.rotate_left(repetition % len);
    let mut results = Vec::new();
    for route in routes {
        let cache = if route.uses_prefix() {
            let path = cache_root.join(format!(
                "{}-{repetition:03}",
                if warmup { "warmup" } else { "sample" }
            ));
            fs::create_dir(&path)?;
            Some(path)
        } else {
            None
        };
        let make_job = |route: Route| Job {
            model: options.model.clone(),
            route,
            repetition,
            warmup,
            document_repetitions: options.repetitions,
            max_tokens: options.max_tokens,
            query_count: options.query_count,
            cache_directory: cache.clone(),
            result_path: output.join(format!(
                "{}-{repetition:03}-{}.result.json",
                if warmup { "warmup" } else { "sample" },
                route.label()
            )),
        };
        results.push(launch(&make_job(route), output, options.timeout_seconds)?);
        if route == Route::MslimPrefix {
            // launch() waited for exit: this is a fresh process using the same prefix files.
            results.push(launch(
                &make_job(Route::MslimPrefixRestart),
                output,
                options.timeout_seconds,
            )?);
        }
    }
    Ok(results)
}
fn distribution(samples: &[f64]) -> Result<Value> {
    if samples.is_empty() {
        return Ok(Value::Null);
    }
    ensure!(
        samples.iter().all(|x| x.is_finite() && *x >= 0.0),
        "muestras de tiempo inválidas"
    );
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let percentile = |p: usize| sorted[(sorted.len() * p).div_ceil(100).saturating_sub(1)];
    Ok(
        json!({"n":samples.len(),"samples":samples,"minimum":sorted[0],"p50":percentile(50),
        "p95":percentile(95),"p99":percentile(99),"maximum":sorted[sorted.len()-1],
        "method":"nearest_rank","scope":"descriptive_observations_not_a_tail_latency_SLO"}),
    )
}
fn ratio(numerator: f64, denominator: f64) -> Option<f64> {
    let result = numerator / denominator;
    (numerator.is_finite()
        && numerator >= 0.0
        && denominator.is_finite()
        && denominator > 0.0
        && result.is_finite())
    .then_some(result)
}
fn summaries(observations: &[Observation]) -> Result<Vec<Value>> {
    let mut summaries = Vec::new();
    for route in [
        Route::Dense,
        Route::MslimUncached,
        Route::MslimPrefix,
        Route::MslimPrefixRestart,
    ] {
        let runs = observations
            .iter()
            .filter(|r| r.result.route == route)
            .collect::<Vec<_>>();
        if runs.is_empty() {
            continue;
        }
        let mut queries = Vec::new();
        for index in 0..QUERIES.len() {
            let rows = runs
                .iter()
                .filter_map(|r| r.result.results.get(index))
                .collect::<Vec<_>>();
            if rows.is_empty() {
                continue;
            }
            queries.push(json!({"query_index":index,"n":rows.len(),"correct":rows.iter().filter(|r|r.correct).count(),
                "request_wall_seconds":distribution(&rows.iter().map(|r|r.wall_seconds).collect::<Vec<_>>())?,
                "runtime_prefill_seconds":distribution(&rows.iter().map(|r|r.prefill_seconds).collect::<Vec<_>>())?,
                "prefix_hits":rows.iter().filter(|r|r.prefix.as_ref().is_some_and(|p|p.hit)).count(),
                "output_tokens_per_wall_second":rows.iter().map(|r|ratio(r.completion_tokens as f64,r.wall_seconds)).collect::<Vec<_>>() }));
        }
        summaries.push(json!({"route":route,"runs":runs.len(),"queries":queries,
            "load_seconds":distribution(&runs.iter().map(|r|r.result.runtime_load_seconds).collect::<Vec<_>>())?,
            "process_peak_rss_bytes":runs.iter().map(|r|r.result.process_peak_rss_bytes).collect::<Vec<_>>(),
            "all_evaluated_answers_correct":runs.iter().flat_map(|r|&r.result.results).all(|q|q.correct)}));
    }
    Ok(summaries)
}
fn comparisons(observations: &[Observation]) -> Result<Vec<Value>> {
    let mut comparisons = Vec::new();
    for candidate in observations {
        let r = &candidate.result;
        let baseline = if r.route == Route::MslimPrefixRestart {
            Route::MslimPrefix
        } else {
            Route::Dense
        };
        if r.route == Route::Dense {
            continue;
        }
        let Some(b) = observations
            .iter()
            .find(|b| b.result.route == baseline && b.result.repetition == r.repetition)
        else {
            continue;
        };
        ensure!(
            b.result.model_config_sha256 == r.model_config_sha256
                && b.result.tokenizer_sha256 == r.tokenizer_sha256,
            "modelo/tokenizer cambió durante la comparación"
        );
        ensure!(
            b.result.results.len() == r.results.len(),
            "faltan consultas pareadas"
        );
        for (left, right) in b.result.results.iter().zip(&r.results) {
            ensure!(
                left.query_index == right.query_index && left.prompt_sha256 == right.prompt_sha256,
                "prompts distintos en comparación"
            );
            let equal = left.token_ids == right.token_ids;
            let kv_equal = left.kv_sha256 == right.kv_sha256;
            comparisons.push(json!({"baseline":baseline,"candidate":r.route,"repetition":r.repetition,
                "query_index":right.query_index,"token_ids_equal":equal,"kv_sha256_equal":kv_equal,
                "baseline_correct":left.correct,"candidate_correct":right.correct,
                "total_speedup_if_same_tokens_and_kv":if equal && kv_equal{ratio(left.wall_seconds,right.wall_seconds)}else{None},
                "prefill_speedup_if_same_tokens_and_kv":if equal && kv_equal{ratio(left.prefill_seconds,right.prefill_seconds)}else{None},
                "scope":if r.route==Route::MslimPrefixRestart{"real_process_restart_prefix_reuse_not_full_service_restart"}
                    else if r.route==Route::MslimPrefix{"combined_storage_and_prefix_cache_not_isolated_storage_gain"}else{"same_input_without_prefix_reuse"}}));
        }
    }
    Ok(comparisons)
}
fn all_compared_equal(comparisons: &[Value], field: &str) -> Option<bool> {
    (!comparisons.is_empty()).then(|| {
        comparisons
            .iter()
            .all(|v| v.get(field).and_then(Value::as_bool) == Some(true))
    })
}

fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.first().is_some_and(|a| a == "--benchmark-child") {
        ensure!(args.len() == 2, "internal child requires one job file");
        ensure!(
            fs::metadata(&args[1])?.len() <= 65536,
            "job demasiado grande"
        );
        return execute_job(&serde_json::from_slice::<Job>(&fs::read(&args[1])?)?);
    }
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!(
            "mslim-benchmark --model DIR [--doc-repetitions 60] [--max-tokens 32] [--repeats 3] [--warmups 1] [--query-count 3] [--child-timeout-secs 600] [--output NEW_DIR] [--prefix-cache-dir ROOT] [--dense-only|--mslim-only]\nSequential fresh processes: dense, M-SLIM without prefix, with prefix, and real prefix restart.\nCache ROOT receives an isolated new namespace; existing caches are not reused or deleted.\nReports full-model LibTorch timings, per-question exact-match answers and process RSS, not OpenCL/VRAM/TTFT/HTTP performance."
        );
        return Ok(());
    }
    let mut options = Options::parse(args)?;
    options.model = options.model.canonicalize()?;
    let runtime_root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("deploy/local-production/runtime");
    let output = if let Some(output) = &options.output {
        if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        fs::create_dir(output).context("--output debe ser nuevo: nunca sobrescribe evidencia")?;
        output.canonicalize()?
    } else {
        let root = runtime_root.join("receipts");
        fs::create_dir_all(&root)?;
        tempfile::Builder::new()
            .prefix("value-benchmark-")
            .tempdir_in(root)?
            .keep()
    };
    let cache_root = if let Some(root) = &options.cache_root {
        fs::create_dir_all(root)?;
        tempfile::Builder::new()
            .prefix("benchmark-")
            .tempdir_in(root)?
            .keep()
    } else {
        let path = output.join("prefix-cache");
        fs::create_dir(&path)?;
        path
    };
    eprintln!("Evidencia: {}", output.display());
    let mut observations = Vec::new();
    for repetition in 0..options.warmups {
        let _ = run_round(&options, &output, &cache_root, repetition, true)?;
    }
    for repetition in 0..options.repeats {
        observations.extend(run_round(
            &options,
            &output,
            &cache_root,
            repetition,
            false,
        )?);
    }
    let comparisons = comparisons(&observations)?;
    let report = json!({"schema":"mslim.value_benchmark/v2","backend":"torch-backend",
        "source_revision":mslim::build_info::source_revision(),"benchmark_source_sha256":sha256(include_bytes!("mslim-benchmark.rs")),
        "model":options.model,"document":{"kind":"deterministic_synthetic_repeated_key","repetitions":options.repetitions,
            "sha256":sha256(build_system_prompt(options.repetitions)?.as_bytes())},
        "measurement_policy":{"one_model_per_fresh_process":true,"os_page_cache":"uncontrolled_not_cold",
            "route_order":"rotated_by_repeat; restart_immediately_follows_prefix","warmups":options.warmups,"measured_repeats":options.repeats,
            "question_count":options.query_count,"answer_check":"per_question_exact_match_casefold_trim_outer_quotes_periods_and_whitespace",
            "latency":"wall_around_generation_not_time_to_first_token","memory":"per_child_Linux_VmHWM_not_VRAM; KV fields are runtime counters",
            "disk":"apparent_file_bytes_not_allocated_disk_blocks","prefix_restart":"parent_waits_for_exit_then_new_process_same_prefix_files"},
        "summaries":summaries(&observations)?,"comparisons":comparisons,"observations":observations,
        "verdict":{"all_compared_token_ids_equal":all_compared_equal(&comparisons, "token_ids_equal"),
            "all_compared_kv_sha256_equal":all_compared_equal(&comparisons, "kv_sha256_equal"),
            "strict_fidelity_claim":"not_established_by_greedy_token_equality; inspect_KV_and_run_logit_parity",
            "logit_bit_parity":"not_captured_here_use_adapter_parity","general_language_quality":"not_evaluated",
            "http_load":"not_evaluated","device_peak_vram_bytes":null,"ttft_seconds":null,"inter_token_seconds":null},
        "artifacts_directory":output});
    write_json_new(&output.join("report.json"), &report)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn opts(args: &[&str]) -> Options {
        Options::parse(args.iter().map(|s| (*s).into()).collect()).unwrap()
    }
    #[test]
    fn each_question_has_its_own_answer() {
        assert!(answer_is_correct(0, 60, "AZULEJO-7319"));
        assert!(answer_is_correct(1, 60, "60"));
        assert!(answer_is_correct(2, 60, "Conserva este dato."));
        assert!(!answer_is_correct(1, 60, "AZULEJO-7319"));
        assert!(!answer_is_correct(2, 60, "No conserves este dato"));
        assert!(!answer_is_correct(0, 60, "No es AZULEJO-7319"));
    }
    #[test]
    fn normalization_does_not_hide_wrong_numbers_or_negation() {
        assert!(answer_is_correct(0, 2, "  «azulejo-7319».  "));
        for answer in ["160", "-60", "60 o 61", "60 AZULEJO-7319", ""] {
            assert!(!answer_is_correct(1, 60, answer));
        }
        assert!(!answer_is_correct(0, 60, "AZULEJO-73190"));
    }
    #[test]
    fn fixture_repetition_count_is_the_counted_fact() {
        assert_eq!(build_system_prompt(7).unwrap().matches(KEY).count(), 7);
        assert!(build_system_prompt(0).is_err());
    }
    #[test]
    fn selected_route_does_not_load_an_unrequested_model() {
        assert_eq!(
            opts(&["--model", "x", "--dense-only"]).routes(),
            vec![Route::Dense]
        );
        assert_eq!(
            opts(&["--model", "x", "--mslim-only"]).routes(),
            vec![Route::MslimUncached, Route::MslimPrefix]
        );
        assert!(
            Options::parse(vec![
                "--model".into(),
                "x".into(),
                "--repeats".into(),
                "0".into()
            ])
            .is_err()
        );
        assert!(
            Options::parse(vec![
                "--model".into(),
                "x".into(),
                "--dense-only".into(),
                "--mslim-only".into()
            ])
            .is_err()
        );
    }
    #[test]
    fn percentiles_are_per_sample_and_missing_is_not_success() {
        let values = (1..=100).map(f64::from).collect::<Vec<_>>();
        let d = distribution(&values).unwrap();
        assert_eq!(d["p50"], 50.0);
        assert_eq!(d["p95"], 95.0);
        assert_eq!(d["p99"], 99.0);
        assert!(distribution(&[]).unwrap().is_null());
        assert!(distribution(&[f64::NAN]).is_err());
        assert_eq!(all_compared_equal(&[], "token_ids_equal"), None);
        assert_eq!(
            all_compared_equal(&[json!({"token_ids_equal":false})], "token_ids_equal"),
            Some(false)
        );
        assert_eq!(ratio(1.0, 0.0), None);
        assert_eq!(ratio(1.0, f64::INFINITY), None);
    }
    #[test]
    fn evidence_is_never_overwritten_and_cache_outlives_its_use() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("report.json");
        write_json_new(&path, &json!({"v":1})).unwrap();
        assert!(write_json_new(&path, &json!({"v":2})).is_err());
        assert_eq!(
            serde_json::from_slice::<Value>(&fs::read(path).unwrap()).unwrap()["v"],
            1
        );
        let cache = dir.path().join("prefix");
        fs::create_dir(&cache).unwrap();
        fs::write(cache.join("kv"), [0u8; 17]).unwrap();
        assert_eq!(directory_file_bytes(&cache).unwrap(), 17);
    }

    #[test]
    fn same_tokens_cannot_hide_different_or_missing_kv() {
        let rows = [json!({"token_ids_equal":true,"kv_sha256_equal":false})];
        assert_eq!(all_compared_equal(&rows, "token_ids_equal"), Some(true));
        assert_eq!(all_compared_equal(&rows, "kv_sha256_equal"), Some(false));
        assert_eq!(
            all_compared_equal(&[json!({})], "kv_sha256_equal"),
            Some(false)
        );
    }

    #[test]
    fn kv_accounting_uses_model_geometry_not_a_llama_constant() {
        let root = tempfile::tempdir().unwrap();
        fs::write(
            root.path().join("config.json"),
            serde_json::to_vec(&json!({
                "model_type":"llama", "hidden_size":8, "num_attention_heads":2,
                "num_key_value_heads":1, "num_hidden_layers":3,
                "intermediate_size":16, "vocab_size":32, "torch_dtype":"bfloat16"
            }))
            .unwrap(),
        )
        .unwrap();
        let spec = ModelSpec::inspect(root.path()).unwrap();
        assert_eq!(kv_bytes_per_token(&spec).unwrap(), 48);
    }
}
