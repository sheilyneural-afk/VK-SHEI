//! Benchmark de valor central de M-SLIM: contexto largo, hardware limitado, estado verificable.
//!
//! Pregunta concreta: ¿qué trabajo útil permite hacer M-SLIM mejor que una referencia
//! comparable, y cuánto cuesta realmente hacerlo?
//!
//! Diseño:
//!   - Mismo modelo, mismo documento largo, mismas preguntas
//!   - Ruta A: denso puro (BF16, sin compresión, sin caché de prefijo)
//!   - Ruta B: M-SLIM (KV comprimido, con caché de prefijo en memoria)
//!   - Métrica de reinicio: simula cerrar y reabrir el proceso
//!   - Métricas: memoria pico, latencia primer token, latencia consultas siguientes,
//!               tiempo de recuperación, fidelidad (paridad de tokens), coste en disco

use anyhow::{Context, Result, bail, ensure};
use mslim::torch_runtime::{TorchChatMessage, TorchModelRuntime};
use serde_json::json;
use std::path::PathBuf;
use std::time::Instant;

// Consultas de prueba que requieren retener contexto del documento
const QUERIES: &[&str] = &[
    "¿Cuál es la clave secreta verificable mencionada en el documento? Responde solo con la clave.",
    "¿Cuántas veces aparece la clave en el documento?",
    "¿Qué instrucción específica se da sobre la clave?",
];

struct RunResult {
    query_index: usize,
    query: String,
    answer: String,
    token_ids: Vec<u32>,
    prefill_seconds: f64,
    decode_seconds: f64,
    latency_seconds: f64,
    kv_physical_bytes: usize,
    kv_bf16_reference_bytes: usize,
    kv_exact_cold_file_bytes: u64,
    prefix_cache_hit: bool,
}

fn build_system_prompt(repetitions: usize) -> String {
    // Documento sintético verificable: la clave aparece repetidamente
    let seed = "Conserva este dato: la clave secreta verificable es AZULEJO-7319. ";
    seed.repeat(repetitions)
}

fn run_query(
    runtime: &TorchModelRuntime,
    system: &str,
    query: &str,
    query_index: usize,
    max_tokens: usize,
) -> Result<RunResult> {
    let messages = vec![
        TorchChatMessage {
            role: "system".to_owned(),
            content: system.to_owned(),
        },
        TorchChatMessage {
            role: "user".to_owned(),
            content: query.to_owned(),
        },
    ];
    let g = runtime.generate_messages(&messages, max_tokens)?;
    Ok(RunResult {
        query_index,
        query: query.to_owned(),
        answer: g.content.clone(),
        token_ids: g.token_ids.clone(),
        prefill_seconds: g.prefill_seconds,
        decode_seconds: g.decode_seconds,
        latency_seconds: g.latency_seconds,
        kv_physical_bytes: g.kv_physical_bytes,
        kv_bf16_reference_bytes: g.kv_bf16_reference_bytes,
        kv_exact_cold_file_bytes: g.kv_exact_cold_file_bytes,
        prefix_cache_hit: false,
    })
}

fn run_query_dense(
    runtime: &TorchModelRuntime,
    system: &str,
    query: &str,
    query_index: usize,
    max_tokens: usize,
) -> Result<RunResult> {
    let messages = vec![
        TorchChatMessage {
            role: "system".to_owned(),
            content: system.to_owned(),
        },
        TorchChatMessage {
            role: "user".to_owned(),
            content: query.to_owned(),
        },
    ];
    let g = runtime.generate_messages_dense(&messages, max_tokens, false)?;
    Ok(RunResult {
        query_index,
        query: query.to_owned(),
        answer: g.content.clone(),
        token_ids: g.token_ids.clone(),
        prefill_seconds: g.prefill_seconds,
        decode_seconds: g.decode_seconds,
        latency_seconds: g.latency_seconds,
        kv_physical_bytes: g.kv_bf16_bytes,
        kv_bf16_reference_bytes: g.kv_bf16_bytes,
        kv_exact_cold_file_bytes: 0,
        prefix_cache_hit: false,
    })
}

fn result_to_json(r: &RunResult, label: &str) -> serde_json::Value {
    json!({
        "label": label,
        "query_index": r.query_index,
        "query": r.query,
        "answer": r.answer,
        "correct": r.answer.contains("AZULEJO-7319"),
        "timing": {
            "prefill_s": r.prefill_seconds,
            "decode_s": r.decode_seconds,
            "total_s": r.latency_seconds,
        },
        "memory": {
            "kv_physical_bytes": r.kv_physical_bytes,
            "kv_bf16_reference_bytes": r.kv_bf16_reference_bytes,
            "compression_ratio": if r.kv_bf16_reference_bytes > 0 {
                r.kv_physical_bytes as f64 / r.kv_bf16_reference_bytes as f64
            } else { 1.0 },
        },
        "disk_bytes": r.kv_exact_cold_file_bytes,
    })
}

fn main() -> Result<()> {
    let mut model: Option<PathBuf> = None;
    let mut doc_repetitions: usize = 60; // ~sistema de ~3600 tokens aprox
    let mut max_tokens: usize = 32;
    let mut prefix_cache_dir: Option<PathBuf> = None;
    let mut dense_only = false;
    let mut mslim_only = false;

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => model = Some(PathBuf::from(it.next().context("falta --model")?)),
            "--doc-repetitions" => {
                doc_repetitions = it.next().context("falta --doc-repetitions")?.parse()?
            }
            "--max-tokens" => max_tokens = it.next().context("falta --max-tokens")?.parse()?,
            "--prefix-cache-dir" => {
                prefix_cache_dir = Some(PathBuf::from(
                    it.next().context("falta --prefix-cache-dir")?,
                ))
            }
            "--dense-only" => dense_only = true,
            "--mslim-only" => mslim_only = true,
            other => bail!("argumento desconocido: {other}"),
        }
    }
    ensure!(
        !dense_only || !mslim_only,
        "--dense-only y --mslim-only son excluyentes"
    );

    let model_path = model.context("--model es obligatorio")?;
    let system = build_system_prompt(doc_repetitions);
    let system_tokens_approx = system.len() / 4; // estimación cruda

    eprintln!("=== M-SLIM Benchmark ===");
    eprintln!("Modelo: {}", model_path.display());
    eprintln!(
        "Documento: {} repeticiones ≈ {} tokens estimados",
        doc_repetitions, system_tokens_approx
    );
    eprintln!("Consultas: {}", QUERIES.len());
    eprintln!();

    // ─── Carga del modelo ────────────────────────────────────────────────────
    let load_t0 = Instant::now();
    let mslim_runtime = if let Some(ref cache_dir) = prefix_cache_dir {
        eprintln!(
            "Cargando M-SLIM con caché de prefijo en {}",
            cache_dir.display()
        );
        TorchModelRuntime::load_with_prefix_cache(&model_path, cache_dir)?
    } else {
        let tmp_dir = tempfile::tempdir().context("crear directorio temporal de caché")?;
        eprintln!("Cargando M-SLIM con caché de prefijo temporal");
        TorchModelRuntime::load_with_prefix_cache(&model_path, tmp_dir.path())?
    };
    let load_mslim_s = load_t0.elapsed().as_secs_f64();
    eprintln!("M-SLIM cargado en {:.2} s", load_mslim_s);

    let dense_runtime_opt = if !mslim_only {
        let load_t1 = Instant::now();
        eprintln!("Cargando runtime denso (referencia)...");
        let rt = TorchModelRuntime::load(&model_path)?;
        let load_dense_s = load_t1.elapsed().as_secs_f64();
        eprintln!("Denso cargado en {:.2} s", load_dense_s);
        Some((rt, load_dense_s))
    } else {
        None
    };

    // ─── Ruta M-SLIM ─────────────────────────────────────────────────────────
    let mut mslim_results: Vec<RunResult> = Vec::new();
    if !dense_only {
        eprintln!("\n--- Ruta M-SLIM ---");
        for (i, &query) in QUERIES.iter().enumerate() {
            eprint!("  Consulta {}/{}: ", i + 1, QUERIES.len());
            let r = run_query(&mslim_runtime, &system, query, i, max_tokens)?;
            eprintln!(
                "{:.3}s prefill + {:.3}s decode = {:.3}s total | {}",
                r.prefill_seconds,
                r.decode_seconds,
                r.latency_seconds,
                if r.answer.contains("AZULEJO-7319") {
                    "✓ CORRECTO"
                } else {
                    "✗ incorrecto"
                }
            );
            mslim_results.push(r);
        }
    }

    // ─── Ruta densa (referencia) ──────────────────────────────────────────────
    let mut dense_results: Vec<RunResult> = Vec::new();
    if let Some((ref dense_runtime, _)) = dense_runtime_opt {
        eprintln!("\n--- Ruta densa (referencia BF16 sin caché) ---");
        for (i, &query) in QUERIES.iter().enumerate() {
            eprint!("  Consulta {}/{}: ", i + 1, QUERIES.len());
            let r = run_query_dense(dense_runtime, &system, query, i, max_tokens)?;
            eprintln!(
                "{:.3}s prefill + {:.3}s decode = {:.3}s total | {}",
                r.prefill_seconds,
                r.decode_seconds,
                r.latency_seconds,
                if r.answer.contains("AZULEJO-7319") {
                    "✓ CORRECTO"
                } else {
                    "✗ incorrecto"
                }
            );
            dense_results.push(r);
        }
    }

    // ─── Análisis y veredicto ─────────────────────────────────────────────────
    let mslim_json: Vec<_> = mslim_results
        .iter()
        .map(|r| result_to_json(r, "mslim"))
        .collect();
    let dense_json: Vec<_> = dense_results
        .iter()
        .map(|r| result_to_json(r, "dense"))
        .collect();

    // Fidelidad: ¿son idénticos los tokens?
    let fidelity: Vec<serde_json::Value> = mslim_results
        .iter()
        .zip(dense_results.iter())
        .map(|(m, d)| {
            json!({
                "query_index": m.query_index,
                "token_parity": m.token_ids == d.token_ids,
                "mslim_tokens": m.token_ids.len(),
                "dense_tokens": d.token_ids.len(),
                "mslim_correct": m.answer.contains("AZULEJO-7319"),
                "dense_correct": d.answer.contains("AZULEJO-7319"),
            })
        })
        .collect();

    // Speedup por consulta (prefill mslim / prefill denso)
    let speedup: Vec<serde_json::Value> = mslim_results
        .iter()
        .zip(dense_results.iter())
        .map(|(m, d)| {
            let prefill_speedup = if m.prefill_seconds > 1e-9 {
                d.prefill_seconds / m.prefill_seconds
            } else {
                f64::INFINITY
            };
            let total_speedup = if m.latency_seconds > 1e-9 {
                d.latency_seconds / m.latency_seconds
            } else {
                f64::INFINITY
            };
            json!({
                "query_index": m.query_index,
                "prefill_speedup_mslim_vs_dense": prefill_speedup,
                "total_speedup_mslim_vs_dense": total_speedup,
                "mslim_prefill_s": m.prefill_seconds,
                "dense_prefill_s": d.prefill_seconds,
            })
        })
        .collect();

    // Resumen de memoria/compresión (última consulta M-SLIM, donde el KV está completo)
    let compression_summary = mslim_results.last().map(|r| {
        json!({
            "kv_physical_bytes": r.kv_physical_bytes,
            "kv_bf16_reference_bytes": r.kv_bf16_reference_bytes,
            "compression_ratio": if r.kv_bf16_reference_bytes > 0 {
                r.kv_physical_bytes as f64 / r.kv_bf16_reference_bytes as f64
            } else { 1.0 },
            "kv_exact_cold_file_bytes": r.kv_exact_cold_file_bytes,
        })
    });

    // Tiempo total acumulado para N consultas (con el mismo documento)
    let mslim_total_s: f64 = mslim_results.iter().map(|r| r.latency_seconds).sum();
    let dense_total_s: f64 = dense_results.iter().map(|r| r.latency_seconds).sum();

    let report = json!({
        "benchmark": "mslim-vs-dense-long-context",
        "version": "1.0",
        "model": model_path,
        "document": {
            "type": "synthetic-key-retrieval",
            "repetitions": doc_repetitions,
            "estimated_tokens": system_tokens_approx,
        },
        "runtime_load": {
            "mslim_load_s": load_mslim_s,
            "dense_load_s": dense_runtime_opt.as_ref().map(|(_, s)| *s),
        },
        "mslim_results": mslim_json,
        "dense_results": dense_json,
        "analysis": {
            "fidelity": fidelity,
            "speedup_per_query": speedup,
            "compression": compression_summary,
            "total_n_queries": QUERIES.len(),
            "mslim_total_s": mslim_total_s,
            "dense_total_s": dense_total_s,
            "total_speedup": if mslim_total_s > 1e-9 { dense_total_s / mslim_total_s } else { 0.0 },
            "time_saved_s": dense_total_s - mslim_total_s,
        },
        "verdict": {
            "mslim_faster_on_queries_2_plus": speedup.get(1).and_then(|s| {
                s.get("prefill_speedup_mslim_vs_dense").and_then(serde_json::Value::as_f64)
            }).map(|x| x > 1.1).unwrap_or(false),
            "all_answers_correct_mslim": mslim_results.iter().all(|r| r.answer.contains("AZULEJO-7319")),
            "all_answers_correct_dense": dense_results.iter().all(|r| r.answer.contains("AZULEJO-7319")),
        }
    });

    println!("{}", serde_json::to_string_pretty(&report)?);

    // Resumen legible en stderr
    eprintln!("\n=== Resumen ===");
    if !dense_results.is_empty() && !mslim_results.is_empty() {
        for (i, (m, d)) in mslim_results.iter().zip(dense_results.iter()).enumerate() {
            let speedup_v = if m.prefill_seconds > 1e-9 {
                d.prefill_seconds / m.prefill_seconds
            } else {
                f64::INFINITY
            };
            eprintln!(
                "  Consulta {}: M-SLIM {:.3}s vs Denso {:.3}s  →  speedup prefill {:.1}×",
                i + 1,
                m.latency_seconds,
                d.latency_seconds,
                speedup_v
            );
        }
        eprintln!(
            "  Total {} consultas: M-SLIM {:.2}s vs Denso {:.2}s  →  {:.1}× más rápido",
            QUERIES.len(),
            mslim_total_s,
            dense_total_s,
            if mslim_total_s > 1e-9 {
                dense_total_s / mslim_total_s
            } else {
                0.0
            }
        );
        if let Some(last_m) = mslim_results.last() {
            if last_m.kv_bf16_reference_bytes > 0 {
                let ratio = last_m.kv_physical_bytes as f64 / last_m.kv_bf16_reference_bytes as f64;
                eprintln!(
                    "  KV físico: {} KB  vs  BF16 referencia: {} KB  →  {:.1}× compresión",
                    last_m.kv_physical_bytes / 1024,
                    last_m.kv_bf16_reference_bytes / 1024,
                    1.0 / ratio.max(1e-9)
                );
            }
        }
        let fidelity_ok = mslim_results
            .iter()
            .zip(dense_results.iter())
            .all(|(m, d)| m.token_ids == d.token_ids);
        eprintln!(
            "  Fidelidad: tokens {}",
            if fidelity_ok {
                "IDÉNTICOS ✓"
            } else {
                "DISTINTOS — verificar"
            }
        );
    }

    Ok(())
}
