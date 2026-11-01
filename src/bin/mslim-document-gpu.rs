//! Document Q&A that uses DocumentGraph page retrieval + GpuLlm with ColdStore KV cache.
//!
//! Pipeline:
//!   DocumentGraph → fused_retrieval (selects relevant pages) →
//!   llama3 prompt → GpuLlm (OpenCL, KV comprimido) → ColdStore (persiste entre llamadas)
//!
//! Primera consulta: prefill completo. Consultas posteriores sobre el mismo documento:
//! prefix replay desde ColdStore (sin repetir el prefill del documento).

use anyhow::{Context, Result, ensure};
use mslim::{
    document_graph::{
        DocumentGraph, RetrievalBudget, RetrievalScope, has_substantive_document_answer,
    },
    gpu_llm::{GpuLlm, WeightMode},
    gpu_region_store::GpuRegionConfig,
};
use std::{path::PathBuf, time::Instant};
use tokenizers::Tokenizer;

fn llama3_chat(system: &str, user: &str) -> String {
    format!(
        "<|begin_of_text|>\
<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>\
<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>\
<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
}

fn main() -> Result<()> {
    let mut model = PathBuf::from("deploy/local-production/runtime/models/active");
    let mut graph_path = None::<PathBuf>;
    let mut questions: Vec<String> = Vec::new();
    let mut max_pages = 8usize;
    let mut max_characters = 24_000usize;
    let mut max_new_tokens = 128usize;
    let mut max_context = 8192usize;
    let mut coldstore: Option<PathBuf> = None;
    let mut scope = RetrievalScope::Focused;
    let mut epsilon = 0.0f64;
    let mut graph_hops = 1usize;
    let mut resident = false;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model = args.next().context("--model")?.into(),
            "--graph" => graph_path = Some(args.next().context("--graph")?.into()),
            "--question" => questions.push(args.next().context("--question")?),
            "--max-pages" => max_pages = args.next().context("--max-pages")?.parse()?,
            "--max-characters" => {
                max_characters = args.next().context("--max-characters")?.parse()?
            }
            "--max-new-tokens" => {
                max_new_tokens = args.next().context("--max-new-tokens")?.parse()?
            }
            "--max-context" => max_context = args.next().context("--max-context")?.parse()?,
            "--coldstore" => coldstore = Some(args.next().context("--coldstore")?.into()),
            "--lod-epsilon" => epsilon = args.next().context("--lod-epsilon")?.parse()?,
            "--graph-hops" => graph_hops = args.next().context("--graph-hops")?.parse()?,
            "--resident" => resident = true,
            "--scope" => {
                scope = match args.next().context("--scope")?.as_str() {
                    "focused" => RetrievalScope::Focused,
                    "exhaustive" => RetrievalScope::Exhaustive,
                    v => anyhow::bail!("scope desconocido: {v}"),
                }
            }
            "--help" => {
                eprintln!(
                    "mslim-document-gpu --graph GRAPH_JSON --question TEXT \
[--model DIR] [--coldstore DIR] [--max-pages 8] [--max-characters 24000] \
[--max-new-tokens 128] [--max-context 8192] [--lod-epsilon 0.0] \
[--graph-hops 1] [--scope focused|exhaustive]"
                );
                return Ok(());
            }
            other => anyhow::bail!("argumento desconocido: {other}"),
        }
    }
    ensure!(!questions.is_empty(), "--question es obligatorio");
    let graph_path = graph_path.context("--graph es obligatorio")?;

    let t0 = Instant::now();
    let document = DocumentGraph::load_verified(&graph_path)?;
    eprintln!(
        "Grafo cargado en {:.2}s ({} páginas)",
        t0.elapsed().as_secs_f64(),
        document.page_count()
    );

    let tokenizer = Tokenizer::from_file(model.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

    let weight_mode = if resident {
        WeightMode::Resident
    } else {
        WeightMode::Stream
    };
    let region_config = GpuRegionConfig {
        epsilon,
        stage_tokens: 64,
        cold_directory: coldstore.clone(),
    };
    let t1 = Instant::now();
    let mut runtime =
        GpuLlm::load_with_gpu_region(&model, max_context, weight_mode, region_config)?;
    if coldstore.is_some() && resident {
        // Persistent replay across invocations requiere Resident weights.
        runtime.set_verified_gpu_prompt_replay(max_context)?;
    }
    // set_verified_gpu_prefix_replay se activa tras la primera consulta (requiere KV previo).
    eprintln!(
        "GPU runtime en {:.2}s | dispositivo: {} | LOD: {} | ColdStore: {}",
        t1.elapsed().as_secs_f64(),
        runtime.device_name(),
        if epsilon > 0.0 {
            format!("ε={epsilon}")
        } else {
            "exacto".to_owned()
        },
        coldstore
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "desactivado".to_owned()),
    );

    let budget = RetrievalBudget {
        max_pages,
        max_characters,
        graph_hops,
    };

    let mut results = Vec::new();

    for (i, question) in questions.iter().enumerate() {
        // Tras la primera consulta el KV está cacheado: activar shared prefix replay.
        if i == 1 && coldstore.is_some() {
            let _ = runtime.set_verified_gpu_prefix_replay(true);
        }
        let t_query = Instant::now();

        let phase = document.retrieve_phase_with_scope(question, budget, scope)?;
        let cited = phase.assemble_cited_context()?;
        let retrieval_s = t_query.elapsed().as_secs_f64();

        let prompt = llama3_chat(
            "Responde de forma directa y breve, en el idioma del usuario.",
            &cited.prompt,
        );
        let enc = tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        let prompt_ids: Vec<u32> = enc.get_ids().to_vec();

        ensure!(
            prompt_ids.len() + max_new_tokens <= max_context,
            "prompt {} + {} > contexto máximo {}",
            prompt_ids.len(),
            max_new_tokens,
            max_context
        );

        let t_gen = Instant::now();
        let generation = runtime.generate(&prompt_ids, max_new_tokens, false)?;
        let gen_s = t_gen.elapsed().as_secs_f64();

        let answer = tokenizer
            .decode(&generation.tokens, true)
            .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
        let answer = answer.trim().to_owned();

        let decode_s = gen_s - generation.prefill_seconds;
        let cache_hit = generation.gpu_verified_prefix_reused_tokens > 0
            || generation.gpu_verified_prompt_replay_hit;
        let correct = has_substantive_document_answer(&answer);

        eprintln!(
            "Q{}: retrieval={:.2}s prefill={:.2}s decode={:.2}s | páginas={} tokens={} caché={} correcto={}",
            i + 1,
            retrieval_s,
            generation.prefill_seconds,
            decode_s,
            cited.evidence_pages.len(),
            prompt_ids.len(),
            if cache_hit { "✓ HIT" } else { "MISS" },
            if correct { "✓" } else { "✗" },
        );

        results.push(serde_json::json!({
            "query_index": i,
            "question": question,
            "answer": answer,
            "correct": correct,
            "evidence_pages": cited.evidence_pages,
            "evidence_characters": cited.evidence_characters,
            "prompt_tokens": prompt_ids.len(),
            "cache_hit": cache_hit,
            "prefix_reused_tokens": generation.gpu_verified_prefix_reused_tokens,
            "prompt_replay_hit": generation.gpu_verified_prompt_replay_hit,
            "lod_active": generation.mslim_lod_active,
            "timing": {
                "retrieval_s": retrieval_s,
                "prefill_s": generation.prefill_seconds,
                "decode_s": decode_s,
                "total_s": t_query.elapsed().as_secs_f64(),
            }
        }));
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "model": model,
            "graph": graph_path,
            "scope": format!("{scope:?}"),
            "coldstore": coldstore,
            "lod_epsilon": epsilon,
            "results": results,
        }))?
    );

    Ok(())
}
