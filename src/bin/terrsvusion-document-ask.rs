use anyhow::{Context, Result, ensure};
use serde_json::json;
use std::{path::PathBuf, time::Instant};
use terrsvusion::{
    document_graph::{DocumentGraph, RetrievalBudget, RetrievalScope},
    torch_runtime::{TorchChatMessage, TorchModelRuntime},
};

fn main() -> Result<()> {
    let mut model = PathBuf::from("/home/yo/models/Llama-3.1-8B-Instruct");
    let mut graph = None;
    let mut question = None;
    let mut max_pages = 8usize;
    let mut max_characters = 32_000usize;
    let mut graph_hops = 1usize;
    let mut max_new_tokens = 64usize;
    let mut scope = RetrievalScope::Focused;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--model" => model = arguments.next().context("missing --model")?.into(),
            "--graph" => graph = arguments.next().map(PathBuf::from),
            "--question" => question = arguments.next(),
            "--max-pages" => {
                max_pages = arguments.next().context("missing --max-pages")?.parse()?
            }
            "--max-characters" => {
                max_characters = arguments
                    .next()
                    .context("missing --max-characters")?
                    .parse()?
            }
            "--graph-hops" => {
                graph_hops = arguments.next().context("missing --graph-hops")?.parse()?
            }
            "--max-new-tokens" => {
                max_new_tokens = arguments
                    .next()
                    .context("missing --max-new-tokens")?
                    .parse()?
            }
            "--scope" => {
                scope = match arguments.next().context("missing --scope")?.as_str() {
                    "focused" => RetrievalScope::Focused,
                    "exhaustive" => RetrievalScope::Exhaustive,
                    value => anyhow::bail!("unknown retrieval scope: {value}"),
                }
            }
            "--help" | "-h" => {
                println!(
                    "terrsvusion-document-ask --graph graph.json --question TEXT [--model DIR] [--scope focused|exhaustive] [--max-pages 8] [--max-characters 32000] [--graph-hops 1] [--max-new-tokens 64]"
                );
                return Ok(());
            }
            other => anyhow::bail!("unknown option: {other}"),
        }
    }
    let graph_path = graph.context("--graph is required")?;
    let question = question.context("--question is required")?;
    ensure!(
        (1..=512).contains(&max_new_tokens),
        "max-new-tokens must be 1..=512"
    );

    let index_started = Instant::now();
    let document = DocumentGraph::load_verified(&graph_path)?;
    let phase = document.retrieve_phase_with_scope(
        &question,
        RetrievalBudget {
            max_pages,
            max_characters,
            graph_hops,
        },
        scope,
    )?;
    let cited = phase.assemble_cited_context()?;
    let retrieval_seconds = index_started.elapsed().as_secs_f64();
    ensure!(
        cited.complete_for_candidate_set,
        "selected phase is incomplete: {} candidate pages remain deferred; widen the budget before answering",
        cited.deferred_candidate_pages.len()
    );

    let load_started = Instant::now();
    let runtime = TorchModelRuntime::load(&model)?;
    let load_seconds = load_started.elapsed().as_secs_f64();
    let messages = vec![TorchChatMessage {
        role: "user".to_owned(),
        content: cited.prompt.clone(),
    }];
    let prompt_tokens = runtime.encode_chat_prompt_token_ids(&messages, false)?;
    let full_context = document
        .full_evidence_phase(&question)?
        .assemble_cited_context()?;
    let full_messages = vec![TorchChatMessage {
        role: "user".to_owned(),
        content: full_context.prompt,
    }];
    let full_document_prompt_tokens = runtime
        .encode_chat_prompt_token_ids(&full_messages, false)?
        .len();
    ensure!(
        prompt_tokens.len() + max_new_tokens <= runtime.context_limit_tokens(),
        "selected evidence exceeds the model context: {} + {} > {}",
        prompt_tokens.len(),
        max_new_tokens,
        runtime.context_limit_tokens()
    );
    let generation = runtime.generate_messages_dense(&messages, max_new_tokens, false)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "operation":"document_graph_selective_dense_answer",
            "model":model,
            "graph":graph_path,
            "question":question,
            "document_pages":document.page_count(),
            "candidate_pages":phase.candidate_pages,
            "broad_match_pages":phase.broad_match_pages,
            "retrieval_scope":phase.scope,
            "evidence_pages":cited.evidence_pages,
            "evidence_characters":cited.evidence_characters,
            "retrieval_complete":cited.complete_for_candidate_set,
            "deferred_candidate_pages":cited.deferred_candidate_pages,
            "retrieval_seconds":retrieval_seconds,
            "model_load_seconds":load_seconds,
            "model_context_limit":runtime.context_limit_tokens(),
            "actual_prompt_tokens":prompt_tokens.len(),
            "full_document_prompt_tokens":full_document_prompt_tokens,
            "transformer_prompt_tokens_avoided":full_document_prompt_tokens.saturating_sub(prompt_tokens.len()),
            "selected_prompt_percent_of_full":100.0 * prompt_tokens.len() as f64 / full_document_prompt_tokens as f64,
            "generation":generation,
            "claim_boundary":{
                "whole_document_transformer_prefill":false,
                "selected_original_evidence_prefill":true,
                "focused_candidate_set_complete":cited.complete_for_candidate_set,
                "semantic_exhaustiveness_certified":phase.semantic_exhaustiveness_certified
            }
        }))?
    );
    Ok(())
}
