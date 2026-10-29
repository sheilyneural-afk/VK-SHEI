use anyhow::{Context, Result, ensure};
use serde_json::json;
use std::{collections::BTreeSet, path::PathBuf, time::Instant};
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
    let mut max_phases = 1usize;
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
            "--max-phases" => {
                max_phases = arguments.next().context("missing --max-phases")?.parse()?
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
                    "terrsvusion-document-ask --graph graph.json --question TEXT [--model DIR] [--scope focused|exhaustive] [--max-pages 8] [--max-characters 32000] [--graph-hops 1] [--max-phases 1] [--max-new-tokens 64]"
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
    ensure!(
        (1..=1024).contains(&max_phases),
        "max-phases must be 1..=1024"
    );

    let index_started = Instant::now();
    let document = DocumentGraph::load_verified(&graph_path)?;
    let phases = document.retrieve_phases_with_scope(
        &question,
        RetrievalBudget {
            max_pages,
            max_characters,
            graph_hops,
        },
        scope,
    )?;
    let retrieval_seconds = index_started.elapsed().as_secs_f64();

    let load_started = Instant::now();
    let runtime = TorchModelRuntime::load(&model)?;
    let load_seconds = load_started.elapsed().as_secs_f64();
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
    let mut phase_receipts = Vec::new();
    let mut final_generation = serde_json::Value::Null;
    let mut total_prompt_tokens = 0usize;
    let mut evidence_pages = BTreeSet::new();
    let mut answered = false;
    for (index, phase) in phases.iter().take(max_phases).enumerate() {
        let cited = phase.assemble_cited_context()?;
        evidence_pages.extend(cited.evidence_pages.iter().copied());
        let messages = vec![TorchChatMessage {
            role: "user".to_owned(),
            content: cited.prompt,
        }];
        let prompt_tokens = runtime.encode_chat_prompt_token_ids(&messages, false)?;
        ensure!(
            prompt_tokens.len() + max_new_tokens <= runtime.context_limit_tokens(),
            "phase {} exceeds the model context: {} + {} > {}",
            index + 1,
            prompt_tokens.len(),
            max_new_tokens,
            runtime.context_limit_tokens()
        );
        let generation = runtime.generate_messages_dense(&messages, max_new_tokens, false)?;
        let insufficient = is_insufficient_evidence(&generation.content);
        total_prompt_tokens = total_prompt_tokens
            .checked_add(prompt_tokens.len())
            .context("cumulative prompt token count overflow")?;
        final_generation = serde_json::to_value(&generation)?;
        phase_receipts.push(json!({
            "phase":index + 1,
            "evidence_pages":cited.evidence_pages,
            "evidence_characters":cited.evidence_characters,
            "prompt_tokens":prompt_tokens.len(),
            "candidate_set_complete_after_phase":cited.complete_for_candidate_set,
            "deferred_candidate_pages":cited.deferred_candidate_pages,
            "insufficient_evidence":insufficient,
            "generation":generation
        }));
        if !insufficient {
            answered = true;
            break;
        }
    }
    let phases_executed = phase_receipts.len();
    let candidate_set_exhausted = phases_executed == phases.len();
    let token_delta_vs_full = i64::try_from(full_document_prompt_tokens)
        .context("full prompt token count exceeds i64")?
        - i64::try_from(total_prompt_tokens).context("phase token count exceeds i64")?;
    let stop_reason = if answered {
        "answer_found"
    } else if candidate_set_exhausted {
        "candidate_set_exhausted_without_answer"
    } else {
        "max_phases_reached_without_answer"
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "operation":"document_graph_selective_dense_answer",
            "model":model,
            "graph":graph_path,
            "question":question,
            "document_pages":document.page_count(),
            "candidate_pages":phases[0].candidate_pages,
            "broad_match_pages":phases[0].broad_match_pages,
            "retrieval_scope":phases[0].scope,
            "evidence_pages":evidence_pages,
            "retrieval_complete":candidate_set_exhausted,
            "phases_available":phases.len(),
            "phases_executed":phases_executed,
            "max_phases":max_phases,
            "stop_reason":stop_reason,
            "model_loads":1,
            "retrieval_seconds":retrieval_seconds,
            "model_load_seconds":load_seconds,
            "model_context_limit":runtime.context_limit_tokens(),
            "actual_prompt_tokens":total_prompt_tokens,
            "cumulative_transformer_prompt_tokens":total_prompt_tokens,
            "full_document_prompt_tokens":full_document_prompt_tokens,
            "transformer_prompt_tokens_avoided":full_document_prompt_tokens.saturating_sub(total_prompt_tokens),
            "transformer_prompt_tokens_extra":total_prompt_tokens.saturating_sub(full_document_prompt_tokens),
            "transformer_prompt_token_delta_vs_full":token_delta_vs_full,
            "selected_prompt_percent_of_full":100.0 * total_prompt_tokens as f64 / full_document_prompt_tokens as f64,
            "generation":final_generation,
            "phases":phase_receipts,
            "claim_boundary":{
                "whole_document_transformer_prefill":false,
                "selected_original_evidence_prefill":true,
                "candidate_set_exhausted":candidate_set_exhausted,
                "semantic_exhaustiveness_certified":phases[0].semantic_exhaustiveness_certified
            }
        }))?
    );
    Ok(())
}

fn is_insufficient_evidence(content: &str) -> bool {
    let normalized = content
        .chars()
        .filter(|character| character.is_alphanumeric())
        .flat_map(char::to_uppercase)
        .collect::<String>();
    normalized == "EVIDENCIAINSUFICIENTE"
}

#[cfg(test)]
mod tests {
    use super::is_insufficient_evidence;

    #[test]
    fn recognizes_only_the_explicit_insufficient_evidence_answer() {
        assert!(is_insufficient_evidence("EVIDENCIA_INSUFICIENTE"));
        assert!(is_insufficient_evidence("Evidencia insuficiente."));
        assert!(!is_insufficient_evidence(
            "La evidencia es insuficiente para una parte, pero el dato es 38."
        ));
        assert!(!is_insufficient_evidence(""));
    }
}
