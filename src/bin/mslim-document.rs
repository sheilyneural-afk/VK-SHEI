use anyhow::{Context, Result, bail};
use mslim::document_graph::{DocumentGraph, RetrievalBudget, RetrievalScope};
use mslim::fused_retrieval::{FusedRetrievalConfig, retrieve_fused};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Ranking used by `query`. `Routing` is the original deterministic ranking
/// and stays the default; `Fused` adds a BM25 channel and fuses both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RankMode {
    Routing,
    Fused,
}

#[derive(Debug, Deserialize)]
struct InputPage {
    page_number: u32,
    #[serde(default)]
    section: Option<String>,
    original_text: String,
}

#[derive(Debug, Serialize)]
struct BuildReceipt {
    operation: &'static str,
    document_id: String,
    version_hash: String,
    page_count: usize,
    graph_micro_spans: usize,
    output: PathBuf,
    original_text_authority_verified: bool,
}

fn main() -> Result<()> {
    let mut arguments = std::env::args().skip(1);
    match arguments.next().as_deref() {
        Some("build") => build(arguments.collect()),
        Some("build-text") => build_text(arguments.collect()),
        Some("query") => query(arguments.collect()),
        Some("--help" | "-h") | None => {
            print_help();
            Ok(())
        }
        Some(command) => bail!("unknown mslim-document command: {command}"),
    }
}

fn build_text(arguments: Vec<String>) -> Result<()> {
    let mut document_id = None;
    let mut version_hash = None;
    let mut text_path = None;
    let mut output = None;
    let mut page_characters = 4_000usize;
    let mut values = arguments.into_iter();
    while let Some(argument) = values.next() {
        match argument.as_str() {
            "--document-id" => document_id = values.next(),
            "--version" => version_hash = values.next(),
            "--text" => text_path = values.next().map(PathBuf::from),
            "--output" => output = values.next().map(PathBuf::from),
            "--page-characters" => {
                page_characters = values
                    .next()
                    .context("--page-characters requires a value")?
                    .parse()?
            }
            other => bail!("unknown build-text option: {other}"),
        }
    }
    let document_id = document_id.context("build-text requires --document-id")?;
    let version_hash = version_hash.context("build-text requires --version")?;
    let text_path = text_path.context("build-text requires --text")?;
    let output = output.context("build-text requires --output")?;
    anyhow::ensure!(page_characters > 0, "page-characters must be positive");
    let text = std::fs::read_to_string(&text_path)
        .with_context(|| format!("read UTF-8 document {}", text_path.display()))?;
    let mut pages = Vec::new();
    let mut current = String::new();
    for line in text.split_inclusive('\n') {
        if !current.is_empty() && current.chars().count() + line.chars().count() > page_characters {
            pages.push(current);
            current = String::new();
        }
        current.push_str(line);
    }
    if !current.is_empty() {
        pages.push(current);
    }
    anyhow::ensure!(!pages.is_empty(), "input text is empty");
    anyhow::ensure!(
        pages.concat() == text,
        "pagination failed to preserve the exact UTF-8 document"
    );
    anyhow::ensure!(
        pages.len() <= u32::MAX as usize,
        "document has more pages than the u32 page identifier supports"
    );
    let numbered_pages = pages
        .into_iter()
        .enumerate()
        .map(|(index, page)| {
            u32::try_from(index + 1)
                .map(|number| (number, None, page))
                .context("page number exceeds u32")
        })
        .collect::<Result<Vec<_>>>()?;
    let graph = DocumentGraph::build(document_id.clone(), version_hash.clone(), numbered_pages)?;
    graph.save_atomic(&output)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&BuildReceipt {
            operation: "document_graph_build_text",
            document_id,
            version_hash,
            page_count: graph.page_count(),
            graph_micro_spans: graph.graph_micro_span_count(),
            output,
            original_text_authority_verified: true,
        })?
    );
    Ok(())
}

fn build(arguments: Vec<String>) -> Result<()> {
    let mut document_id = None;
    let mut version_hash = None;
    let mut pages_path = None;
    let mut output = None;
    let mut values = arguments.into_iter();
    while let Some(argument) = values.next() {
        match argument.as_str() {
            "--document-id" => document_id = values.next(),
            "--version" => version_hash = values.next(),
            "--pages" => pages_path = values.next().map(PathBuf::from),
            "--output" => output = values.next().map(PathBuf::from),
            other => bail!("unknown build option: {other}"),
        }
    }
    let document_id = document_id.context("build requires --document-id")?;
    let version_hash = version_hash.context("build requires --version")?;
    let pages_path = pages_path.context("build requires --pages")?;
    let output = output.context("build requires --output")?;
    let bytes = std::fs::read(&pages_path)
        .with_context(|| format!("read page input {}", pages_path.display()))?;
    let pages: Vec<InputPage> = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse page input {}", pages_path.display()))?;
    let graph = DocumentGraph::build(
        document_id.clone(),
        version_hash.clone(),
        pages
            .into_iter()
            .map(|page| (page.page_number, page.section, page.original_text))
            .collect(),
    )?;
    graph.save_atomic(&output)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&BuildReceipt {
            operation: "document_graph_build",
            document_id,
            version_hash,
            page_count: graph.page_count(),
            graph_micro_spans: graph.graph_micro_span_count(),
            output,
            original_text_authority_verified: true,
        })?
    );
    Ok(())
}

fn query(arguments: Vec<String>) -> Result<()> {
    let mut graph_path = None;
    let mut text = None;
    let mut max_pages = 8usize;
    let mut max_characters = 32_000usize;
    let mut graph_hops = 1usize;
    let mut scope = RetrievalScope::Focused;
    let mut rank = RankMode::Routing;
    let mut values = arguments.into_iter();
    while let Some(argument) = values.next() {
        match argument.as_str() {
            "--rank" => {
                rank = match values.next().context("--rank requires a value")?.as_str() {
                    "routing" => RankMode::Routing,
                    "fused" => RankMode::Fused,
                    value => bail!("unknown rank mode: {value}"),
                }
            }
            "--graph" => graph_path = values.next().map(PathBuf::from),
            "--text" => text = values.next(),
            "--max-pages" => {
                max_pages = values
                    .next()
                    .context("--max-pages requires a value")?
                    .parse()?
            }
            "--max-characters" => {
                max_characters = values
                    .next()
                    .context("--max-characters requires a value")?
                    .parse()?
            }
            "--graph-hops" => {
                graph_hops = values
                    .next()
                    .context("--graph-hops requires a value")?
                    .parse()?
            }
            "--scope" => {
                scope = match values.next().context("--scope requires a value")?.as_str() {
                    "focused" => RetrievalScope::Focused,
                    "exhaustive" => RetrievalScope::Exhaustive,
                    value => bail!("unknown retrieval scope: {value}"),
                }
            }
            other => bail!("unknown query option: {other}"),
        }
    }
    let graph_path = graph_path.context("query requires --graph")?;
    let text = text.context("query requires --text")?;
    let graph = DocumentGraph::load_verified(&graph_path)?;
    let budget = RetrievalBudget {
        max_pages,
        max_characters,
        graph_hops,
    };
    match rank {
        RankMode::Routing => {
            let phase = graph.retrieve_phase_with_scope(&text, budget, scope)?;
            println!("{}", serde_json::to_string_pretty(&phase)?);
        }
        RankMode::Fused => {
            let retrieval = retrieve_fused(
                &graph,
                &text,
                budget,
                scope,
                FusedRetrievalConfig::default(),
            )?;
            println!("{}", serde_json::to_string_pretty(&retrieval)?);
        }
    }
    Ok(())
}

fn print_help() {
    println!("M-SLIM original-document graph and micro-metadata router");
    println!(
        "  mslim-document build --document-id ID --version HASH --pages pages.json --output graph.json"
    );
    println!(
        "  mslim-document build-text --document-id ID --version HASH --text document.txt --output graph.json [--page-characters 4000]"
    );
    println!(
        "  mslim-document query --graph graph.json --text QUESTION [--scope focused|exhaustive] [--rank routing|fused] [--max-pages 8] [--max-characters 32000] [--graph-hops 1]"
    );
    println!(
        "  --rank fused adds a BM25 channel and fuses it with the routing ranking; output is {{\"phase\":..., \"fusion\":...}}"
    );
}
