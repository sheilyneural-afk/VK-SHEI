//! Document retrieval that fuses deterministic routing with a BM25 channel.
//!
//! The routing channel is the existing `DocumentGraph` ranking: co-located
//! terms inside one sentence span, exact numbers and exact phrases, plus graph
//! neighbours. Its focused scope deliberately drops pages whose query terms are
//! scattered across different sentences, and it uses neither term frequency nor
//! page length. The BM25 channel scores exactly those signals. The two rankings
//! are fused by [`crate::page_fusion`], so every admitted page is checked
//! against the digest of its original text and the result carries an
//! explanation of which channel ranked each page where.
//!
//! The graph's original pages remain the only text authority; nothing here
//! summarizes or rewrites page text. Fusion can surface a page that routing
//! alone excluded, and it can reorder pages. It does not certify coverage
//! (`full_scope_coverage_verified` stays false), and no relevance benchmark has
//! been run: the tests show behaviour on constructed documents, not a measured
//! improvement in answer quality.

use crate::{
    document_graph::{
        DocumentGraph, EvidencePage, RetrievalBudget, RetrievalPhase, RetrievalScope, signal_terms,
        tokenize,
    },
    page_fusion::{
        MAXIMUM_LIMIT, PageCandidate, PageChannel, PageFusion, PageFusionRequest, fuse_pages,
    },
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

const BM25_K1: f64 = 1.2;
const BM25_B: f64 = 0.75;

pub const ROUTING_CHANNEL: &str = "routing";
pub const BM25_CHANNEL: &str = "bm25";

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FusedRetrievalConfig {
    pub routing_weight: f64,
    pub bm25_weight: f64,
    pub rrf_k: u32,
    /// Maximum pages the BM25 channel proposes.
    pub bm25_candidates: usize,
}

impl Default for FusedRetrievalConfig {
    fn default() -> Self {
        Self {
            routing_weight: 1.0,
            bm25_weight: 1.0,
            rrf_k: 60,
            bm25_candidates: 64,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FusedRetrieval {
    /// Evidence in fused order. Pages located only by BM25 carry
    /// `routing_score` 0 and `graph_distance` 0; their channel ranks are in
    /// `fusion`.
    pub phase: RetrievalPhase,
    pub fusion: PageFusion,
}

/// BM25 ranking of the original pages (k1 = 1.2, b = 0.75, Lucene-style IDF).
/// Only pages with a positive score are returned, best first, ties by page
/// number. A query without signal terms returns an empty ranking.
pub fn bm25_ranking(graph: &DocumentGraph, query: &str, limit: usize) -> Result<Vec<(u32, f64)>> {
    ensure!(!query.trim().is_empty(), "retrieval query cannot be empty");
    ensure!(limit > 0, "limit must be positive");
    graph.verify_original_authority()?;
    Ok(bm25_scores(graph, query, limit))
}

fn bm25_scores(graph: &DocumentGraph, query: &str, limit: usize) -> Vec<(u32, f64)> {
    let terms = signal_terms(query);
    if terms.is_empty() {
        return Vec::new();
    }
    let mut stats = Vec::new();
    for page in graph.pages() {
        let tokens = tokenize(&page.original_text);
        let mut frequencies = BTreeMap::<&str, u32>::new();
        for token in &tokens {
            if let Some(term) = terms.get(token.as_str()) {
                *frequencies.entry(term.as_str()).or_default() += 1;
            }
        }
        stats.push((page.page_number, tokens.len() as f64, frequencies));
    }
    let pages = stats.len() as f64;
    let total_length = stats.iter().map(|(_, length, _)| *length).sum::<f64>();
    let average_length = if total_length > 0.0 {
        total_length / pages
    } else {
        1.0
    };
    let mut document_frequency = BTreeMap::<&str, f64>::new();
    for (_, _, frequencies) in &stats {
        for term in frequencies.keys() {
            *document_frequency.entry(term).or_default() += 1.0;
        }
    }
    let mut scored = stats
        .iter()
        .filter_map(|(page_number, length, frequencies)| {
            let mut score = 0.0;
            for (term, &frequency) in frequencies {
                let df = document_frequency[term];
                let idf = (1.0 + (pages - df + 0.5) / (df + 0.5)).ln();
                let tf = f64::from(frequency);
                score += idf * tf * (BM25_K1 + 1.0)
                    / (tf + BM25_K1 * (1.0 - BM25_B + BM25_B * length / average_length));
            }
            (score > 0.0).then_some((*page_number, score))
        })
        .collect::<Vec<_>>();
    scored.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    scored.truncate(limit);
    scored
}

/// Retrieves a bounded evidence phase from the fused order of the routing and
/// BM25 channels. The budget applies to the fused order exactly as the
/// routing-only phase applies it: pages are taken in rank order until the next
/// one does not fit, and the rest are reported as deferred candidates.
pub fn retrieve_fused(
    graph: &DocumentGraph,
    query: &str,
    budget: RetrievalBudget,
    scope: RetrievalScope,
    config: FusedRetrievalConfig,
) -> Result<FusedRetrieval> {
    ensure!(!query.trim().is_empty(), "retrieval query cannot be empty");
    ensure!(
        config.bm25_candidates > 0,
        "bm25_candidates must be positive"
    );
    let budget = budget.validate()?;
    graph.verify_original_authority()?;

    // Routing: every candidate of the existing ranking, in rank order. The
    // phases partition one ranking without repeating a page.
    let phases = graph.retrieve_phases_with_scope(query, budget, scope)?;
    let broad_match_pages = phases
        .first()
        .context("routing produced no phases")?
        .broad_match_pages;
    let routing = phases
        .iter()
        .flat_map(|phase| phase.evidence.iter())
        .collect::<Vec<_>>();
    let routing_metadata = routing
        .iter()
        .map(|page| (page.page_number, (page.routing_score, page.graph_distance)))
        .collect::<BTreeMap<_, _>>();

    let bm25 = bm25_scores(graph, query, config.bm25_candidates);
    let routing_channel = PageChannel {
        channel_id: ROUTING_CHANNEL.into(),
        weight: config.routing_weight,
        candidates: routing
            .iter()
            .map(|page| PageCandidate {
                page_number: page.page_number,
                sha256: page.sha256.clone(),
            })
            .collect(),
    };
    let bm25_channel = PageChannel {
        channel_id: BM25_CHANNEL.into(),
        weight: config.bm25_weight,
        candidates: bm25
            .iter()
            .filter_map(|&(page_number, _)| {
                graph.page(page_number).map(|page| PageCandidate {
                    page_number,
                    sha256: page.sha256.clone(),
                })
            })
            .collect(),
    };
    let limit =
        (routing_channel.candidates.len() + bm25_channel.candidates.len()).clamp(1, MAXIMUM_LIMIT);
    let fusion = fuse_pages(
        graph,
        PageFusionRequest {
            query_id: format!("sha256:{}", hex::encode(Sha256::digest(query.as_bytes()))),
            rrf_k: config.rrf_k,
            limit,
            channels: vec![routing_channel, bm25_channel],
        },
    )?;

    let mut evidence = Vec::new();
    let mut selected_characters = 0_usize;
    for fused in &fusion.pages {
        let page = graph
            .page(fused.page_number)
            .context("fused page lost from document authority")?;
        let characters = page.original_text.chars().count();
        let fits = evidence.len() < budget.max_pages
            && selected_characters
                .checked_add(characters)
                .is_some_and(|total| total <= budget.max_characters);
        if !fits {
            break;
        }
        selected_characters += characters;
        let (routing_score, graph_distance) = routing_metadata
            .get(&fused.page_number)
            .copied()
            .unwrap_or((0, 0));
        evidence.push(EvidencePage {
            page_number: page.page_number,
            section: page.section.clone(),
            original_text: page.original_text.clone(),
            sha256: page.sha256.clone(),
            routing_score,
            graph_distance,
        });
    }
    ensure!(
        !evidence.is_empty(),
        "retrieval budget cannot fit any candidate page"
    );
    let deferred_candidate_pages = fusion.pages[evidence.len()..]
        .iter()
        .map(|page| page.page_number)
        .collect::<Vec<_>>();
    let phase = RetrievalPhase {
        query: query.to_owned(),
        scope,
        candidate_pages: fusion.pages.len(),
        broad_match_pages,
        selected_characters,
        complete_for_candidate_set: deferred_candidate_pages.is_empty(),
        evidence,
        deferred_candidate_pages,
        original_text_authority: true,
        full_scope_coverage_verified: false,
    };
    Ok(FusedRetrieval { phase, fusion })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(pages: &[&str]) -> DocumentGraph {
        DocumentGraph::build(
            "doc",
            "v1",
            pages
                .iter()
                .enumerate()
                .map(|(index, text)| (index as u32 + 1, None, (*text).to_owned()))
                .collect(),
        )
        .unwrap()
    }

    fn pages_of(ranking: &[(u32, f64)]) -> Vec<u32> {
        ranking.iter().map(|&(page, _)| page).collect()
    }

    fn evidence_pages(phase: &RetrievalPhase) -> Vec<u32> {
        phase.evidence.iter().map(|page| page.page_number).collect()
    }

    fn budget(max_pages: usize) -> RetrievalBudget {
        RetrievalBudget {
            max_pages,
            max_characters: 10_000,
            graph_hops: 0,
        }
    }

    fn warranty() -> DocumentGraph {
        build(&[
            "La garantía de calidad cubre defectos de fabricación.",
            "La garantía se extiende un año. La calidad se revisa cada mes. \
             La garantía cubre piezas. La calidad mejora cada trimestre. \
             La garantía incluye mano de obra.",
            "El envío tarda tres días.",
            "El precio total es de 40 euros.",
        ])
    }

    #[test]
    fn term_frequency_saturates_and_missing_terms_are_absent() {
        let graph = build(&[
            "garantía garantía garantía garantía envío",
            "garantía envío envío envío",
            "precio total",
        ]);
        let ranking = bm25_ranking(&graph, "garantía", 10).unwrap();
        assert_eq!(pages_of(&ranking), vec![1, 2]);
        assert!(
            ranking
                .iter()
                .all(|&(_, score)| score.is_finite() && score > 0.0)
        );
        assert!(ranking[0].1 > ranking[1].1);
    }

    #[test]
    fn rare_terms_outweigh_common_ones() {
        let graph = build(&["alfa beta", "alfa gamma", "alfa delta", "omega alfa"]);
        let ranking = bm25_ranking(&graph, "omega alfa", 10).unwrap();
        assert_eq!(ranking[0].0, 4);
    }

    #[test]
    fn longer_pages_are_normalized_down() {
        let filler = "relleno ".repeat(50);
        let long = format!("garantía {filler}");
        let graph = build(&["garantía relleno relleno", long.as_str()]);
        let ranking = bm25_ranking(&graph, "garantía", 10).unwrap();
        assert_eq!(pages_of(&ranking), vec![1, 2]);
        assert!(ranking[0].1 > ranking[1].1);
    }

    #[test]
    fn equal_scores_break_ties_by_page_number() {
        let graph = build(&["garantía total", "garantía total", "garantía total"]);
        let ranking = bm25_ranking(&graph, "garantía", 2).unwrap();
        assert_eq!(pages_of(&ranking), vec![1, 2]);
    }

    #[test]
    fn short_words_and_stopwords_carry_no_signal_but_numbers_do() {
        let graph = build(&["el precio de la casa", "cuesta 38 millones"]);
        assert!(bm25_ranking(&graph, "el de la", 10).unwrap().is_empty());
        assert_eq!(
            pages_of(&bm25_ranking(&graph, "cuánto 38", 10).unwrap()),
            vec![2]
        );
    }

    #[test]
    fn invalid_arguments_are_rejected() {
        let graph = build(&["texto"]);
        assert!(bm25_ranking(&graph, "  ", 5).is_err());
        assert!(bm25_ranking(&graph, "texto", 0).is_err());
    }

    #[test]
    fn fusion_recovers_a_page_the_focused_routing_excludes_and_keeps_the_best_first() {
        let graph = warranty();
        let routing_only = graph
            .retrieve_phase("garantía de calidad", budget(8))
            .unwrap();
        assert_eq!(evidence_pages(&routing_only), vec![1]);

        let fused = retrieve_fused(
            &graph,
            "garantía de calidad",
            budget(8),
            RetrievalScope::Focused,
            FusedRetrievalConfig::default(),
        )
        .unwrap();
        assert_eq!(evidence_pages(&fused.phase), vec![1, 2]);
        assert_eq!(fused.phase.candidate_pages, 2);
        assert!(fused.phase.complete_for_candidate_set);
        assert!(fused.phase.original_text_authority);
        assert!(!fused.phase.full_scope_coverage_verified);

        let first = &fused.fusion.pages[0];
        let channels = first
            .contributions
            .iter()
            .map(|c| c.channel_id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(channels, vec![BM25_CHANNEL, ROUTING_CHANNEL]);
        let second = &fused.fusion.pages[1];
        assert_eq!(second.contributions.len(), 1);
        assert_eq!(second.contributions[0].channel_id, BM25_CHANNEL);
        fused.fusion.verify_against(&graph).unwrap();
    }

    #[test]
    fn an_exact_identifier_keeps_its_page_first() {
        let graph = build(&[
            "La empresa deberá pagar 38 millones antes de 2028.",
            "La obligación general se explica en la página 1.",
            "Excepción: no habrá pago cuando exista fuerza mayor.",
            "El identificador inesperado ZK-991 figura solamente aquí.",
        ]);
        let routing_only = graph.retrieve_phase("ZK-991", budget(4)).unwrap();
        let fused = retrieve_fused(
            &graph,
            "ZK-991",
            budget(4),
            RetrievalScope::Focused,
            FusedRetrievalConfig::default(),
        )
        .unwrap();
        assert_eq!(routing_only.evidence[0].page_number, 4);
        assert_eq!(fused.phase.evidence[0].page_number, 4);
    }

    #[test]
    fn the_budget_applies_to_the_fused_order_and_reports_deferred_pages() {
        let graph = warranty();
        let fused = retrieve_fused(
            &graph,
            "garantía de calidad",
            budget(1),
            RetrievalScope::Focused,
            FusedRetrievalConfig::default(),
        )
        .unwrap();
        assert_eq!(evidence_pages(&fused.phase), vec![1]);
        assert_eq!(fused.phase.deferred_candidate_pages, vec![2]);
        assert!(!fused.phase.complete_for_candidate_set);
        let context = fused.phase.assemble_cited_context().unwrap();
        assert!(
            context
                .prompt
                .contains("AVISO: existen páginas candidatas diferidas")
        );
    }

    #[test]
    fn every_evidence_page_is_the_original_text_with_its_digest() {
        let graph = warranty();
        let fused = retrieve_fused(
            &graph,
            "garantía de calidad",
            budget(8),
            RetrievalScope::Exhaustive,
            FusedRetrievalConfig::default(),
        )
        .unwrap();
        assert!(!fused.phase.evidence.is_empty());
        for page in &fused.phase.evidence {
            let original = graph.page(page.page_number).unwrap();
            assert_eq!(page.original_text, original.original_text);
            assert_eq!(page.sha256, original.sha256);
        }
        let context = fused.phase.assemble_cited_context().unwrap();
        assert_eq!(context.evidence_pages, evidence_pages(&fused.phase));
    }

    #[test]
    fn the_result_is_deterministic() {
        let graph = warranty();
        let run = || {
            retrieve_fused(
                &graph,
                "garantía de calidad",
                budget(8),
                RetrievalScope::Focused,
                FusedRetrievalConfig::default(),
            )
            .unwrap()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn invalid_configuration_and_queries_are_rejected() {
        let graph = warranty();
        let call = |query: &str, config| {
            retrieve_fused(&graph, query, budget(8), RetrievalScope::Focused, config)
        };
        let default = FusedRetrievalConfig::default();
        assert!(call("  ", default).is_err());
        assert!(
            call(
                "garantía",
                FusedRetrievalConfig {
                    bm25_candidates: 0,
                    ..default
                }
            )
            .is_err()
        );
        assert!(
            call(
                "garantía",
                FusedRetrievalConfig {
                    bm25_weight: 0.0,
                    ..default
                }
            )
            .is_err()
        );
        assert!(
            call(
                "garantía",
                FusedRetrievalConfig {
                    rrf_k: 0,
                    ..default
                }
            )
            .is_err()
        );
    }

    #[test]
    fn a_query_without_signal_terms_falls_back_to_routing_alone() {
        let graph = warranty();
        let routing_only = graph.retrieve_phase("el de la", budget(8)).unwrap();
        let fused = retrieve_fused(
            &graph,
            "el de la",
            budget(8),
            RetrievalScope::Focused,
            FusedRetrievalConfig::default(),
        )
        .unwrap();
        assert_eq!(evidence_pages(&fused.phase), evidence_pages(&routing_only));
    }
}
