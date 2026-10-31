//! Weighted reciprocal-rank fusion of page rankings over one document graph.
//!
//! Ranking channels (deterministic lexical routing, graph expansion, an order
//! produced by a model runtime...) are advisory indexes. A candidate is
//! admitted only when its page exists in the graph and the SHA-256 it claims
//! equals the digest of the page's original text, so a stale or forged index
//! entry can reorder nothing. The graph's original pages stay the only text
//! authority: this module never creates, summarizes or edits page text.
//!
//! Fusion does not retrieve anything and does not certify coverage. It cannot
//! surface a page that no channel proposed.

use crate::document_graph::{DocumentGraph, RetrievalPhase};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const PAGE_FUSION_SCHEMA: &str = "mslim.page_fusion.weighted_rrf/v1";

const MAXIMUM_CHANNELS: usize = 64;
const MAXIMUM_CANDIDATES_PER_CHANNEL: usize = 100_000;
pub const MAXIMUM_LIMIT: usize = 10_000;
const MAXIMUM_RRF_K: u32 = 1_000_000;
const MAXIMUM_WEIGHT: f64 = 1_000.0;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PageCandidate {
    pub page_number: u32,
    /// SHA-256 of the original page text the index believes it points to.
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PageChannel {
    pub channel_id: String,
    pub weight: f64,
    /// Best first; the rank is the one-based position.
    pub candidates: Vec<PageCandidate>,
}

impl PageChannel {
    /// Turns a ranked retrieval phase into a channel. Use only phases that
    /// rank pages (`retrieve_phase*`); `full_evidence_phase` is in page order
    /// and carries no ranking.
    pub fn from_phase(channel_id: impl Into<String>, weight: f64, phase: &RetrievalPhase) -> Self {
        Self {
            channel_id: channel_id.into(),
            weight,
            candidates: phase
                .evidence
                .iter()
                .map(|page| PageCandidate {
                    page_number: page.page_number,
                    sha256: page.sha256.clone(),
                })
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PageFusionRequest {
    pub query_id: String,
    pub rrf_k: u32,
    pub limit: usize,
    pub channels: Vec<PageChannel>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChannelContribution {
    pub channel_id: String,
    pub rank: u32,
    pub weight: f64,
    pub reciprocal_score: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RejectionReason {
    UnknownPage,
    OriginalDigestMismatch,
    DuplicateWithinChannel,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RejectedPage {
    pub channel_id: String,
    pub page_number: u32,
    pub reason: RejectionReason,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FusedPage {
    pub page_number: u32,
    pub sha256: String,
    pub rrf_score: f64,
    pub best_rank: u32,
    pub contributions: Vec<ChannelContribution>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PageFusion {
    pub schema: String,
    pub query_id: String,
    pub document_id: String,
    pub version_hash: String,
    pub rrf_k: u32,
    pub requested_limit: usize,
    pub pages: Vec<FusedPage>,
    pub rejected: Vec<RejectedPage>,
    pub digest_sha256: String,
}

impl PageFusion {
    /// Whether the stored digest matches the content. Detects edits; it does
    /// not by itself prove the pages still match a graph.
    pub fn verify_digest(&self) -> bool {
        self.schema == PAGE_FUSION_SCHEMA
            && digest(self).is_ok_and(|expected| expected == self.digest_sha256)
    }

    /// Verifies the digest, the document identity and that every fused page
    /// still equals the graph's original page.
    pub fn verify_against(&self, graph: &DocumentGraph) -> Result<()> {
        ensure!(self.verify_digest(), "page fusion digest mismatch");
        ensure!(
            self.document_id == graph.document_id && self.version_hash == graph.version_hash,
            "page fusion belongs to another document version"
        );
        graph.verify_original_authority()?;
        for fused in &self.pages {
            let original = graph
                .page(fused.page_number)
                .with_context(|| format!("fused page {} is not in the graph", fused.page_number))?;
            ensure!(
                original.sha256 == fused.sha256,
                "fused page {} no longer matches the original",
                fused.page_number
            );
        }
        Ok(())
    }
}

struct Aggregate {
    sha256: String,
    score: f64,
    best_rank: u32,
    contributions: Vec<ChannelContribution>,
}

/// Fuses ranked channels with weighted reciprocal-rank fusion.
///
/// Channels are processed in `channel_id` order, so the result does not depend
/// on the order of `request.channels`. Rejected candidates are reported rather
/// than failing the request: an index entry may be stale without the request
/// being invalid. The original pages are verified once before anything is
/// admitted.
pub fn fuse_pages(graph: &DocumentGraph, mut request: PageFusionRequest) -> Result<PageFusion> {
    validate_request(&request)?;
    graph.verify_original_authority()?;
    request
        .channels
        .sort_by(|left, right| left.channel_id.cmp(&right.channel_id));

    let mut merged = BTreeMap::<u32, Aggregate>::new();
    let mut rejected = Vec::new();
    for channel in &request.channels {
        let mut seen = BTreeSet::new();
        for (offset, candidate) in channel.candidates.iter().enumerate() {
            let rank = u32::try_from(offset + 1).context("candidate rank overflow")?;
            let reject = |reason| RejectedPage {
                channel_id: channel.channel_id.clone(),
                page_number: candidate.page_number,
                reason,
            };
            if !seen.insert(candidate.page_number) {
                rejected.push(reject(RejectionReason::DuplicateWithinChannel));
                continue;
            }
            let Some(original) = graph.page(candidate.page_number) else {
                rejected.push(reject(RejectionReason::UnknownPage));
                continue;
            };
            if original.sha256 != candidate.sha256 {
                rejected.push(reject(RejectionReason::OriginalDigestMismatch));
                continue;
            }
            let reciprocal_score = channel.weight / (f64::from(request.rrf_k) + f64::from(rank));
            let aggregate = merged
                .entry(candidate.page_number)
                .or_insert_with(|| Aggregate {
                    sha256: original.sha256.clone(),
                    score: 0.0,
                    best_rank: rank,
                    contributions: Vec::new(),
                });
            aggregate.score += reciprocal_score;
            aggregate.best_rank = aggregate.best_rank.min(rank);
            aggregate.contributions.push(ChannelContribution {
                channel_id: channel.channel_id.clone(),
                rank,
                weight: channel.weight,
                reciprocal_score,
            });
        }
    }

    let mut pages = merged
        .into_iter()
        .map(|(page_number, aggregate)| FusedPage {
            page_number,
            sha256: aggregate.sha256,
            rrf_score: aggregate.score,
            best_rank: aggregate.best_rank,
            contributions: aggregate.contributions,
        })
        .collect::<Vec<_>>();
    pages.sort_by(|left, right| {
        right
            .rrf_score
            .total_cmp(&left.rrf_score)
            .then_with(|| left.best_rank.cmp(&right.best_rank))
            .then_with(|| left.page_number.cmp(&right.page_number))
    });
    pages.truncate(request.limit);
    rejected.sort();

    let mut fusion = PageFusion {
        schema: PAGE_FUSION_SCHEMA.into(),
        query_id: request.query_id,
        document_id: graph.document_id.clone(),
        version_hash: graph.version_hash.clone(),
        rrf_k: request.rrf_k,
        requested_limit: request.limit,
        pages,
        rejected,
        digest_sha256: String::new(),
    };
    fusion.digest_sha256 = digest(&fusion)?;
    Ok(fusion)
}

fn validate_request(request: &PageFusionRequest) -> Result<()> {
    ensure!(
        !request.query_id.trim().is_empty(),
        "query_id cannot be empty"
    );
    ensure!(
        (1..=MAXIMUM_RRF_K).contains(&request.rrf_k),
        "rrf_k must be in 1..={MAXIMUM_RRF_K}"
    );
    ensure!(
        (1..=MAXIMUM_LIMIT).contains(&request.limit),
        "limit must be in 1..={MAXIMUM_LIMIT}"
    );
    ensure!(
        (1..=MAXIMUM_CHANNELS).contains(&request.channels.len()),
        "channels must be in 1..={MAXIMUM_CHANNELS}"
    );
    let mut ids = BTreeSet::new();
    for channel in &request.channels {
        ensure!(
            !channel.channel_id.trim().is_empty(),
            "channel_id cannot be empty"
        );
        ensure!(
            channel.weight.is_finite() && channel.weight > 0.0 && channel.weight <= MAXIMUM_WEIGHT,
            "channel {} weight must be finite in (0, {MAXIMUM_WEIGHT}]",
            channel.channel_id
        );
        ensure!(
            channel.candidates.len() <= MAXIMUM_CANDIDATES_PER_CHANNEL,
            "channel {} has too many candidates",
            channel.channel_id
        );
        ensure!(
            ids.insert(channel.channel_id.as_str()),
            "duplicate channel {}",
            channel.channel_id
        );
    }
    Ok(())
}

fn digest(fusion: &PageFusion) -> Result<String> {
    let mut unsigned = fusion.clone();
    unsigned.digest_sha256.clear();
    Ok(hex::encode(Sha256::digest(serde_json::to_vec(&unsigned)?)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document_graph::RetrievalBudget;

    fn graph(version: &str) -> DocumentGraph {
        DocumentGraph::build(
            "contract",
            version,
            vec![
                (
                    1,
                    Some("Obligaciones".to_owned()),
                    "La empresa deberá pagar 38 millones antes de 2028.".to_owned(),
                ),
                (
                    2,
                    Some("Obligaciones".to_owned()),
                    "La obligación general se explica en la página 1.".to_owned(),
                ),
                (
                    3,
                    Some("Excepciones".to_owned()),
                    "Excepción: no habrá pago cuando exista fuerza mayor.".to_owned(),
                ),
                (
                    4,
                    Some("Anexo".to_owned()),
                    "El identificador inesperado ZK-991 figura solamente aquí.".to_owned(),
                ),
            ],
        )
        .unwrap()
    }

    fn candidate(graph: &DocumentGraph, page: u32) -> PageCandidate {
        PageCandidate {
            page_number: page,
            sha256: graph.page(page).unwrap().sha256.clone(),
        }
    }

    fn channel(graph: &DocumentGraph, id: &str, weight: f64, pages: &[u32]) -> PageChannel {
        PageChannel {
            channel_id: id.into(),
            weight,
            candidates: pages.iter().map(|&p| candidate(graph, p)).collect(),
        }
    }

    fn request(channels: Vec<PageChannel>) -> PageFusionRequest {
        PageFusionRequest {
            query_id: "q1".into(),
            rrf_k: 60,
            limit: 10,
            channels,
        }
    }

    fn order(fusion: &PageFusion) -> Vec<u32> {
        fusion.pages.iter().map(|p| p.page_number).collect()
    }

    #[test]
    fn equal_weights_fuse_by_reciprocal_rank() {
        let g = graph("v1");
        let fusion = fuse_pages(
            &g,
            request(vec![
                channel(&g, "lexical", 1.0, &[1, 2, 3]),
                channel(&g, "model", 1.0, &[3, 1]),
            ]),
        )
        .unwrap();
        // page 1: 1/61 + 1/62; page 3: 1/63 + 1/61; page 2: 1/62.
        assert_eq!(order(&fusion), vec![1, 3, 2]);
        let first = &fusion.pages[0];
        assert!((first.rrf_score - (1.0 / 61.0 + 1.0 / 62.0)).abs() < 1e-12);
        assert_eq!(first.best_rank, 1);
        let ids = first
            .contributions
            .iter()
            .map(|c| c.channel_id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(ids, vec!["lexical", "model"]);
        assert!(fusion.rejected.is_empty());
    }

    #[test]
    fn a_heavier_channel_can_overturn_the_order() {
        let g = graph("v1");
        let fusion = fuse_pages(
            &g,
            request(vec![
                channel(&g, "lexical", 1.0, &[1, 2, 3]),
                channel(&g, "model", 3.0, &[3, 1]),
            ]),
        )
        .unwrap();
        // page 3: 1/63 + 3/61; page 1: 1/61 + 3/62.
        assert_eq!(order(&fusion), vec![3, 1, 2]);
    }

    #[test]
    fn stale_or_forged_entries_reorder_nothing() {
        let g = graph("v1");
        let good = candidate(&g, 1);
        let fusion = fuse_pages(
            &g,
            request(vec![PageChannel {
                channel_id: "index".into(),
                weight: 1.0,
                candidates: vec![
                    good.clone(),
                    PageCandidate {
                        page_number: 99,
                        sha256: good.sha256.clone(),
                    },
                    PageCandidate {
                        page_number: 2,
                        sha256: "0".repeat(64),
                    },
                    good,
                ],
            }]),
        )
        .unwrap();
        assert_eq!(order(&fusion), vec![1]);
        let reasons = fusion
            .rejected
            .iter()
            .map(|r| (r.page_number, r.reason))
            .collect::<Vec<_>>();
        assert_eq!(
            reasons,
            vec![
                (1, RejectionReason::DuplicateWithinChannel),
                (2, RejectionReason::OriginalDigestMismatch),
                (99, RejectionReason::UnknownPage),
            ]
        );
    }

    #[test]
    fn the_limit_applies_after_ranking() {
        let g = graph("v1");
        let mut limited = request(vec![channel(&g, "lexical", 1.0, &[4, 3, 2, 1])]);
        limited.limit = 2;
        let fusion = fuse_pages(&g, limited).unwrap();
        assert_eq!(order(&fusion), vec![4, 3]);
        assert_eq!(fusion.requested_limit, 2);
    }

    #[test]
    fn result_is_independent_of_channel_order_and_the_digest_detects_edits() {
        let g = graph("v1");
        let a = channel(&g, "a", 1.0, &[1, 2]);
        let b = channel(&g, "b", 2.0, &[2, 3]);
        let forward = fuse_pages(&g, request(vec![a.clone(), b.clone()])).unwrap();
        let backward = fuse_pages(&g, request(vec![b, a])).unwrap();
        assert_eq!(forward, backward);
        assert!(forward.verify_digest());
        forward.verify_against(&g).unwrap();

        let mut edited = forward.clone();
        edited.pages[0].rrf_score += 1.0;
        assert!(!edited.verify_digest());
        assert!(edited.verify_against(&g).is_err());
    }

    #[test]
    fn a_fusion_does_not_verify_against_another_document_version() {
        let fusion = {
            let g = graph("v1");
            fuse_pages(&g, request(vec![channel(&g, "a", 1.0, &[1, 2])])).unwrap()
        };
        assert!(fusion.verify_against(&graph("v2")).is_err());
        fusion.verify_against(&graph("v1")).unwrap();
    }

    #[test]
    fn invalid_requests_fail_closed() {
        let g = graph("v1");
        let valid = channel(&g, "a", 1.0, &[1]);
        let cases: Vec<PageFusionRequest> = vec![
            request(vec![]),
            PageFusionRequest {
                rrf_k: 0,
                ..request(vec![valid.clone()])
            },
            PageFusionRequest {
                limit: 0,
                ..request(vec![valid.clone()])
            },
            PageFusionRequest {
                query_id: "  ".into(),
                ..request(vec![valid.clone()])
            },
            request(vec![valid.clone(), valid.clone()]),
            request(vec![PageChannel {
                weight: 0.0,
                ..valid.clone()
            }]),
            request(vec![PageChannel {
                weight: f64::NAN,
                ..valid.clone()
            }]),
            request(vec![PageChannel {
                weight: -1.0,
                ..valid.clone()
            }]),
            request(vec![PageChannel {
                channel_id: String::new(),
                ..valid
            }]),
        ];
        for (index, case) in cases.into_iter().enumerate() {
            assert!(fuse_pages(&g, case).is_err(), "case {index}");
        }
    }

    #[test]
    fn a_ranked_phase_becomes_a_channel_and_one_channel_keeps_its_order() {
        let g = graph("v1");
        let phase = g
            .retrieve_phase(
                "obligación general",
                RetrievalBudget {
                    max_pages: 4,
                    max_characters: 1_000,
                    graph_hops: 1,
                },
            )
            .unwrap();
        assert!(phase.evidence.len() >= 2);
        let lexical = PageChannel::from_phase("lexical", 1.0, &phase);
        for (candidate, evidence) in lexical.candidates.iter().zip(&phase.evidence) {
            assert_eq!(candidate.page_number, evidence.page_number);
            assert_eq!(candidate.sha256, evidence.sha256);
        }
        let fusion = fuse_pages(&g, request(vec![lexical])).unwrap();
        let expected = phase
            .evidence
            .iter()
            .map(|e| e.page_number)
            .collect::<Vec<_>>();
        assert_eq!(order(&fusion), expected);
        assert!(fusion.rejected.is_empty());
    }

    #[test]
    fn fusion_roundtrips_through_json_and_rejects_unknown_fields() {
        let g = graph("v1");
        let fusion = fuse_pages(&g, request(vec![channel(&g, "a", 1.0, &[1, 2])])).unwrap();
        let text = serde_json::to_string(&fusion).unwrap();
        let restored: PageFusion = serde_json::from_str(&text).unwrap();
        assert_eq!(restored, fusion);
        restored.verify_against(&g).unwrap();
        let mut value: serde_json::Value = serde_json::from_str(&text).unwrap();
        value["unexpected"] = serde_json::json!(true);
        assert!(serde_json::from_value::<PageFusion>(value).is_err());
    }
}
