//! Lossless document-page authority plus compact deterministic routing data.
//!
//! This is deliberately not an LLM summary store. Micro-metadata and graph
//! edges select candidate pages; every returned item carries the original
//! text and its digest. A bounded phase reports deferred candidates instead
//! of pretending that a partial context covered the whole document.

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    io::Write,
    path::Path,
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DocumentPage {
    pub page_number: u32,
    pub section: Option<String>,
    pub original_text: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PageMicroMetadata {
    pub page_number: u32,
    pub character_count: usize,
    pub token_count: usize,
    pub terms: BTreeSet<String>,
    pub numbers: BTreeSet<String>,
    pub explicit_page_references: BTreeSet<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum DocumentEdgeKind {
    PreviousPage,
    NextPage,
    ExplicitReference,
    SameSection,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DocumentEdge {
    pub from_page: u32,
    pub to_page: u32,
    pub kind: DocumentEdgeKind,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DocumentGraph {
    pub document_id: String,
    pub version_hash: String,
    pages: BTreeMap<u32, DocumentPage>,
    metadata: BTreeMap<u32, PageMicroMetadata>,
    inverted_terms: BTreeMap<String, BTreeSet<u32>>,
    inverted_numbers: BTreeMap<String, BTreeSet<u32>>,
    edges: Vec<DocumentEdge>,
    neighbours: BTreeMap<u32, BTreeSet<u32>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RetrievalBudget {
    pub max_pages: usize,
    pub max_characters: usize,
    pub graph_hops: usize,
}

impl RetrievalBudget {
    pub fn validate(self) -> Result<Self> {
        ensure!(self.max_pages > 0, "retrieval max_pages must be positive");
        ensure!(
            self.max_characters > 0,
            "retrieval max_characters must be positive"
        );
        ensure!(self.graph_hops <= 8, "retrieval graph_hops must be <= 8");
        Ok(self)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvidencePage {
    pub page_number: u32,
    pub section: Option<String>,
    pub original_text: String,
    pub sha256: String,
    pub routing_score: u64,
    pub graph_distance: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RetrievalPhase {
    pub query: String,
    pub evidence: Vec<EvidencePage>,
    pub deferred_candidate_pages: Vec<u32>,
    pub candidate_pages: usize,
    pub selected_characters: usize,
    pub complete_for_candidate_set: bool,
    pub original_text_authority: bool,
}

impl DocumentGraph {
    pub fn build(
        document_id: impl Into<String>,
        version_hash: impl Into<String>,
        mut pages: Vec<(u32, Option<String>, String)>,
    ) -> Result<Self> {
        let document_id = document_id.into();
        let version_hash = version_hash.into();
        ensure!(
            !document_id.trim().is_empty(),
            "document_id cannot be empty"
        );
        ensure!(
            !version_hash.trim().is_empty(),
            "version_hash cannot be empty"
        );
        ensure!(!pages.is_empty(), "document must contain at least one page");
        pages.sort_by_key(|(number, _, _)| *number);

        let mut authority = BTreeMap::new();
        let mut metadata = BTreeMap::new();
        let mut inverted_terms = BTreeMap::<String, BTreeSet<u32>>::new();
        let mut inverted_numbers = BTreeMap::<String, BTreeSet<u32>>::new();
        for (page_number, section, original_text) in pages {
            ensure!(page_number > 0, "page numbers are one-based");
            ensure!(
                !original_text.trim().is_empty(),
                "page {page_number} has no text"
            );
            ensure!(
                !authority.contains_key(&page_number),
                "duplicate page {page_number}"
            );
            let sha256 = sha256_hex(original_text.as_bytes());
            let micro = micro_metadata(page_number, &original_text);
            for term in &micro.terms {
                inverted_terms
                    .entry(term.clone())
                    .or_default()
                    .insert(page_number);
            }
            for number in &micro.numbers {
                inverted_numbers
                    .entry(number.clone())
                    .or_default()
                    .insert(page_number);
            }
            authority.insert(
                page_number,
                DocumentPage {
                    page_number,
                    section,
                    original_text,
                    sha256,
                },
            );
            metadata.insert(page_number, micro);
        }

        let mut graph = Self {
            document_id,
            version_hash,
            pages: authority,
            metadata,
            inverted_terms,
            inverted_numbers,
            edges: Vec::new(),
            neighbours: BTreeMap::new(),
        };
        graph.build_edges();
        Ok(graph)
    }

    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    pub fn page(&self, page_number: u32) -> Option<&DocumentPage> {
        self.pages.get(&page_number)
    }

    pub fn page_metadata(&self, page_number: u32) -> Option<&PageMicroMetadata> {
        self.metadata.get(&page_number)
    }

    pub fn edges(&self) -> &[DocumentEdge] {
        &self.edges
    }

    pub fn verify_original_authority(&self) -> Result<()> {
        for page in self.pages.values() {
            ensure!(
                sha256_hex(page.original_text.as_bytes()) == page.sha256,
                "document page {} digest mismatch",
                page.page_number
            );
        }
        Ok(())
    }

    /// Atomically persists the complete original-page authority and routing
    /// metadata. A crash before `persist` leaves the previous file unchanged.
    pub fn save_atomic(&self, path: impl AsRef<Path>) -> Result<()> {
        self.verify_original_authority()?;
        let path = path.as_ref();
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create document graph directory {}", parent.display()))?;
        let mut temporary = tempfile::NamedTempFile::new_in(parent)
            .with_context(|| format!("create temporary graph in {}", parent.display()))?;
        serde_json::to_writer(&mut temporary, self).context("serialize document graph")?;
        temporary.flush().context("flush document graph")?;
        temporary
            .as_file()
            .sync_all()
            .context("sync document graph")?;
        temporary
            .persist(path)
            .map_err(|error| error.error)
            .with_context(|| format!("atomically persist document graph {}", path.display()))?;
        Ok(())
    }

    pub fn load_verified(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .with_context(|| format!("read document graph {}", path.display()))?;
        let graph: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("parse document graph {}", path.display()))?;
        graph.verify_original_authority()?;
        let pages = graph
            .pages
            .values()
            .map(|page| {
                (
                    page.page_number,
                    page.section.clone(),
                    page.original_text.clone(),
                )
            })
            .collect();
        // Routing metadata is acceleration data, never authority. Rebuild it
        // from the verified original pages so edited/stale indexes cannot hide
        // evidence after restart.
        Self::build(graph.document_id, graph.version_hash, pages)
    }

    pub fn retrieve_phase(&self, query: &str, budget: RetrievalBudget) -> Result<RetrievalPhase> {
        let budget = budget.validate()?;
        ensure!(!query.trim().is_empty(), "retrieval query cannot be empty");
        self.verify_original_authority()?;
        let query_micro = micro_metadata(1, query);
        let normalized_query = normalize_text(query);
        let mut scores = BTreeMap::<u32, u64>::new();
        for term in &query_micro.terms {
            if let Some(pages) = self.inverted_terms.get(term) {
                let inverse_frequency = 10_000_u64 / pages.len() as u64;
                for page in pages {
                    *scores.entry(*page).or_default() += inverse_frequency.max(1);
                }
            }
        }
        for number in &query_micro.numbers {
            if let Some(pages) = self.inverted_numbers.get(number) {
                for page in pages {
                    *scores.entry(*page).or_default() += 20_000;
                }
            }
        }
        for page in self.pages.values() {
            if normalize_text(&page.original_text).contains(&normalized_query) {
                *scores.entry(page.page_number).or_default() += 50_000;
            }
        }

        // If deterministic metadata found nothing, fail open to every page.
        // The bounded phase will expose the remainder as deferred evidence.
        if scores.is_empty() {
            scores.extend(self.pages.keys().map(|page| (*page, 1)));
        }
        let seeds = scores.keys().copied().collect::<Vec<_>>();
        let mut distance = seeds
            .iter()
            .map(|page| (*page, 0usize))
            .collect::<BTreeMap<_, _>>();
        let mut queue = VecDeque::from(seeds);
        while let Some(page) = queue.pop_front() {
            let current = distance[&page];
            if current >= budget.graph_hops {
                continue;
            }
            if let Some(next) = self.neighbours.get(&page) {
                for neighbour in next {
                    if distance.contains_key(neighbour) {
                        continue;
                    }
                    let next_distance = current + 1;
                    distance.insert(*neighbour, next_distance);
                    // Graph expansion is useful context but never outranks an
                    // exact lexical/number match by itself.
                    scores.entry(*neighbour).or_insert(
                        100_u64.saturating_sub(u64::try_from(next_distance).unwrap_or(100)),
                    );
                    queue.push_back(*neighbour);
                }
            }
        }

        let mut candidates = scores
            .into_iter()
            .map(|(page, score)| (page, score, *distance.get(&page).unwrap_or(&0)))
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| {
            right
                .1
                .cmp(&left.1)
                .then(left.2.cmp(&right.2))
                .then(left.0.cmp(&right.0))
        });
        let candidate_pages = candidates.len();
        let mut evidence = Vec::new();
        let mut deferred = Vec::new();
        let mut selected_characters = 0usize;
        for (page_number, score, graph_distance) in candidates {
            let page = self
                .pages
                .get(&page_number)
                .context("candidate page lost from document authority")?;
            let chars = page.original_text.chars().count();
            let fits = evidence.len() < budget.max_pages
                && selected_characters
                    .checked_add(chars)
                    .is_some_and(|total| total <= budget.max_characters);
            if fits {
                selected_characters += chars;
                evidence.push(EvidencePage {
                    page_number,
                    section: page.section.clone(),
                    original_text: page.original_text.clone(),
                    sha256: page.sha256.clone(),
                    routing_score: score,
                    graph_distance,
                });
            } else {
                deferred.push(page_number);
            }
        }
        ensure!(
            !evidence.is_empty(),
            "retrieval budget cannot fit any candidate page"
        );
        Ok(RetrievalPhase {
            query: query.to_owned(),
            evidence,
            deferred_candidate_pages: deferred.clone(),
            candidate_pages,
            selected_characters,
            complete_for_candidate_set: deferred.is_empty(),
            original_text_authority: true,
        })
    }

    fn build_edges(&mut self) {
        let page_numbers = self.pages.keys().copied().collect::<Vec<_>>();
        for pair in page_numbers.windows(2) {
            self.add_edge(pair[0], pair[1], DocumentEdgeKind::NextPage);
            self.add_edge(pair[1], pair[0], DocumentEdgeKind::PreviousPage);
        }
        let sections = self
            .pages
            .values()
            .filter_map(|page| page.section.as_ref().map(|s| (s.clone(), page.page_number)))
            .fold(
                BTreeMap::<String, Vec<u32>>::new(),
                |mut map, (section, page)| {
                    map.entry(section).or_default().push(page);
                    map
                },
            );
        for pages in sections.values() {
            for pair in pages.windows(2) {
                self.add_edge(pair[0], pair[1], DocumentEdgeKind::SameSection);
                self.add_edge(pair[1], pair[0], DocumentEdgeKind::SameSection);
            }
        }
        let explicit = self
            .metadata
            .values()
            .flat_map(|meta| {
                meta.explicit_page_references
                    .iter()
                    .map(move |target| (meta.page_number, *target))
            })
            .collect::<Vec<_>>();
        for (from, to) in explicit {
            if self.pages.contains_key(&to) && from != to {
                self.add_edge(from, to, DocumentEdgeKind::ExplicitReference);
                self.add_edge(to, from, DocumentEdgeKind::ExplicitReference);
            }
        }
    }

    fn add_edge(&mut self, from_page: u32, to_page: u32, kind: DocumentEdgeKind) {
        self.edges.push(DocumentEdge {
            from_page,
            to_page,
            kind,
        });
        self.neighbours
            .entry(from_page)
            .or_default()
            .insert(to_page);
    }
}

fn micro_metadata(page_number: u32, text: &str) -> PageMicroMetadata {
    let tokens = tokenize(text);
    let mut terms = BTreeSet::new();
    let mut numbers = BTreeSet::new();
    let mut references = BTreeSet::new();
    for (index, token) in tokens.iter().enumerate() {
        if token.chars().any(|c| c.is_ascii_digit()) {
            numbers.insert(token.clone());
        }
        if token.len() >= 3 && !is_stopword(token) {
            terms.insert(token.clone());
        }
        if matches!(token.as_str(), "pagina" | "página" | "page")
            && let Some(next) = tokens.get(index + 1)
            && let Ok(target) = next.parse::<u32>()
            && target > 0
        {
            references.insert(target);
        }
    }
    PageMicroMetadata {
        page_number,
        character_count: text.chars().count(),
        token_count: tokens.len(),
        terms,
        numbers,
        explicit_page_references: references,
    }
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(|token| token.to_lowercase())
        .collect()
}

fn normalize_text(text: &str) -> String {
    tokenize(text).join(" ")
}

fn is_stopword(term: &str) -> bool {
    matches!(
        term,
        "con"
            | "del"
            | "desde"
            | "el"
            | "ella"
            | "en"
            | "esta"
            | "este"
            | "la"
            | "las"
            | "los"
            | "para"
            | "por"
            | "que"
            | "the"
            | "una"
            | "uno"
            | "unos"
            | "y"
    )
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> DocumentGraph {
        DocumentGraph::build(
            "contract",
            "v1",
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

    #[test]
    fn exact_unexpected_detail_returns_original_page_and_digest() {
        let graph = sample();
        let phase = graph
            .retrieve_phase(
                "ZK-991",
                RetrievalBudget {
                    max_pages: 2,
                    max_characters: 500,
                    graph_hops: 0,
                },
            )
            .unwrap();
        assert_eq!(phase.evidence[0].page_number, 4);
        assert!(phase.evidence[0].original_text.contains("ZK-991"));
        assert_eq!(
            phase.evidence[0].sha256,
            sha256_hex(phase.evidence[0].original_text.as_bytes())
        );
        assert!(phase.original_text_authority);
    }

    #[test]
    fn explicit_graph_reference_expands_without_replacing_evidence() {
        let graph = sample();
        let phase = graph
            .retrieve_phase(
                "obligación general",
                RetrievalBudget {
                    max_pages: 4,
                    max_characters: 1_000,
                    graph_hops: 1,
                },
            )
            .unwrap();
        let pages = phase
            .evidence
            .iter()
            .map(|page| page.page_number)
            .collect::<BTreeSet<_>>();
        assert!(pages.contains(&1));
        assert!(pages.contains(&2));
        assert!(
            phase
                .evidence
                .iter()
                .all(|page| !page.original_text.is_empty())
        );
    }

    #[test]
    fn bounded_phase_exposes_deferred_candidates_instead_of_false_coverage() {
        let graph = sample();
        let phase = graph
            .retrieve_phase(
                "contenido que no existe",
                RetrievalBudget {
                    max_pages: 1,
                    max_characters: 100,
                    graph_hops: 0,
                },
            )
            .unwrap();
        assert_eq!(phase.candidate_pages, 4);
        assert_eq!(phase.evidence.len(), 1);
        assert_eq!(phase.deferred_candidate_pages.len(), 3);
        assert!(!phase.complete_for_candidate_set);
    }

    #[test]
    fn corrupted_original_page_fails_before_retrieval() {
        let mut graph = sample();
        graph
            .pages
            .get_mut(&3)
            .unwrap()
            .original_text
            .push_str(" alterado");
        assert!(graph.verify_original_authority().is_err());
        assert!(
            graph
                .retrieve_phase(
                    "excepción",
                    RetrievalBudget {
                        max_pages: 2,
                        max_characters: 500,
                        graph_hops: 1,
                    }
                )
                .is_err()
        );
    }

    #[test]
    fn atomic_persistence_roundtrip_reopens_original_authority() {
        let mut graph = sample();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("contract.document-graph.json");
        // Simulate stale/corrupted acceleration metadata. Reopen must rebuild
        // it from original pages rather than accepting false negative routes.
        graph.inverted_terms.clear();
        graph.neighbours.clear();
        graph.save_atomic(&path).unwrap();
        let reopened = DocumentGraph::load_verified(&path).unwrap();
        assert_eq!(reopened.page_count(), 4);
        assert_eq!(reopened.page(4), graph.page(4));
        let phase = reopened
            .retrieve_phase(
                "38 millones",
                RetrievalBudget {
                    max_pages: 1,
                    max_characters: 200,
                    graph_hops: 0,
                },
            )
            .unwrap();
        assert_eq!(phase.evidence[0].page_number, 1);
    }
}
