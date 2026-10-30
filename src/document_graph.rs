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

type RankedDocumentCandidate = (u32, u64, usize);
type RankedDocumentCandidates = (Vec<RankedDocumentCandidate>, usize);

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
    /// Sentence/line hyperedges: every span connects all terms and numbers
    /// that coexist in one exact original-text region without materializing
    /// the quadratic term-to-term edge set.
    #[serde(default)]
    pub graph_spans: Vec<GraphMicroSpan>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct GraphMicroSpan {
    pub span_index: u32,
    pub byte_start: usize,
    pub byte_end: usize,
    pub sha256: String,
    pub terms: BTreeSet<String>,
    pub numbers: BTreeSet<String>,
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

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalScope {
    /// Select the pages with the strongest joint deterministic evidence and
    /// graph context. Completion applies only to that focused candidate set.
    Focused,
    /// Retain every page matching any deterministic query signal. This is
    /// broader, but still does not certify semantic exhaustiveness.
    Exhaustive,
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
    pub scope: RetrievalScope,
    pub evidence: Vec<EvidencePage>,
    pub deferred_candidate_pages: Vec<u32>,
    pub candidate_pages: usize,
    pub broad_match_pages: usize,
    pub selected_characters: usize,
    pub complete_for_candidate_set: bool,
    pub original_text_authority: bool,
    pub semantic_exhaustiveness_certified: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CitedEvidenceContext {
    pub question: String,
    pub prompt: String,
    pub evidence_pages: Vec<u32>,
    pub evidence_characters: usize,
    pub complete_for_candidate_set: bool,
    pub deferred_candidate_pages: Vec<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct FullDocumentContext {
    pub text: String,
    pub pages: usize,
    pub original_characters: usize,
    pub assembled_sha256: String,
}

impl RetrievalPhase {
    /// Builds a model-facing context without changing or summarizing evidence.
    /// Exact tokenizer accounting belongs to the selected model runtime.
    pub fn assemble_cited_context(&self) -> Result<CitedEvidenceContext> {
        ensure!(!self.evidence.is_empty(), "cannot assemble empty evidence");
        let mut prompt = String::from(
            "Responde la pregunta usando exclusivamente las evidencias originales citadas.\n\
             Distingue hechos textuales de inferencias. Escribe primero la respuesta y añade después [página N].\n\
             La cita no sustituye la respuesta y no debes copiar el SHA256.\n\
             Si las evidencias no bastan, responde exactamente EVIDENCIA_INSUFICIENTE.\n\n",
        );
        prompt.push_str("PREGUNTA:\n");
        prompt.push_str(&self.query);
        prompt.push_str("\n\nEVIDENCIAS ORIGINALES:\n");
        let mut evidence_pages = Vec::with_capacity(self.evidence.len());
        for evidence in &self.evidence {
            evidence_pages.push(evidence.page_number);
            prompt.push_str(&format!(
                "\n[PÁGINA {} | SHA256 {} | SECCIÓN {}]\n{}\n",
                evidence.page_number,
                evidence.sha256,
                evidence.section.as_deref().unwrap_or("sin sección"),
                evidence.original_text
            ));
        }
        prompt.push_str("\nPREGUNTA A RESPONDER DESPUÉS DE LEER LAS EVIDENCIAS:\n");
        prompt.push_str(&self.query);
        prompt.push_str(
            "\nSi una evidencia contiene directamente la respuesta, responde de forma concisa y cítala.\n",
        );
        if !self.complete_for_candidate_set {
            prompt.push_str(
                "\nAVISO: existen páginas candidatas diferidas. No afirmes cobertura exhaustiva.\n",
            );
        }
        Ok(CitedEvidenceContext {
            question: self.query.clone(),
            prompt,
            evidence_pages,
            evidence_characters: self.selected_characters,
            complete_for_candidate_set: self.complete_for_candidate_set,
            deferred_candidate_pages: self.deferred_candidate_pages.clone(),
        })
    }
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

    pub fn graph_micro_span_count(&self) -> usize {
        self.metadata
            .values()
            .map(|metadata| metadata.graph_spans.len())
            .sum()
    }

    pub fn page(&self, page_number: u32) -> Option<&DocumentPage> {
        self.pages.get(&page_number)
    }

    pub fn page_metadata(&self, page_number: u32) -> Option<&PageMicroMetadata> {
        self.metadata.get(&page_number)
    }

    /// Assemble the complete verified document for one persistent Transformer
    /// prefix. This is not retrieval: every page is included in authority
    /// order and later questions reuse the resulting immutable BF16 K/V.
    pub fn full_document_context(&self) -> Result<FullDocumentContext> {
        self.verify_original_authority()?;
        let original_characters = self
            .pages
            .values()
            .map(|page| page.original_text.chars().count())
            .sum();
        let mut text = String::new();
        for page in self.pages.values() {
            text.push_str(&format!("\n[PÁGINA {}]\n", page.page_number));
            text.push_str(&page.original_text);
            text.push('\n');
        }
        let assembled_sha256 = sha256_hex(text.as_bytes());
        Ok(FullDocumentContext {
            text,
            pages: self.pages.len(),
            original_characters,
            assembled_sha256,
        })
    }

    pub fn full_evidence_phase(&self, query: &str) -> Result<RetrievalPhase> {
        ensure!(!query.trim().is_empty(), "retrieval query cannot be empty");
        self.verify_original_authority()?;
        let evidence = self
            .pages
            .values()
            .map(|page| EvidencePage {
                page_number: page.page_number,
                section: page.section.clone(),
                original_text: page.original_text.clone(),
                sha256: page.sha256.clone(),
                routing_score: 0,
                graph_distance: 0,
            })
            .collect::<Vec<_>>();
        let selected_characters = evidence
            .iter()
            .map(|page| page.original_text.chars().count())
            .sum();
        Ok(RetrievalPhase {
            query: query.to_owned(),
            scope: RetrievalScope::Exhaustive,
            candidate_pages: evidence.len(),
            broad_match_pages: evidence.len(),
            evidence,
            deferred_candidate_pages: Vec::new(),
            selected_characters,
            complete_for_candidate_set: true,
            original_text_authority: true,
            semantic_exhaustiveness_certified: false,
        })
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
            let metadata = self
                .metadata
                .get(&page.page_number)
                .context("page micro-metadata missing from document graph")?;
            for span in &metadata.graph_spans {
                let original = page
                    .original_text
                    .get(span.byte_start..span.byte_end)
                    .with_context(|| {
                        format!(
                            "page {} graph span {} has invalid UTF-8 byte boundaries",
                            page.page_number, span.span_index
                        )
                    })?;
                ensure!(
                    sha256_hex(original.as_bytes()) == span.sha256,
                    "page {} graph span {} digest mismatch",
                    page.page_number,
                    span.span_index
                );
            }
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
        self.retrieve_phase_with_scope(query, budget, RetrievalScope::Focused)
    }

    pub fn retrieve_phase_with_scope(
        &self,
        query: &str,
        budget: RetrievalBudget,
        scope: RetrievalScope,
    ) -> Result<RetrievalPhase> {
        let budget = budget.validate()?;
        let (candidates, broad_match_pages) =
            self.rank_candidates(query, budget.graph_hops, scope)?;
        self.materialize_candidate_phase(query, scope, &candidates, broad_match_pages, budget, 0)
            .map(|(phase, _)| phase)
    }

    /// Partitions one immutable ranking into bounded phases. The ranking and
    /// original-page authority are verified once; phases never reload or
    /// silently repeat a candidate. Completion becomes true only on the final
    /// phase.
    pub fn retrieve_phases_with_scope(
        &self,
        query: &str,
        budget: RetrievalBudget,
        scope: RetrievalScope,
    ) -> Result<Vec<RetrievalPhase>> {
        let budget = budget.validate()?;
        let (candidates, broad_match_pages) =
            self.rank_candidates(query, budget.graph_hops, scope)?;
        let mut phases = Vec::new();
        let mut offset = 0usize;
        while offset < candidates.len() {
            let (phase, next_offset) = self.materialize_candidate_phase(
                query,
                scope,
                &candidates,
                broad_match_pages,
                budget,
                offset,
            )?;
            ensure!(next_offset > offset, "retrieval phase made no progress");
            phases.push(phase);
            offset = next_offset;
        }
        ensure!(!phases.is_empty(), "retrieval produced no phases");
        Ok(phases)
    }

    fn rank_candidates(
        &self,
        query: &str,
        graph_hops: usize,
        scope: RetrievalScope,
    ) -> Result<RankedDocumentCandidates> {
        ensure!(!query.trim().is_empty(), "retrieval query cannot be empty");
        self.verify_original_authority()?;
        let query_micro = micro_metadata(1, query);
        let query_tokens = tokenize(query);
        let asks_for_quantity = query_tokens.iter().any(|token| {
            matches!(
                token.as_str(),
                "cuanto" | "cuantos" | "cuánta" | "cuántas" | "cuánto" | "cuántos"
            )
        });
        let normalized_query = normalize_text(query);
        let mut broad_scores = BTreeMap::<u32, u64>::new();
        let mut matched_terms = BTreeMap::<u32, usize>::new();
        for term in &query_micro.terms {
            if let Some(pages) = self.inverted_terms.get(term) {
                let inverse_frequency = 10_000_u64 / pages.len() as u64;
                for page in pages {
                    *broad_scores.entry(*page).or_default() += inverse_frequency.max(1);
                }
            }
        }
        // A sentence/line span acts as a compact graph hyperedge. Focused
        // ranking uses the strongest co-located query evidence, not merely
        // terms that happen to occur somewhere on the same coarse page.
        for metadata in self.metadata.values() {
            let strongest_span_match = metadata
                .graph_spans
                .iter()
                .map(|span| {
                    let term_matches = span.terms.intersection(&query_micro.terms).count();
                    term_matches
                        + usize::from(
                            asks_for_quantity && term_matches > 0 && !span.numbers.is_empty(),
                        )
                })
                .max()
                .unwrap_or_else(|| metadata.terms.intersection(&query_micro.terms).count());
            if strongest_span_match > 0 {
                matched_terms.insert(metadata.page_number, strongest_span_match);
                *broad_scores.entry(metadata.page_number).or_default() +=
                    u64::try_from(strongest_span_match)
                        .unwrap_or(u64::MAX)
                        .saturating_mul(1_000);
            }
        }
        let mut exact_number_pages = BTreeSet::new();
        for number in &query_micro.numbers {
            if let Some(pages) = self.inverted_numbers.get(number) {
                for page in pages {
                    *broad_scores.entry(*page).or_default() += 20_000;
                    exact_number_pages.insert(*page);
                }
            }
        }
        let mut exact_phrase_pages = BTreeSet::new();
        for page in self.pages.values() {
            if normalize_text(&page.original_text).contains(&normalized_query) {
                *broad_scores.entry(page.page_number).or_default() += 50_000;
                exact_phrase_pages.insert(page.page_number);
            }
        }

        // If deterministic metadata found nothing, fail open to every page.
        // The bounded phase will expose the remainder as deferred evidence.
        if broad_scores.is_empty() {
            broad_scores.extend(self.pages.keys().map(|page| (*page, 1)));
        }
        let broad_match_pages = broad_scores.len();
        let max_term_matches = matched_terms.values().copied().max().unwrap_or(0);
        let focused_pages = broad_scores
            .keys()
            .filter(|page| {
                exact_phrase_pages.contains(page)
                    || exact_number_pages.contains(page)
                    || (max_term_matches > 0
                        && matched_terms.get(page).copied().unwrap_or(0) == max_term_matches)
            })
            .copied()
            .collect::<BTreeSet<_>>();
        let mut scores = match scope {
            RetrievalScope::Focused if !focused_pages.is_empty() => focused_pages
                .into_iter()
                .map(|page| (page, broad_scores[&page]))
                .collect::<BTreeMap<_, _>>(),
            RetrievalScope::Focused | RetrievalScope::Exhaustive => broad_scores,
        };
        let seeds = scores.keys().copied().collect::<Vec<_>>();
        let mut distance = seeds
            .iter()
            .map(|page| (*page, 0usize))
            .collect::<BTreeMap<_, _>>();
        let mut queue = VecDeque::from(seeds);
        while let Some(page) = queue.pop_front() {
            let current = distance[&page];
            if current >= graph_hops {
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
        Ok((candidates, broad_match_pages))
    }

    fn materialize_candidate_phase(
        &self,
        query: &str,
        scope: RetrievalScope,
        candidates: &[(u32, u64, usize)],
        broad_match_pages: usize,
        budget: RetrievalBudget,
        offset: usize,
    ) -> Result<(RetrievalPhase, usize)> {
        ensure!(
            offset < candidates.len(),
            "retrieval phase offset is exhausted"
        );
        let candidate_pages = candidates.len();
        let mut evidence = Vec::new();
        let mut selected_characters = 0usize;
        let mut next_offset = offset;
        for &(page_number, score, graph_distance) in &candidates[offset..] {
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
                next_offset += 1;
            } else {
                break;
            }
        }
        ensure!(
            !evidence.is_empty(),
            "retrieval budget cannot fit any candidate page"
        );
        let deferred = candidates[next_offset..]
            .iter()
            .map(|(page, _, _)| *page)
            .collect::<Vec<_>>();
        Ok((
            RetrievalPhase {
                query: query.to_owned(),
                scope,
                evidence,
                deferred_candidate_pages: deferred.clone(),
                candidate_pages,
                broad_match_pages,
                selected_characters,
                complete_for_candidate_set: deferred.is_empty(),
                original_text_authority: true,
                semantic_exhaustiveness_certified: false,
            },
            next_offset,
        ))
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
        if is_number_token(token) {
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
        graph_spans: graph_micro_spans(text),
    }
}

fn graph_micro_spans(text: &str) -> Vec<GraphMicroSpan> {
    let mut ranges = Vec::new();
    let mut start = 0usize;
    for (index, character) in text.char_indices() {
        let end = index + character.len_utf8();
        if matches!(character, '.' | '?' | '!' | '\n') {
            if !text[start..end].trim().is_empty() {
                ranges.push((start, end));
            }
            start = end;
        }
    }
    if start < text.len() && !text[start..].trim().is_empty() {
        ranges.push((start, text.len()));
    }
    ranges
        .into_iter()
        .enumerate()
        .filter_map(|(index, (byte_start, byte_end))| {
            let original = &text[byte_start..byte_end];
            let tokens = tokenize(original);
            let terms = tokens
                .iter()
                .filter(|token| token.len() >= 3 && !is_stopword(token))
                .cloned()
                .collect::<BTreeSet<_>>();
            let numbers = tokens
                .iter()
                .filter(|token| is_number_token(token))
                .cloned()
                .collect::<BTreeSet<_>>();
            if terms.is_empty() && numbers.is_empty() {
                return None;
            }
            Some(GraphMicroSpan {
                span_index: u32::try_from(index).unwrap_or(u32::MAX),
                byte_start,
                byte_end,
                sha256: sha256_hex(original.as_bytes()),
                terms,
                numbers,
            })
        })
        .collect()
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

fn is_number_token(token: &str) -> bool {
    token.chars().any(|character| character.is_ascii_digit())
        || matches!(
            token,
            "cero"
                | "dos"
                | "tres"
                | "cuatro"
                | "cinco"
                | "seis"
                | "siete"
                | "ocho"
                | "nueve"
                | "diez"
                | "once"
                | "doce"
                | "trece"
                | "catorce"
                | "quince"
                | "veinte"
                | "treinta"
                | "cuarenta"
                | "cincuenta"
                | "cien"
                | "ciento"
                | "mil"
                | "millon"
                | "millón"
                | "millones"
        )
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub fn is_insufficient_evidence_answer(content: &str) -> bool {
    let normalized = content
        .chars()
        .filter(|character| character.is_alphanumeric())
        .flat_map(char::to_uppercase)
        .collect::<String>();
    normalized == "EVIDENCIAINSUFICIENTE"
}

pub fn has_substantive_document_answer(content: &str) -> bool {
    content
        .split(|character: char| !character.is_alphanumeric())
        .map(str::to_lowercase)
        .any(|token| {
            token.len() >= 2
                && !matches!(
                    token.as_str(),
                    "pagina" | "page" | "página" | "sha" | "sha256"
                )
                && !token.chars().all(|character| character.is_ascii_digit())
        })
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

    #[test]
    fn cited_context_preserves_every_selected_original_byte_and_partial_warning() {
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
        let context = phase.assemble_cited_context().unwrap();
        assert!(!context.complete_for_candidate_set);
        assert!(context.prompt.contains("páginas candidatas diferidas"));
        for evidence in &phase.evidence {
            assert!(context.prompt.contains(&evidence.original_text));
            assert!(context.prompt.contains(&evidence.sha256));
        }
        assert!(context.prompt.ends_with(
            "AVISO: existen páginas candidatas diferidas. No afirmes cobertura exhaustiva.\n"
        ));
    }

    #[test]
    fn full_evidence_phase_contains_every_page_in_order() {
        let graph = sample();
        let phase = graph.full_evidence_phase("resumen global").unwrap();
        assert_eq!(
            phase
                .evidence
                .iter()
                .map(|page| page.page_number)
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 4]
        );
        assert!(phase.complete_for_candidate_set);
        assert!(phase.deferred_candidate_pages.is_empty());
    }

    #[test]
    fn full_document_context_contains_every_original_page_once_in_order() {
        let graph = sample();
        let context = graph.full_document_context().unwrap();
        assert_eq!(context.pages, 4);
        assert_eq!(
            context.original_characters,
            (1..=4)
                .map(|page| graph.page(page).unwrap().original_text.chars().count())
                .sum::<usize>()
        );
        let mut previous = 0usize;
        for page_number in 1..=4 {
            let page = graph.page(page_number).unwrap();
            assert_eq!(context.text.matches(&page.original_text).count(), 1);
            let position = context.text.find(&page.original_text).unwrap();
            assert!(position >= previous);
            previous = position;
        }
        assert_eq!(
            context.assembled_sha256,
            sha256_hex(context.text.as_bytes())
        );
    }

    #[test]
    fn focused_scope_requires_the_strongest_joint_term_match() {
        let graph = DocumentGraph::build(
            "town",
            "v1",
            vec![
                (
                    1,
                    None,
                    "document-place era una población de dos mil habitantes.".to_owned(),
                ),
                (2, None, "El tren regresó a document-place.".to_owned()),
                (3, None, "Otros habitantes cerraron las puertas.".to_owned()),
            ],
        )
        .unwrap();
        let phase = graph
            .retrieve_phase(
                "¿Cuántos habitantes tenía document-place?",
                RetrievalBudget {
                    max_pages: 3,
                    max_characters: 1_000,
                    graph_hops: 0,
                },
            )
            .unwrap();
        assert_eq!(phase.scope, RetrievalScope::Focused);
        assert_eq!(phase.broad_match_pages, 3);
        assert_eq!(phase.candidate_pages, 1);
        assert_eq!(phase.evidence[0].page_number, 1);
        assert!(phase.complete_for_candidate_set);
        assert!(!phase.semantic_exhaustiveness_certified);
    }

    #[test]
    fn exhaustive_scope_retains_every_broad_deterministic_match() {
        let graph = DocumentGraph::build(
            "town",
            "v1",
            vec![
                (
                    1,
                    None,
                    "document-place era una población de dos mil habitantes.".to_owned(),
                ),
                (2, None, "El tren regresó a document-place.".to_owned()),
                (3, None, "Otros habitantes cerraron las puertas.".to_owned()),
            ],
        )
        .unwrap();
        let phase = graph
            .retrieve_phase_with_scope(
                "¿Cuántos habitantes tenía document-place?",
                RetrievalBudget {
                    max_pages: 3,
                    max_characters: 1_000,
                    graph_hops: 0,
                },
                RetrievalScope::Exhaustive,
            )
            .unwrap();
        assert_eq!(phase.scope, RetrievalScope::Exhaustive);
        assert_eq!(phase.broad_match_pages, 3);
        assert_eq!(phase.candidate_pages, 3);
        assert!(phase.complete_for_candidate_set);
        assert!(!phase.semantic_exhaustiveness_certified);
    }

    #[test]
    fn bounded_phases_preserve_one_ranking_without_repeating_candidates() {
        let graph = sample();
        let phases = graph
            .retrieve_phases_with_scope(
                "contenido que no existe",
                RetrievalBudget {
                    max_pages: 1,
                    max_characters: 500,
                    graph_hops: 0,
                },
                RetrievalScope::Exhaustive,
            )
            .unwrap();
        assert_eq!(phases.len(), 4);
        let pages = phases
            .iter()
            .flat_map(|phase| phase.evidence.iter().map(|page| page.page_number))
            .collect::<Vec<_>>();
        assert_eq!(pages, vec![1, 2, 3, 4]);
        assert_eq!(pages.iter().copied().collect::<BTreeSet<_>>().len(), 4);
        assert!(
            phases[..3]
                .iter()
                .all(|phase| !phase.complete_for_candidate_set)
        );
        assert!(phases[3].complete_for_candidate_set);
        assert_eq!(phases[0].deferred_candidate_pages, vec![2, 3, 4]);
        assert!(phases[3].deferred_candidate_pages.is_empty());
    }

    #[test]
    fn graph_micro_spans_resolve_to_exact_original_utf8_and_digest() {
        let graph = DocumentGraph::build(
            "utf8",
            "v1",
            vec![(
                1,
                None,
                "entity-a llegó al document_fixture. ¿Cuántos años tenía? 38 años.".to_owned(),
            )],
        )
        .unwrap();
        let page = graph.page(1).unwrap();
        let metadata = graph.page_metadata(1).unwrap();
        assert_eq!(metadata.graph_spans.len(), 3);
        for span in &metadata.graph_spans {
            let original = &page.original_text[span.byte_start..span.byte_end];
            assert_eq!(span.sha256, sha256_hex(original.as_bytes()));
        }
        graph.verify_original_authority().unwrap();
    }

    #[test]
    fn focused_ranking_prefers_terms_connected_by_one_graph_span() {
        let graph = DocumentGraph::build(
            "relations",
            "v1",
            vec![
                (
                    1,
                    None,
                    "document-place aparece aquí. Los habitantes aparecen en otra situación."
                        .to_owned(),
                ),
                (
                    2,
                    None,
                    "Los habitantes de document-place eran dos mil.".to_owned(),
                ),
            ],
        )
        .unwrap();
        let phase = graph
            .retrieve_phase(
                "habitantes document-place",
                RetrievalBudget {
                    max_pages: 1,
                    max_characters: 500,
                    graph_hops: 0,
                },
            )
            .unwrap();
        assert_eq!(phase.candidate_pages, 1);
        assert_eq!(phase.evidence[0].page_number, 2);
        assert!(phase.complete_for_candidate_set);
        assert_eq!(phase.broad_match_pages, 2);
    }

    #[test]
    fn quantity_question_prefers_related_span_with_number_words() {
        let graph = DocumentGraph::build(
            "quantity",
            "v1",
            vec![
                (
                    1,
                    None,
                    "document-place tenía dos mil habitantes.".to_owned(),
                ),
                (
                    2,
                    None,
                    "Los habitantes de document-place cambiaron su historia.".to_owned(),
                ),
            ],
        )
        .unwrap();
        let phase = graph
            .retrieve_phase(
                "¿Cuántos habitantes tenía document-place?",
                RetrievalBudget {
                    max_pages: 2,
                    max_characters: 500,
                    graph_hops: 0,
                },
            )
            .unwrap();
        assert_eq!(phase.broad_match_pages, 2);
        assert_eq!(phase.candidate_pages, 1);
        assert_eq!(phase.evidence[0].page_number, 1);
        let metadata = graph.page_metadata(1).unwrap();
        assert!(metadata.graph_spans[0].numbers.contains("dos"));
        assert!(metadata.graph_spans[0].numbers.contains("mil"));
    }
}
