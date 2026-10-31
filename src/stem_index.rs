//! Stem-level BM25 over the original pages of a [`DocumentGraph`].
//!
//! Natural-language questions rarely repeat the document's exact word forms
//! ("descartar" against "descarte", "evaluar" against "evaluación"). This index
//! conflates inflections with a light deterministic stemmer, drops the
//! function words of a question, and can optionally add pseudo-relevance
//! feedback: the most characteristic stems of the first pass's top pages are
//! added to the query with a reduced weight. Everything is derived from the
//! document itself; there is no model, no dictionary and no network.
//!
//! It does not understand synonyms: "gráfica" and "GPU" stay unrelated unless
//! the document itself uses them together. The index stores no text and is
//! never an authority; evidence always comes from the graph's original pages.

use crate::document_graph::{DocumentGraph, tokenize};
use std::collections::{BTreeMap, HashMap};

const BM25_K1: f64 = 1.2;
const BM25_B: f64 = 0.75;
/// Weight of a feedback term relative to an original query term (1.0).
const EXPANSION_WEIGHT: f64 = 0.3;
const FEEDBACK_PAGES: usize = 3;

/// Accent-folded function words of questions and prose.
const FUNCTION_WORDS: &[&str] = &[
    "que", "cual", "cuales", "cuanto", "cuantos", "cuanta", "cuantas", "como", "cuando", "donde",
    "quien", "quienes", "hay", "ser", "son", "esta", "estan", "este", "estos", "estas", "hacer",
    "hace", "hacen", "puede", "pueden", "poder", "debe", "deben", "tiene", "tienen", "tener",
    "sobre", "entre", "hasta", "desde", "tambien", "pero", "sino", "aunque", "cada", "otro",
    "otra", "otros", "otras", "muy", "mas", "menos", "solo", "ese", "esa", "eso", "esos", "esas",
    "mismo", "misma", "algun", "alguna", "ningun", "ninguna", "todo", "toda", "todos", "todas",
    "durante", "antes", "despues", "ahora", "con", "del", "los", "las", "una", "uno", "unos",
    "unas", "por", "para", "sin", "fue", "han", "ella", "ellos", "ellas", "cuya", "cuyo", "the",
    "and", "what", "which", "how", "does", "are", "was", "were", "with", "from", "that", "this",
    "have", "has", "for", "not", "can",
];

fn fold_accents(token: &str) -> String {
    token
        .chars()
        .map(|character| match character {
            'á' | 'à' | 'ä' | 'â' => 'a',
            'é' | 'è' | 'ë' | 'ê' => 'e',
            'í' | 'ì' | 'ï' | 'î' => 'i',
            'ó' | 'ò' | 'ö' | 'ô' => 'o',
            'ú' | 'ù' | 'ü' | 'û' => 'u',
            other => other,
        })
        .collect()
}

/// Light, deterministic suffix stripping for Spanish and English. Conflates
/// inflections and common derivations and never alters tokens with digits.
pub fn light_stem(token: &str) -> String {
    let folded = fold_accents(&token.to_lowercase());
    if folded.chars().any(|character| character.is_ascii_digit()) {
        return folded;
    }
    const DERIVATIONAL: &[&str] = &[
        "amientos", "imientos", "aciones", "uciones", "adores", "adoras", "idades", "amiento",
        "imiento", "ancias", "logias", "acion", "ucion", "mente", "idad", "ables", "ibles", "ador",
        "adora", "able", "ible", "ings", "ing",
    ];
    const INFLECTIONAL: &[&str] = &[
        "ando", "iendo", "ados", "adas", "idos", "idas", "ante", "ar", "er", "ir", "as", "es",
        "os", "an", "en", "ed", "ly", "a", "e", "o", "s",
    ];
    for suffixes in [DERIVATIONAL, INFLECTIONAL] {
        for suffix in suffixes {
            if let Some(stem) = folded.strip_suffix(suffix)
                && stem.chars().count() >= 4
            {
                return stem.to_owned();
            }
        }
    }
    folded
}

fn is_signal_token(token: &str) -> bool {
    let folded = fold_accents(token);
    folded.chars().any(|character| character.is_ascii_digit())
        || (folded.chars().count() >= 3 && !FUNCTION_WORDS.contains(&folded.as_str()))
}

fn query_stems(query: &str) -> Vec<String> {
    let mut stems = Vec::<String>::new();
    for token in tokenize(query) {
        if is_signal_token(&token) {
            let stem = light_stem(&token);
            if !stems.contains(&stem) {
                stems.push(stem);
            }
        }
    }
    stems
}

struct IndexedPage {
    page_number: u32,
    length: f64,
    frequencies: BTreeMap<String, u32>,
}

/// Stem statistics of the original pages. Build once per graph and reuse it
/// across queries.
pub struct StemIndex {
    pages: Vec<IndexedPage>,
    document_frequency: BTreeMap<String, f64>,
    average_length: f64,
}

impl StemIndex {
    pub fn build(graph: &DocumentGraph) -> Self {
        let mut pages = Vec::new();
        let mut document_frequency = BTreeMap::<String, f64>::new();
        // Text repeats a small vocabulary; stem each distinct token once.
        let mut stems = HashMap::<String, Option<String>>::new();
        for page in graph.pages() {
            let tokens = tokenize(&page.original_text);
            let mut counts = HashMap::<&str, u32>::new();
            for token in &tokens {
                *counts.entry(token.as_str()).or_default() += 1;
            }
            let mut frequencies = BTreeMap::<String, u32>::new();
            for (token, count) in counts {
                let stem = match stems.get(token) {
                    Some(stem) => stem.clone(),
                    None => {
                        let stem = is_signal_token(token).then(|| light_stem(token));
                        stems.insert(token.to_owned(), stem.clone());
                        stem
                    }
                };
                if let Some(stem) = stem {
                    *frequencies.entry(stem).or_default() += count;
                }
            }
            for stem in frequencies.keys() {
                *document_frequency.entry(stem.clone()).or_default() += 1.0;
            }
            pages.push(IndexedPage {
                page_number: page.page_number,
                length: tokens.len() as f64,
                frequencies,
            });
        }
        let total = pages.iter().map(|page| page.length).sum::<f64>();
        let average_length = if total > 0.0 {
            total / pages.len() as f64
        } else {
            1.0
        };
        Self {
            pages,
            document_frequency,
            average_length,
        }
    }

    fn idf(&self, stem: &str) -> f64 {
        let pages = self.pages.len() as f64;
        let df = self.document_frequency.get(stem).copied().unwrap_or(0.0);
        (1.0 + (pages - df + 0.5) / (df + 0.5)).ln()
    }

    fn score(&self, terms: &[(String, f64)]) -> Vec<(u32, f64)> {
        self.pages
            .iter()
            .filter_map(|page| {
                let mut score = 0.0;
                for (stem, weight) in terms {
                    if let Some(&frequency) = page.frequencies.get(stem) {
                        let tf = f64::from(frequency);
                        let normalization =
                            1.0 - BM25_B + BM25_B * page.length / self.average_length;
                        score += weight * self.idf(stem) * tf * (BM25_K1 + 1.0)
                            / (tf + BM25_K1 * normalization);
                    }
                }
                (score > 0.0).then_some((page.page_number, score))
            })
            .collect()
    }

    fn rank(mut scored: Vec<(u32, f64)>, limit: usize) -> Vec<(u32, f64)> {
        scored.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        scored.truncate(limit);
        scored
    }

    /// Best pages first, ties by page number; only pages with a positive score.
    /// With `expansion_terms > 0` a second pass adds that many feedback stems.
    pub fn search(&self, query: &str, limit: usize, expansion_terms: usize) -> Vec<(u32, f64)> {
        let original = query_stems(query);
        if original.is_empty() {
            return Vec::new();
        }
        let mut terms = original
            .iter()
            .map(|stem| (stem.clone(), 1.0))
            .collect::<Vec<_>>();
        let first = Self::rank(self.score(&terms), usize::MAX);
        if expansion_terms == 0 || first.is_empty() {
            return Self::rank(first, limit);
        }
        let mut feedback = BTreeMap::<&str, f64>::new();
        for &(page_number, _) in first.iter().take(FEEDBACK_PAGES) {
            if let Some(page) = self
                .pages
                .iter()
                .find(|page| page.page_number == page_number)
            {
                for (stem, &frequency) in &page.frequencies {
                    if original.contains(stem) || stem.chars().count() < 4 {
                        continue;
                    }
                    *feedback.entry(stem.as_str()).or_default() +=
                        f64::from(frequency) / page.length.max(1.0) * self.idf(stem);
                }
            }
        }
        let mut candidates = feedback.into_iter().collect::<Vec<_>>();
        candidates
            .sort_by(|left, right| right.1.total_cmp(&left.1).then_with(|| left.0.cmp(right.0)));
        terms.extend(
            candidates
                .into_iter()
                .take(expansion_terms)
                .map(|(stem, _)| (stem.to_owned(), EXPANSION_WEIGHT)),
        );
        Self::rank(self.score(&terms), limit)
    }
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

    #[test]
    fn inflections_and_derivations_share_a_stem() {
        assert_eq!(light_stem("descartar"), light_stem("descarte"));
        assert_eq!(light_stem("descarte"), light_stem("descartes"));
        assert_eq!(light_stem("evaluar"), light_stem("evaluación"));
        assert_eq!(light_stem("restauración"), light_stem("restauraciones"));
        assert_ne!(light_stem("gráfica"), light_stem("gpu"));
    }

    #[test]
    fn numbers_short_words_and_accents_are_handled_safely() {
        assert_eq!(light_stem("4096"), "4096");
        assert_eq!(light_stem("v1"), "v1");
        assert_eq!(light_stem("casa"), "casa");
        assert_eq!(light_stem("Árbol"), "arbol");
    }

    #[test]
    fn a_question_matches_other_word_forms_and_ignores_interrogatives() {
        let graph = build(&[
            "El envío tarda tres días hábiles.",
            "Se evalúan los candidatos y se descarta el peor.",
            "El precio total es de 40 euros.",
        ]);
        let index = StemIndex::build(&graph);
        let ranking = index.search("¿Qué hay que evaluar y descartar?", 10, 0);
        assert_eq!(pages_of(&ranking), vec![2]);
    }

    #[test]
    fn a_query_made_only_of_function_words_returns_nothing() {
        let graph = build(&["alfa beta", "gamma delta"]);
        let index = StemIndex::build(&graph);
        assert!(index.search("¿Qué hay para cuando?", 10, 3).is_empty());
    }

    #[test]
    fn results_are_deterministic_and_ties_break_by_page_number() {
        let graph = build(&["alfa beta", "alfa beta", "alfa beta"]);
        let index = StemIndex::build(&graph);
        let first = index.search("alfa", 10, 2);
        assert_eq!(pages_of(&first), vec![1, 2, 3]);
        assert_eq!(first, index.search("alfa", 10, 2));
    }

    #[test]
    fn feedback_surfaces_a_page_that_shares_vocabulary_but_not_the_query_term() {
        let graph = build(&[
            "garantía cubre defectos fabricación defectos fabricación",
            "defectos fabricación revisión taller",
            "envío tarda tres días",
        ]);
        let index = StemIndex::build(&graph);
        assert_eq!(pages_of(&index.search("garantía", 10, 0)), vec![1]);
        let expanded = pages_of(&index.search("garantía", 10, 4));
        assert_eq!(expanded.first(), Some(&1));
        assert!(expanded.contains(&2));
        assert!(!expanded.contains(&3));
    }
}
