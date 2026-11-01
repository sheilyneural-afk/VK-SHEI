//! Deterministic extractive compression: keeps whole source sentences, ranked by
//! salience, in original order. Never rewrites text.
//! Port of `sheily-execution-kernel-rs/src/linguistic_capabilities.rs`
//! (`compression_units`, `compression_salience`, `execute_language_compression`).

use std::collections::BTreeSet;

/// Split on sentence terminators, keeping the terminator with its sentence.
pub fn compression_units(source: &str) -> Vec<String> {
    let mut units = Vec::new();
    let mut start = 0usize;
    for (idx, ch) in source.char_indices() {
        if matches!(ch, '.' | '!' | '?' | '\n') {
            let end = idx + ch.len_utf8();
            let unit = source[start..end].trim();
            if !unit.is_empty() {
                units.push(unit.to_string());
            }
            start = end;
        }
    }
    let tail = source[start..].trim();
    if !tail.is_empty() {
        units.push(tail.to_string());
    }
    units
}

pub fn compression_salience(unit: &str, index: usize, total: usize) -> i64 {
    let words = unit
        .split_whitespace()
        .map(|word| word.trim_matches(|ch: char| !ch.is_alphanumeric()))
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>();
    let unique = words
        .iter()
        .map(|word| word.to_ascii_lowercase())
        .collect::<BTreeSet<_>>()
        .len() as i64;
    let evidence_terms = words
        .iter()
        .filter(|word| word.chars().any(|ch| ch.is_ascii_digit()))
        .count() as i64;
    let boundary = if index == 0 || index + 1 == total {
        24
    } else {
        0
    };
    boundary + unique * 2 + evidence_terms * 3 + (unit.chars().count().min(240) / 24) as i64
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Compressed {
    pub text: String,
    pub retained_units: usize,
    pub omitted_units: usize,
    pub output_chars: usize,
}

/// Compress `input` to at most `max_chars` characters. Returns `None` when the
/// input already fits (exact passthrough).
pub fn compress_extractive(
    input: &str,
    max_chars: usize,
) -> Result<Option<Compressed>, &'static str> {
    if input.chars().count() <= max_chars {
        return Ok(None);
    }
    let units = compression_units(input);
    if units.is_empty() {
        return Err("language_compression_no_units");
    }
    let mut ranked = units
        .iter()
        .enumerate()
        .map(|(index, unit)| (index, compression_salience(unit, index, units.len())))
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));

    let mut selected = BTreeSet::new();
    let mut used = 0usize;
    for (index, _) in ranked {
        let chars = units[index].chars().count();
        let separator = usize::from(!selected.is_empty());
        if used + separator + chars > max_chars {
            continue;
        }
        selected.insert(index);
        used += separator + chars;
    }
    if selected.is_empty() {
        return Err("language_compression_budget_too_small_for_complete_unit");
    }
    let text = selected
        .iter()
        .map(|index| units[*index].as_str())
        .collect::<Vec<_>>()
        .join(" ");
    Ok(Some(Compressed {
        text,
        retained_units: selected.len(),
        omitted_units: units.len() - selected.len(),
        output_chars: used,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_when_it_fits() {
        assert_eq!(compress_extractive("Hola.", 100).unwrap(), None);
    }

    #[test]
    fn keeps_whole_sentences_in_original_order() {
        let input = "Intro general sobre el tema. Dato 42 relevante aquí. Relleno que no aporta nada. Conclusión final importante.";
        let out = compress_extractive(input, 60).unwrap().unwrap();
        assert!(out.text.chars().count() <= 60);
        // Every retained sentence must be an exact source span.
        for unit in compression_units(&out.text) {
            assert!(input.contains(&unit));
        }
        assert!(out.retained_units >= 1);
    }

    #[test]
    fn budget_too_small_fails_closed() {
        let err = compress_extractive("Una frase bastante larga que no cabe.", 5).unwrap_err();
        assert_eq!(
            err,
            "language_compression_budget_too_small_for_complete_unit"
        );
    }
}
