//! Detect the local host and rank remote catalog candidates.
//! This command recommends only; it never downloads or activates a model.

use anyhow::{Context, Result, bail};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let query = args.next().unwrap_or_else(|| "local-inference".into());
    let limit = args
        .next()
        .map(|value| value.parse::<u16>())
        .transpose()
        .context("limit inválido")?
        .unwrap_or(20);
    if !(1..=100).contains(&limit) {
        bail!("limit debe estar entre 1 y 100");
    }
    let host = mslim::adapter_catalog::host_capabilities();
    let search_query = mslim::model_resolver::hardware_search_query(&host, &query);
    let candidates = mslim::model_catalog::huggingface::search_with_details(&search_query, limit)?;
    let recommendations = mslim::model_resolver::rank_candidates(&host, candidates);
    println!(
        "{}",
        serde_json::json!({
            "host": host,
            "recommendations": recommendations,
            "activation": "requires_local_import_and_real_runtime_validation"
        })
    );
    Ok(())
}
