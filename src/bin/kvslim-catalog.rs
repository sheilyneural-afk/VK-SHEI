//! Explicit optional discovery. It never imports or executes a model.

use anyhow::{Context, Result, bail};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let output = match args.next().as_deref() {
        Some("search") => {
            let query = args
                .next()
                .context("uso: kvslim-catalog search <texto> [limit]")?;
            let limit = args
                .next()
                .map(|value| value.parse())
                .transpose()
                .context("limit inválido")?
                .unwrap_or(10);
            kvslim::model_catalog::huggingface::search(&query, limit)?
        }
        Some("inspect") => {
            let id = args
                .next()
                .context("uso: kvslim-catalog inspect <org/model>")?;
            vec![kvslim::model_catalog::huggingface::inspect(&id)?]
        }
        _ => bail!("uso: kvslim-catalog <search|inspect> ..."),
    };
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}
