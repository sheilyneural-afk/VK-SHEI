//! Project-local Hugging Face token management.
//! Passwords are never accepted or persisted.

use anyhow::{Context, Result, bail, ensure};
use std::{
    fs,
    io::{self, Read},
    path::PathBuf,
};

fn token_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("deploy/local-production/runtime/secrets/huggingface.token")
}

fn main() -> Result<()> {
    match std::env::args().nth(1).as_deref() {
        Some("store") => store(),
        Some("status") => {
            let path = token_path();
            println!(
                "{}",
                if path.is_file() {
                    "configured"
                } else {
                    "not-configured"
                }
            );
            Ok(())
        }
        Some("logout") => {
            let path = token_path();
            if path.exists() {
                fs::remove_file(&path).with_context(|| format!("eliminar {}", path.display()))?;
            }
            println!("credential-removed");
            Ok(())
        }
        Some("validate") => {
            let valid = kvslim::model_catalog::huggingface::validate_project_token()?;
            println!("{}", if valid { "valid" } else { "invalid" });
            if valid {
                Ok(())
            } else {
                bail!("la credencial de Hugging Face no es válida")
            }
        }
        _ => bail!(
            "uso: kvslim-auth <store|status|validate|logout>; store acepta un token HF de lectura por stdin"
        ),
    }
}

fn store() -> Result<()> {
    let mut token = String::new();
    io::stdin()
        .read_to_string(&mut token)
        .context("leer token")?;
    let token = token.trim();
    ensure!(!token.is_empty(), "el token está vacío");
    ensure!(
        token.starts_with("hf_"),
        "KVSLIM solo acepta tokens de Hugging Face (prefijo hf_)"
    );
    ensure!(
        !token.chars().any(char::is_whitespace),
        "el token contiene espacios"
    );
    let path = token_path();
    fs::create_dir_all(path.parent().unwrap())?;
    fs::write(&path, format!("{token}\n"))?;
    restrict_permissions(&path)?;
    println!("credential-stored-in-project");
    Ok(())
}

fn restrict_permissions(path: &std::path::Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}
