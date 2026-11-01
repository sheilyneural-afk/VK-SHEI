//! Records the source revision in the binary so validation receipts can name
//! the commit that produced them. Missing git (for example a source tarball)
//! yields "unknown", never a guessed value.

use std::process::Command;

fn git(arguments: &[&str]) -> Option<String> {
    let output = Command::new("git").args(arguments).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/index");
    if let Some(reference) = git(&["rev-parse", "--symbolic-full-name", "HEAD"])
        && reference.starts_with("refs/")
    {
        println!("cargo:rerun-if-changed=.git/{reference}");
    }
    let commit = git(&["rev-parse", "HEAD"])
        .filter(|value| value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit()));
    // Untracked source files are compiled inputs too, so they count as dirty.
    // Ignored build products (target/, deploy artifacts) never appear here.
    let dirty = git(&["status", "--porcelain"]).map(|status| !status.is_empty());
    println!(
        "cargo:rustc-env=MSLIM_SOURCE_COMMIT={}",
        commit.as_deref().unwrap_or("unknown")
    );
    println!(
        "cargo:rustc-env=MSLIM_SOURCE_DIRTY={}",
        match dirty {
            Some(true) => "true",
            Some(false) => "false",
            None => "unknown",
        }
    );
}
