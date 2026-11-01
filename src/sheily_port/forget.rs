//! Forget blocklist (tombstones) for KV pages.
//!
//! ColdStore is append-only, so "forget" does not rewrite pages. It records
//! tombstoned page ids per scope (user / document). Readers consult the
//! blocklist before using a page; a later compaction can drop tombstoned
//! pages. Port of the blocklist semantics in
//! `sheily-memory-engine-rs/src/canonical_store.rs` (`execute_forget_request`).

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ScopeTombstones {
    pub page_ids: BTreeSet<u64>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ForgetBlocklist {
    pub scopes: BTreeMap<String, ScopeTombstones>,
}

#[derive(Debug)]
pub struct ForgetStore {
    path: PathBuf,
    doc: ForgetBlocklist,
}

impl ForgetStore {
    /// Load the blocklist from `path`, or start empty when the file is absent.
    pub fn open(path: impl Into<PathBuf>) -> io::Result<Self> {
        let path = path.into();
        let doc = if path.exists() {
            serde_json::from_slice(&fs::read(&path)?)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?
        } else {
            ForgetBlocklist::default()
        };
        Ok(Self { path, doc })
    }

    /// Tombstone pages for a scope. Returns how many were newly added.
    pub fn forget(&mut self, scope: &str, page_ids: &[u64]) -> io::Result<usize> {
        if scope.trim().is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "forget_scope_required",
            ));
        }
        if page_ids.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "forget_request_empty",
            ));
        }
        let entry = self.doc.scopes.entry(scope.to_string()).or_default();
        let before = entry.page_ids.len();
        entry.page_ids.extend(page_ids.iter().copied());
        let added = entry.page_ids.len() - before;
        self.persist()?;
        Ok(added)
    }

    pub fn is_forgotten(&self, scope: &str, page_id: u64) -> bool {
        self.doc
            .scopes
            .get(scope)
            .is_some_and(|s| s.page_ids.contains(&page_id))
    }

    fn persist(&self) -> io::Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let tmp = self.path.with_extension("json.tmp");
        let bytes = serde_json::to_vec_pretty(&self.doc)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        fs::write(&tmp, bytes)?;
        fs::rename(tmp, &self.path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tombstones_survive_reopen_and_are_scoped() {
        let dir = std::env::temp_dir().join(format!("kv-forget-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let path = dir.join("forget.json");
        let mut store = ForgetStore::open(&path).unwrap();
        assert_eq!(store.forget("user:ana", &[3, 4, 4]).unwrap(), 2);
        drop(store);
        let reopened = ForgetStore::open(&path).unwrap();
        assert!(reopened.is_forgotten("user:ana", 3));
        assert!(!reopened.is_forgotten("user:bob", 3));
        assert!(!reopened.is_forgotten("user:ana", 5));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_request_and_scope_fail_closed() {
        let dir = std::env::temp_dir().join(format!("kv-forget-empty-{}", std::process::id()));
        let mut store = ForgetStore::open(dir.join("f.json")).unwrap();
        assert!(store.forget("", &[1]).is_err());
        assert!(store.forget("s", &[]).is_err());
        let _ = fs::remove_dir_all(&dir);
    }
}

/// Result of compacting a cold store: live pages were rewritten into a new
/// file, in order, and `remap` gives `old_page_id -> new_page_id`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Compaction {
    pub kept: usize,
    pub dropped: usize,
    pub remap: BTreeMap<u64, u64>,
}

/// Rewrite every page of `source` that is not in `forgotten` into a fresh
/// ColdStore at `dest`, then audit the result. Nothing is replaced here: the
/// caller switches to `dest` (and remaps page ids) only after this returns Ok.
/// `dest` must not already exist. On error `dest` is removed.
pub fn compact_excluding(
    source: &mut crate::cold_store::ColdStore,
    forgotten: &BTreeSet<u64>,
    dest: &Path,
) -> io::Result<Compaction> {
    use crate::cold_store::ColdStore;
    if dest.exists() {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "compaction_destination_exists",
        ));
    }
    let metas = source.metadata().to_vec();
    let mut out = ColdStore::create(dest, 0)?;
    let result = (|| -> io::Result<Compaction> {
        let mut remap = BTreeMap::new();
        for meta in &metas {
            if forgotten.contains(&meta.page_id) {
                continue;
            }
            let payload = source.read_page(meta.page_id)?;
            let prepared = ColdStore::prepare_page(
                out.metadata().len() as u64,
                meta.geometry,
                &payload.key,
                &payload.value,
            )?;
            let new_meta = out.append_prepared(prepared, true)?;
            remap.insert(meta.page_id, new_meta.page_id);
        }
        out.audit()?;
        Ok(Compaction {
            kept: remap.len(),
            dropped: metas.len() - remap.len(),
            remap,
        })
    })();
    if result.is_err() {
        drop(out);
        let _ = fs::remove_file(dest);
    }
    result
}

#[cfg(test)]
mod compaction_tests {
    use super::*;
    use crate::cold_store::{ColdStore, PageGeometry};

    fn geom(layer: u32, start: u64) -> PageGeometry {
        PageGeometry {
            layer,
            token_start: start,
            tokens: 2,
            heads: 1,
            head_dim: 2,
            tile: 2,
        }
    }

    #[test]
    fn compaction_drops_forgotten_pages_and_keeps_bytes_exact() {
        let dir = std::env::temp_dir().join(format!("kv-compact-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let src_path = dir.join("pages.kv");
        let mut src = ColdStore::create(&src_path, 0).unwrap();
        for i in 0..4u64 {
            let key = vec![i as u8; 8];
            let value = vec![(10 + i) as u8; 8];
            src.append_page(geom(0, i * 2), &key, &value, true).unwrap();
        }
        let forgotten: BTreeSet<u64> = [1, 3].into_iter().collect();
        let dest = dir.join("pages.compact.kv");
        let c = compact_excluding(&mut src, &forgotten, &dest).unwrap();
        assert_eq!((c.kept, c.dropped), (2, 2));
        assert_eq!(c.remap.get(&0), Some(&0));
        assert_eq!(c.remap.get(&2), Some(&1));

        let mut out = ColdStore::open_read_only(&dest, 0).unwrap();
        assert_eq!(out.metadata().len(), 2);
        assert_eq!(out.read_page(1).unwrap().value, vec![12u8; 8]);
        assert_eq!(out.read_page(0).unwrap().key, vec![0u8; 8]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn existing_destination_is_refused() {
        let dir = std::env::temp_dir().join(format!("kv-compact-x-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let mut src = ColdStore::create(dir.join("a.kv"), 0).unwrap();
        src.append_page(geom(0, 0), &[1; 8], &[2; 8], true).unwrap();
        fs::write(dir.join("b.kv"), b"x").unwrap();
        assert!(compact_excluding(&mut src, &BTreeSet::new(), &dir.join("b.kv")).is_err());
        let _ = fs::remove_dir_all(&dir);
    }
}
