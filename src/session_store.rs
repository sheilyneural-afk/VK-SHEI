//! Persistent semantic session store.
//!
//! This store owns continuity metadata and opaque, verified session payloads.
//! It never interprets or mutates physical KV tensors; a runtime owner may store
//! a provider snapshot here only after validating its receipt and identity.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{fs, io, path::PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SessionGeneration {
    pub session_handle: String,
    pub generation: u64,
    pub state: serde_json::Value,
    pub state_hash: String,
    pub previous_hash: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoreError {
    Invalid(String),
    NotFound,
    DivergentReplay,
    Io(String),
    Corrupt(String),
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for StoreError {}

#[derive(Debug, Clone)]
pub struct SessionStore {
    root: PathBuf,
}

impl SessionStore {
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, StoreError> {
        let root = root.into();
        fs::create_dir_all(&root).map_err(ioe)?;
        Ok(Self { root })
    }

    fn dir(&self, handle: &str) -> Result<PathBuf, StoreError> {
        validate_handle(handle)?;
        let d = self.root.join(handle);
        fs::create_dir_all(&d).map_err(ioe)?;
        Ok(d)
    }

    pub fn latest(&self, handle: &str) -> Result<Option<SessionGeneration>, StoreError> {
        let d = self.dir(handle)?;
        let current = d.join("CURRENT");
        if !current.exists() {
            return Ok(None);
        }
        let name = fs::read_to_string(current).map_err(ioe)?.trim().to_owned();
        if name.is_empty() {
            return Err(StoreError::Corrupt("empty CURRENT".into()));
        }
        let bytes = fs::read(d.join(name)).map_err(ioe)?;
        let g: SessionGeneration =
            serde_json::from_slice(&bytes).map_err(|e| StoreError::Corrupt(e.to_string()))?;
        verify(&g)?;
        Ok(Some(g))
    }

    pub fn prepare(
        &self,
        handle: &str,
        generation: u64,
        state: serde_json::Value,
    ) -> Result<SessionGeneration, StoreError> {
        let d = self.dir(handle)?;
        let previous = self.latest(handle)?;
        let expected = previous.as_ref().map(|x| x.generation + 1).unwrap_or(0);
        if generation != expected {
            return Err(StoreError::Invalid(format!(
                "expected generation {expected}, got {generation}"
            )));
        }
        let previous_hash = previous.map(|x| x.state_hash);
        let state_hash = hash_json(&state)?;
        let g = SessionGeneration {
            session_handle: handle.to_owned(),
            generation,
            state,
            state_hash,
            previous_hash,
        };
        verify(&g)?;
        let name = format!("generation-{generation:020}.json");
        let tmp = d.join(format!(".{name}.tmp"));
        let bytes =
            serde_json::to_vec_pretty(&g).map_err(|e| StoreError::Invalid(e.to_string()))?;
        fs::write(&tmp, bytes).map_err(ioe)?;
        let file = fs::OpenOptions::new().read(true).open(&tmp).map_err(ioe)?;
        file.sync_all().map_err(ioe)?;
        fs::rename(&tmp, d.join(&name)).map_err(ioe)?;
        Ok(g)
    }

    pub fn commit(&self, handle: &str, generation: u64) -> Result<SessionGeneration, StoreError> {
        let d = self.dir(handle)?;
        let name = format!("generation-{generation:020}.json");
        let path = d.join(&name);
        let bytes = fs::read(&path).map_err(ioe)?;
        let g: SessionGeneration =
            serde_json::from_slice(&bytes).map_err(|e| StoreError::Corrupt(e.to_string()))?;
        verify(&g)?;
        let tmp = d.join(".CURRENT.tmp");
        fs::write(&tmp, format!("{name}\n")).map_err(ioe)?;
        let file = fs::OpenOptions::new().read(true).open(&tmp).map_err(ioe)?;
        file.sync_all().map_err(ioe)?;
        fs::rename(&tmp, d.join("CURRENT")).map_err(ioe)?;
        let dir = fs::File::open(&d).map_err(ioe)?;
        dir.sync_all().map_err(ioe)?;
        Ok(g)
    }

    pub fn prepare_and_commit(
        &self,
        handle: &str,
        generation: u64,
        state: serde_json::Value,
    ) -> Result<SessionGeneration, StoreError> {
        self.prepare(handle, generation, state)?;
        self.commit(handle, generation)
    }
}

fn validate_handle(h: &str) -> Result<(), StoreError> {
    if h.is_empty()
        || h.len() > 128
        || h == "."
        || h == ".."
        || h.contains('/')
        || h.contains('\\')
        || h.contains("..")
    {
        return Err(StoreError::Invalid("invalid session handle".into()));
    }
    Ok(())
}
fn hash_json(v: &serde_json::Value) -> Result<String, StoreError> {
    let b = serde_json::to_vec(v).map_err(|e| StoreError::Invalid(e.to_string()))?;
    let mut h = Sha256::new();
    h.update(b);
    Ok(format!("{:x}", h.finalize()))
}
fn verify(g: &SessionGeneration) -> Result<(), StoreError> {
    if g.state_hash != hash_json(&g.state)? {
        return Err(StoreError::Corrupt("state hash mismatch".into()));
    }
    if g.session_handle.is_empty() {
        return Err(StoreError::Corrupt("empty handle".into()));
    }
    Ok(())
}
fn ioe(e: io::Error) -> StoreError {
    StoreError::Io(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn restart_roundtrip_and_atomic_current() {
        let d = std::env::temp_dir().join(format!("kvslim-session-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        let s = SessionStore::open(&d).unwrap();
        s.prepare_and_commit("abc", 0, serde_json::json!({"next":"one"}))
            .unwrap();
        let s2 = SessionStore::open(&d).unwrap();
        assert_eq!(s2.latest("abc").unwrap().unwrap().generation, 0);
        let g = s2
            .prepare("abc", 1, serde_json::json!({"next":"two"}))
            .unwrap();
        assert_eq!(s2.latest("abc").unwrap().unwrap().generation, 0);
        s2.commit("abc", g.generation).unwrap();
        assert_eq!(s2.latest("abc").unwrap().unwrap().generation, 1);
        let _ = fs::remove_dir_all(&d);
    }
    #[test]
    fn rejects_generation_replay() {
        let d = std::env::temp_dir().join(format!("kvslim-session-replay-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        let s = SessionStore::open(&d).unwrap();
        s.prepare_and_commit("abc", 0, serde_json::json!({"x":1}))
            .unwrap();
        assert!(matches!(
            s.prepare("abc", 0, serde_json::json!({"x":2})),
            Err(StoreError::Invalid(_))
        ));
        let _ = fs::remove_dir_all(&d);
    }
}
