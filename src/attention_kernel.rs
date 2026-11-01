//! Bounded attention-kernel configuration, not a second memory manager.
//! Reference/strict and experimental numeric contracts must never be conflated.
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionKernelMode {
    #[default]
    Reference,
    StrictTiled,
    OnlineExperimental,
    SplitKvExperimental,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AttentionKernelConfig {
    pub mode: AttentionKernelMode,
    pub block_queries: usize,
    pub block_keys: usize,
    pub work_group: usize,
    pub splits: usize,
}
impl Default for AttentionKernelConfig {
    fn default() -> Self {
        Self {
            mode: AttentionKernelMode::Reference,
            block_queries: 1,
            block_keys: 64,
            work_group: 64,
            splits: 1,
        }
    }
}
impl AttentionKernelConfig {
    pub fn experimental(self) -> bool {
        matches!(
            self.mode,
            AttentionKernelMode::OnlineExperimental | AttentionKernelMode::SplitKvExperimental
        )
    }
    pub fn validate(self) -> Result<()> {
        ensure!(
            [1, 2, 4].contains(&self.block_queries),
            "attention block_queries must be 1,2,4"
        );
        ensure!(
            [32, 64, 128].contains(&self.block_keys),
            "attention block_keys must be 32,64,128"
        );
        ensure!(
            [32, 64, 128].contains(&self.work_group),
            "attention work_group must be 32,64,128"
        );
        ensure!(
            (1..=64).contains(&self.splits),
            "attention splits must be 1..=64"
        );
        if self.mode == AttentionKernelMode::SplitKvExperimental {
            ensure!(
                self.block_queries == 1 && self.splits >= 2,
                "split-KV requires one query per group and at least two splits"
            );
        } else {
            ensure!(
                self.splits == 1,
                "only explicit split-KV may request multiple splits"
            );
        }
        if self.mode == AttentionKernelMode::Reference {
            ensure!(
                self == Self::default(),
                "reference attention cannot carry ignored tuning values"
            );
        }
        Ok(())
    }
    /// Legacy snapshots predate tiled kernels and therefore belong to reference mode.
    pub fn check_snapshot(self, recorded: Option<&serde_json::Value>) -> Result<()> {
        let recorded: Self = recorded
            .map(|v| serde_json::from_value(v.clone()))
            .transpose()?
            .unwrap_or_default();
        recorded.validate()?;
        ensure!(
            self == recorded,
            "snapshot attention numeric configuration mismatch"
        );
        Ok(())
    }

    pub fn validate_activation(self, allow_experimental: bool) -> Result<()> {
        self.validate()?;
        ensure!(
            !self.experimental() || allow_experimental,
            "experimental attention requires explicit numeric-contract opt-in"
        );
        Ok(())
    }
    pub fn id(self) -> String {
        if self.mode == AttentionKernelMode::Reference {
            return "reference".into();
        }
        format!(
            "{:?}-q{}-k{}-wg{}-s{}",
            self.mode, self.block_queries, self.block_keys, self.work_group, self.splits
        )
        .to_ascii_lowercase()
    }
    /// Storage for local K-or-V tile, Q tile, score tile and shared row state.
    /// This is a lower bound; the driver-reported per-kernel amount is checked too.
    pub fn local_bytes(self, head_dim: usize) -> Result<usize> {
        self.validate()?;
        ensure!(
            (1..=256).contains(&head_dim),
            "tiled attention head dimension must be 1..=256"
        );
        Ok(2 * self.block_keys * head_dim
            + 2 * self.block_queries * head_dim
            + 4 * self.block_queries * self.block_keys
            + 4 * self.block_queries * 4)
    }
    pub fn decode_scratch_bytes(self, heads: usize, head_dim: usize) -> Result<usize> {
        self.validate()?;
        if self.mode != AttentionKernelMode::SplitKvExperimental {
            return Ok(0);
        }
        let bytes = heads
            .checked_mul(self.splits)
            .and_then(|x| x.checked_mul(head_dim.checked_add(2)?))
            .and_then(|x| x.checked_mul(4));
        ensure!(
            bytes.is_some_and(|n| n > 0 && n <= 16 * 1024 * 1024),
            "split-KV scratch exceeds its 16 MiB bound or overflows"
        );
        Ok(bytes.unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn default_is_reference_and_experimental_opt_in_is_mandatory() {
        assert_eq!(
            serde_json::from_str::<AttentionKernelConfig>("{}").unwrap(),
            AttentionKernelConfig::default()
        );
        let c = AttentionKernelConfig {
            mode: AttentionKernelMode::OnlineExperimental,
            ..Default::default()
        };
        assert!(c.validate_activation(false).is_err());
        assert!(c.validate_activation(true).is_ok());
    }
    #[test]
    fn reject_ignored_or_unbounded_configuration() {
        for c in [
            AttentionKernelConfig {
                work_group: 0,
                ..Default::default()
            },
            AttentionKernelConfig {
                block_keys: 33,
                ..Default::default()
            },
            AttentionKernelConfig {
                splits: 2,
                ..Default::default()
            },
            AttentionKernelConfig {
                mode: AttentionKernelMode::StrictTiled,
                splits: 2,
                ..Default::default()
            },
            AttentionKernelConfig {
                mode: AttentionKernelMode::SplitKvExperimental,
                splits: 65,
                ..Default::default()
            },
        ] {
            assert!(c.validate().is_err());
        }
        assert!(serde_json::from_str::<AttentionKernelConfig>(r#"{"epsilon":0.1}"#).is_err());
    }
    #[test]
    fn scratch_is_bounded_and_strict_has_no_global_scores() {
        let c = AttentionKernelConfig {
            mode: AttentionKernelMode::StrictTiled,
            ..Default::default()
        };
        assert_eq!(c.decode_scratch_bytes(32, 128).unwrap(), 0);
        assert!(c.local_bytes(128).unwrap() < 64 * 1024);
        let split = AttentionKernelConfig {
            mode: AttentionKernelMode::SplitKvExperimental,
            splits: 8,
            ..Default::default()
        };
        assert_eq!(
            split.decode_scratch_bytes(32, 128).unwrap(),
            32 * 8 * 130 * 4
        );
        assert!(split.decode_scratch_bytes(usize::MAX, 128).is_err());
    }
}
