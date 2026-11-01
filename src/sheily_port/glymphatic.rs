//! Cleanup pressure model for the KV cold store.
//!
//! Sheily's glymphatic cleaner accumulates a "toxin pressure" over time and
//! purges during sleep phases. Here the pressure is driven by the real
//! measured cold-store size against its byte budget, not by elapsed time alone.
//! Port of the pressure/purge arithmetic in
//! `sheily-memory-engine-rs/src/glymphatic.rs`.

pub const TOXIN_ALERT: f64 = 0.70;
pub const NREM_PURGE_RATE: f64 = 0.40;
pub const REM_PURGE_RATE: f64 = 0.10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SleepPhase {
    Wake,
    Nrem,
    Rem,
}

/// Pressure in [0,1] from bytes in use over budget. Above the budget it saturates.
pub fn pressure_from_bytes(used_bytes: u64, budget_bytes: u64) -> f64 {
    if budget_bytes == 0 {
        return 1.0;
    }
    (used_bytes as f64 / budget_bytes as f64).clamp(0.0, 1.0)
}

/// Apply one purge cycle. Returns the new pressure.
pub fn purge_cycle(pressure: f64, phase: SleepPhase) -> f64 {
    let rate = match phase {
        SleepPhase::Rem => REM_PURGE_RATE,
        SleepPhase::Nrem => NREM_PURGE_RATE,
        SleepPhase::Wake => return pressure,
    };
    (pressure * (1.0 - rate)).clamp(0.0, 1.0)
}

pub fn should_alert(pressure: f64) -> bool {
    pressure >= TOXIN_ALERT
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pressure_tracks_real_bytes() {
        assert_eq!(pressure_from_bytes(0, 100), 0.0);
        assert!((pressure_from_bytes(75, 100) - 0.75).abs() < 1e-12);
        assert_eq!(pressure_from_bytes(500, 100), 1.0);
        assert_eq!(pressure_from_bytes(1, 0), 1.0);
    }

    #[test]
    fn nrem_purges_more_than_rem_and_wake_does_nothing() {
        assert!(purge_cycle(0.8, SleepPhase::Nrem) < purge_cycle(0.8, SleepPhase::Rem));
        assert_eq!(purge_cycle(0.8, SleepPhase::Wake), 0.8);
        assert!(should_alert(0.7));
        assert!(!should_alert(0.69));
    }
}
