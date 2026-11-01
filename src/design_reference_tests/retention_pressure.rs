//! Retention-pressure model for a bounded K/V store.
//!
//! Pressure is driven by measured store size against its byte budget. The
//! policies model different reclamation rates without biological terminology.

pub const RETENTION_PRESSURE_ALERT: f64 = 0.70;
pub const AGGRESSIVE_RECLAIM_RATE: f64 = 0.40;
pub const CONSERVATIVE_RECLAIM_RATE: f64 = 0.10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReclaimPolicy {
    Disabled,
    Aggressive,
    Conservative,
}

/// Pressure in [0,1] from bytes in use over budget. Above the budget it saturates.
pub fn pressure_from_bytes(used_bytes: u64, budget_bytes: u64) -> f64 {
    if budget_bytes == 0 {
        return 1.0;
    }
    (used_bytes as f64 / budget_bytes as f64).clamp(0.0, 1.0)
}

/// Apply one purge cycle. Returns the new pressure.
pub fn reclaim_cycle(pressure: f64, policy: ReclaimPolicy) -> f64 {
    let rate = match policy {
        ReclaimPolicy::Conservative => CONSERVATIVE_RECLAIM_RATE,
        ReclaimPolicy::Aggressive => AGGRESSIVE_RECLAIM_RATE,
        ReclaimPolicy::Disabled => return pressure,
    };
    (pressure * (1.0 - rate)).clamp(0.0, 1.0)
}

pub fn should_alert(pressure: f64) -> bool {
    pressure >= RETENTION_PRESSURE_ALERT
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
    fn aggressive_reclaims_more_than_conservative_and_disabled_does_nothing() {
        assert!(
            reclaim_cycle(0.8, ReclaimPolicy::Aggressive)
                < reclaim_cycle(0.8, ReclaimPolicy::Conservative)
        );
        assert_eq!(reclaim_cycle(0.8, ReclaimPolicy::Disabled), 0.8);
        assert!(should_alert(0.7));
        assert!(!should_alert(0.69));
    }
}
