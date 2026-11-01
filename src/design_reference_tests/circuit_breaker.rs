//! Closed / open / half-open circuit breaker with an explicit clock.
//! Port of `sheily-execution-kernel-rs/src/circuit_breaker.rs` (state machine only;
//! the persisted registry is left to the caller).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: u64,
    pub success_threshold: u64,
    /// Seconds the breaker stays open before a half-open probe is allowed.
    pub timeout: f64,
    pub half_open_max_calls: u64,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 2,
            timeout: 60.0,
            half_open_max_calls: 3,
        }
    }
}

impl CircuitBreakerConfig {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.failure_threshold == 0 {
            return Err("circuit_breaker_failure_threshold_must_be_positive");
        }
        if self.success_threshold == 0 {
            return Err("circuit_breaker_success_threshold_must_be_positive");
        }
        if !self.timeout.is_finite() || self.timeout < 0.0 {
            return Err("circuit_breaker_timeout_invalid");
        }
        if self.half_open_max_calls == 0 {
            return Err("circuit_breaker_half_open_max_calls_must_be_positive");
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    state: CircuitState,
    failure_count: u64,
    success_count: u64,
    half_open_calls: u64,
    last_failure_at: Option<f64>,
    pub total_calls: u64,
    pub total_successes: u64,
    pub total_failures: u64,
    pub total_rejected: u64,
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Result<Self, &'static str> {
        config.validate()?;
        Ok(Self {
            config,
            state: CircuitState::Closed,
            failure_count: 0,
            success_count: 0,
            half_open_calls: 0,
            last_failure_at: None,
            total_calls: 0,
            total_successes: 0,
            total_failures: 0,
            total_rejected: 0,
        })
    }

    pub fn state(&self) -> CircuitState {
        self.state
    }

    fn refresh(&mut self, now: f64) {
        if self.state == CircuitState::Open
            && self
                .last_failure_at
                .is_some_and(|at| now - at >= self.config.timeout)
        {
            self.state = CircuitState::HalfOpen;
            self.half_open_calls = 0;
            self.success_count = 0;
        }
    }

    /// Ask permission before a call. `now` is seconds on any monotonic-ish clock.
    pub fn before_call(&mut self, now: f64) -> Result<(), &'static str> {
        self.total_calls += 1;
        self.refresh(now);
        match self.state {
            CircuitState::Closed => Ok(()),
            CircuitState::Open => {
                self.total_rejected += 1;
                Err("circuit_breaker_open")
            }
            CircuitState::HalfOpen => {
                if self.half_open_calls >= self.config.half_open_max_calls {
                    self.total_rejected += 1;
                    return Err("circuit_breaker_half_open_max_calls_reached");
                }
                self.half_open_calls += 1;
                Ok(())
            }
        }
    }

    pub fn record_success(&mut self) {
        self.total_successes += 1;
        match self.state {
            CircuitState::HalfOpen => {
                self.success_count += 1;
                if self.success_count >= self.config.success_threshold {
                    self.reset();
                }
            }
            CircuitState::Closed => self.failure_count = 0,
            CircuitState::Open => {}
        }
    }

    pub fn record_failure(&mut self, now: f64) {
        self.total_failures += 1;
        self.last_failure_at = Some(now);
        match self.state {
            CircuitState::HalfOpen => self.open(),
            CircuitState::Closed => {
                self.failure_count += 1;
                if self.failure_count >= self.config.failure_threshold {
                    self.open();
                }
            }
            CircuitState::Open => {}
        }
    }

    fn open(&mut self) {
        self.state = CircuitState::Open;
        self.success_count = 0;
        self.half_open_calls = 0;
    }

    fn reset(&mut self) {
        self.state = CircuitState::Closed;
        self.failure_count = 0;
        self.success_count = 0;
        self.half_open_calls = 0;
        self.last_failure_at = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> CircuitBreakerConfig {
        CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 2,
            timeout: 10.0,
            half_open_max_calls: 1,
        }
    }

    #[test]
    fn opens_after_threshold_and_rejects() {
        let mut cb = CircuitBreaker::new(cfg()).unwrap();
        cb.before_call(0.0).unwrap();
        cb.record_failure(0.0);
        cb.before_call(0.0).unwrap();
        cb.record_failure(0.0);
        assert_eq!(cb.state(), CircuitState::Open);
        assert_eq!(cb.before_call(5.0), Err("circuit_breaker_open"));
        assert_eq!(cb.total_rejected, 1);
    }

    #[test]
    fn half_open_probe_closes_after_successes() {
        let mut cb = CircuitBreaker::new(cfg()).unwrap();
        for _ in 0..2 {
            cb.before_call(0.0).unwrap();
            cb.record_failure(0.0);
        }
        cb.before_call(10.0).unwrap();
        assert_eq!(cb.state(), CircuitState::HalfOpen);
        assert_eq!(
            cb.before_call(10.0),
            Err("circuit_breaker_half_open_max_calls_reached")
        );
        cb.record_success();
        cb.before_call(10.0).err();
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn failed_probe_reopens() {
        let mut cb = CircuitBreaker::new(cfg()).unwrap();
        for _ in 0..2 {
            cb.before_call(0.0).unwrap();
            cb.record_failure(0.0);
        }
        cb.before_call(10.0).unwrap();
        cb.record_failure(10.0);
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn invalid_config_is_rejected() {
        let bad = CircuitBreakerConfig {
            failure_threshold: 0,
            ..cfg()
        };
        assert!(CircuitBreaker::new(bad).is_err());
    }
}
