//! Per-task resource budget (iterations, tokens, wall time, cost, repeated tools).
//! Port of the admission checks in
//! `sheily-execution-kernel-rs/src/execution/budget_control.rs` (`check_task`, `forecast`),
//! with time passed in explicitly.

#[derive(Debug, Clone, PartialEq)]
pub struct Limits {
    pub max_iters: u64,
    pub token_budget: u64,
    pub cost_limit: f64,
    pub tool_repeat_limit: u64,
    pub wall_time_limit: f64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_iters: 20,
            token_budget: 50_000,
            cost_limit: 1.0,
            tool_repeat_limit: 5,
            wall_time_limit: 300.0,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Task {
    pub limits: Limits,
    pub started_at: f64,
    pub iterations: u64,
    pub tokens_used: u64,
    pub cost_usd: f64,
    pub tool_counts: std::collections::BTreeMap<String, u64>,
    pub exhausted: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Decision {
    Allowed {
        throttle: bool,
        steps_remaining: u64,
    },
    Denied {
        reason: String,
    },
}

/// Steps left before any limit is hit, using the observed per-iteration rate.
pub fn steps_remaining(task: &Task, now: f64) -> u64 {
    let iter_steps = task.limits.max_iters.saturating_sub(task.iterations);
    if task.iterations < 2 {
        return iter_steps;
    }
    let elapsed = (now - task.started_at).max(0.0);
    let tokens_per_iter = task.tokens_used as f64 / task.iterations as f64;
    let token_steps = (task.limits.token_budget.saturating_sub(task.tokens_used) as f64
        / tokens_per_iter.max(1.0))
    .floor()
    .max(0.0) as u64;
    let time_per_iter = elapsed / task.iterations as f64;
    let time_steps = if time_per_iter <= 0.0 {
        task.limits.max_iters
    } else {
        ((task.limits.wall_time_limit - elapsed).max(0.0) / time_per_iter)
            .floor()
            .max(0.0) as u64
    };
    token_steps.min(time_steps).min(iter_steps)
}

/// Admission check for one tool call. Mutates `task` on denial (sticky exhaustion).
pub fn check(task: &mut Task, tool: &str, now: f64) -> Decision {
    if let Some(reason) = &task.exhausted {
        return Decision::Denied {
            reason: reason.clone(),
        };
    }
    let deny = |task: &mut Task, reason: String| {
        task.exhausted = Some(reason.clone());
        Decision::Denied { reason }
    };
    if task.iterations >= task.limits.max_iters {
        return deny(task, format!("max_iters={} reached", task.limits.max_iters));
    }
    let elapsed = (now - task.started_at).max(0.0);
    if elapsed > task.limits.wall_time_limit {
        return deny(
            task,
            format!("wall_time={elapsed:.0}s > {}s", task.limits.wall_time_limit),
        );
    }
    if task.tokens_used >= task.limits.token_budget {
        return deny(
            task,
            format!("token_budget={} exhausted", task.limits.token_budget),
        );
    }
    if task.cost_usd >= task.limits.cost_limit {
        return deny(
            task,
            format!("cost_limit_usd={:.2} reached", task.limits.cost_limit),
        );
    }
    let count = task.tool_counts.get(tool).copied().unwrap_or(0);
    if count >= task.limits.tool_repeat_limit {
        return deny(task, format!("tool '{tool}' repeated {count} times"));
    }
    let budget_pct = task.tokens_used as f64 / task.limits.token_budget.max(1) as f64;
    let steps = steps_remaining(task, now);
    let throttle = budget_pct > 0.8 || steps <= 3;
    Decision::Allowed {
        throttle,
        steps_remaining: steps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_tool_is_denied_and_sticky() {
        let mut t = Task {
            limits: Limits {
                tool_repeat_limit: 2,
                ..Limits::default()
            },
            ..Task::default()
        };
        t.tool_counts.insert("search".into(), 2);
        assert!(matches!(
            check(&mut t, "search", 0.0),
            Decision::Denied { .. }
        ));
        assert!(matches!(
            check(&mut t, "other", 0.0),
            Decision::Denied { .. }
        ));
    }

    #[test]
    fn near_token_exhaustion_throttles() {
        let mut t = Task {
            limits: Limits {
                token_budget: 1000,
                ..Limits::default()
            },
            ..Task::default()
        };
        t.tokens_used = 900;
        match check(&mut t, "x", 0.0) {
            Decision::Allowed { throttle, .. } => assert!(throttle),
            other => panic!("expected allowed, got {other:?}"),
        }
    }
}
