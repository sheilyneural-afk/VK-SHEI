//! Workload classifier with hysteresis. Chooses a serving profile
//! (context size, threads, batch) from the request shape.
//! Port of `sheily-inference-rs/src/scheduling/workload_classification.rs`,
//! with the clock passed in explicitly instead of read from `Instant::now`.

use std::collections::VecDeque;
use std::time::Duration;

const HYSTERESIS_MIN_REQUESTS: usize = 3;
const HISTORY_WINDOW: Duration = Duration::from_secs(120);
const AGENT_LOOP_WINDOW: Duration = Duration::from_secs(30);
const AGENT_LOOP_MIN_REQUESTS: usize = 5;
const REASONING_COST_THRESHOLD: u64 = 800;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Workload {
    ChatShort,
    Reasoning,
    AgentLoop,
    Rag,
    Batch,
}

impl Workload {
    pub fn from_hint(value: &str) -> Option<Self> {
        match value {
            "chat_short" => Some(Self::ChatShort),
            "reasoning" => Some(Self::Reasoning),
            "agent_loop" => Some(Self::AgentLoop),
            "rag" => Some(Self::Rag),
            "batch" => Some(Self::Batch),
            _ => None,
        }
    }

    /// Serving adjustment for this workload.
    pub fn adjustment(self) -> Adjustment {
        match self {
            Self::ChatShort => Adjustment {
                n_ctx_override: 1024,
                n_threads_delta: 0,
                n_batch_delta: 0,
                reason: "short_chat",
            },
            Self::Reasoning => Adjustment {
                n_ctx_override: 4096,
                n_threads_delta: -2,
                n_batch_delta: -256,
                reason: "long_reasoning",
            },
            Self::AgentLoop => Adjustment {
                n_ctx_override: 2048,
                n_threads_delta: 0,
                n_batch_delta: 0,
                reason: "sequential_agent_loop",
            },
            Self::Rag => Adjustment {
                n_ctx_override: 3072,
                n_threads_delta: -1,
                n_batch_delta: -128,
                reason: "retrieval_augmented_generation",
            },
            Self::Batch => Adjustment {
                n_ctx_override: 1024,
                n_threads_delta: 0,
                n_batch_delta: 256,
                reason: "concurrent_batch",
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Adjustment {
    pub n_ctx_override: u32,
    pub n_threads_delta: i32,
    pub n_batch_delta: i32,
    pub reason: &'static str,
}

#[derive(Debug, Clone, Copy)]
struct Observation {
    at: f64,
}

/// Classifier memory. `now` values are seconds on one consistent clock.
#[derive(Debug, Default)]
pub struct WorkloadClassifier {
    history: VecDeque<Observation>,
    current: Option<Workload>,
    pending: Option<Workload>,
    pending_streak: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct Request<'a> {
    pub prompt: &'a str,
    pub hint: Option<&'a str>,
    pub concurrent_requests: u64,
    pub output_tokens: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Classification {
    pub workload: Workload,
    pub confidence: f64,
    pub signals: Vec<String>,
    pub forced: bool,
    pub switched: bool,
}

pub fn estimate_prompt_tokens(prompt: &str) -> u64 {
    ((prompt.split_whitespace().count() as f64 * 1.3).ceil() as u64).max(1)
}

pub fn estimate_cost(prompt_tokens: u64, output_tokens: u64) -> u64 {
    (prompt_tokens as f64 * 0.3 + output_tokens as f64).floor() as u64
}

impl WorkloadClassifier {
    fn prune(&mut self, now: f64) {
        let window = HISTORY_WINDOW.as_secs_f64();
        while self
            .history
            .front()
            .is_some_and(|row| now - row.at > window)
        {
            self.history.pop_front();
        }
    }

    fn agent_loop(&self, now: f64) -> bool {
        let window = AGENT_LOOP_WINDOW.as_secs_f64();
        self.history
            .iter()
            .filter(|row| now - row.at <= window)
            .count()
            >= AGENT_LOOP_MIN_REQUESTS
    }

    fn apply_hysteresis(&mut self, candidate: Workload, forced: bool) -> (Workload, bool) {
        if forced {
            self.current = Some(candidate);
            self.pending = None;
            self.pending_streak = 0;
            return (candidate, false);
        }
        let Some(current) = self.current else {
            self.current = Some(candidate);
            return (candidate, false);
        };
        if current == candidate {
            self.pending = None;
            self.pending_streak = 0;
            return (current, false);
        }
        if self.pending == Some(candidate) {
            self.pending_streak = self.pending_streak.saturating_add(1);
        } else {
            self.pending = Some(candidate);
            self.pending_streak = 1;
        }
        if self.pending_streak >= HYSTERESIS_MIN_REQUESTS {
            self.current = Some(candidate);
            self.pending = None;
            self.pending_streak = 0;
            (candidate, true)
        } else {
            (current, false)
        }
    }

    pub fn classify(&mut self, now: f64, request: Request<'_>) -> Classification {
        self.prune(now);
        let prompt_tokens = estimate_prompt_tokens(request.prompt);
        let cost = estimate_cost(prompt_tokens, request.output_tokens);
        let (raw, confidence, signals, forced) =
            self.classify_raw(now, &request, prompt_tokens, cost);
        self.history.push_back(Observation { at: now });
        let (workload, switched) = self.apply_hysteresis(raw, forced);
        Classification {
            workload,
            confidence,
            signals,
            forced,
            switched,
        }
    }

    fn classify_raw(
        &self,
        now: f64,
        request: &Request<'_>,
        prompt_tokens: u64,
        cost: u64,
    ) -> (Workload, f64, Vec<String>, bool) {
        let mut signals = Vec::new();
        if let Some(hint) = request.hint {
            if let Some(workload) = Workload::from_hint(hint) {
                return (workload, 1.0, vec![format!("hint={hint}")], true);
            }
            signals.push(format!("hint_invalid={hint}"));
        }
        if request.concurrent_requests > 1 {
            signals.push(format!("concurrent={}", request.concurrent_requests));
            return (Workload::Batch, 0.95, signals, false);
        }
        if self.agent_loop(now) {
            signals.push("agent_loop_by_history".into());
            if self.current == Some(Workload::AgentLoop) {
                return (Workload::AgentLoop, 0.85, signals, false);
            }
        }
        let lower = request.prompt.to_ascii_lowercase();
        let rag_hits = [
            "context:",
            "retrieved:",
            "document:",
            "passage:",
            "[doc",
            "###",
            "<doc",
        ]
        .iter()
        .filter(|needle| lower.contains(**needle))
        .count();
        if prompt_tokens > 800 && rag_hits >= 2 {
            signals.push(format!("rag:tokens={prompt_tokens}:patterns={rag_hits}"));
            return (Workload::Rag, 0.80, signals, false);
        }
        if cost >= REASONING_COST_THRESHOLD {
            signals.push(format!("reasoning:cost={cost}:tokens={prompt_tokens}"));
            return (Workload::Reasoning, 0.75, signals, false);
        }
        signals.push(format!("chat_short:tokens={prompt_tokens}:cost={cost}"));
        (Workload::ChatShort, 0.70, signals, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(prompt: &str) -> Request<'_> {
        Request {
            prompt,
            hint: None,
            concurrent_requests: 1,
            output_tokens: 256,
        }
    }

    #[test]
    fn explicit_hint_forces_workload() {
        let mut c = WorkloadClassifier::default();
        let out = c.classify(
            0.0,
            Request {
                hint: Some("reasoning"),
                ..req("hola")
            },
        );
        assert_eq!(out.workload, Workload::Reasoning);
        assert!(out.forced);
    }

    #[test]
    fn concurrency_means_batch() {
        let mut c = WorkloadClassifier::default();
        let out = c.classify(
            0.0,
            Request {
                concurrent_requests: 4,
                ..req("hola")
            },
        );
        assert_eq!(out.workload, Workload::Batch);
    }

    #[test]
    fn switch_needs_three_consecutive_requests() {
        let mut c = WorkloadClassifier::default();
        assert_eq!(c.classify(0.0, req("hola")).workload, Workload::ChatShort);
        // Large RAG prompts would switch, but hysteresis keeps ChatShort for two.
        let big = format!("{} context: x document: y", "palabra ".repeat(900));
        let r1 = c.classify(1.0, req(&big));
        assert_eq!(r1.workload, Workload::ChatShort);
        let r2 = c.classify(2.0, req(&big));
        assert_eq!(r2.workload, Workload::ChatShort);
        let r3 = c.classify(3.0, req(&big));
        assert_eq!(r3.workload, Workload::Rag);
        assert!(r3.switched);
    }

    #[test]
    fn adjustments_match_table() {
        assert_eq!(Workload::Reasoning.adjustment().n_ctx_override, 4096);
        assert_eq!(Workload::Batch.adjustment().n_batch_delta, 256);
    }
}
