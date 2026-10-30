//! Bounded background handoff pipeline for copied BF16 KV pages.
//!
//! The worker is deliberately generic over the background operation and its
//! output. The canonical runtime transfers owned BF16 page bits without
//! inventing a second K/V representation.

use std::collections::BTreeMap;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread::{self, JoinHandle};
use std::time::Instant;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PageShape {
    pub layer: usize,
    pub token_start: usize,
    pub heads: usize,
    pub tokens: usize,
    pub dim: usize,
    pub tile: usize,
}

/// Owned copy of a BF16 page. Values are raw IEEE BF16 bits, not converted KV.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Bf16PageJob {
    pub sequence: u64,
    pub shape: PageShape,
    pub key_bf16: Vec<u16>,
    pub value_bf16: Vec<u16>,
}

impl Bf16PageJob {
    pub fn copied(shape: PageShape, key_bf16: &[u16], value_bf16: &[u16]) -> Result<Self, String> {
        validate(shape, key_bf16, value_bf16)?;
        Ok(Self {
            sequence: 0,
            shape,
            key_bf16: key_bf16.to_vec(),
            value_bf16: value_bf16.to_vec(),
        })
    }

    /// Transfers already-owned exact BF16 bits into the bounded worker queue
    /// without copying the complete K/V payload a second time.
    pub fn owned(
        shape: PageShape,
        key_bf16: Vec<u16>,
        value_bf16: Vec<u16>,
    ) -> Result<Self, String> {
        validate(shape, &key_bf16, &value_bf16)?;
        Ok(Self {
            sequence: 0,
            shape,
            key_bf16,
            value_bf16,
        })
    }
}

#[derive(Debug)]
pub struct ConversionResult<O> {
    pub sequence: u64,
    pub shape: PageShape,
    pub worker_nanos: u64,
    pub encoded: Result<O, String>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ConversionStats {
    pub submitted: u64,
    pub delivered: u64,
    pub buffered_out_of_order: usize,
    pub closed: bool,
    pub worker_nanos: u64,
}

/// Bounded producer queue plus ordered consumer view over parallel encoders.
pub struct ConversionWorker<O: Send + 'static> {
    sender: Option<SyncSender<Bf16PageJob>>,
    results: Receiver<ConversionResult<O>>,
    handles: Vec<JoinHandle<()>>,
    next_submit: u64,
    next_deliver: u64,
    pending: BTreeMap<u64, ConversionResult<O>>,
    worker_nanos: u64,
    closed: bool,
}

impl<O: Send + 'static> ConversionWorker<O> {
    pub fn spawn<E>(queue_capacity: usize, worker_count: usize, encoder: E) -> Result<Self, String>
    where
        E: Fn(Bf16PageJob) -> Result<O, String> + Send + Sync + 'static,
    {
        if queue_capacity == 0 || worker_count == 0 {
            return Err("queue_capacity y worker_count deben ser positivos".into());
        }
        Self::spawn_from(queue_capacity, worker_count, 0, encoder)
    }

    /// Start an ordered worker whose sequence continues an already committed
    /// persistent store. This is required when a prefix cache is restored:
    /// newly generated pages must append after the immutable prefix page IDs.
    pub fn spawn_from<E>(
        queue_capacity: usize,
        worker_count: usize,
        first_sequence: u64,
        encoder: E,
    ) -> Result<Self, String>
    where
        E: Fn(Bf16PageJob) -> Result<O, String> + Send + Sync + 'static,
    {
        if queue_capacity == 0 || worker_count == 0 {
            return Err("queue_capacity y worker_count deben ser positivos".into());
        }
        let (sender, jobs) = mpsc::sync_channel::<Bf16PageJob>(queue_capacity);
        let (result_sender, results) = mpsc::channel();
        let jobs = Arc::new(std::sync::Mutex::new(jobs));
        let encoder = Arc::new(encoder);
        let mut handles = Vec::with_capacity(worker_count);
        for index in 0..worker_count {
            let jobs = jobs.clone();
            let result_sender = result_sender.clone();
            let encoder = encoder.clone();
            handles.push(
                thread::Builder::new()
                    .name(format!("kvslim-kv-converter-{index}"))
                    .spawn(move || {
                        loop {
                            let job = match jobs.lock() {
                                Ok(receiver) => receiver.recv(),
                                Err(_) => return,
                            };
                            let job = match job {
                                Ok(job) => job,
                                Err(_) => return,
                            };
                            let sequence = job.sequence;
                            let shape = job.shape;
                            let started = Instant::now();
                            // A third-party/backend encoder must not kill one
                            // worker and silently lose its sequence. Losing a
                            // sequence while sibling workers remain alive would
                            // leave the ordered consumer blocked forever. Both
                            // dev and release profiles use panic=unwind
                            // (Cargo.toml), so this catch is always active; a
                            // panic=abort build would end the whole process.
                            let encoded = panic::catch_unwind(AssertUnwindSafe(|| encoder(job)))
                                .unwrap_or_else(|_| {
                                    Err("worker de conversión terminó con panic".to_owned())
                                });
                            if result_sender
                                .send(ConversionResult {
                                    sequence,
                                    shape,
                                    worker_nanos: started
                                        .elapsed()
                                        .as_nanos()
                                        .min(u128::from(u64::MAX))
                                        as u64,
                                    encoded,
                                })
                                .is_err()
                            {
                                return;
                            }
                        }
                    })
                    .map_err(|error| format!("no se pudo iniciar worker KV: {error}"))?,
            );
        }
        drop(result_sender);
        Ok(Self {
            sender: Some(sender),
            results,
            handles,
            next_submit: first_sequence,
            next_deliver: first_sequence,
            pending: BTreeMap::new(),
            worker_nanos: 0,
            closed: false,
        })
    }

    /// Blocks when the bounded queue is full: explicit producer backpressure.
    pub fn submit(&mut self, mut job: Bf16PageJob) -> Result<u64, String> {
        let sequence = self.next_submit;
        job.sequence = sequence;
        self.sender
            .as_ref()
            .ok_or_else(|| "worker de conversión cerrado".to_owned())?
            .send(job)
            .map_err(|_| "workers de conversión terminados".to_owned())?;
        self.next_submit += 1;
        Ok(sequence)
    }

    /// Non-blocking: returns only the contiguous ordered prefix already ready.
    pub fn try_collect_ordered(&mut self) -> Vec<ConversionResult<O>> {
        while let Ok(result) = self.results.try_recv() {
            self.pending.insert(result.sequence, result);
        }
        self.take_contiguous()
    }

    /// Wait for the next sequence, buffering faster later conversions.
    pub fn wait_next(&mut self) -> Result<ConversionResult<O>, String> {
        if let Some(result) = self.pending.remove(&self.next_deliver) {
            self.worker_nanos = self.worker_nanos.saturating_add(result.worker_nanos);
            self.next_deliver += 1;
            return Ok(result);
        }
        loop {
            let result = self.results.recv().map_err(|_| {
                "canal de resultados cerrado antes del resultado esperado".to_owned()
            })?;
            if result.sequence == self.next_deliver {
                self.worker_nanos = self.worker_nanos.saturating_add(result.worker_nanos);
                self.next_deliver += 1;
                return Ok(result);
            }
            self.pending.insert(result.sequence, result);
        }
    }

    /// Stop accepting jobs, finish every accepted conversion, and join threads.
    pub fn close(&mut self) -> Result<(), String> {
        if self.closed {
            return Ok(());
        }
        self.sender.take();
        for handle in self.handles.drain(..) {
            handle
                .join()
                .map_err(|_| "worker de conversión terminó con panic".to_owned())?;
        }
        self.closed = true;
        Ok(())
    }

    pub fn stats(&self) -> ConversionStats {
        ConversionStats {
            submitted: self.next_submit,
            delivered: self.next_deliver,
            buffered_out_of_order: self.pending.len(),
            closed: self.closed,
            worker_nanos: self.worker_nanos,
        }
    }

    fn take_contiguous(&mut self) -> Vec<ConversionResult<O>> {
        let mut ordered = Vec::new();
        while let Some(result) = self.pending.remove(&self.next_deliver) {
            self.worker_nanos = self.worker_nanos.saturating_add(result.worker_nanos);
            self.next_deliver += 1;
            ordered.push(result);
        }
        ordered
    }
}

impl<O: Send + 'static> Drop for ConversionWorker<O> {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

fn validate(shape: PageShape, key: &[u16], value: &[u16]) -> Result<(), String> {
    if shape.heads == 0 || shape.tokens == 0 || shape.dim == 0 || shape.tile == 0 {
        return Err("geometría BF16 contiene cero".into());
    }
    if !shape.dim.is_multiple_of(shape.tile) || !shape.dim.is_multiple_of(2) {
        return Err("dim debe ser par y múltiplo de tile".into());
    }
    let expected = shape
        .heads
        .checked_mul(shape.tokens)
        .and_then(|value| value.checked_mul(shape.dim))
        .ok_or_else(|| "geometría BF16 desborda usize".to_owned())?;
    if key.len() != expected || value.len() != expected {
        return Err(format!(
            "longitud BF16 incorrecta: esperada {expected}, K={}, V={}",
            key.len(),
            value.len()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc::TryRecvError;
    use std::time::Duration;

    fn shape() -> PageShape {
        PageShape {
            layer: 2,
            token_start: 64,
            heads: 1,
            tokens: 2,
            dim: 16,
            tile: 16,
        }
    }

    fn job(fill: u16) -> Bf16PageJob {
        Bf16PageJob::copied(shape(), &[fill; 32], &[fill + 1; 32]).unwrap()
    }

    #[test]
    fn copied_bf16_is_independent_and_preserves_exact_bits() {
        let mut source = vec![0x3f80_u16; 32];
        let copied = Bf16PageJob::copied(shape(), &source, &source).unwrap();
        source.fill(0);
        let mut worker =
            ConversionWorker::spawn(2, 1, |job: Bf16PageJob| Ok(job.key_bf16)).unwrap();
        worker.submit(copied).unwrap();
        let result = worker.wait_next().unwrap().encoded.unwrap();
        assert!(result.iter().all(|&bits| bits == 0x3f80));
        worker.close().unwrap();
    }

    #[test]
    fn owned_bf16_preserves_payload_without_changing_geometry() {
        let key = vec![0x3f80_u16; 32];
        let value = vec![0x4000_u16; 32];
        let key_pointer = key.as_ptr();
        let value_pointer = value.as_ptr();
        let job = Bf16PageJob::owned(shape(), key, value).unwrap();
        assert_eq!(job.key_bf16.as_ptr(), key_pointer);
        assert_eq!(job.value_bf16.as_ptr(), value_pointer);
        assert!(job.key_bf16.iter().all(|bits| *bits == 0x3f80));
        assert!(job.value_bf16.iter().all(|bits| *bits == 0x4000));
    }

    #[test]
    fn parallel_results_are_delivered_in_submission_order() {
        let mut worker = ConversionWorker::spawn(4, 3, |job: Bf16PageJob| {
            if job.sequence == 0 {
                thread::sleep(Duration::from_millis(40));
            }
            Ok(job.sequence)
        })
        .unwrap();
        for value in 0..4 {
            worker.submit(job(value)).unwrap();
        }
        let results = (0..4)
            .map(|_| worker.wait_next().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            results
                .iter()
                .map(|result| result.sequence)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        worker.close().unwrap();
    }

    #[test]
    fn encoder_panic_is_reported_without_losing_order_or_worker_capacity() {
        let mut worker = ConversionWorker::spawn(2, 1, |job: Bf16PageJob| {
            if job.sequence == 0 {
                panic!("panic de prueba");
            }
            Ok(job.sequence)
        })
        .unwrap();
        worker.submit(job(1)).unwrap();
        worker.submit(job(2)).unwrap();

        let failed = worker.wait_next().unwrap();
        assert_eq!(failed.sequence, 0);
        assert_eq!(
            failed.encoded.unwrap_err(),
            "worker de conversión terminó con panic"
        );
        let recovered = worker.wait_next().unwrap();
        assert_eq!(recovered.sequence, 1);
        assert_eq!(recovered.encoded.unwrap(), 1);
        worker.close().unwrap();
    }

    #[test]
    fn worker_time_is_accumulated_when_results_are_delivered() {
        let mut worker = ConversionWorker::spawn(2, 2, |job: Bf16PageJob| {
            thread::sleep(Duration::from_millis(1));
            Ok(job.sequence)
        })
        .unwrap();
        worker.submit(job(1)).unwrap();
        worker.submit(job(2)).unwrap();
        assert_eq!(worker.stats().worker_nanos, 0);
        let _ = worker.wait_next().unwrap();
        let first = worker.stats().worker_nanos;
        assert!(first > 0, "el trabajador debe registrar trabajo real");
        let _ = worker.wait_next().unwrap();
        assert!(worker.stats().worker_nanos >= first);
        worker.close().unwrap();
    }

    #[test]
    fn bounded_queue_applies_backpressure() {
        let gate = Arc::new(Barrier::new(2));
        let entered = Arc::new(AtomicBool::new(false));
        let worker_gate = gate.clone();
        let worker_entered = entered.clone();
        let (events, observed) = mpsc::channel();
        let producer = thread::spawn(move || {
            let mut worker = ConversionWorker::spawn(1, 1, move |job: Bf16PageJob| {
                if job.sequence == 0 {
                    worker_entered.store(true, Ordering::SeqCst);
                    worker_gate.wait();
                }
                Ok(job.sequence)
            })
            .unwrap();
            worker.submit(job(1)).unwrap();
            while !entered.load(Ordering::SeqCst) {
                thread::yield_now();
            }
            worker.submit(job(2)).unwrap(); // sole queue slot
            events.send("before-third").unwrap();
            worker.submit(job(3)).unwrap(); // must block until gate opens
            events.send("after-third").unwrap();
            worker.close().unwrap();
        });
        assert_eq!(observed.recv().unwrap(), "before-third");
        thread::sleep(Duration::from_millis(30));
        assert!(matches!(observed.try_recv(), Err(TryRecvError::Empty)));
        gate.wait();
        assert_eq!(observed.recv().unwrap(), "after-third");
        producer.join().unwrap();
    }

    #[test]
    fn close_drains_accepted_jobs_and_rejects_new_work() {
        let mut worker =
            ConversionWorker::spawn(2, 1, |job: Bf16PageJob| Ok(job.sequence)).unwrap();
        worker.submit(job(1)).unwrap();
        worker.submit(job(2)).unwrap();
        worker.close().unwrap();
        assert!(worker.submit(job(3)).is_err());
        assert_eq!(worker.wait_next().unwrap().sequence, 0);
        assert_eq!(worker.wait_next().unwrap().sequence, 1);
        assert!(worker.stats().closed);
    }
}
