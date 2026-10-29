//! Bounded, single-owner inference worker.
//!
//! The model executor is moved into exactly one OS thread. Request handlers
//! only clone [`InferenceWorker`] (a bounded queue sender); they never share or
//! lock the model. [`InferenceResponse`] is a standard `Future`, so Axum/Tokio
//! may await it without blocking an async worker thread.

use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::future::Future;
use std::panic::{self, AssertUnwindSafe};
use std::pin::Pin;
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll, Waker};
use std::thread;

/// Synchronous model operation owned and called only by the worker thread.
pub trait InferenceExecutor: Send + 'static {
    type Request: Send + 'static;
    type Response: Send + 'static;
    type Error: Send + 'static;

    fn execute(&mut self, request: Self::Request) -> Result<Self::Response, Self::Error>;
}

/// Submission failed before ownership of the request was accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitError {
    QueueFull,
    WorkerStopped,
}

impl Display for SubmitError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::QueueFull => formatter.write_str("inference queue is full"),
            Self::WorkerStopped => formatter.write_str("inference worker has stopped"),
        }
    }
}

impl Error for SubmitError {}

/// Completion failure after the worker accepted a request.
#[derive(Debug, PartialEq, Eq)]
pub enum InferenceError<E> {
    Executor(E),
    ExecutorPanicked,
    WorkerStopped,
}

impl<E: Display> Display for InferenceError<E> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Executor(error) => write!(formatter, "inference failed: {error}"),
            Self::ExecutorPanicked => formatter.write_str("inference executor panicked"),
            Self::WorkerStopped => formatter.write_str("inference worker stopped before replying"),
        }
    }
}

impl<E: Error + 'static> Error for InferenceError<E> {}

struct ResponseState<R, E> {
    result: Option<Result<R, InferenceError<E>>>,
    waker: Option<Waker>,
}

struct ResponseCell<R, E> {
    state: Mutex<ResponseState<R, E>>,
    ready: Condvar,
}

impl<R, E> ResponseCell<R, E> {
    fn pending() -> Self {
        Self {
            state: Mutex::new(ResponseState {
                result: None,
                waker: None,
            }),
            ready: Condvar::new(),
        }
    }

    fn complete(&self, result: Result<R, InferenceError<E>>) {
        let waker = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if state.result.is_some() {
                return;
            }
            state.result = Some(result);
            state.waker.take()
        };
        self.ready.notify_all();
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}

struct Completion<R, E> {
    cell: Arc<ResponseCell<R, E>>,
    completed: bool,
}

impl<R, E> Completion<R, E> {
    fn finish(mut self, result: Result<R, InferenceError<E>>) {
        self.cell.complete(result);
        self.completed = true;
    }
}

impl<R, E> Drop for Completion<R, E> {
    fn drop(&mut self) {
        if !self.completed {
            self.cell.complete(Err(InferenceError::WorkerStopped));
        }
    }
}

struct Job<Q, R, E> {
    request: Q,
    completion: Completion<R, E>,
}

/// Awaitable response for one accepted inference request.
pub struct InferenceResponse<R, E> {
    cell: Arc<ResponseCell<R, E>>,
}

impl<R, E> InferenceResponse<R, E> {
    /// Blocking counterpart intended for synchronous callers and unit tests.
    pub fn wait(self) -> Result<R, InferenceError<E>> {
        let mut state = self
            .cell
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        while state.result.is_none() {
            state = self
                .cell
                .ready
                .wait(state)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        state.result.take().expect("result checked above")
    }
}

impl<R, E> Future for InferenceResponse<R, E> {
    type Output = Result<R, InferenceError<E>>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self
            .cell
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(result) = state.result.take() {
            Poll::Ready(result)
        } else {
            let replace = state
                .waker
                .as_ref()
                .is_none_or(|registered| !registered.will_wake(context.waker()));
            if replace {
                state.waker = Some(context.waker().clone());
            }
            Poll::Pending
        }
    }
}

/// Cloneable, non-blocking submission handle for the single model owner.
pub struct InferenceWorker<Q, R, E> {
    sender: SyncSender<Job<Q, R, E>>,
}

impl<Q, R, E> Clone for InferenceWorker<Q, R, E> {
    fn clone(&self) -> Self {
        Self {
            sender: self.sender.clone(),
        }
    }
}

impl<Q, R, E> InferenceWorker<Q, R, E>
where
    Q: Send + 'static,
    R: Send + 'static,
    E: Send + 'static,
{
    /// Start one named owner thread and a queue with exactly `queue_capacity`
    /// waiting slots. Capacity excludes the request currently executing.
    pub fn spawn<X>(executor: X, queue_capacity: usize) -> std::io::Result<Self>
    where
        X: InferenceExecutor<Request = Q, Response = R, Error = E>,
    {
        if queue_capacity == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "inference queue capacity must be greater than zero",
            ));
        }
        let (sender, receiver) = mpsc::sync_channel::<Job<Q, R, E>>(queue_capacity);
        thread::Builder::new()
            .name("terrsvusion-inference".to_owned())
            .spawn(move || {
                let mut executor = executor;
                while let Ok(job) = receiver.recv() {
                    let result =
                        panic::catch_unwind(AssertUnwindSafe(|| executor.execute(job.request)));
                    match result {
                        Ok(Ok(response)) => job.completion.finish(Ok(response)),
                        Ok(Err(error)) => {
                            job.completion.finish(Err(InferenceError::Executor(error)))
                        }
                        Err(_) => {
                            job.completion.finish(Err(InferenceError::ExecutorPanicked));
                            break;
                        }
                    }
                }
            })?;
        Ok(Self { sender })
    }

    /// Submit without waiting for queue space. `QueueFull` is explicit
    /// backpressure suitable for mapping to HTTP 429 or 503 plus Retry-After.
    pub fn try_submit(&self, request: Q) -> Result<InferenceResponse<R, E>, SubmitError> {
        let cell = Arc::new(ResponseCell::pending());
        let job = Job {
            request,
            completion: Completion {
                cell: Arc::clone(&cell),
                completed: false,
            },
        };
        match self.sender.try_send(job) {
            Ok(()) => Ok(InferenceResponse { cell }),
            Err(TrySendError::Full(_)) => Err(SubmitError::QueueFull),
            Err(TrySendError::Disconnected(_)) => Err(SubmitError::WorkerStopped),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    struct RecordingExecutor {
        owner: Arc<Mutex<Option<thread::ThreadId>>>,
        calls: Arc<AtomicUsize>,
    }

    impl InferenceExecutor for RecordingExecutor {
        type Request = usize;
        type Response = usize;
        type Error = &'static str;

        fn execute(&mut self, request: usize) -> Result<usize, Self::Error> {
            let current = thread::current().id();
            let mut owner = self.owner.lock().unwrap();
            match *owner {
                Some(expected) => assert_eq!(expected, current),
                None => *owner = Some(current),
            }
            self.calls.fetch_add(1, Ordering::SeqCst);
            if request == 13 {
                Err("bad request")
            } else {
                Ok(request * 2)
            }
        }
    }

    #[test]
    fn one_thread_owns_executor_and_errors_are_request_local() {
        let owner = Arc::new(Mutex::new(None));
        let calls = Arc::new(AtomicUsize::new(0));
        let worker = InferenceWorker::spawn(
            RecordingExecutor {
                owner: Arc::clone(&owner),
                calls: Arc::clone(&calls),
            },
            4,
        )
        .unwrap();

        assert_eq!(worker.try_submit(3).unwrap().wait(), Ok(6));
        assert_eq!(
            worker.try_submit(13).unwrap().wait(),
            Err(InferenceError::Executor("bad request"))
        );
        assert_eq!(worker.try_submit(5).unwrap().wait(), Ok(10));
        assert_eq!(calls.load(Ordering::SeqCst), 3);
        assert_ne!(*owner.lock().unwrap(), Some(thread::current().id()));
    }

    struct GatedExecutor {
        started: mpsc::Sender<()>,
        release: mpsc::Receiver<()>,
    }

    impl InferenceExecutor for GatedExecutor {
        type Request = usize;
        type Response = usize;
        type Error = ();

        fn execute(&mut self, request: usize) -> Result<usize, ()> {
            if request == 1 {
                self.started.send(()).unwrap();
                self.release.recv().unwrap();
            }
            Ok(request)
        }
    }

    #[test]
    fn bounded_queue_rejects_excess_without_blocking_submitter() {
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let worker = InferenceWorker::spawn(
            GatedExecutor {
                started: started_tx,
                release: release_rx,
            },
            1,
        )
        .unwrap();

        let running = worker.try_submit(1).unwrap();
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let queued = worker.try_submit(2).unwrap();
        assert!(matches!(worker.try_submit(3), Err(SubmitError::QueueFull)));
        release_tx.send(()).unwrap();
        assert_eq!(running.wait(), Ok(1));
        assert_eq!(queued.wait(), Ok(2));
    }

    struct PanicExecutor;

    impl InferenceExecutor for PanicExecutor {
        type Request = ();
        type Response = ();
        type Error = ();

        fn execute(&mut self, (): ()) -> Result<(), ()> {
            panic!("synthetic executor panic")
        }
    }

    #[test]
    fn panic_is_reported_and_later_submissions_see_stopped_worker() {
        let worker = InferenceWorker::spawn(PanicExecutor, 1).unwrap();
        assert_eq!(
            worker.try_submit(()).unwrap().wait(),
            Err(InferenceError::ExecutorPanicked)
        );
        for _ in 0..100 {
            if matches!(worker.try_submit(()), Err(SubmitError::WorkerStopped)) {
                return;
            }
            thread::yield_now();
        }
        panic!("worker sender did not observe receiver shutdown");
    }

    #[test]
    fn zero_capacity_is_rejected() {
        let owner = Arc::new(Mutex::new(None));
        let calls = Arc::new(AtomicUsize::new(0));
        let result = InferenceWorker::spawn(RecordingExecutor { owner, calls }, 0);
        assert!(matches!(result, Err(error) if error.kind() == std::io::ErrorKind::InvalidInput));
    }
}
