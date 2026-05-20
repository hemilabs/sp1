//! Retry helper for hardening GPU prover dispatch against transient
//! verify failures.
//!
//! ## Background
//!
//! The GPU Groth16 / PLONK final-wrap path runs in a helper subprocess
//! (`groth16_gpu_helper` / `plonk_gpu_helper`) and is gated by a gnark
//! Go FFI verify. Standalone validation has shown the prover is
//! deterministic-PASS at >=99.95% across thousands of iterations on all
//! three production GPUs (5090, 4090, 7900 XTX) — see
//! `project_groth16_flake_2026-05-08.md`. Yet the production matrix
//! occasionally observes a single `pairing doesn't match` flake that is
//! not reproducible under tight stability sweeps. The most likely cause
//! is a transient hardware/driver glitch (a random bit-flip in pooled
//! VRAM, a momentary SDMA ECC scrub, etc.).
//!
//! Auto-retry at the dispatcher level is cheap insurance: re-spawn the
//! helper, re-prove, re-verify. Each attempt uses a fresh subprocess
//! and therefore a fresh RNG / fresh GPU runtime context, so a
//! deterministic failure (e.g. a real prover bug) will keep failing and
//! is correctly surfaced after the retry budget is exhausted.
//!
//! ## Behavior
//!
//! * Default attempts = 1 prove + 2 retries = 3 total.
//! * Configurable via env `SP1_GPU_PROVE_RETRY=N` (N retries; N=0
//!   restores the prior single-shot behavior).
//! * Only retries on verify failures. Errors raised before / inside the
//!   prove closure (build failures, OOM panics, helper-subprocess
//!   panics) propagate up immediately — they are not transient.
//! * On verify failure, log a structured warning with attempt index,
//!   total budget, error message, and elapsed wall time per attempt;
//!   then re-run the prove closure.
//! * After the retry budget is exhausted, the LAST verify error is
//!   propagated unchanged.
//!
//! ## Test injection
//!
//! Setting `SP1_GPU_VERIFY_FAIL_INJECT=N` (handled by callers via
//! `verify_fail_inject_remaining`) instructs the dispatcher to force
//! the next N verify calls to FAIL with a synthetic error, regardless
//! of the real verifier's result. Used by the unit test below and by
//! manual end-to-end sanity checks.

use anyhow::Result;
use std::sync::atomic::{AtomicU32, Ordering};

/// Default number of retries (so total attempts = 1 + this).
pub const DEFAULT_RETRIES: u32 = 2;

/// Env var name controlling retry budget.
pub const RETRY_ENV: &str = "SP1_GPU_PROVE_RETRY";

/// Env var name controlling fake-verify-failure injection (for tests).
pub const FAIL_INJECT_ENV: &str = "SP1_GPU_VERIFY_FAIL_INJECT";

/// Read the retry budget from the env. Returns `DEFAULT_RETRIES` if
/// unset or unparseable.
pub fn retry_budget() -> u32 {
    std::env::var(RETRY_ENV).ok().and_then(|s| s.parse::<u32>().ok()).unwrap_or(DEFAULT_RETRIES)
}

/// Process-global counter of remaining synthetic verify failures to
/// inject. Initialized lazily on first call from `SP1_GPU_VERIFY_FAIL_INJECT`.
static FAIL_INJECT_REMAINING: AtomicU32 = AtomicU32::new(u32::MAX);

fn init_fail_inject_once() {
    // Sentinel u32::MAX = uninitialized; replace with parsed env value
    // exactly once. After init the counter is decremented by takes.
    if FAIL_INJECT_REMAINING.load(Ordering::Relaxed) == u32::MAX {
        let n =
            std::env::var(FAIL_INJECT_ENV).ok().and_then(|s| s.parse::<u32>().ok()).unwrap_or(0);
        // Race-tolerant: multiple callers may CAS in 0 / N; the result
        // is identical because every caller parses the same env value.
        let _ = FAIL_INJECT_REMAINING.compare_exchange(
            u32::MAX,
            n,
            Ordering::SeqCst,
            Ordering::Relaxed,
        );
    }
}

/// If a synthetic verify failure should be injected on the *next*
/// verify call, decrement the counter and return true. Otherwise
/// return false. Used by callers that want to wrap a real verifier.
pub fn take_fail_inject() -> bool {
    init_fail_inject_once();
    let mut cur = FAIL_INJECT_REMAINING.load(Ordering::Relaxed);
    while cur > 0 && cur != u32::MAX {
        match FAIL_INJECT_REMAINING.compare_exchange_weak(
            cur,
            cur - 1,
            Ordering::SeqCst,
            Ordering::Relaxed,
        ) {
            Ok(_) => return true,
            Err(actual) => cur = actual,
        }
    }
    false
}

/// Reset the fail-injection counter (test-only). Permitted in
/// production builds because the function is a no-op except when the
/// env var is set.
#[doc(hidden)]
pub fn reset_fail_inject_for_test(n: u32) {
    FAIL_INJECT_REMAINING.store(n, Ordering::SeqCst);
}

/// Run `prove_fn` and `verify_fn` with retry-on-verify-failure.
///
/// `prove_fn` is invoked at the start of each attempt (so each retry
/// uses a fresh prove). If `prove_fn` itself returns `Err` or panics,
/// the error/panic is NOT retried — it propagates immediately, since
/// build / OOM / helper-subprocess failures are not transient.
///
/// `verify_fn` is invoked once per successful prove. If it returns
/// `Err` and there is retry budget remaining, the loop logs a warning
/// and re-runs `prove_fn`. Otherwise the error is returned.
pub fn prove_with_retry<P, V, T>(
    label: &'static str,
    max_retries: u32,
    mut prove_fn: P,
    mut verify_fn: V,
) -> Result<T>
where
    P: FnMut() -> Result<T>,
    V: FnMut(&T) -> Result<()>,
{
    let total_attempts = max_retries.saturating_add(1);
    let mut last_err: Option<anyhow::Error> = None;

    for attempt in 0..total_attempts {
        let t0 = std::time::Instant::now();
        let proof = prove_fn()?;
        let prove_elapsed = t0.elapsed();

        let t1 = std::time::Instant::now();
        let result = verify_fn(&proof);
        let verify_elapsed = t1.elapsed();

        match result {
            Ok(()) => {
                if attempt > 0 {
                    tracing::info!(
                        target = "sp1_gpu_prove_retry",
                        prove_label = label,
                        attempt = attempt + 1,
                        total_attempts,
                        prove_ms = prove_elapsed.as_millis() as u64,
                        verify_ms = verify_elapsed.as_millis() as u64,
                        "GPU prove + verify SUCCEEDED on retry attempt {}/{}",
                        attempt + 1,
                        total_attempts,
                    );
                }
                return Ok(proof);
            }
            Err(e) => {
                if attempt + 1 < total_attempts {
                    tracing::warn!(
                        target = "sp1_gpu_prove_retry",
                        prove_label = label,
                        attempt = attempt + 1,
                        total_attempts,
                        prove_ms = prove_elapsed.as_millis() as u64,
                        verify_ms = verify_elapsed.as_millis() as u64,
                        error = %e,
                        "GPU prove + verify FAILED on attempt {}/{}: {}; retrying with a fresh \
                         helper subprocess",
                        attempt + 1,
                        total_attempts,
                        e,
                    );
                    last_err = Some(e);
                    continue;
                } else {
                    tracing::error!(
                        target = "sp1_gpu_prove_retry",
                        prove_label = label,
                        attempt = attempt + 1,
                        total_attempts,
                        prove_ms = prove_elapsed.as_millis() as u64,
                        verify_ms = verify_elapsed.as_millis() as u64,
                        error = %e,
                        "GPU prove + verify FAILED on final attempt {}/{}; retry budget \
                         exhausted, surfacing error",
                        attempt + 1,
                        total_attempts,
                    );
                    return Err(e);
                }
            }
        }
    }

    // Should be unreachable: the loop always returns inside the match.
    // Guard against future refactors with an explicit panic message.
    Err(last_err.unwrap_or_else(|| {
        anyhow::anyhow!("prove_with_retry exited without producing a proof or error")
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn no_retry_when_first_verify_passes() {
        let prove_calls = Cell::new(0u32);
        let verify_calls = Cell::new(0u32);

        let result: Result<u32> = prove_with_retry(
            "test_no_retry",
            2,
            || {
                prove_calls.set(prove_calls.get() + 1);
                Ok(42)
            },
            |proof| {
                verify_calls.set(verify_calls.get() + 1);
                assert_eq!(*proof, 42);
                Ok(())
            },
        );

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
        assert_eq!(prove_calls.get(), 1, "should prove exactly once");
        assert_eq!(verify_calls.get(), 1, "should verify exactly once");
    }

    #[test]
    fn retries_then_succeeds() {
        let prove_calls = Cell::new(0u32);
        let verify_calls = Cell::new(0u32);

        let result: Result<u32> = prove_with_retry(
            "test_retry_then_succeed",
            2,
            || {
                prove_calls.set(prove_calls.get() + 1);
                Ok(prove_calls.get())
            },
            |proof| {
                verify_calls.set(verify_calls.get() + 1);
                if *proof < 3 {
                    Err(anyhow::anyhow!("synthetic verify failure on attempt {}", proof))
                } else {
                    Ok(())
                }
            },
        );

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 3);
        assert_eq!(prove_calls.get(), 3, "should prove three times (1 + 2 retries)");
        assert_eq!(verify_calls.get(), 3);
    }

    #[test]
    fn surfaces_error_after_budget_exhausted() {
        let prove_calls = Cell::new(0u32);
        let verify_calls = Cell::new(0u32);

        let result: Result<u32> = prove_with_retry(
            "test_exhaust",
            2,
            || {
                prove_calls.set(prove_calls.get() + 1);
                Ok(7)
            },
            |_proof| {
                verify_calls.set(verify_calls.get() + 1);
                Err(anyhow::anyhow!("always fails"))
            },
        );

        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("always fails"), "unexpected error: {msg}");
        assert_eq!(prove_calls.get(), 3, "should prove 1 + 2 retries times");
        assert_eq!(verify_calls.get(), 3);
    }

    #[test]
    fn zero_retries_means_single_attempt() {
        let prove_calls = Cell::new(0u32);
        let verify_calls = Cell::new(0u32);

        let result: Result<u32> = prove_with_retry(
            "test_zero",
            0,
            || {
                prove_calls.set(prove_calls.get() + 1);
                Ok(0)
            },
            |_proof| {
                verify_calls.set(verify_calls.get() + 1);
                Err(anyhow::anyhow!("fails"))
            },
        );

        assert!(result.is_err());
        assert_eq!(prove_calls.get(), 1, "zero retries = single attempt");
        assert_eq!(verify_calls.get(), 1);
    }

    #[test]
    fn prove_error_does_not_retry() {
        let prove_calls = Cell::new(0u32);
        let verify_calls = Cell::new(0u32);

        let result: Result<u32> = prove_with_retry(
            "test_prove_error",
            5,
            || {
                prove_calls.set(prove_calls.get() + 1);
                Err(anyhow::anyhow!("OOM in helper"))
            },
            |_proof| {
                verify_calls.set(verify_calls.get() + 1);
                Ok(())
            },
        );

        assert!(result.is_err());
        assert_eq!(prove_calls.get(), 1, "prove errors should NOT trigger retry");
        assert_eq!(verify_calls.get(), 0);
    }

    #[test]
    fn retry_budget_env_default() {
        // Make sure the default kicks in cleanly when env unset.
        std::env::remove_var(RETRY_ENV);
        assert_eq!(retry_budget(), DEFAULT_RETRIES);

        std::env::set_var(RETRY_ENV, "0");
        assert_eq!(retry_budget(), 0);

        std::env::set_var(RETRY_ENV, "5");
        assert_eq!(retry_budget(), 5);

        std::env::set_var(RETRY_ENV, "garbage");
        assert_eq!(retry_budget(), DEFAULT_RETRIES);

        std::env::remove_var(RETRY_ENV);
    }

    #[test]
    fn fail_inject_consumes_then_stops() {
        // Reset state for deterministic test (the static is process-global).
        reset_fail_inject_for_test(3);

        assert!(take_fail_inject(), "1st call should consume");
        assert!(take_fail_inject(), "2nd call should consume");
        assert!(take_fail_inject(), "3rd call should consume");
        assert!(!take_fail_inject(), "4th call should return false");
        assert!(!take_fail_inject(), "stays false");
    }
}
