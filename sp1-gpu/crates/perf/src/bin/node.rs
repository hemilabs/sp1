use std::time::Duration;

use clap::Parser;
use sp1_core_executor::SP1Context;
use sp1_gpu_perf::{get_program_and_input, Measurement};
use sp1_gpu_prover::cuda_worker_builder_with_machine;
use sp1_prover::worker::{SP1LocalNodeBuilder, SP1Proof};
use sp1_prover_types::network_base_types::ProofMode;
use sp1_sdk::RiscvAir;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "local-fibonacci")]
    pub program: String,
    #[arg(long, default_value = "")]
    pub param: String,
    #[arg(long, default_value = "false")]
    pub telemetry: bool,
    #[arg(long, default_value = "core")]
    pub mode: String,
    #[arg(long, short, default_value = "1")]
    pub num_iterations: usize,
}

fn proof_mode_from_string(s: &str) -> ProofMode {
    match s {
        "core" => ProofMode::Core,
        "compressed" => ProofMode::Compressed,
        "groth16" => ProofMode::Groth16,
        "plonk" => ProofMode::Plonk,
        _ => panic!("invalid proof mode provided: {s}"),
    }
}

// mimalloc with large OS pages on hosts with enough RAM; see `sp1_gpu_host_alloc`.
#[global_allocator]
static GLOBAL: sp1_gpu_host_alloc::HostAlloc = sp1_gpu_host_alloc::HostAlloc;

fn main() {
    // First: when started as its own Groth16 CPU helper, prove and exit here. See
    // `sp1_recursion_gnark_ffi::run_groth16_cpu_helper_if_requested`.
    sp1_recursion_gnark_ffi::run_groth16_cpu_helper_if_requested();
    // A failed proof ends the run with its error and exit code 1, once the runtime is down:
    // exiting while its tasks still use the GPU crashes in the runtime's teardown.
    if let Err(e) = run() {
        eprintln!("proof failed: {e}");
        sp1_core_executor::unlink_owned_shm();
        std::process::exit(1);
    }
}

#[tokio::main]
#[allow(clippy::field_reassign_with_default)]
#[allow(clippy::print_stdout)]
async fn run() -> Result<(), String> {
    let args = Args::parse();

    // Load the environment variables.
    dotenv::dotenv().ok();

    // Initialize the tracer.
    #[cfg(not(feature = "tokio-blocked"))]
    if args.telemetry {
        use opentelemetry::KeyValue;
        use opentelemetry_sdk::Resource;
        use sp1_gpu_perf::telemetry;

        let resource = Resource::new(vec![KeyValue::new("service.name", "sp1-gpu-node")]);
        telemetry::init(resource);
    } else {
        sp1_gpu_tracing::init_tracer();
    }
    #[cfg(feature = "tokio-blocked")]
    {
        use tokio_blocked::TokioBlockedLayer;
        use tracing_subscriber::{prelude::*, EnvFilter};

        let fmt = tracing_subscriber::fmt::layer().with_filter(EnvFilter::from_default_env());

        let duration = std::env::var("TOKIO_BLOCKED_WARN_DURATION_MICROS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(1000);
        let blocked = TokioBlockedLayer::new()
            .with_warn_busy_single_poll(Some(Duration::from_micros(duration)));

        tracing_subscriber::registry().with(fmt).with(blocked).init();
    }

    tracing::info!("host allocator: {}", sp1_gpu_host_alloc::host_allocator_name());

    // Get the program and input.
    let (elf, stdin) = get_program_and_input(args.program.clone(), args.param);
    let machine = RiscvAir::machine();

    // Initialize the AirProver and permits
    let measurements = sp1_gpu_cudart::spawn(move |t| async move {
        let base_builder = cuda_worker_builder_with_machine(t.clone(), machine).await;
        #[cfg(feature = "mprotect")]
        let worker_builder = base_builder.without_vk_verification();
        #[cfg(not(feature = "mprotect"))]
        let worker_builder = base_builder;
        let client =
            SP1LocalNodeBuilder::from_worker_client_builder(worker_builder).build().await.unwrap();

        let context = SP1Context::default();

        // Skip separate execute() — the Controller re-executes internally anyway.
        // We'll get cycles from the proof's public values after proving.

        let time = tokio::time::Instant::now();
        let vk = client.setup(&elf).await.unwrap();
        let setup_time = time.elapsed();
        tracing::info!("setup time: {:?}", setup_time);

        // Run the prover for a number of iterations.
        let mut measurements = Vec::with_capacity(args.num_iterations);
        for _ in 0..args.num_iterations {
            let mode = proof_mode_from_string(&args.mode);
            let stdin = stdin.clone();
            let context = context.clone();
            let time = tokio::time::Instant::now();
            tracing::info!("proving with mode: {mode:?}");
            let proved = client.prove_with_mode_timed(&elf, stdin, context, mode).await;
            let (proof, final_wrap) = match proved {
                Ok(proof) => proof,
                // A proof refused because an earlier one reset the GPU (SP1_GPU_RESET_BEFORE_WRAP):
                // exit as below, since unwinding would run destructors that segfault on it.
                #[cfg(feature = "native-gnark")]
                Err(e) if sp1_recursion_gnark_ffi::gpu_was_reset() => {
                    eprintln!("proof failed: {e:#}");
                    std::io::Write::flush(&mut std::io::stdout()).ok();
                    sp1_core_executor::unlink_owned_shm();
                    std::process::exit(1);
                }
                // Failed with its retries, if any, spent: end the run (see `main`), not panic.
                Err(e) => return Err(format!("{e:#}")),
            };
            let proof_time = time.elapsed();
            tracing::info!("proof time: {:?}", proof_time);
            // A Groth16 or PLONK proof's time until its compressed proof was done.
            let compressed_time = final_wrap.map(|wrap| {
                let compressed =
                    tokio::time::Instant::from_std(wrap.compressed).saturating_duration_since(time);
                tracing::info!(
                    "final wrap: {:?} on top of the compressed proof ({compressed:?}): shrink and \
                     wrap {:?}, {mode:?} {:?}",
                    proof_time.saturating_sub(compressed),
                    wrap.shrink_wrap,
                    wrap.snark,
                );
                compressed
            });

            let (num_shards, cycles) = if let SP1Proof::Core(ref shard_proofs) = &proof.proof {
                use sp1_hypercube::air::PublicValues;
                use std::borrow::Borrow;
                let max_ts = shard_proofs
                    .iter()
                    .map(|p| {
                        let pv: &PublicValues<[_; 4], [_; 3], [_; 4], _> =
                            p.public_values.as_slice().borrow();
                        pv.range().timestamp_range.1
                    })
                    .max()
                    .unwrap_or(0);
                (shard_proofs.len(), max_ts as usize)
            } else {
                (0, 0)
            };

            // Verify the proof
            tokio::task::spawn_blocking({
                let client = client.clone();
                let vk = vk.clone();
                move || client.verify(&vk, &proof.proof)
            })
            .await
            .unwrap()
            .unwrap();
            if !sp1_gpu_host_alloc::keep_warm() {
                sp1_core_executor::release_recycled_event_buffers();
            }
            sp1_gpu_host_alloc::release_free_memory();

            let (core_time, compress_time, shrink_time, wrap_time) = match mode {
                ProofMode::Core => (Some(proof_time), None, Duration::ZERO, Duration::ZERO),
                ProofMode::Compressed => (None, Some(proof_time), Duration::ZERO, Duration::ZERO),
                // Compress Time: until the compressed proof was done; Wrap Time: the final wrap on
                // top of it (the shrink and wrap STARKs, then the Groth16 or PLONK proof).
                ProofMode::Groth16 | ProofMode::Plonk => match compressed_time {
                    Some(compressed) => (
                        None,
                        Some(compressed),
                        Duration::ZERO,
                        proof_time.saturating_sub(compressed),
                    ),
                    None => (None, None, Duration::ZERO, proof_time),
                },
                _ => panic!("invalid proof mode: {mode:?}"),
            };
            let measurement = Measurement {
                name: args.program.clone(),
                cycles,
                num_shards,
                core_time,
                compress_time,
                shrink_time,
                wrap_time,
            };
            println!("{measurement}");
            measurements.push(measurement);
        }

        // Exit explicitly once all proofs are produced and verified, while still
        // inside the GPU task scope. The Groth16/PLONK wrap path can call
        // `cudaDeviceReset` in the parent to free VRAM for the helper subprocess
        // (SP1_GPU_RESET_BEFORE_WRAP, where the helper would not fit otherwise),
        // which invalidates the parent's remaining CUDA-backed objects (caching
        // allocator, device buffers, task scope). Letting the scope unwind then
        // runs their `Drop` impls, which call `cudaFree` on dead pointers and
        // segfault. All work is done here, so skip destructors and exit cleanly.
        // Skipping them would leave the executor shared memory still owned (the one the fixed-PK
        // mode caches for the next proof) in /dev/shm: unlink it first.
        println!("All {} measurements done", measurements.len());
        std::io::Write::flush(&mut std::io::stdout()).ok();
        unsafe { sp1_gpu_cudart::sys::runtime::sp1_profile_flush() };
        sp1_core_executor::unlink_owned_shm();
        std::process::exit(0);
        #[allow(unreachable_code)]
        Ok(measurements)
    })
    .await
    .unwrap()?;

    if args.telemetry {
        tokio::task::spawn_blocking(opentelemetry::global::shutdown_tracer_provider).await.unwrap();
    }

    println!("All {} measurements done", measurements.len());
    Ok(())
}
