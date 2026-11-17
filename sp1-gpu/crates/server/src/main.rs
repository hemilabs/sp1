// mimalloc with large OS pages on hosts with enough RAM; see `sp1_gpu_host_alloc`.
#[global_allocator]
static GLOBAL: sp1_gpu_host_alloc::HostAlloc = sp1_gpu_host_alloc::HostAlloc;

use clap::Parser;
use server::Server;
use sp1_gpu_cudart::run_in_place;

mod server;

#[derive(Debug, Parser)]
struct Args {
    #[clap(long)]
    version: bool,
    /// Print what this server can do, one capability per line ([`CAPABILITIES`]), then
    /// `commit=<sha>` when the build knew its commit, and exit. A server older than this flag
    /// refuses it.
    #[clap(long)]
    capabilities: bool,
}

/// What `--capabilities` reports, one per line. Each name is stable: a client looks for the line.
/// - `fd-shared-memory`: the executor shares the guest's memory and trace ring with its runner
///   through file descriptors (memfds), never through names in `/dev/shm`.
const CAPABILITIES: &[&str] = &["fd-shared-memory"];

fn main() {
    // First, before the runtime or CUDA start: when started as its own Groth16 CPU helper, prove
    // and exit here. See `sp1_recursion_gnark_ffi::run_groth16_cpu_helper_if_requested`.
    sp1_recursion_gnark_ffi::run_groth16_cpu_helper_if_requested();
    serve();
}

#[tokio::main]
#[allow(clippy::print_stdout)]
async fn serve() {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    if args.version {
        println!("{}", sp1_primitives::SP1_CRATE_VERSION);
        return;
    }
    if args.capabilities {
        for capability in CAPABILITIES {
            println!("{capability}");
        }
        if let Some(commit) = option_env!("SP1_GPU_SERVER_COMMIT") {
            println!("commit={commit}");
        }
        return;
    }

    tracing::info!("host allocator: {}", sp1_gpu_host_alloc::host_allocator_name());

    let cuda_device_id = std::env::var("CUDA_VISIBLE_DEVICES")
        .expect("CUDA_VISIBLE_DEVICES must be set")
        .parse()
        .expect("Expected only one CUDA device as a u32");

    // The HIP and SYCL builds are pinned to their card by HIP_VISIBLE_DEVICES or ZE_AFFINITY_MASK,
    // so the id here only names the socket. HIP also filters devices by CUDA_VISIBLE_DEVICES: leave
    // it set and the id would hide the card (e.g. id 1 of the one device HIP_VISIBLE_DEVICES shows).
    if sp1_gpu_cudart::sys::BACKEND != "cuda" {
        std::env::remove_var("CUDA_VISIBLE_DEVICES");
    }

    let server = Server { cuda_device_id };

    if let Err(e) = run_in_place(|scope| server.run(scope)).await.await {
        eprintln!("Error running server: {e}");
    }
}
