#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

use clap::Parser;
use server::Server;
use sp1_gpu_cudart::run_in_place;

mod server;

#[derive(Debug, Parser)]
struct Args {
    #[clap(long)]
    version: bool,
}

#[tokio::main]
#[allow(clippy::print_stdout)]
async fn main() {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    if args.version {
        println!("{}", sp1_primitives::SP1_CRATE_VERSION);
        return;
    }

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
