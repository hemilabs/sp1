//! Runs gnark's CPU Groth16 prover for one proof and exits, so the prover that started it gets all
//! of that memory back. See `sp1_recursion_gnark_ffi::cpu_helper_main` for the arguments and exit
//! codes.
//!
//! Binaries that call `run_groth16_cpu_helper_if_requested` first thing in `main`, such as
//! `sp1-gpu-server`, are their own helper and do not need this one.

fn main() {
    std::process::exit(sp1_recursion_gnark_ffi::cpu_helper_main(std::env::args_os().skip(1)));
}
