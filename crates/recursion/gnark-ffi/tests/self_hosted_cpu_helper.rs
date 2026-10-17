//! The self-hosted CPU helper, end to end: like `sp1-gpu-server`, this binary calls
//! `run_groth16_cpu_helper_if_requested` first thing in `main`, and is then started as its own
//! helper. Needs no circuit: it checks what a prover relies on before any proving starts.

// A test without the harness: its result is its output.
#![allow(clippy::print_stdout)]

use std::process::{Command, Output};

use sp1_recursion_gnark_ffi::CPU_HELPER_ARG;

fn main() {
    sp1_recursion_gnark_ffi::run_groth16_cpu_helper_if_requested();

    let helper = |args: &[&str]| -> Output {
        Command::new(std::env::current_exe().unwrap())
            .arg(CPU_HELPER_ARG)
            .args(args)
            .output()
            .unwrap()
    };
    let text = |bytes: &[u8]| String::from_utf8_lossy(bytes).into_owned();

    // A binary that can serve says so: what a deployment can probe for.
    let help = helper(&["--help"]);
    assert_eq!(help.status.code(), Some(0), "{help:?}");
    assert!(text(&help.stdout).contains("Runs gnark's CPU Groth16 prover"), "{help:?}");

    // Bad arguments are usage errors, before any Go runs.
    let bogus = helper(&["--build-dir", "/nonexistent", "--bogus"]);
    assert_eq!(bogus.status.code(), Some(64), "{bogus:?}");
    assert!(text(&bogus.stderr).contains("--bogus"), "{bogus:?}");

    let dir = std::env::temp_dir();
    let empty = dir.to_str().unwrap();
    let prepare = helper(&["--prepare", "--build-dir", empty]);
    assert_eq!(prepare.status.code(), Some(64), "{prepare:?}");
    assert!(text(&prepare.stderr).contains("no groth16_vk.bin"), "{prepare:?}");

    println!("self-hosted CPU helper: ok");
}
