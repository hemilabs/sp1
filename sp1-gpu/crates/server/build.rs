//! Names the commit the server is built from, for `--capabilities`, when the source is a git
//! checkout and `git` can say (`cargo install --git` builds from one). Otherwise nothing.

use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn main() {
    let Some(commit) = git(&["rev-parse", "HEAD"]).filter(|c| c.len() == 40) else {
        return;
    };
    println!("cargo:rustc-env=SP1_GPU_SERVER_COMMIT={commit}");
    // Again when HEAD moves: the file naming the branch, and the branch's own ref.
    for path in ["HEAD".to_string()]
        .into_iter()
        .chain(git(&["symbolic-ref", "-q", "HEAD"]))
        .chain(["packed-refs".to_string()])
    {
        if let Some(file) = git(&["rev-parse", "--git-path", &path]) {
            println!("cargo:rerun-if-changed={file}");
        }
    }
}
