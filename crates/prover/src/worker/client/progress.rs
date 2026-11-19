//! A proof's progress, as task counts, for a process that runs this one and wants to show it.
//!
//! A proof's request is answered once, with the proof, and nothing in between says how far it has
//! got. When [`PROGRESS_FILE_ENV`] names a file, the local worker client keeps, for each proof,
//! how many tasks of each kind were submitted and how many succeeded, and rewrites the file with
//! the counts of the proof a task last moved in. A file, not a message on the request's stream, so
//! a server and an SDK of different versions still talk as before, and the reader is whoever set
//! the variable (the process that starts `sp1-gpu-server` passes its environment on).
//!
//! The file is one line, rewritten whole:
//!
//! ```text
//! proof=<id> CORE_EXECUTE=0/1 PROVE_SHARD=12/40 RECURSION_REDUCE=3/9 SHRINK_WRAP=0/0 GROTH16_WRAP=0/0
//! ```
//!
//! `KIND=done/submitted`, one for each kind the proof has submitted. The shard total is known
//! once `CORE_EXECUTE` is done (execution submits shards as it goes), the reduce total once the
//! shards are all proven. A reader may see a partly written file: it should skip a line it cannot
//! parse and read again.

use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::Mutex,
    time::{Duration, Instant},
};

use hashbrown::HashMap;
use sp1_prover_types::TaskType;

use crate::worker::{ProofId, TaskId};

/// The environment variable naming the progress file. Unset: no counts are kept.
pub const PROGRESS_FILE_ENV: &str = "SP1_PROGRESS_FILE";

/// The least time between two writes for a shard or a reduce task; any other kind is written at
/// once, as it marks a new stage.
const MIN_WRITE_GAP: Duration = Duration::from_millis(100);

#[derive(Default)]
struct Counts {
    /// (done, submitted) per kind.
    by_kind: BTreeMap<i32, (u32, u32)>,
}

impl Counts {
    fn line(&self, proof_id: &ProofId) -> String {
        let mut line = format!("proof={proof_id}");
        for (kind, (done, submitted)) in &self.by_kind {
            let name = TaskType::try_from(*kind).map(|k| k.as_str_name()).unwrap_or("UNKNOWN");
            line.push_str(&format!(" {name}={done}/{submitted}"));
        }
        line
    }
}

#[derive(Default)]
struct State {
    tasks: HashMap<TaskId, (ProofId, TaskType)>,
    proofs: HashMap<ProofId, Counts>,
    last_write: Option<Instant>,
}

/// The per-proof task counts and the file they go to.
pub(crate) struct ProgressFile {
    path: Option<PathBuf>,
    state: Mutex<State>,
}

impl ProgressFile {
    /// Reads [`PROGRESS_FILE_ENV`].
    pub(crate) fn from_env() -> Self {
        let path = std::env::var_os(PROGRESS_FILE_ENV).filter(|p| !p.is_empty()).map(PathBuf::from);
        Self::at(path)
    }

    pub(crate) fn at(path: Option<PathBuf>) -> Self {
        Self { path, state: Mutex::new(State::default()) }
    }

    /// A task of `kind` was submitted for `proof_id`.
    pub(crate) fn submitted(&self, task_id: &TaskId, proof_id: &ProofId, kind: TaskType) {
        let Some(path) = &self.path else { return };
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.tasks.insert(task_id.clone(), (proof_id.clone(), kind));
        let counts = state.proofs.entry(proof_id.clone()).or_default();
        counts.by_kind.entry(kind as i32).or_default().1 += 1;
        Self::write(path, &mut state, proof_id, kind);
    }

    /// `task_id` succeeded. A retried task is counted when its last attempt succeeds.
    pub(crate) fn succeeded(&self, task_id: &TaskId) {
        let Some(path) = &self.path else { return };
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let Some((proof_id, kind)) = state.tasks.remove(task_id) else { return };
        if let Some(counts) = state.proofs.get_mut(&proof_id) {
            counts.by_kind.entry(kind as i32).or_default().0 += 1;
        }
        Self::write(path, &mut state, &proof_id, kind);
    }

    /// `proof_id` ended: its counts are dropped.
    pub(crate) fn ended(&self, proof_id: &ProofId) {
        if self.path.is_none() {
            return;
        }
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.proofs.remove(proof_id);
        state.tasks.retain(|_, (p, _)| p != proof_id);
    }

    /// The line [`write`](Self::write) would put in the file for `proof_id`, if it has counts.
    #[cfg(test)]
    pub(crate) fn line(&self, proof_id: &ProofId) -> Option<String> {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.proofs.get(proof_id).map(|c| c.line(proof_id))
    }

    fn write(path: &PathBuf, state: &mut State, proof_id: &ProofId, kind: TaskType) {
        let frequent = matches!(kind, TaskType::ProveShard | TaskType::RecursionReduce);
        let now = Instant::now();
        if frequent && state.last_write.is_some_and(|t| now.duration_since(t) < MIN_WRITE_GAP) {
            return;
        }
        let Some(counts) = state.proofs.get(proof_id) else { return };
        // Best effort: a reader that cannot follow along only shows less.
        let _ = std::fs::write(path, counts.line(proof_id));
        state.last_write = Some(now);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(n: usize) -> Vec<TaskId> {
        (0..n).map(|i| TaskId::new(format!("task{i}"))).collect()
    }

    #[test]
    fn counts_each_kind_of_a_proof_and_writes_the_file() {
        let dir = std::env::temp_dir().join(format!("sp1-progress-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("progress.txt");
        let progress = ProgressFile::at(Some(path.clone()));
        let proof = ProofId::new("p1".to_string());
        let tasks = ids(4);
        progress.submitted(&tasks[0], &proof, TaskType::CoreExecute);
        progress.submitted(&tasks[1], &proof, TaskType::ProveShard);
        progress.submitted(&tasks[2], &proof, TaskType::ProveShard);
        progress.succeeded(&tasks[1]);
        progress.succeeded(&tasks[0]);
        progress.submitted(&tasks[3], &proof, TaskType::Groth16Wrap);
        let line = progress.line(&proof).unwrap();
        assert!(line.starts_with("proof=p1 "), "{line}");
        assert!(line.contains("PROVE_SHARD=1/2"), "{line}");
        assert!(line.contains("CORE_EXECUTE=1/1"), "{line}");
        assert!(line.contains("GROTH16_WRAP=0/1"), "{line}");
        // The stage markers are written at once, so the file holds the last line.
        assert_eq!(std::fs::read_to_string(&path).unwrap(), line);
        // A task succeeding twice (or one never submitted) is not counted again.
        progress.succeeded(&tasks[1]);
        assert!(progress.line(&proof).unwrap().contains("PROVE_SHARD=1/2"));
        progress.ended(&proof);
        assert!(progress.line(&proof).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn keeps_nothing_without_a_file() {
        let progress = ProgressFile::at(None);
        let proof = ProofId::new("p2".to_string());
        let task = TaskId::new("t".to_string());
        progress.submitted(&task, &proof, TaskType::ProveShard);
        assert!(progress.line(&proof).is_none());
    }
}
