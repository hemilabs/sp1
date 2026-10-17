//! Exclusive advisory file locks (`flock`) for the host-wide Groth16 queue and the per-circuit
//! caches.
//!
//! A lock belongs to an open file description, so threads that each open the file exclude each
//! other, and the kernel releases the lock when the last descriptor closes, however the holder
//! exits. Lock files are never deleted: a process locking a deleted file and one locking its
//! replacement would both believe they hold the lock. [`is_current`] lets a holder detect that.

use std::fs::File;
use std::io;
use std::path::Path;

/// Opens (creating if needed) a lock file every local user can lock. A file another user created
/// is opened read-only if need be, which `flock` accepts on local filesystems (not on NFS).
#[cfg(unix)]
pub(crate) fn open(path: &Path) -> io::Result<File> {
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
    match std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o666)
        .open(path)
    {
        Ok(file) => {
            // The mode above is masked by the umask; fchmod is not. Only the owner may, so other
            // users' attempts fail harmlessly.
            let _ = file.set_permissions(std::fs::Permissions::from_mode(0o666));
            Ok(file)
        }
        Err(write_err) => File::open(path).map_err(|_| write_err),
    }
}

/// Tries to take the lock without blocking. `Ok(false)` means another holder has it.
#[cfg(unix)]
pub(crate) fn try_lock(file: &File) -> io::Result<bool> {
    flock(file, libc::LOCK_EX | libc::LOCK_NB).map(|()| true).or_else(|err| {
        if err.raw_os_error() == Some(libc::EWOULDBLOCK) {
            Ok(false)
        } else {
            Err(err)
        }
    })
}

/// Takes the lock, waiting as long as it takes.
#[cfg(unix)]
pub(crate) fn lock(file: &File) -> io::Result<()> {
    flock(file, libc::LOCK_EX)
}

#[cfg(unix)]
fn flock(file: &File, operation: libc::c_int) -> io::Result<()> {
    use std::os::fd::AsRawFd;
    loop {
        // SAFETY: flock on a descriptor we own; no memory is passed.
        if unsafe { libc::flock(file.as_raw_fd(), operation) } == 0 {
            return Ok(());
        }
        let err = io::Error::last_os_error();
        if err.kind() != io::ErrorKind::Interrupted {
            return Err(err);
        }
    }
}

/// Whether `path` still names the file `file` has open. False once someone has deleted or
/// replaced it, in which case a lock on `file` excludes nobody who opens `path` from now on.
#[cfg(unix)]
pub(crate) fn is_current(file: &File, path: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    match (file.metadata(), std::fs::metadata(path)) {
        (Ok(open), Ok(named)) => open.dev() == named.dev() && open.ino() == named.ino(),
        _ => false,
    }
}

// Without flock there is nothing to coordinate with: locking succeeds at once, as it did for
// these callers before they were locked at all.
#[cfg(not(unix))]
pub(crate) fn open(path: &Path) -> io::Result<File> {
    std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(false).open(path)
}

#[cfg(not(unix))]
pub(crate) fn try_lock(_file: &File) -> io::Result<bool> {
    Ok(true)
}

#[cfg(not(unix))]
pub(crate) fn lock(_file: &File) -> io::Result<()> {
    Ok(())
}

#[cfg(not(unix))]
pub(crate) fn is_current(_file: &File, _path: &Path) -> bool {
    true
}
