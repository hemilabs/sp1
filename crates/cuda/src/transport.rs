//! The connection between a [`crate::CudaProver`] and `sp1-gpu-server`.
//!
//! A Unix domain socket at [`crate::client::socket_path`] on Unix. Windows has no
//! `tokio::net::UnixStream`, so there it is a named pipe, `\\.\pipe\sp1-cuda-<id>`. Either way it
//! is a byte stream carrying the same length-delimited bincode messages.

use std::{io, path::Path};

/// The client's end of a connection.
#[cfg(unix)]
pub type ClientStream = tokio::net::UnixStream;
/// The client's end of a connection.
#[cfg(windows)]
pub type ClientStream = tokio::net::windows::named_pipe::NamedPipeClient;

/// The server's end of a connection.
#[cfg(unix)]
pub type ServerStream = tokio::net::UnixStream;
/// The server's end of a connection.
#[cfg(windows)]
pub type ServerStream = tokio::net::windows::named_pipe::NamedPipeServer;

/// Connects to the server listening at `path`.
pub async fn connect(path: &Path) -> io::Result<ClientStream> {
    #[cfg(unix)]
    {
        tokio::net::UnixStream::connect(path).await
    }
    #[cfg(windows)]
    {
        tokio::net::windows::named_pipe::ClientOptions::new().open(path)
    }
}

/// Accepts connections at a server's path.
pub struct Listener {
    #[cfg(unix)]
    inner: tokio::net::UnixListener,
    /// The pipe instance the next client will connect to. A named pipe has no listener as such:
    /// each instance serves one client, so one is always kept waiting.
    #[cfg(windows)]
    next: tokio::net::windows::named_pipe::NamedPipeServer,
    path: std::path::PathBuf,
}

impl Listener {
    /// Starts listening at `path`, replacing a socket file a dead server left behind.
    ///
    /// On Windows this fails if another server already listens at `path`, as binding a live
    /// socket would; a dead server's pipe disappears with its process.
    pub fn bind(path: &Path) -> io::Result<Self> {
        #[cfg(unix)]
        {
            if let Err(e) = std::fs::remove_file(path) {
                if e.kind() != io::ErrorKind::NotFound {
                    tracing::warn!("Failed to remove orphaned socket: {}", e);
                }
            }
            Ok(Self { inner: tokio::net::UnixListener::bind(path)?, path: path.to_owned() })
        }
        #[cfg(windows)]
        {
            let next = tokio::net::windows::named_pipe::ServerOptions::new()
                .first_pipe_instance(true)
                .create(path)?;
            Ok(Self { next, path: path.to_owned() })
        }
    }

    /// Waits for the next client.
    pub async fn accept(&mut self) -> io::Result<ServerStream> {
        #[cfg(unix)]
        {
            Ok(self.inner.accept().await?.0)
        }
        #[cfg(windows)]
        {
            self.next.connect().await?;
            let fresh = tokio::net::windows::named_pipe::ServerOptions::new().create(&self.path)?;
            Ok(std::mem::replace(&mut self.next, fresh))
        }
    }

    /// The path clients connect to.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Removes what the listener leaves behind on shutdown: the socket file, on Unix.
    pub fn cleanup(&self) {
        #[cfg(unix)]
        if let Err(e) = std::fs::remove_file(&self.path) {
            tracing::error!("Failed to remove orphaned socket: {}", e);
        }
    }
}
