//! Whether the GPU Groth16 prover streams its G1 point sets (see `prover::stream_g1_srs`): the
//! one rule both the prover and the process that starts its helper (`sp1-recursion-gnark-ffi`,
//! which sizes the helper's GPU memory by it) decide by. Host-only, without the `cuda` feature.

/// Total GPU memory below which the prover streams its G1 point sets by default: the 16 GB tier.
/// A 24 GB card reports just under 24 GiB (RTX 4090 23.65, RX 7900 XTX 23.98), and keeps them.
pub const STREAM_SRS_BELOW_BYTES: u64 = 20 << 30;

/// Whether the prover streams its G1 point sets on a card with `total` memory: as
/// `SP1_GROTH16_STREAM_SRS` (`setting`) says (`1`/`true` or `0`/`false`), else on cards below
/// [`STREAM_SRS_BELOW_BYTES`]. An unknown total counts as a card that keeps them.
pub fn streams(setting: Option<&str>, total: Option<u64>) -> bool {
    match setting.map(str::trim) {
        Some("1" | "true") => true,
        Some("0" | "false") => false,
        _ => total.is_some_and(|total| total < STREAM_SRS_BELOW_BYTES),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    #[test]
    fn streams_on_16_gb_cards_and_as_the_setting_says() {
        // RTX 5060 Ti / RX 9060 XT (16 GB), RTX 4090 (24 GB), RTX 5090 (32 GB).
        assert!(streams(None, Some(16 * GIB - GIB / 8)));
        assert!(!streams(None, Some(24 * GIB - GIB / 2)));
        assert!(!streams(None, Some(32 * GIB)));
        assert!(streams(None, Some(STREAM_SRS_BELOW_BYTES - 1)));
        assert!(!streams(None, Some(STREAM_SRS_BELOW_BYTES)));
        assert!(!streams(None, None), "an unknown card keeps its point sets");
        assert!(streams(Some("1"), Some(48 * GIB)));
        assert!(streams(Some(" true"), None));
        assert!(!streams(Some(" false "), Some(16 * GIB)));
        assert!(!streams(Some("0"), Some(16 * GIB)));
        // Anything else is the default.
        assert!(streams(Some("yes"), Some(16 * GIB)));
        assert!(!streams(Some(""), Some(32 * GIB)));
    }
}
