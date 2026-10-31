//! Data types and binary loading for the GPU Groth16 prover.
//!
//! The proving data is exported from Go/gnark via a flat binary format,
//! following the same pattern as the PLONK GPU prover (LE canonical bytes).

use crate::fq2::Fq2;
use crate::g2::G2Affine;
use crate::{BN254Fr, BN254G1Affine, Fq, Fr};
use rayon::prelude::*;

/// Groth16 proving data loaded from exported binary files.
pub struct Groth16ProvingData {
    /// Domain cardinality (number of constraints, rounded to power of 2)
    pub domain_size: usize,
    pub lg_domain_size: u32,

    /// Number of wires (public + private)
    pub nb_wires: usize,
    /// Number of public inputs
    pub nb_public: usize,
    /// Number of points at infinity in A
    pub nb_infinity_a: usize,
    /// Number of points at infinity in B
    pub nb_infinity_b: usize,

    /// Proving key G1 points
    pub pk_g1_a: Vec<BN254G1Affine>,
    pub pk_g1_b: Vec<BN254G1Affine>,
    pub pk_g1_z: Vec<BN254G1Affine>,
    pub pk_g1_k: Vec<BN254G1Affine>,

    /// Proving key G2 points
    pub pk_g2_b: Vec<G2Affine>,
    /// Pre-converted arkworks G2 bases for fast MSM. Converted once at load
    /// time to avoid paying the ~6s Fq→ark-Fq conversion on every proof.
    pub pk_g2_b_ark: Vec<ark_bn254::G2Affine>,

    /// Scalar proving key elements
    pub pk_g1_alpha: BN254G1Affine,
    pub pk_g1_beta: BN254G1Affine,
    pub pk_g1_delta: BN254G1Affine,
    pub pk_g2_beta: G2Affine,
    pub pk_g2_delta: G2Affine,

    /// Infinity masks
    pub infinity_a: Vec<bool>,
    pub infinity_b: Vec<bool>,

    /// Wire indices to exclude from K MSM (PrivateCommitted + CommitmentIndexes)
    pub k_wire_filter: Vec<usize>,

    /// NTT domain generator (omega) in Fr
    pub omega: Fr,
}

/// Solved Groth16 witness data (exported from gnark R1CS solver).
///
/// Its vectors are not pinned: the prover reads them only on the host, copying them into its
/// own pinned buffers (the wire-value scatters, `compute_h`'s copies), and pinning their 2 GB
/// made each wrap ~35 ms slower.
pub struct Groth16WitnessData {
    /// All wire values (public + private), pre-converted to Montgomery Fr at
    /// load time to avoid ~350ms per-proof conversion in the hot prove path.
    pub wire_values: Vec<Fr>,
    /// A constraint evaluation vector
    pub solution_a: Vec<BN254Fr>,
    /// B constraint evaluation vector
    pub solution_b: Vec<BN254Fr>,
    /// C constraint evaluation vector
    pub solution_c: Vec<BN254Fr>,
    /// Pre-computed H polynomial coefficients (natural order, canonical Fr).
    /// Computed by gnark's computeH and bit-reversed to natural order on export.
    pub h_coefficients: Vec<Fr>,
    /// Pre-computed Pedersen commitments (from gnark BSB22)
    pub commitments: Vec<BN254G1Affine>,
    /// Pedersen commitment proof-of-knowledge
    pub commitment_pok: BN254G1Affine,
}

/// Groth16 proof (BN254).
#[derive(Debug)]
pub struct Groth16Proof {
    /// π_A ∈ G1
    pub ar: BN254G1Affine,
    /// π_B ∈ G2
    pub bs: G2Affine,
    /// π_C ∈ G1
    pub krs: BN254G1Affine,
    /// Pedersen commitments
    pub commitments: Vec<BN254G1Affine>,
    /// Pedersen commitment proof-of-knowledge
    pub commitment_pok: BN254G1Affine,
}

impl Groth16ProvingData {
    /// Load Groth16 proving data from the exported binary directory.
    pub fn load(dir: &str) -> anyhow::Result<Self> {
        let dir = std::path::Path::new(dir);
        let t_load = std::time::Instant::now();

        // Read metadata
        let meta = std::fs::read(dir.join("groth16_metadata.bin"))?;
        let mut off = 0;
        let read_u64 = |data: &[u8], offset: &mut usize| -> u64 {
            let val = u64::from_le_bytes(data[*offset..*offset + 8].try_into().unwrap());
            *offset += 8;
            val
        };

        let domain_size = read_u64(&meta, &mut off) as usize;
        let nb_wires = read_u64(&meta, &mut off) as usize;
        let nb_public = read_u64(&meta, &mut off) as usize;
        let nb_infinity_a = read_u64(&meta, &mut off) as usize;
        let nb_infinity_b = read_u64(&meta, &mut off) as usize;
        let _nb_commitments = read_u64(&meta, &mut off) as usize;
        let _len_g1_a = read_u64(&meta, &mut off) as usize;
        let _len_g1_b = read_u64(&meta, &mut off) as usize;
        let _len_g1_z = read_u64(&meta, &mut off) as usize;
        let _len_g1_k = read_u64(&meta, &mut off) as usize;
        let _len_g2_b = read_u64(&meta, &mut off) as usize;

        // Omega (32 bytes LE canonical Fr)
        let omega = load_single_fr(&meta[off..off + 32]);
        off += 32;

        // Infinity masks
        let infinity_a: Vec<bool> = meta[off..off + nb_wires].iter().map(|&b| b != 0).collect();
        off += nb_wires;
        let infinity_b: Vec<bool> = meta[off..off + nb_wires].iter().map(|&b| b != 0).collect();
        off += nb_wires;

        // K wire filter indices
        let num_filter = read_u64(&meta, &mut off) as usize;
        let mut k_wire_filter = Vec::with_capacity(num_filter);
        for _ in 0..num_filter {
            k_wire_filter.push(read_u64(&meta, &mut off) as usize);
        }

        // Domain sanity: must be a non-zero power of two so trailing_zeros(), `domain_size - 1`,
        // and all downstream NTT indexing are well-defined.
        anyhow::ensure!(
            domain_size > 0 && domain_size.is_power_of_two(),
            "domain_size ({domain_size}) must be a non-zero power of two",
        );
        anyhow::ensure!(
            nb_wires >= nb_public,
            "nb_wires ({nb_wires}) must be >= nb_public ({nb_public})",
        );
        anyhow::ensure!(
            nb_infinity_a <= nb_wires,
            "nb_infinity_a ({nb_infinity_a}) must be <= nb_wires ({nb_wires})",
        );
        anyhow::ensure!(
            nb_infinity_b <= nb_wires,
            "nb_infinity_b ({nb_infinity_b}) must be <= nb_wires ({nb_wires})",
        );

        let lg_domain_size = domain_size.trailing_zeros();

        // Cross-validate nb_infinity_{a,b} counters against the infinity masks to
        // detect metadata/export corruption early.
        let mask_infinity_a = infinity_a.iter().filter(|b| **b).count();
        let mask_infinity_b = infinity_b.iter().filter(|b| **b).count();
        anyhow::ensure!(
            mask_infinity_a == nb_infinity_a,
            "infinity_a mask count ({mask_infinity_a}) does not match nb_infinity_a ({nb_infinity_a})",
        );
        anyhow::ensure!(
            mask_infinity_b == nb_infinity_b,
            "infinity_b mask count ({mask_infinity_b}) does not match nb_infinity_b ({nb_infinity_b})",
        );

        // Load PK G1 points
        let pk_g1_a = load_g1_points(&dir.join("pk_g1_a.bin"))?;
        let pk_g1_b = load_g1_points(&dir.join("pk_g1_b.bin"))?;
        let pk_g1_z = load_g1_points(&dir.join("pk_g1_z.bin"))?;
        let pk_g1_k = load_g1_points(&dir.join("pk_g1_k.bin"))?;

        // Validate PK sizes against metadata so a length mismatch surfaces here rather
        // than as an MSM-time `assert_eq!` panic.
        anyhow::ensure!(
            pk_g1_a.len() == nb_wires - nb_infinity_a,
            "pk_g1_a.len() ({}) != nb_wires - nb_infinity_a ({} - {} = {})",
            pk_g1_a.len(),
            nb_wires,
            nb_infinity_a,
            nb_wires - nb_infinity_a,
        );
        anyhow::ensure!(
            pk_g1_b.len() == nb_wires - nb_infinity_b,
            "pk_g1_b.len() ({}) != nb_wires - nb_infinity_b ({} - {} = {})",
            pk_g1_b.len(),
            nb_wires,
            nb_infinity_b,
            nb_wires - nb_infinity_b,
        );
        anyhow::ensure!(
            pk_g1_z.len() == domain_size - 1,
            "pk_g1_z.len() ({}) != domain_size - 1 ({})",
            pk_g1_z.len(),
            domain_size - 1,
        );
        // pk_g1_k length matches gnark setup: nbWires - nbPublic - len(k_wire_filter),
        // where k_wire_filter is PrivateCommitted wires + CommitmentIndex wires.
        let expected_k_len = nb_wires - nb_public - k_wire_filter.len();
        anyhow::ensure!(
            pk_g1_k.len() == expected_k_len,
            "pk_g1_k.len() ({}) != nb_wires - nb_public - k_wire_filter.len() ({} - {} - {} = {})",
            pk_g1_k.len(),
            nb_wires,
            nb_public,
            k_wire_filter.len(),
            expected_k_len,
        );

        eprintln!("[groth16-load] G1 points loaded: {:?}", t_load.elapsed());
        // Load PK G2 points
        let pk_g2_b = load_g2_points(&dir.join("pk_g2_b.bin"))?;
        anyhow::ensure!(
            pk_g2_b.len() == nb_wires - nb_infinity_b,
            "pk_g2_b.len() ({}) != nb_wires - nb_infinity_b ({} - {} = {})",
            pk_g2_b.len(),
            nb_wires,
            nb_infinity_b,
            nb_wires - nb_infinity_b,
        );
        // Pre-convert G2 bases to arkworks format (pays ~6s one-time, saves ~6s
        // on every proof). Parallel conversion.
        let t = std::time::Instant::now();
        let pk_g2_b_ark = crate::g2::g2_affine_to_ark_batch(&pk_g2_b);
        eprintln!("[groth16-load] pk_g2_b → arkworks conversion: {:?}", t.elapsed());

        // Load scalar PK elements (each file must contain exactly one point).
        let alpha_pts = load_g1_points(&dir.join("pk_g1_alpha.bin"))?;
        let beta_pts = load_g1_points(&dir.join("pk_g1_beta.bin"))?;
        let delta_pts = load_g1_points(&dir.join("pk_g1_delta.bin"))?;
        let g2_beta_pts = load_g2_points(&dir.join("pk_g2_beta.bin"))?;
        let g2_delta_pts = load_g2_points(&dir.join("pk_g2_delta.bin"))?;
        for (name, len) in [
            ("pk_g1_alpha.bin", alpha_pts.len()),
            ("pk_g1_beta.bin", beta_pts.len()),
            ("pk_g1_delta.bin", delta_pts.len()),
            ("pk_g2_beta.bin", g2_beta_pts.len()),
            ("pk_g2_delta.bin", g2_delta_pts.len()),
        ] {
            anyhow::ensure!(len == 1, "{name} must contain exactly 1 point, got {len}");
        }

        eprintln!("[groth16-load] proving data loaded: {:?}", t_load.elapsed());
        Ok(Self {
            domain_size,
            lg_domain_size,
            nb_wires,
            nb_public,
            nb_infinity_a,
            nb_infinity_b,
            pk_g1_a,
            pk_g1_b,
            pk_g1_z,
            pk_g1_k,
            pk_g2_b,
            pk_g2_b_ark,
            pk_g1_alpha: alpha_pts[0],
            pk_g1_beta: beta_pts[0],
            pk_g1_delta: delta_pts[0],
            pk_g2_beta: g2_beta_pts[0],
            pk_g2_delta: g2_delta_pts[0],
            infinity_a,
            infinity_b,
            k_wire_filter,
            omega,
        })
    }
}

impl Groth16ProvingData {
    /// [`Self::load`] on at most `threads` threads (all of the global pool's when `None`), for a
    /// helper that loads the key while the prover it serves still needs the CPU.
    pub fn load_with_threads(dir: &str, threads: Option<usize>) -> anyhow::Result<Self> {
        match threads {
            Some(n) => {
                rayon::ThreadPoolBuilder::new().num_threads(n).build()?.install(|| Self::load(dir))
            }
            None => Self::load(dir),
        }
    }
}

impl Groth16WitnessData {
    /// Load solved witness data from the exported binary directory.
    pub fn load(dir: &str) -> anyhow::Result<Self> {
        let dir = std::path::Path::new(dir);

        // Load wire_values using auto-detection: if the file has the "MFr1"
        // Montgomery magic header, the Fr values are used directly (zero
        // conversion cost). Otherwise falls back to canonical-to-Montgomery
        // conversion.
        // The four ~0.5 GB files are independent: read them at once.
        let ((wire_values, solution_a), (solution_b, solution_c)) = rayon::join(
            || {
                rayon::join(
                    || load_fr_elements_auto(&dir.join("wire_values.bin")),
                    || load_fr_elements(&dir.join("solution_a.bin")),
                )
            },
            || {
                rayon::join(
                    || load_fr_elements(&dir.join("solution_b.bin")),
                    || load_fr_elements(&dir.join("solution_c.bin")),
                )
            },
        );
        let (wire_values, solution_a, solution_b, solution_c) =
            (wire_values?, solution_a?, solution_b?, solution_c?);

        // H is computed on GPU; load if present (backward compat), else empty.
        let h_path = dir.join("h_coefficients.bin");
        let h_coefficients =
            if h_path.exists() { load_fr_elements_auto(&h_path)? } else { Vec::new() };

        let commitments = if dir.join("commitments.bin").exists() {
            load_g1_points(&dir.join("commitments.bin"))?
        } else {
            Vec::new()
        };

        let commitment_pok = if dir.join("commitment_pok.bin").exists() {
            let pts = load_g1_points(&dir.join("commitment_pok.bin"))?;
            if pts.is_empty() {
                BN254G1Affine {
                    x: crate::BN254Fq { limbs: [0; 8] },
                    y: crate::BN254Fq { limbs: [0; 8] },
                }
            } else {
                pts[0]
            }
        } else {
            BN254G1Affine {
                x: crate::BN254Fq { limbs: [0; 8] },
                y: crate::BN254Fq { limbs: [0; 8] },
            }
        };

        Ok(Self {
            wire_values,
            solution_a,
            solution_b,
            solution_c,
            h_coefficients,
            commitments,
            commitment_pok,
        })
    }
}

// === Binary loading helpers (matching PLONK pattern) ===

fn load_single_fr(data: &[u8]) -> Fr {
    // Data is LE canonical bytes. Convert to BN254Fr then to Montgomery Fr.
    let mut elem = BN254Fr { limbs: [0; 8] };
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), elem.limbs.as_mut_ptr() as *mut u8, 32);
    }
    Fr::from_bn254fr(&elem)
}

fn load_g1_points(path: &std::path::Path) -> anyhow::Result<Vec<BN254G1Affine>> {
    let data = std::fs::read(path)?;
    anyhow::ensure!(
        data.len() % 64 == 0,
        "{}: size {} not multiple of 64",
        path.display(),
        data.len()
    );
    // LE canonical bytes → BN254Fq (canonical) → Fq (Montgomery) → BN254Fq (Montgomery), in
    // parallel: ~16M points per file, which a serial loop took seconds over.
    let points = data
        .par_chunks_exact(64)
        .with_min_len(4096)
        .map(|chunk| {
            let mut canonical_x = crate::BN254Fq { limbs: [0; 8] };
            let mut canonical_y = crate::BN254Fq { limbs: [0; 8] };
            unsafe {
                std::ptr::copy_nonoverlapping(
                    chunk.as_ptr(),
                    canonical_x.limbs.as_mut_ptr() as *mut u8,
                    32,
                );
                std::ptr::copy_nonoverlapping(
                    chunk[32..].as_ptr(),
                    canonical_y.limbs.as_mut_ptr() as *mut u8,
                    32,
                );
            }
            let mont_x = Fq::from_bn254fq_canonical(&canonical_x);
            let mont_y = Fq::from_bn254fq_canonical(&canonical_y);
            BN254G1Affine { x: mont_x.to_bn254fq_raw(), y: mont_y.to_bn254fq_raw() }
        })
        .collect();
    Ok(points)
}

fn load_g2_points(path: &std::path::Path) -> anyhow::Result<Vec<G2Affine>> {
    let data = std::fs::read(path)?;
    anyhow::ensure!(
        data.len() % 128 == 0,
        "{}: size {} not multiple of 128",
        path.display(),
        data.len()
    );
    let points = data
        .par_chunks_exact(128)
        .with_min_len(2048)
        .map(|chunk| {
            // Each G2 point: [X.A1_LE, X.A0_LE, Y.A1_LE, Y.A0_LE] (A1 first), 32 bytes each
            let load_fq = |off: usize| -> Fq {
                let mut canonical = crate::BN254Fq { limbs: [0; 8] };
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        chunk[off..].as_ptr(),
                        canonical.limbs.as_mut_ptr() as *mut u8,
                        32,
                    );
                }
                Fq::from_bn254fq_canonical(&canonical)
            };
            // gnark G2Affine.RawBytes() serializes as [X.A1, X.A0, Y.A1, Y.A0] (each 32 bytes).
            // After reverseBytes: file layout is [X.A1_LE, X.A0_LE, Y.A1_LE, Y.A0_LE].
            // Fq2 = c0 + c1*u, where c0 = A0 (real), c1 = A1 (imaginary).
            let x = Fq2::new(load_fq(32), load_fq(0)); // c0=A0, c1=A1
            let y = Fq2::new(load_fq(96), load_fq(64)); // c0=A0, c1=A1
            G2Affine { x, y }
        })
        .collect();
    Ok(points)
}

fn load_fr_elements(path: &std::path::Path) -> anyhow::Result<Vec<BN254Fr>> {
    read_elements(path, 0)
}

/// Magic header for Montgomery-form Fr files written by `writeFrFileMontgomery`.
const FR_MONTGOMERY_MAGIC: &[u8; 4] = b"MFr1";

/// Load Fr elements that may be in either canonical LE or raw Montgomery format.
///
/// If the file starts with the "MFr1" magic header, the remaining bytes are
/// interpreted as raw Montgomery `Fr` values (4 x u64 LE limbs each) with no
/// conversion needed. This is possible because gnark's `fr.Element` and our
/// `Fr` type use identical Montgomery representations (same R = 2^256 mod r).
///
/// If the magic header is absent, the file is treated as canonical LE format
/// (the legacy path) and each element is converted via `Fr::from_bn254fr`.
///
/// This auto-detection makes the loader backward-compatible with old exports.
fn load_fr_elements_auto(path: &std::path::Path) -> anyhow::Result<Vec<Fr>> {
    use std::io::Read;
    let mut magic = [0u8; 4];
    let has_magic =
        std::fs::File::open(path)?.read_exact(&mut magic).is_ok() && &magic == FR_MONTGOMERY_MAGIC;

    if has_magic {
        let elements: Vec<Fr> = read_elements(path, FR_MONTGOMERY_MAGIC.len() as u64)?;
        eprintln!(
            "[groth16-load] {}: loaded {} Fr elements (Montgomery, no conversion)",
            path.display(),
            elements.len()
        );
        Ok(elements)
    } else {
        // Legacy canonical LE format: load as BN254Fr, convert to Montgomery Fr.
        let raw: Vec<BN254Fr> = read_elements(path, 0)?;
        let elements: Vec<Fr> = raw.par_iter().map(Fr::from_bn254fr).collect();
        eprintln!(
            "[groth16-load] {}: loaded {} Fr elements (canonical, converted to Montgomery)",
            path.display(),
            elements.len()
        );
        Ok(elements)
    }
}

/// Field elements that are 32 bytes of plain data, little-endian limbs, and valid when all-zero:
/// what [`read_elements`] may read straight from a file.
///
/// # Safety
///
/// Implementors must be exactly 32 bytes, with no padding, and every bit pattern must be a value
/// of the type.
unsafe trait RawElement: Copy + Send + Sync {}

// SAFETY: `#[repr(C)]` over `[u32; 8]`.
unsafe impl RawElement for BN254Fr {}
// SAFETY: `#[repr(transparent)]` over `[u64; 4]`.
unsafe impl RawElement for Fr {}

/// Reads the elements in `path`, from byte `skip` on, straight into the vector that is returned,
/// in parallel chunks. The witness files are ~0.5 GB each in RAM-backed `/dev/shm`, so their cost
/// is the copy and the page faults, which reading into an intermediate buffer doubled.
fn read_elements<T: RawElement>(path: &std::path::Path, skip: u64) -> anyhow::Result<Vec<T>> {
    const CHUNK_BYTES: usize = 16 << 20;
    assert_eq!(std::mem::size_of::<T>(), 32);
    let file = std::fs::File::open(path)?;
    let len = file.metadata()?.len();
    anyhow::ensure!(
        len >= skip && (len - skip).is_multiple_of(32),
        "{}: size {} (after a {skip}-byte header) not multiple of 32",
        path.display(),
        len
    );
    let n = ((len - skip) / 32) as usize;
    if n == 0 {
        return Ok(Vec::new());
    }
    let layout = std::alloc::Layout::array::<T>(n)?;
    // SAFETY: `layout` has a non-zero size, and an all-zero `T` is a valid value (`RawElement`),
    // so the zeroed allocation holds `n` initialized elements; it was made by the global
    // allocator with `T`'s layout, as `Vec::from_raw_parts` requires.
    let mut elements = unsafe {
        let ptr = std::alloc::alloc_zeroed(layout) as *mut T;
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Vec::from_raw_parts(ptr, n, n)
    };
    // SAFETY: `T` is 32 bytes without padding and any bytes are a valid `T` (`RawElement`).
    let bytes = unsafe { std::slice::from_raw_parts_mut(elements.as_mut_ptr() as *mut u8, n * 32) };
    bytes.par_chunks_mut(CHUNK_BYTES).enumerate().try_for_each(|(i, chunk)| {
        // Windows serializes I/O on one synchronous handle: each chunk reads through its own.
        #[cfg(windows)]
        let file =
            std::fs::File::open(path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
        read_exact_at(&file, chunk, skip + (i * CHUNK_BYTES) as u64)
            .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))
    })?;
    Ok(elements)
}

/// Fills `buf` from `file` at `offset`, without the file's cursor: chunks read in parallel.
#[cfg(unix)]
fn read_exact_at(file: &std::fs::File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(file, buf, offset)
}

/// Fills `buf` from `file` at `offset`. Each read names its offset, so chunks read in parallel
/// don't depend on the file's cursor.
#[cfg(windows)]
fn read_exact_at(file: &std::fs::File, mut buf: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        match file.seek_read(buf, offset) {
            Ok(0) => return Err(std::io::ErrorKind::UnexpectedEof.into()),
            Ok(n) => {
                buf = &mut buf[n..];
                offset += n as u64;
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &std::path::Path, name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn elements_are_read_exactly_as_the_old_loader_copied_them() {
        let dir = tempfile::tempdir().unwrap();
        // Several read chunks, the last one partial.
        let n = (3 << 20) / 2 + 7;
        let bytes: Vec<u8> = (0..n * 32).map(|i| (i * 7 + i / 251) as u8).collect();
        let canonical = write(dir.path(), "canonical.bin", &bytes);
        let read = load_fr_elements(&canonical).unwrap();
        assert_eq!(read.len(), n);
        for (i, e) in read.iter().enumerate() {
            let mut expected = BN254Fr { limbs: [0; 8] };
            for (j, limb) in expected.limbs.iter_mut().enumerate() {
                let o = i * 32 + j * 4;
                *limb = u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
            }
            assert_eq!(*e, expected, "element {i}");
        }

        let mut with_magic = FR_MONTGOMERY_MAGIC.to_vec();
        with_magic.extend_from_slice(&bytes);
        let montgomery = write(dir.path(), "montgomery.bin", &with_magic);
        let read = load_fr_elements_auto(&montgomery).unwrap();
        assert_eq!(read.len(), n);
        for (i, e) in read.iter().enumerate() {
            let mut expected = [0u64; 4];
            for (j, limb) in expected.iter_mut().enumerate() {
                let o = i * 32 + j * 8;
                *limb = u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
            }
            assert_eq!(e.0, expected, "element {i}");
        }
    }

    #[test]
    fn empty_and_ragged_files() {
        let dir = tempfile::tempdir().unwrap();
        assert!(load_fr_elements(&write(dir.path(), "empty.bin", &[])).unwrap().is_empty());
        assert!(load_fr_elements_auto(&write(dir.path(), "magic.bin", FR_MONTGOMERY_MAGIC))
            .unwrap()
            .is_empty());
        assert!(load_fr_elements(&write(dir.path(), "ragged.bin", &[0; 33])).is_err());
    }
}
