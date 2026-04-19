package sp1

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"os"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark/backend/plonk"
	plonk_bn254 "github.com/consensys/gnark/backend/plonk/bn254"
	cs "github.com/consensys/gnark/constraint/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr/fft"
)

// ExportPlonkData loads the PLONK proving key and constraint system, builds
// the trace (selectors + permutation polynomials), and exports everything as
// flat binary arrays for GPU consumption.
//
// Output files (written to outputDir):
//   - srs_g1_lagrange.bin: N affine G1 points (64 bytes each, LE)
//   - trace_ql.bin through trace_s3.bin: N Fr elements each (32 bytes, LE)
//   - plonk_domain_info.bin: domain metadata
func ExportPlonkData(dataDir string, outputDir string) {
	os.MkdirAll(outputDir, 0755)
	totalStart := time.Now()

	// Load constraint system (SCS)
	start := time.Now()
	fmt.Println("Loading SCS...")
	scsFile, err := os.Open(dataDir + "/" + plonkCircuitPath)
	if err != nil {
		panic(err)
	}
	defer scsFile.Close()

	scsGeneric := plonk.NewCS(ecc.BN254)
	if _, err := scsGeneric.ReadFrom(scsFile); err != nil {
		panic(fmt.Sprintf("failed to read SCS: %v", err))
	}
	spr := scsGeneric.(*cs.SparseR1CS)
	fmt.Printf("Loaded SCS in %s (%d constraints, %d public)\n",
		time.Since(start), spr.GetNbConstraints(), len(spr.Public))

	// Load proving key
	start = time.Now()
	fmt.Println("Loading proving key...")
	pkFile, err := os.Open(dataDir + "/" + plonkPkPath)
	if err != nil {
		panic(err)
	}
	defer pkFile.Close()

	pk := plonk.NewProvingKey(ecc.BN254)
	bufReader := bufio.NewReaderSize(pkFile, 1024*1024)
	if _, err := pk.UnsafeReadFrom(bufReader); err != nil {
		panic(fmt.Sprintf("failed to read PK: %v", err))
	}
	pk_bn254 := pk.(*plonk_bn254.ProvingKey)
	fmt.Printf("Loaded PK in %s\n", time.Since(start))

	// Build the trace (selectors + permutation)
	start = time.Now()
	fmt.Println("Building trace...")
	domainSize := ecc.NextPowerOfTwo(uint64(spr.GetNbConstraints() + len(spr.Public)))
	domain := fft.NewDomain(domainSize)
	trace := plonk_bn254.NewTrace(spr, domain)
	fmt.Printf("Built trace in %s (domain size %d = 2^%d)\n",
		time.Since(start), domain.Cardinality, log2(domain.Cardinality))

	// Export Lagrange SRS
	start = time.Now()
	lagrangePoints := pk_bn254.KzgLagrange.G1
	fmt.Printf("Exporting %d Lagrange SRS points...\n", len(lagrangePoints))
	writeG1File(outputDir+"/srs_g1_lagrange.bin", lagrangePoints)
	fmt.Printf("Exported Lagrange SRS in %s\n", time.Since(start))

	// Export canonical SRS (for monomial commitments in Rounds 3-5)
	start = time.Now()
	canonicalPoints := pk_bn254.Kzg.G1
	fmt.Printf("Exporting %d canonical SRS points...\n", len(canonicalPoints))
	writeG1File(outputDir+"/srs_g1_canonical.bin", canonicalPoints)
	fmt.Printf("Exported canonical SRS in %s\n", time.Since(start))

	// Export selector polynomials (in Lagrange form)
	n := int(domain.Cardinality)
	start = time.Now()
	fmt.Printf("Exporting selector polynomials (N=%d)...\n", n)
	writeFrFile(outputDir+"/trace_ql.bin", trace.Ql.Coefficients()[:n])
	writeFrFile(outputDir+"/trace_qr.bin", trace.Qr.Coefficients()[:n])
	writeFrFile(outputDir+"/trace_qm.bin", trace.Qm.Coefficients()[:n])
	writeFrFile(outputDir+"/trace_qo.bin", trace.Qo.Coefficients()[:n])
	writeFrFile(outputDir+"/trace_qk.bin", trace.Qk.Coefficients()[:n])
	fmt.Printf("Exported 5 selectors in %s\n", time.Since(start))

	// Export permutation polynomials
	start = time.Now()
	writeFrFile(outputDir+"/trace_s1.bin", trace.S1.Coefficients()[:n])
	writeFrFile(outputDir+"/trace_s2.bin", trace.S2.Coefficients()[:n])
	writeFrFile(outputDir+"/trace_s3.bin", trace.S3.Coefficients()[:n])
	fmt.Printf("Exported 3 permutation polys in %s\n", time.Since(start))

	// Export BSB22 commitment polynomials (SP1 has 1)
	start = time.Now()
	for i, qcp := range trace.Qcp {
		writeFrFile(fmt.Sprintf("%s/trace_qcp_%d.bin", outputDir, i), qcp.Coefficients()[:n])
	}
	fmt.Printf("Exported %d BSB22 polys in %s\n", len(trace.Qcp), time.Since(start))

	// Export domain info
	domainInfoFile, err := os.Create(outputDir + "/plonk_domain_info.bin")
	if err != nil {
		panic(err)
	}
	defer domainInfoFile.Close()
	binary.Write(domainInfoFile, binary.LittleEndian, domain.Cardinality)

	// Write domain generator (omega) as 32 bytes LE
	var genBytes [32]byte
	genLE := domain.Generator.Bytes() // big-endian canonical
	reverseBytes(genBytes[:], genLE[:])
	domainInfoFile.Write(genBytes[:])

	// Write NbPublicVariables as uint64 LE
	// Needed by GPU prover to fill Qk with witness public inputs
	nbPublic := uint64(len(spr.Public))
	binary.Write(domainInfoFile, binary.LittleEndian, nbPublic)
	fmt.Printf("NbPublicVariables: %d\n", nbPublic)

	// Write CommitmentConstraintIndexes
	// Needed for BSB22 commitment handling
	// CommitmentIndexes() is part of the constraint.Commitments interface
	// (both PlonkCommitments and Groth16Commitments implement it)
	commitmentIndexes := spr.CommitmentInfo.CommitmentIndexes()
	numCommitments := uint64(len(trace.Qcp))
	binary.Write(domainInfoFile, binary.LittleEndian, numCommitments)

	// Write FrMultiplicativeGen (coset shift k1) as 32 bytes LE
	// k1 = 5 for BN254 Fr (hardcoded in gnark-crypto)
	// k2 = k1^2 (computed by prover)
	var k1Bytes [32]byte
	k1LE := domain.FrMultiplicativeGen.Bytes() // big-endian canonical
	reverseBytes(k1Bytes[:], k1LE[:])
	domainInfoFile.Write(k1Bytes[:])
	fmt.Printf("FrMultiplicativeGen (coset shift): exported\n")

	// Write commitment constraint indexes (at offset 88)
	// These tell the GPU prover where to inject BSB22 hash values in the PI polynomial
	for _, idx := range commitmentIndexes {
		binary.Write(domainInfoFile, binary.LittleEndian, uint64(idx))
	}
	fmt.Printf("CommitmentConstraintIndexes: %v\n", commitmentIndexes)

	fmt.Printf("\nTotal export time: %s\n", time.Since(totalStart))
	fmt.Printf("Output directory: %s\n", outputDir)
}

func writeG1File(path string, points []bn254.G1Affine) {
	f, err := os.Create(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	w := bufio.NewWriterSize(f, 1024*1024)
	defer w.Flush()

	buf := make([]byte, 64)
	for i := range points {
		raw := points[i].RawBytes()
		reverseBytes(buf[:32], raw[:32])
		reverseBytes(buf[32:], raw[32:])
		w.Write(buf)
	}
}

func writeFrFile(path string, elems []fr.Element) {
	f, err := os.Create(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	w := bufio.NewWriterSize(f, 1024*1024)
	defer w.Flush()

	buf := make([]byte, 32)
	for i := range elems {
		raw := elems[i].Bytes() // big-endian canonical form
		reverseBytes(buf, raw[:])
		w.Write(buf)
	}
}

// writeFrFileMontgomery writes Fr elements in raw Montgomery form (no fromMont
// conversion). On x86 little-endian, each fr.Element is [4]uint64 in Montgomery
// representation. We write a 4-byte magic header ("MFr1") followed by the raw
// bytes of each element. The Rust loader detects this header and skips the
// expensive canonical-to-Montgomery conversion (mont_mul with R^2).
//
// This saves ~79M fromMont calls in Go and ~31.7M mont_mul calls in Rust for
// typical Groth16 witnesses (~2.5M wire values + ~1M H coefficients).
func writeFrFileMontgomery(path string, elems []fr.Element) {
	f, err := os.Create(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	w := bufio.NewWriterSize(f, 1024*1024)
	defer w.Flush()

	// Magic header: "MFr1" (Montgomery Fr format version 1)
	w.Write([]byte("MFr1"))

	// Write raw Montgomery limbs. fr.Element is [4]uint64; on little-endian
	// x86, unsafe.Slice gives us the LE byte representation directly, which
	// matches our Rust Fr([u64; 4]) layout exactly.
	buf := make([]byte, 32)
	for i := range elems {
		// Access the [4]uint64 directly and write as LE bytes.
		for j := 0; j < 4; j++ {
			limb := elems[i][j]
			buf[j*8+0] = byte(limb)
			buf[j*8+1] = byte(limb >> 8)
			buf[j*8+2] = byte(limb >> 16)
			buf[j*8+3] = byte(limb >> 24)
			buf[j*8+4] = byte(limb >> 32)
			buf[j*8+5] = byte(limb >> 40)
			buf[j*8+6] = byte(limb >> 48)
			buf[j*8+7] = byte(limb >> 56)
		}
		w.Write(buf)
	}
}

func reverseBytes(dst, src []byte) {
	for i, j := 0, len(src)-1; i < len(src); i, j = i+1, j-1 {
		dst[i] = src[j]
	}
}

func log2(n uint64) int {
	r := 0
	for n > 1 {
		n >>= 1
		r++
	}
	return r
}
