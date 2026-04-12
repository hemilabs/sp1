package sp1

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"math/big"
	"os"
	"path/filepath"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fp"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark/backend/groth16"
	groth16_bn254 "github.com/consensys/gnark/backend/groth16/bn254"
	"github.com/consensys/gnark/constraint"
	"github.com/consensys/gnark/frontend"
)

// ExportGroth16GpuData exports the Groth16 proving key and solved witness
// as flat binary files for the Rust GPU prover.
//
// The Rust prover handles: H polynomial (7 NTTs), 4 G1 MSMs, 1 G2 MSM.
// Go handles: R1CS solving, Pedersen commitments (BSB22).
func ExportGroth16GpuData(dataDir string, witnessPath string, outputDir string) {
	start := time.Now()

	// Load R1CS
	r1cs := groth16.NewCS(ecc.BN254)
	r1csFile, err := os.Open(dataDir + "/" + groth16CircuitPath)
	if err != nil {
		panic(err)
	}
	r1csReader := bufio.NewReaderSize(r1csFile, 1024*1024)
	r1cs.ReadFrom(r1csReader)
	r1csFile.Close()
	fmt.Printf("Reading R1CS took %s\n", time.Since(start))

	// Load proving key
	start = time.Now()
	pk := groth16.NewProvingKey(ecc.BN254)
	pkFile, err := os.Open(dataDir + "/" + groth16PkPath)
	if err != nil {
		panic(err)
	}
	pkReader := bufio.NewReaderSize(pkFile, 1024*1024)
	pk.ReadDump(pkReader)
	pkFile.Close()
	fmt.Printf("Reading proving key took %s\n", time.Since(start))

	// Load witness
	start = time.Now()
	witnessFile, err := os.ReadFile(witnessPath)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Reading witness file took %s\n", time.Since(start))

	// Deserialize witness JSON
	start = time.Now()
	os.Setenv("CONSTRAINTS_JSON", dataDir+"/"+constraintsJsonFile)
	os.Setenv("GROTH16", "1")
	witnessInput, err := DeserializeWitnessInput(witnessFile)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Deserializing witness took %s\n", time.Since(start))

	// Generate full witness
	start = time.Now()
	assignment := NewCircuit(witnessInput)
	witness, err := frontend.NewWitness(assignment, ecc.BN254.ScalarField())
	if err != nil {
		panic(err)
	}
	fmt.Printf("Generating witness took %s\n", time.Since(start))

	// Solve R1CS
	start = time.Now()
	_r1cs := r1cs.(*constraint.R1CS)
	solution, err := _r1cs.Solve(witness)
	if err != nil {
		panic(fmt.Sprintf("R1CS solve failed: %v", err))
	}
	fmt.Printf("Solving R1CS took %s\n", time.Since(start))

	// Get typed proving key
	_pk := pk.(*groth16_bn254.ProvingKey)

	// Create output directory
	os.MkdirAll(outputDir, 0755)

	// Export metadata
	start = time.Now()
	exportGroth16Metadata(outputDir, _pk, _r1cs)

	// Export proving key G1/G2 points
	exportGroth16PK(outputDir, _pk)

	// Export solved witness vectors
	exportGroth16Solution(outputDir, solution, _pk, _r1cs)

	fmt.Printf("Exporting data took %s\n", time.Since(start))
}

func exportGroth16Metadata(dir string, pk *groth16_bn254.ProvingKey, r1cs *constraint.R1CS) {
	f, _ := os.Create(filepath.Join(dir, "groth16_metadata.bin"))
	defer f.Close()

	// Domain size (cardinality)
	binary.Write(f, binary.LittleEndian, uint64(pk.Domain.Cardinality))
	// Number of wires
	binary.Write(f, binary.LittleEndian, uint64(r1cs.NbPublicVariables+r1cs.NbSecretVariables))
	// Number of public variables
	binary.Write(f, binary.LittleEndian, uint64(r1cs.NbPublicVariables))
	// Infinity counts
	binary.Write(f, binary.LittleEndian, uint64(pk.NbInfinityA))
	binary.Write(f, binary.LittleEndian, uint64(pk.NbInfinityB))
	// Number of commitments (BSB22)
	binary.Write(f, binary.LittleEndian, uint64(len(pk.CommitmentKeys)))
	// Sizes of MSM arrays
	binary.Write(f, binary.LittleEndian, uint64(len(pk.G1.A)))
	binary.Write(f, binary.LittleEndian, uint64(len(pk.G1.B)))
	binary.Write(f, binary.LittleEndian, uint64(len(pk.G1.Z)))
	binary.Write(f, binary.LittleEndian, uint64(len(pk.G1.K)))
	binary.Write(f, binary.LittleEndian, uint64(len(pk.G2.B)))

	// Domain generator (omega) as 32-byte Fr element
	var omegaBytes [32]byte
	omegaBig := pk.Domain.Generator.BigInt(new(big.Int))
	omegaBig.FillBytes(omegaBytes[:])
	f.Write(omegaBytes[:])

	// Infinity masks
	for _, v := range pk.InfinityA {
		if v {
			f.Write([]byte{1})
		} else {
			f.Write([]byte{0})
		}
	}
	for _, v := range pk.InfinityB {
		if v {
			f.Write([]byte{1})
		} else {
			f.Write([]byte{0})
		}
	}
}

func exportGroth16PK(dir string, pk *groth16_bn254.ProvingKey) {
	// Export G1 points as raw LE Montgomery Fq coordinates (64 bytes each)
	writeG1Points(filepath.Join(dir, "pk_g1_a.bin"), pk.G1.A)
	writeG1Points(filepath.Join(dir, "pk_g1_b.bin"), pk.G1.B)
	writeG1Points(filepath.Join(dir, "pk_g1_z.bin"), pk.G1.Z)
	writeG1Points(filepath.Join(dir, "pk_g1_k.bin"), pk.G1.K)

	// Export G2 points (128 bytes each: 4 Fq coordinates in LE Montgomery)
	writeG2Points(filepath.Join(dir, "pk_g2_b.bin"), pk.G2.B)

	// Export scalar PK elements
	writeSingleG1(filepath.Join(dir, "pk_g1_alpha.bin"), &pk.G1.Alpha)
	writeSingleG1(filepath.Join(dir, "pk_g1_beta.bin"), &pk.G1.Beta)
	writeSingleG1(filepath.Join(dir, "pk_g1_delta.bin"), &pk.G1.Delta)
	writeSingleG2(filepath.Join(dir, "pk_g2_beta.bin"), &pk.G2.Beta)
	writeSingleG2(filepath.Join(dir, "pk_g2_delta.bin"), &pk.G2.Delta)
}

func exportGroth16Solution(dir string, solution constraint.R1CSSolution, pk *groth16_bn254.ProvingKey, r1cs *constraint.R1CS) {
	w := solution.W
	a := solution.A
	b := solution.B
	c := solution.C

	// Export wire values as raw LE Fr elements (32 bytes each)
	writeFrSlice(filepath.Join(dir, "wire_values.bin"), w)
	writeFrSlice(filepath.Join(dir, "solution_a.bin"), a)
	writeFrSlice(filepath.Join(dir, "solution_b.bin"), b)
	writeFrSlice(filepath.Join(dir, "solution_c.bin"), c)
}

// Helper: write a slice of G1Affine points as raw LE Montgomery bytes
func writeG1Points(path string, points []bn254.G1Affine) {
	f, err := os.Create(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	w := bufio.NewWriterSize(f, 1024*1024)
	for i := range points {
		// Each G1Affine is (X, Y) in Fq, each Fq is 4 uint64 in Montgomery form
		var buf [64]byte
		xBytes := points[i].X.Marshal() // 32 bytes, big-endian canonical
		yBytes := points[i].Y.Marshal()
		copy(buf[0:32], xBytes)
		copy(buf[32:64], yBytes)
		w.Write(buf[:])
	}
	w.Flush()
}

func writeG2Points(path string, points []bn254.G2Affine) {
	f, err := os.Create(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	w := bufio.NewWriterSize(f, 1024*1024)
	for i := range points {
		// G2Affine: X = (A0, A1) in Fq2, Y = (A0, A1) in Fq2
		// Each component is 32 bytes big-endian canonical
		var buf [128]byte
		xa0 := points[i].X.A0.Marshal()
		xa1 := points[i].X.A1.Marshal()
		ya0 := points[i].Y.A0.Marshal()
		ya1 := points[i].Y.A1.Marshal()
		copy(buf[0:32], xa0)
		copy(buf[32:64], xa1)
		copy(buf[64:96], ya0)
		copy(buf[96:128], ya1)
		w.Write(buf[:])
	}
	w.Flush()
}

func writeSingleG1(path string, pt *bn254.G1Affine) {
	writeG1Points(path, []bn254.G1Affine{*pt})
}

func writeSingleG2(path string, pt *bn254.G2Affine) {
	writeG2Points(path, []bn254.G2Affine{*pt})
}

func writeFrSlice(path string, vals []fr.Element) {
	f, err := os.Create(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	w := bufio.NewWriterSize(f, 1024*1024)
	for i := range vals {
		b := vals[i].Marshal() // 32 bytes, big-endian canonical
		w.Write(b)
	}
	w.Flush()
}

// Unused placeholders for fp to avoid import errors
var _ fp.Element
