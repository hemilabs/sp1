package sp1

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr/fft"
	"github.com/consensys/gnark/backend/groth16"
	groth16_bn254 "github.com/consensys/gnark/backend/groth16/bn254"
	"github.com/consensys/gnark/constraint"
	cs "github.com/consensys/gnark/constraint/bn254"
	"github.com/consensys/gnark/debug"
)

// ExportGroth16GpuData exports the Groth16 proving key as flat binary files
// for the Rust GPU prover. This only exports the circuit-static data (PK).
// The witness-specific data (solved A/B/C, commitments) is exported separately
// by ExportGroth16GpuWitness after gnark's Prove() has been called.
func ExportGroth16GpuData(dataDir string, outputDir string) {
	start := time.Now()

	// Load R1CS for metadata
	os.Setenv("CONSTRAINTS_JSON", dataDir+"/"+constraintsJsonFile)
	os.Setenv("GROTH16", "1")

	r1cs := groth16.NewCS(ecc.BN254)
	r1csFile, err := os.Open(dataDir + "/" + groth16CircuitPath)
	if err != nil {
		panic(fmt.Sprintf("Failed to open R1CS: %v", err))
	}
	r1csReader := bufio.NewReaderSize(r1csFile, 1024*1024)
	r1cs.ReadFrom(r1csReader)
	r1csFile.Close()
	fmt.Printf("[groth16-export] Reading R1CS took %s\n", time.Since(start))

	// Create stripped R1CS (without debug data) for faster loading in ExportGroth16GpuWitness.
	// DebugInfo + MDebug + SymbolTable account for ~33% of the 2.4GB file (~817MB) and are
	// unused during proving. Stripping them reduces load time from ~20s to ~5s.
	os.MkdirAll(outputDir, 0755)
	strippedPath := filepath.Join(outputDir, "groth16_circuit_stripped.bin")
	start = time.Now()
	_r1cs := r1cs.(*cs.R1CS)
	_r1cs.DebugInfo = nil
	_r1cs.MDebug = nil
	_r1cs.SymbolTable = debug.SymbolTable{}
	strippedFile, err := os.Create(strippedPath)
	if err != nil {
		panic(fmt.Sprintf("Failed to create stripped R1CS: %v", err))
	}
	strippedWriter := bufio.NewWriterSize(strippedFile, 1024*1024)
	r1cs.WriteTo(strippedWriter)
	strippedWriter.Flush()
	strippedFile.Close()
	fmt.Printf("[groth16-export] Wrote stripped R1CS (%s) in %s\n", strippedPath, time.Since(start))

	// Load proving key
	start = time.Now()
	pk := groth16.NewProvingKey(ecc.BN254)
	pkFile, err := os.Open(dataDir + "/" + groth16PkPath)
	if err != nil {
		panic(fmt.Sprintf("Failed to open PK: %v", err))
	}
	pkReader := bufio.NewReaderSize(pkFile, 1024*1024)
	pk.ReadDump(pkReader)
	pkFile.Close()
	fmt.Printf("[groth16-export] Reading PK took %s\n", time.Since(start))

	// Get typed proving key
	_pk := pk.(*groth16_bn254.ProvingKey)

	// Get commitment info for K MSM filtering
	commitmentInfo := r1cs.GetCommitments().(constraint.Groth16Commitments)

	// Create output directory
	os.MkdirAll(outputDir, 0755)
	start = time.Now()

	// Export metadata
	exportGr16Metadata(outputDir, _pk, r1cs, commitmentInfo)

	// Export PK G1/G2 points
	exportGr16PK(outputDir, _pk)

	fmt.Printf("[groth16-export] Exporting PK data took %s\n", time.Since(start))
}

func exportGr16Metadata(dir string, pk *groth16_bn254.ProvingKey, r1cs constraint.ConstraintSystem, commitmentInfo constraint.Groth16Commitments) {
	f, _ := os.Create(filepath.Join(dir, "groth16_metadata.bin"))
	defer f.Close()

	nbPublic := r1cs.GetNbPublicVariables()
	// Total wire count = len(InfinityA) = public + secret + internal
	nbWires := len(pk.InfinityA)

	binary.Write(f, binary.LittleEndian, uint64(pk.Domain.Cardinality))
	binary.Write(f, binary.LittleEndian, uint64(nbWires))
	binary.Write(f, binary.LittleEndian, uint64(nbPublic))
	binary.Write(f, binary.LittleEndian, uint64(pk.NbInfinityA))
	binary.Write(f, binary.LittleEndian, uint64(pk.NbInfinityB))
	binary.Write(f, binary.LittleEndian, uint64(len(pk.CommitmentKeys)))
	binary.Write(f, binary.LittleEndian, uint64(len(pk.G1.A)))
	binary.Write(f, binary.LittleEndian, uint64(len(pk.G1.B)))
	binary.Write(f, binary.LittleEndian, uint64(len(pk.G1.Z)))
	binary.Write(f, binary.LittleEndian, uint64(len(pk.G1.K)))
	binary.Write(f, binary.LittleEndian, uint64(len(pk.G2.B)))

	// Domain generator (omega) as LE canonical Fr
	var omegaBuf [32]byte
	raw := pk.Domain.Generator.Bytes()
	reverseBytes(omegaBuf[:], raw[:])
	f.Write(omegaBuf[:])

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

	// Commitment filtering info for K MSM:
	// Wire indices to EXCLUDE from wire_values[nb_public:] before K MSM.
	// These are PrivateCommitted indices + CommitmentIndexes (per gnark prove.go).
	allToRemove := make([]uint64, 0)
	for _, ci := range commitmentInfo {
		for _, idx := range ci.PrivateCommitted {
			allToRemove = append(allToRemove, uint64(idx))
		}
	}
	for _, ci := range commitmentInfo {
		allToRemove = append(allToRemove, uint64(ci.CommitmentIndex))
	}
	binary.Write(f, binary.LittleEndian, uint64(len(allToRemove)))
	for _, idx := range allToRemove {
		binary.Write(f, binary.LittleEndian, idx)
	}
}

func exportGr16PK(dir string, pk *groth16_bn254.ProvingKey) {
	// G1 points as LE canonical Fq coordinates (using reverseBytes, matching PLONK export)
	writeG1File(filepath.Join(dir, "pk_g1_a.bin"), pk.G1.A)
	writeG1File(filepath.Join(dir, "pk_g1_b.bin"), pk.G1.B)
	// gnark's setup stores pk.G1.Z in BIT-REVERSED order (setup.go:247
	// calls bitReverse before slicing). gnark's computeH produces H in
	// bit-reversed order too (DIF FFTInverse output), so gnark's MSM is
	// consistent. But our GPU NTT produces NATURAL-ORDER H coefficients,
	// so we need to export Z in natural order to match.
	// gnark stores pk.G1.Z in bit-reversed order (setup.go:247). Our GPU
	// prover uses gnark's pre-computed H which we export in natural order
	// (un-bit-reversed in ExportGroth16GpuWitness). So Z must also be in
	// natural order to match.
	zNatural := make([]bn254.G1Affine, len(pk.G1.Z))
	copy(zNatural, pk.G1.Z)
	// pk.G1.Z has domain.Cardinality-1 elements (not power of 2).
	// Pad to next power of 2 for fft.BitReverse, then truncate.
	zLen := len(zNatural)
	if zLen > 0 && (zLen&(zLen-1)) == 0 {
		fft.BitReverse(zNatural)
	} else {
		nbits := uint(0)
		for (1 << nbits) < uint64(zLen) {
			nbits++
		}
		padded := make([]bn254.G1Affine, 1<<nbits)
		copy(padded, zNatural)
		fft.BitReverse(padded)
		copy(zNatural, padded[:zLen])
	}
	writeG1File(filepath.Join(dir, "pk_g1_z.bin"), zNatural)
	writeG1File(filepath.Join(dir, "pk_g1_k.bin"), pk.G1.K)

	// G2 points as LE canonical Fq2 coordinates
	writeG2File(filepath.Join(dir, "pk_g2_b.bin"), pk.G2.B)

	// Scalar PK elements
	writeG1File(filepath.Join(dir, "pk_g1_alpha.bin"), []bn254.G1Affine{pk.G1.Alpha})
	writeG1File(filepath.Join(dir, "pk_g1_beta.bin"), []bn254.G1Affine{pk.G1.Beta})
	writeG1File(filepath.Join(dir, "pk_g1_delta.bin"), []bn254.G1Affine{pk.G1.Delta})
	writeG2File(filepath.Join(dir, "pk_g2_beta.bin"), []bn254.G2Affine{pk.G2.Beta})
	writeG2File(filepath.Join(dir, "pk_g2_delta.bin"), []bn254.G2Affine{pk.G2.Delta})
}

// writeG2File writes G2Affine points as LE canonical bytes (128 bytes each).
// Layout per gnark RawBytes(): X.A1 (32 LE) + X.A0 (32 LE) + Y.A1 (32 LE) + Y.A0 (32 LE)
func writeG2File(path string, points []bn254.G2Affine) {
	f, err := os.Create(path)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	w := bufio.NewWriterSize(f, 1024*1024)
	defer w.Flush()

	buf := make([]byte, 128)
	for i := range points {
		raw := points[i].RawBytes() // 128 bytes big-endian
		reverseBytes(buf[0:32], raw[0:32])
		reverseBytes(buf[32:64], raw[32:64])
		reverseBytes(buf[64:96], raw[64:96])
		reverseBytes(buf[96:128], raw[96:128])
		w.Write(buf)
	}
}
