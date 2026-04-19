package sp1

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math/big"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr/fft"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr/hash_to_field"
	"github.com/consensys/gnark/backend/groth16"
	groth16_bn254 "github.com/consensys/gnark/backend/groth16/bn254"
	cs "github.com/consensys/gnark/constraint/bn254"
	"github.com/consensys/gnark/constraint"
	"github.com/consensys/gnark/constraint/solver"
	fcs "github.com/consensys/gnark/frontend/cs"
	"github.com/consensys/gnark/frontend"
)

// ExportGroth16GpuWitness solves the Groth16 R1CS and exports the solved
// witness vectors (W, A, B, C) plus BSB22 Pedersen commitments as flat
// binary files for the Rust GPU prover.
//
// This must be called with the same dataDir used for ExportGroth16GpuData.
func ExportGroth16GpuWitness(dataDir string, witnessPath string, outputDir string) {
	start := time.Now()

	// Load R1CS
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
	fmt.Printf("[groth16-witness] Reading R1CS took %s\n", time.Since(start))

	// Load proving key (needed for BSB22 commitment keys)
	start = time.Now()
	pk := groth16.NewProvingKey(ecc.BN254)
	pkFile, err := os.Open(dataDir + "/" + groth16PkPath)
	if err != nil {
		panic(fmt.Sprintf("Failed to open PK: %v", err))
	}
	pkReader := bufio.NewReaderSize(pkFile, 1024*1024)
	pk.ReadDump(pkReader)
	pkFile.Close()
	fmt.Printf("[groth16-witness] Reading PK took %s\n", time.Since(start))

	// Load and parse witness JSON
	start = time.Now()
	witnessFile, err := os.ReadFile(witnessPath)
	if err != nil {
		panic(fmt.Sprintf("Failed to read witness: %v", err))
	}
	var witnessInput WitnessInput
	if err := json.Unmarshal(witnessFile, &witnessInput); err != nil {
		panic(fmt.Sprintf("Failed to unmarshal witness: %v", err))
	}
	assignment := NewCircuit(witnessInput)
	witness, err := frontend.NewWitness(&assignment, ecc.BN254.ScalarField())
	if err != nil {
		panic(fmt.Sprintf("Failed to create witness: %v", err))
	}
	fmt.Printf("[groth16-witness] Witness creation took %s\n", time.Since(start))

	// Get typed proving key and R1CS
	_pk := pk.(*groth16_bn254.ProvingKey)
	_r1cs := r1cs.(*cs.R1CS)

	// Prepare BSB22 commitment capture (matching gnark prove.go exactly)
	commitmentInfo := _r1cs.CommitmentInfo.(constraint.Groth16Commitments)
	commitments := make([]bn254.G1Affine, len(commitmentInfo))
	privateCommittedValues := make([][]fr.Element, len(commitmentInfo))

	// Override BSB22 hint to capture commitments
	bsb22ID := solver.GetHintID(fcs.Bsb22CommitmentComputePlaceholder)
	customBsb22Hint := func(_ *big.Int, in []*big.Int, out []*big.Int) error {
		i := int(in[0].Int64())
		in = in[1:]
		privateCommittedValues[i] = make([]fr.Element, len(commitmentInfo[i].PrivateCommitted))
		hashed := in[:len(commitmentInfo[i].PublicAndCommitmentCommitted)]
		committed := in[+len(hashed):]
		for j, inJ := range committed {
			privateCommittedValues[i][j].SetBigInt(inJ)
		}

		var err error
		if commitments[i], err = _pk.CommitmentKeys[i].Commit(privateCommittedValues[i]); err != nil {
			return err
		}

		// Hash commitment (same DST as gnark's Groth16 prover)
		hashData := constraint.SerializeCommitment(commitments[i].Marshal(), hashed, (fr.Bits-1)/8+1)
		htf := hash_to_field.New([]byte(constraint.CommitmentDst))
		htf.Write(hashData)
		hashBts := htf.Sum(nil)
		htf.Reset()
		nbBuf := fr.Bytes
		if htf.Size() < fr.Bytes {
			nbBuf = htf.Size()
		}
		var res fr.Element
		res.SetBytes(hashBts[:nbBuf])
		res.BigInt(out[0])
		return nil
	}

	// Solve R1CS with BSB22 override
	start = time.Now()
	fmt.Println("[groth16-witness] Solving R1CS...")
	_solution, err := _r1cs.Solve(witness, solver.OverrideHint(bsb22ID, customBsb22Hint))
	if err != nil {
		panic(fmt.Sprintf("R1CS solve failed: %v", err))
	}
	solution := _solution.(*cs.R1CSSolution)
	fmt.Printf("[groth16-witness] Solved in %s (W=%d, A=%d, B=%d, C=%d)\n",
		time.Since(start), len(solution.W), len(solution.A), len(solution.B), len(solution.C))

	// Compute PoK for commitments (matching gnark prove.go:110-127 exactly)
	var commitmentPok bn254.G1Affine
	if len(commitmentInfo) > 0 {
		poks := make([]bn254.G1Affine, len(_pk.CommitmentKeys))
		for i := range _pk.CommitmentKeys {
			if poks[i], err = _pk.CommitmentKeys[i].ProveKnowledge(privateCommittedValues[i]); err != nil {
				panic(fmt.Sprintf("ProveKnowledge failed: %v", err))
			}
		}
		// Fold PoKs: challenge is hash of ALL commitment wire values (not G1 points!)
		// gnark prove.go uses wireValues[commitmentInfo[i].CommitmentIndex].Marshal()
		commitmentsSerialized := make([]byte, fr.Bytes*len(commitmentInfo))
		for i := range commitmentInfo {
			copy(commitmentsSerialized[fr.Bytes*i:], solution.W[commitmentInfo[i].CommitmentIndex].Marshal())
		}
		challenge, err := fr.Hash(commitmentsSerialized, []byte("G16-BSB22"), 1)
		if err != nil {
			panic(fmt.Sprintf("PoK challenge failed: %v", err))
		}
		if _, err = commitmentPok.Fold(poks, challenge[0], ecc.MultiExpConfig{NbTasks: 1}); err != nil {
			panic(fmt.Sprintf("PoK fold failed: %v", err))
		}
	}

	// Compute H using gnark's algorithm (copied from prove.go:346 since it's unexported).
	// Save copies of A, B, C since computeH modifies them in-place.
	start = time.Now()
	aCopy := make([]fr.Element, len(solution.A))
	copy(aCopy, solution.A)
	bCopy := make([]fr.Element, len(solution.B))
	copy(bCopy, solution.B)
	cCopy := make([]fr.Element, len(solution.C))
	copy(cCopy, solution.C)

	h := localComputeH(aCopy, bCopy, cCopy, &_pk.Domain)
	fmt.Printf("[groth16-witness] computeH took %s (len=%d)\n", time.Since(start), len(h))

	// gnark's localComputeH produces H in BIT-REVERSED order (DIF FFTInverse
	// output). gnark's pk.G1.Z is also in bit-reversed order, so their MSM is
	// consistent. But we export Z in NATURAL order for the GPU prover, so we
	// must also export H in natural order. Un-bit-reverse H here.
	// H has domain.Cardinality elements (power of 2), so fft.BitReverse works.
	fft.BitReverse(h)

	// Export
	os.MkdirAll(outputDir, 0755)
	start = time.Now()

	writeFrFile(filepath.Join(outputDir, "wire_values.bin"), solution.W)
	writeFrFile(filepath.Join(outputDir, "solution_a.bin"), solution.A)
	writeFrFile(filepath.Join(outputDir, "solution_b.bin"), solution.B)
	writeFrFile(filepath.Join(outputDir, "solution_c.bin"), solution.C)
	writeG1File(filepath.Join(outputDir, "commitments.bin"), commitments)
	writeG1File(filepath.Join(outputDir, "commitment_pok.bin"), []bn254.G1Affine{commitmentPok})
	writeFrFile(filepath.Join(outputDir, "h_coefficients.bin"), h)

	fmt.Printf("[groth16-witness] Exported witness data in %s\n", time.Since(start))
}

// localComputeH is a copy of gnark's unexported computeH (prove.go:346).
// It computes the H polynomial for Groth16: H = (A*B - C) / Z where
// Z(x) = x^n - 1 is the vanishing polynomial.
//
// IMPORTANT: The output is in BIT-REVERSED order because the final
// FFTInverse uses DIF decimation, which produces bit-reversed output.
// gnark's pk.G1.Z is also stored in bit-reversed order (see setup.go:247),
// so the MSM sum(h[i] * Z[i]) is consistent.
func localComputeH(a, b, c []fr.Element, domain *fft.Domain) []fr.Element {
	n := len(a)

	// Pad to domain cardinality
	padding := make([]fr.Element, int(domain.Cardinality)-n)
	a = append(a, padding...)
	b = append(b, padding...)
	c = append(c, padding...)
	n = len(a)

	// Step 1: iFFT (DIF → output is bit-reversed)
	domain.FFTInverse(a, fft.DIF)
	domain.FFTInverse(b, fft.DIF)
	domain.FFTInverse(c, fft.DIF)

	// Step 2: coset FFT (DIT on bit-reversed input → output is natural-order)
	domain.FFT(a, fft.DIT, fft.OnCoset())
	domain.FFT(b, fft.DIT, fft.OnCoset())
	domain.FFT(c, fft.DIT, fft.OnCoset())

	// den = (g^N - 1)^(-1)
	var den, one fr.Element
	one.SetOne()
	den.Exp(domain.FrMultiplicativeGen, big.NewInt(int64(domain.Cardinality)))
	den.Sub(&den, &one).Inverse(&den)

	// Pointwise: h[i] = (a[i]*b[i] - c[i]) * den
	localParallelize(n, func(start, end int) {
		for i := start; i < end; i++ {
			a[i].Mul(&a[i], &b[i]).
				Sub(&a[i], &c[i]).
				Mul(&a[i], &den)
		}
	})

	// Step 3: coset iFFT (DIF → output is BIT-REVERSED)
	domain.FFTInverse(a, fft.DIF, fft.OnCoset())

	return a
}

// localParallelize is a minimal replacement for gnark's internal utils.Parallelize.
func localParallelize(n int, work func(start, end int)) {
	nbTasks := runtime.NumCPU()
	if nbTasks > n {
		nbTasks = n
	}
	if nbTasks <= 1 {
		work(0, n)
		return
	}
	chunkSize := (n + nbTasks - 1) / nbTasks
	var wg sync.WaitGroup
	for start := 0; start < n; start += chunkSize {
		end := start + chunkSize
		if end > n {
			end = n
		}
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			work(s, e)
		}(start, end)
	}
	wg.Wait()
}
