package sp1

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math/big"
	"os"
	"path/filepath"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
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

	// Compute PoK for commitments
	var commitmentPok bn254.G1Affine
	if len(commitmentInfo) > 0 {
		poks := make([]bn254.G1Affine, len(_pk.CommitmentKeys))
		for i := range _pk.CommitmentKeys {
			if poks[i], err = _pk.CommitmentKeys[i].ProveKnowledge(privateCommittedValues[i]); err != nil {
				panic(fmt.Sprintf("ProveKnowledge failed: %v", err))
			}
		}
		// Fold PoKs
		challenge, err := fr.Hash(commitments[0].Marshal(), []byte("G16-BSB22"), 1)
		if err != nil {
			panic(fmt.Sprintf("PoK challenge failed: %v", err))
		}
		if _, err = commitmentPok.Fold(poks, challenge[0], ecc.MultiExpConfig{NbTasks: 1}); err != nil {
			panic(fmt.Sprintf("PoK fold failed: %v", err))
		}
	}

	// Export
	os.MkdirAll(outputDir, 0755)
	start = time.Now()

	writeFrFile(filepath.Join(outputDir, "wire_values.bin"), solution.W)
	writeFrFile(filepath.Join(outputDir, "solution_a.bin"), solution.A)
	writeFrFile(filepath.Join(outputDir, "solution_b.bin"), solution.B)
	writeFrFile(filepath.Join(outputDir, "solution_c.bin"), solution.C)
	writeG1File(filepath.Join(outputDir, "commitments.bin"), commitments)
	writeG1File(filepath.Join(outputDir, "commitment_pok.bin"), []bn254.G1Affine{commitmentPok})

	fmt.Printf("[groth16-witness] Exported witness data in %s\n", time.Since(start))
}
