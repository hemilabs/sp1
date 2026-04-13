package sp1

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math/big"
	"os"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr/hash_to_field"
	"github.com/consensys/gnark-crypto/ecc/bn254/kzg"
	"github.com/consensys/gnark/backend/plonk"
	plonk_bn254 "github.com/consensys/gnark/backend/plonk/bn254"
	"github.com/consensys/gnark/constraint"
	cs "github.com/consensys/gnark/constraint/bn254"
	"github.com/consensys/gnark/constraint/solver"
	"github.com/consensys/gnark/frontend"
	fcs "github.com/consensys/gnark/frontend/cs"
)

// ExportSolvedWitness solves the PLONK constraint system and exports
// per-proof data (wire assignments L, R, O and BSB22 committed polynomials)
// as flat binary files for the GPU PLONK prover.
//
// This replicates gnark's internal solving + BSB22 hint processing,
// capturing the committed polynomial and KZG commitment that are normally
// discarded after proving.
//
// Output files (written to outputDir):
//   - witness_l.bin: L wire values (N Fr elements, 32 bytes each, canonical LE)
//   - witness_r.bin: R wire values
//   - witness_o.bin: O wire values
//   - bsb22_poly_<i>.bin: BSB22 committed polynomial i (N Fr elements)
//   - bsb22_commitment_<i>.bin: BSB22 KZG commitment i (1 G1 affine point, 64 bytes)
//   - witness_info.bin: metadata (domain size, num public, num BSB22)
func ExportSolvedWitness(dataDir string, witnessPath string, outputDir string) {
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

	// Load proving key (for KZG Lagrange SRS)
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
	pkBn254 := pk.(*plonk_bn254.ProvingKey)
	fmt.Printf("Loaded PK in %s\n", time.Since(start))

	// Build witness from JSON input
	start = time.Now()
	fmt.Println("Building witness...")
	witnessData, err := os.ReadFile(witnessPath)
	if err != nil {
		panic(err)
	}
	var witnessInput WitnessInput
	if err := json.Unmarshal(witnessData, &witnessInput); err != nil {
		panic(fmt.Sprintf("failed to parse witness: %v", err))
	}
	assignment := NewCircuit(witnessInput)
	witness, err := frontend.NewWitness(&assignment, ecc.BN254.ScalarField())
	if err != nil {
		panic(fmt.Sprintf("failed to create witness: %v", err))
	}
	fmt.Printf("Built witness in %s\n", time.Since(start))

	// Prepare BSB22 capture
	commitmentInfo := spr.CommitmentInfo.(constraint.PlonkCommitments)
	numCommitments := len(commitmentInfo)
	domainSize := ecc.NextPowerOfTwo(uint64(spr.GetNbConstraints() + len(spr.Public)))
	nbPublic := spr.GetNbPublicVariables()

	capturedPolys := make([][]fr.Element, numCommitments)
	capturedCommitments := make([]bn254.G1Affine, numCommitments)
	fmt.Printf("BSB22: %d commitments, domain size %d, %d public variables\n",
		numCommitments, domainSize, nbPublic)

	// Custom BSB22 hint that captures the committed polynomial and KZG commitment.
	// This replicates gnark's bsb22Hint (prove.go:280-315) but stores data
	// for export instead of keeping it in the prover's internal state.
	customBsb22Hint := func(_ *big.Int, ins, outs []*big.Int) error {
		commDepth := int(ins[0].Int64())
		ins = ins[1:]

		ci := commitmentInfo[commDepth]

		// Build committed polynomial in Lagrange basis (domain-size vector)
		committedValues := make([]fr.Element, domainSize)
		offset := nbPublic
		for i := range ins {
			committedValues[offset+ci.Committed[i]].SetBigInt(ins[i])
		}

		// Random blinding at positions where Qcp = 0 (safe for ZK)
		if _, err := committedValues[offset+ci.CommitmentIndex].SetRandom(); err != nil {
			return err
		}
		if _, err := committedValues[offset+spr.GetNbConstraints()-1].SetRandom(); err != nil {
			return err
		}

		// Save committed polynomial
		capturedPolys[commDepth] = committedValues

		// KZG commit using Lagrange SRS
		digest, err := kzg.Commit(committedValues, pkBn254.KzgLagrange)
		if err != nil {
			return fmt.Errorf("BSB22 KZG commit %d: %w", commDepth, err)
		}
		capturedCommitments[commDepth] = digest

		// Hash commitment to field element (DST = "BSB22-Plonk")
		htf := hash_to_field.New([]byte("BSB22-Plonk"))
		htf.Write(digest.Marshal())
		hashBts := htf.Sum(nil)
		htf.Reset()

		nbBuf := fr.Bytes
		if htf.Size() < fr.Bytes {
			nbBuf = htf.Size()
		}

		var res fr.Element
		res.SetBytes(hashBts[:nbBuf])
		res.BigInt(outs[0])

		fmt.Printf("BSB22 hint %d: committed %d values, commitment computed\n",
			commDepth, len(ins))
		return nil
	}

	// Solve constraint system with custom BSB22 hint
	start = time.Now()
	fmt.Println("Solving constraint system...")
	bsb22ID := solver.GetHintID(fcs.Bsb22CommitmentComputePlaceholder)
	solution, err := spr.Solve(witness, solver.OverrideHint(bsb22ID, customBsb22Hint))
	if err != nil {
		panic(fmt.Sprintf("failed to solve: %v", err))
	}
	sol := solution.(*cs.SparseR1CSSolution)
	fmt.Printf("Solved in %s (L=%d, R=%d, O=%d)\n",
		time.Since(start), len(sol.L), len(sol.R), len(sol.O))

	// Export wire assignments
	start = time.Now()
	fmt.Println("Exporting wire assignments...")
	writeFrFile(outputDir+"/witness_l.bin", sol.L)
	writeFrFile(outputDir+"/witness_r.bin", sol.R)
	writeFrFile(outputDir+"/witness_o.bin", sol.O)
	fmt.Printf("Exported L, R, O in %s\n", time.Since(start))

	// Export BSB22 committed polynomials and commitment points
	start = time.Now()
	for i := 0; i < numCommitments; i++ {
		writeFrFile(fmt.Sprintf("%s/bsb22_poly_%d.bin", outputDir, i), capturedPolys[i])
		writeG1File(fmt.Sprintf("%s/bsb22_commitment_%d.bin", outputDir, i),
			[]bn254.G1Affine{capturedCommitments[i]})
	}
	fmt.Printf("Exported %d BSB22 polynomials + commitments in %s\n",
		numCommitments, time.Since(start))

	// Export witness metadata
	infoFile, err := os.Create(outputDir + "/witness_info.bin")
	if err != nil {
		panic(err)
	}
	defer infoFile.Close()

	// Witness info layout:
	//   [0..8)   uint64 LE: domain size (N)
	//   [8..16)  uint64 LE: number of public variables
	//   [16..24) uint64 LE: number of BSB22 commitments
	binary.Write(infoFile, binary.LittleEndian, uint64(len(sol.L)))
	binary.Write(infoFile, binary.LittleEndian, uint64(nbPublic))
	binary.Write(infoFile, binary.LittleEndian, uint64(numCommitments))

	fmt.Printf("\nTotal export time: %s\n", time.Since(totalStart))
	fmt.Printf("Output directory: %s\n", outputDir)
	fmt.Printf("Wire assignments: %d elements × 3 wires × 32 bytes = %d MB\n",
		len(sol.L), len(sol.L)*3*32/1024/1024)
}
