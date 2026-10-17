package sp1

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"testing"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/plonk"
	plonk_bn254 "github.com/consensys/gnark/backend/plonk/bn254"
	cs "github.com/consensys/gnark/constraint/bn254"
	"github.com/consensys/gnark/frontend"
)

// TestPlonkProveTiming loads the SP1 PLONK circuit and times the proving step.
// This is a test (not a benchmark) because we only need one run with real data.
func TestPlonkProveTiming(t *testing.T) {
	dataDir := getPlonkDataDir(t)
	witnessPath := dataDir + "/plonk_witness.json"
	if _, err := os.Stat(witnessPath); os.IsNotExist(err) {
		t.Skipf("PLONK witness not found at %s", witnessPath)
	}

	t.Logf("System: %d CPUs, GOMAXPROCS=%d", runtime.NumCPU(), runtime.GOMAXPROCS(0))

	// ---------- Load SCS ----------
	start := time.Now()
	os.Setenv("CONSTRAINTS_JSON", dataDir+"/"+constraintsJsonFile)
	scsFile, err := os.Open(dataDir + "/" + plonkCircuitPath)
	if err != nil {
		t.Fatal(err)
	}
	scsGeneric := plonk.NewCS(ecc.BN254)
	scsGeneric.ReadFrom(scsFile)
	scsFile.Close()
	spr := scsGeneric.(*cs.SparseR1CS)
	nbConstraints := spr.GetNbConstraints()
	nbPublic := len(spr.Public)
	domainSize := ecc.NextPowerOfTwo(uint64(nbConstraints + nbPublic))
	t.Logf("Loaded SCS in %s (%d constraints, %d public, domain=2^%d=%d)",
		time.Since(start), nbConstraints, nbPublic, log2(domainSize), domainSize)

	// ---------- Load proving key ----------
	start = time.Now()
	pkFile, err := os.Open(dataDir + "/" + plonkPkPath)
	if err != nil {
		t.Fatal(err)
	}
	pk := plonk.NewProvingKey(ecc.BN254)
	bufReader := bufio.NewReaderSize(pkFile, 1024*1024)
	pk.UnsafeReadFrom(bufReader)
	pkFile.Close()
	t.Logf("Loaded PK in %s", time.Since(start))

	// Log PK details
	pkBn254 := pk.(*plonk_bn254.ProvingKey)
	t.Logf("PK Lagrange SRS points: %d", len(pkBn254.KzgLagrange.G1))
	t.Logf("PK Canonical SRS points: %d", len(pkBn254.Kzg.G1))

	// ---------- Load verifying key ----------
	start = time.Now()
	vkFile, err := os.Open(dataDir + "/" + plonkVkPath)
	if err != nil {
		t.Fatal(err)
	}
	vk := plonk.NewVerifyingKey(ecc.BN254)
	vk.ReadFrom(vkFile)
	vkFile.Close()
	t.Logf("Loaded VK in %s", time.Since(start))

	// ---------- Load & parse witness ----------
	start = time.Now()
	data, err := os.ReadFile(witnessPath)
	if err != nil {
		t.Fatal(err)
	}
	var witnessInput WitnessInput
	err = json.Unmarshal(data, &witnessInput)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Loaded witness JSON in %s", time.Since(start))

	// ---------- Build witness ----------
	start = time.Now()
	assignment := NewCircuit(witnessInput)
	witness, err := frontend.NewWitness(&assignment, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatal(err)
	}
	publicWitness, err := witness.Public()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Built witness in %s", time.Since(start))

	// ---------- PROVE (this is the key measurement) ----------
	t.Log("Starting plonk.Prove()...")
	runtime.GC() // clean up before timing
	proveStart := time.Now()
	proof, err := plonk.Prove(scsGeneric, pk, witness)
	proveElapsed := time.Since(proveStart)
	if err != nil {
		t.Fatalf("plonk.Prove failed: %v", err)
	}
	t.Logf("============================================")
	t.Logf("plonk.Prove() took: %s", proveElapsed)
	t.Logf("============================================")

	// ---------- VERIFY ----------
	start = time.Now()
	err = plonk.Verify(proof, vk, publicWitness)
	if err != nil {
		t.Fatalf("Verification failed: %v", err)
	}
	t.Logf("Verification took: %s", time.Since(start))

	// ---------- Summary ----------
	t.Log("")
	t.Log("=== SUMMARY ===")
	t.Logf("CPU: %d cores", runtime.NumCPU())
	t.Logf("Circuit: %d constraints, domain size 2^%d = %d", nbConstraints, log2(domainSize), domainSize)
	t.Logf("PLONK Prove time: %s", proveElapsed)

	// Estimate MSM size: PLONK does ~4-5 MSMs of size N during proving
	// Round 1: 3 MSMs of size N (Lagrange basis) for L, R, O commitments
	// Round 2: 1 MSM of size N+2 for Z(X)
	// Round 3-5: additional MSMs in monomial basis
	msmN := domainSize
	t.Logf("Each MSM is ~%d points (2^%d)", msmN, log2(uint64(msmN)))
	t.Logf("Comparison: GPU MSM at ~8.4s/call would need ~%d calls = ~%.0fs total",
		5, float64(5)*8.4)

	fmt.Printf("\n[RESULT] gnark CPU PLONK prove: %s on %d cores, domain 2^%d\n",
		proveElapsed, runtime.NumCPU(), log2(domainSize))
}
