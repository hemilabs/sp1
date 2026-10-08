package sp1

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"runtime/debug"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/constraint"
	"github.com/consensys/gnark/frontend"
)

// ProveGroth16WithR1cs proves with the R1CS at r1csPath and the proving key in dataDir, and keeps
// nothing afterwards.
//
// ProveGroth16 keeps the R1CS and proving key in process globals for the life of the process,
// which suits a dedicated prover but makes every long-lived process that has proved once ~12 GB
// larger forever. This is for a short-lived process instead: groth16_cpu_helper runs one proof and
// exits, which gives all of its memory back to the host. r1csPath is typically the stripped
// circuit (see ExportGroth16StrippedR1cs), which loads several times faster than the full one.
//
// The one error it returns is an R1CS it cannot read, so that the caller can try another (a
// stripped copy can be corrupt, or written by a gnark that serialized differently). Anything else
// panics.
func ProveGroth16WithR1cs(dataDir string, r1csPath string, witnessPath string) (Proof, error) {
	os.Setenv("CONSTRAINTS_JSON", dataDir+"/"+constraintsJsonFile)
	os.Setenv("GROTH16", "1")

	start := time.Now()
	r1cs, err := readR1cs(r1csPath)
	if err != nil {
		return Proof{}, err
	}
	fmt.Printf("Reading R1CS (%s) took %s\n", r1csPath, time.Since(start))

	start = time.Now()
	pk := groth16.NewProvingKey(ecc.BN254)
	pkFile, err := os.Open(dataDir + "/" + groth16PkPath)
	if err != nil {
		panic(fmt.Sprintf("failed to open proving key: %v", err))
	}
	if err := pk.ReadDump(bufio.NewReaderSize(pkFile, 1024*1024)); err != nil {
		pkFile.Close()
		panic(fmt.Sprintf("failed to read proving key: %v", err))
	}
	pkFile.Close()
	fmt.Printf("Reading proving key took %s\n", time.Since(start))

	data, err := os.ReadFile(witnessPath)
	if err != nil {
		panic(fmt.Sprintf("failed to read witness %s: %v", witnessPath, err))
	}
	var witnessInput WitnessInput
	if err := json.Unmarshal(data, &witnessInput); err != nil {
		panic(fmt.Sprintf("failed to parse witness %s: %v", witnessPath, err))
	}

	start = time.Now()
	assignment := NewCircuit(witnessInput)
	witness, err := frontend.NewWitness(&assignment, ecc.BN254.ScalarField())
	if err != nil {
		panic(err)
	}
	fmt.Printf("Generating witness took %s\n", time.Since(start))

	start = time.Now()
	proof, err := groth16.Prove(r1cs, pk, witness)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Generating proof took %s\n", time.Since(start))

	return NewSP1Groth16Proof(&proof, witnessInput), nil
}

// readR1cs reads the constraint system at path, returning a panic in gnark's reader as an error
// too: a corrupt file can make it index out of bounds rather than fail cleanly.
func readR1cs(path string) (r1cs constraint.ConstraintSystem, err error) {
	defer func() {
		if r := recover(); r != nil {
			r1cs, err = nil, fmt.Errorf("failed to read R1CS %s: %v", path, r)
		}
	}()
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open R1CS %s: %w", path, err)
	}
	defer file.Close()
	r1cs = groth16.NewCS(ecc.BN254)
	if _, err := r1cs.ReadFrom(bufio.NewReaderSize(file, 1024*1024)); err != nil {
		return nil, fmt.Errorf("failed to read R1CS %s: %w", path, err)
	}
	return r1cs, nil
}

// ExportGroth16StrippedR1cs writes the R1CS in dataDir to outputPath without its debug
// information, which the prover does not use. The result loads several times faster.
func ExportGroth16StrippedR1cs(dataDir string, outputPath string) {
	os.Setenv("CONSTRAINTS_JSON", dataDir+"/"+constraintsJsonFile)
	os.Setenv("GROTH16", "1")

	start := time.Now()
	r1cs := groth16.NewCS(ecc.BN254)
	r1csFile, err := os.Open(dataDir + "/" + groth16CircuitPath)
	if err != nil {
		panic(fmt.Sprintf("failed to open R1CS: %v", err))
	}
	if _, err := r1cs.ReadFrom(bufio.NewReaderSize(r1csFile, 1024*1024)); err != nil {
		r1csFile.Close()
		panic(fmt.Sprintf("failed to read R1CS: %v", err))
	}
	r1csFile.Close()
	fmt.Printf("[groth16-strip] Reading R1CS took %s\n", time.Since(start))
	writeStrippedR1cs(r1cs, outputPath)
}

// ReleaseCaches drops the circuit and proving key that ProveGroth16 and ExportGroth16GpuWitness
// keep in package globals (~12 GB on the v6.1.0 circuit), and returns the freed memory to the
// host. A long-lived prover calls it after a proof, so that a process that has proved once does
// not stay that much larger. The next call reloads what it needs.
func ReleaseCaches() {
	releaseProverCache()
	gpuWitnessMutex.Lock()
	gpuWitnessR1cs = nil
	gpuWitnessR1csDataDir = ""
	gpuWitnessPk = nil
	gpuWitnessPkDataDir = ""
	gpuWitnessMutex.Unlock()
	runtime.GC()
	debug.FreeOSMemory()
}
