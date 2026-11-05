//go:build !icicle

package sp1

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	groth16_bn254 "github.com/consensys/gnark/backend/groth16/bn254"
	cs "github.com/consensys/gnark/constraint/bn254"
)

// What ProveGroth16 keeps between proofs, for the circuit in the data directory each names, under
// globalMutex: the parsed R1CS (nil when not read: with the GPU R1CS solver armed only its
// metadata is), the R1CS's metadata, and the proving key.
var globalMutex sync.RWMutex
var globalR1cs *cs.R1CS
var globalR1csDataDir string
var globalR1csMeta *r1csMeta
var globalR1csMetaDataDir string
var globalPk groth16.ProvingKey
var globalPkDataDir string

func ProveGroth16(dataDir string, witnessPath string) Proof {
	// Sanity check the required arguments have been provided.
	if dataDir == "" {
		panic("dataDirStr is required")
	}

	start := time.Now()
	os.Setenv("CONSTRAINTS_JSON", dataDir+"/"+constraintsJsonFile)
	os.Setenv("GROTH16", "1")
	fmt.Printf("Setting environment variables took %s\n", time.Since(start))

	circuit, pk, err := loadGroth16Circuit(dataDir, gpuR1csSolverArmed(dataDir))
	if err != nil {
		panic(err)
	}

	start = time.Now()
	// Read the file.
	data, err := os.ReadFile(witnessPath)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Reading witness file took %s\n", time.Since(start))

	start = time.Now()
	// Deserialize the JSON data into a slice of Instruction structs
	var witnessInput WitnessInput
	err = json.Unmarshal(data, &witnessInput)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Deserializing JSON data took %s\n", time.Since(start))

	start = time.Now()
	// Generate the witness.
	witness, err := newWitness(witnessInput)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Generating witness took %s\n", time.Since(start))

	start = time.Now()
	// Generate the proof: on the GPU when the host registered it (groth16_gpu.go), checked against
	// the verifying key, with gnark's CPU prover as the fallback.
	var proof groth16.Proof
	if useGpuProver() {
		proof = proveGroth16OnGpu(circuit, pk, witness)
	}
	if proof == nil {
		// gnark's CPU prover, on the parsed R1CS: read now if only the metadata was.
		r1cs, err := circuit.fullR1cs()
		if err != nil {
			panic(err)
		}
		proof, err = groth16.Prove(r1cs, pk, witness)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			panic(err)
		}
	}
	fmt.Printf("Generating proof took %s\n", time.Since(start))

	return NewSP1Groth16Proof(&proof, witnessInput)
}

// loadGroth16Circuit returns the circuit in dataDir and its proving key, read under one lock
// (ReleaseCaches must not reset one between the two; deferred, so a panic cannot leave the lock
// held), the cached ones where they are for dataDir. Prove with what it returns, not the globals,
// which ReleaseCaches may replace meanwhile.
//
// With gpuSolverArmed (gpuR1csSolverArmed), it reads only the R1CS's metadata (readR1csMeta,
// ~1.4 s for the v6.1.0 circuit, alongside the proving key) and not the R1CS (~19 s), which only
// gnark's CPU solver needs; the circuit then parses it if a proof falls back to that
// (groth16Circuit.fullR1cs). It reads the R1CS itself where the metadata cannot be read, says
// there are BSB22 commitments (which the GPU solver does not handle), or does not match the
// proving key (needFullR1cs).
func loadGroth16Circuit(dataDir string, gpuSolverArmed bool) (*groth16Circuit, groth16.ProvingKey, error) {
	globalMutex.Lock()
	defer globalMutex.Unlock()

	// The proving key, alongside the R1CS or its metadata.
	pk := globalPk
	pkErr := make(chan error, 1)
	if pk == nil || globalPkDataDir != dataDir {
		go func() {
			start := time.Now()
			var err error
			if pk, err = readGroth16Pk(dataDir + "/" + groth16PkPath); err == nil {
				fmt.Printf("Reading proving key took %s\n", time.Since(start))
			}
			pkErr <- err
		}()
	} else {
		pkErr <- nil
	}

	var meta *r1csMeta
	var metaErr error
	r1cs := globalR1cs
	if r1cs == nil || globalR1csDataDir != dataDir {
		r1cs = nil
		if !gpuSolverArmed {
			// The R1CS, as gnark's solver needs it (needFullR1cs), alongside the proving key.
			var err error
			if r1cs, err = readGroth16R1cs(dataDir); err != nil {
				<-pkErr
				return nil, nil, err
			}
			globalR1cs, globalR1csDataDir = r1cs, dataDir
		} else {
			if globalR1csMeta != nil && globalR1csMetaDataDir == dataDir {
				meta = globalR1csMeta
			} else {
				start := time.Now()
				if meta, metaErr = readR1csMeta(dataDir + "/" + groth16CircuitPath); metaErr == nil {
					fmt.Printf("Reading R1CS metadata took %s (the GPU R1CS solver does not need the R1CS)\n", time.Since(start))
				}
			}
		}
	}
	if err := <-pkErr; err != nil {
		return nil, nil, err
	}
	globalPk, globalPkDataDir = pk, dataDir

	if r1cs == nil {
		if full, why := needFullR1cs(gpuSolverArmed, meta, metaErr, pk); full {
			if gpuSolverArmed {
				fmt.Printf("Reading the R1CS: %s\n", why)
			}
			var err error
			if r1cs, err = readGroth16R1cs(dataDir); err != nil {
				return nil, nil, err
			}
			globalR1cs, globalR1csDataDir = r1cs, dataDir
		}
	}
	if r1cs != nil {
		var err error
		if meta, err = r1csMetaOf(r1cs); err != nil {
			return nil, nil, err
		}
	}
	globalR1csMeta, globalR1csMetaDataDir = meta, dataDir
	return &groth16Circuit{
		dataDir:  dataDir,
		meta:     meta,
		r1cs:     r1cs,
		loadR1cs: func() (*cs.R1CS, error) { return loadGroth16R1cs(dataDir) },
	}, pk, nil
}

// needFullR1cs: whether ProveGroth16 parses the R1CS before proving rather than proving with
// its metadata (meta, or metaErr where that could not be read), and if so why: the GPU R1CS
// solver not armed, metadata that could not be read, BSB22 commitments (whose hint the GPU solver
// does not run), or metadata that does not fit the proving key pk.
func needFullR1cs(gpuSolverArmed bool, meta *r1csMeta, metaErr error, pk groth16.ProvingKey) (bool, string) {
	switch {
	case !gpuSolverArmed:
		return true, "the GPU R1CS solver is not armed"
	case metaErr != nil || meta == nil:
		return true, fmt.Sprintf("no metadata (%v)", metaErr)
	case len(meta.commitments) != 0:
		return true, fmt.Sprintf("%d BSB22 commitments", len(meta.commitments))
	}
	bnPk, ok := pk.(*groth16_bn254.ProvingKey)
	if !ok {
		return true, fmt.Sprintf("a proving key of type %T", pk)
	}
	if len(bnPk.InfinityA) != meta.nbWires() || len(bnPk.InfinityB) != meta.nbWires() {
		return true, fmt.Sprintf("the metadata's %d wires do not fit the proving key's %d", meta.nbWires(), len(bnPk.InfinityA))
	}
	return false, ""
}

// loadGroth16R1cs is the parsed R1CS in dataDir, the cached one if there is one.
func loadGroth16R1cs(dataDir string) (*cs.R1CS, error) {
	globalMutex.Lock()
	defer globalMutex.Unlock()
	if globalR1cs != nil && globalR1csDataDir == dataDir {
		return globalR1cs, nil
	}
	r1cs, err := readGroth16R1cs(dataDir)
	if err != nil {
		return nil, err
	}
	globalR1cs, globalR1csDataDir = r1cs, dataDir
	return r1cs, nil
}

// readGroth16R1cs parses the R1CS in dataDir.
func readGroth16R1cs(dataDir string) (*cs.R1CS, error) {
	start := time.Now()
	r1cs, err := readR1cs(dataDir + "/" + groth16CircuitPath)
	if err != nil {
		return nil, err
	}
	bn, ok := r1cs.(*cs.R1CS)
	if !ok {
		return nil, fmt.Errorf("R1CS in %s is a %T", dataDir, r1cs)
	}
	fmt.Printf("Reading R1CS took %s\n", time.Since(start))
	return bn, nil
}

// readGroth16Pk reads the proving key (gnark's dump) at path, returning a panic in gnark's
// reader as an error too.
func readGroth16Pk(path string) (pk groth16.ProvingKey, err error) {
	defer func() {
		if r := recover(); r != nil {
			pk, err = nil, fmt.Errorf("failed to read proving key %s: %v", path, r)
		}
	}()
	pkFile, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open proving key %s: %w", path, err)
	}
	defer pkFile.Close()
	pk = groth16.NewProvingKey(ecc.BN254)
	if err := pk.ReadDump(bufio.NewReaderSize(pkFile, 1024*1024)); err != nil {
		return nil, fmt.Errorf("failed to read proving key %s: %w", path, err)
	}
	return pk, nil
}

// releaseProverCache drops the R1CS, its metadata and the proving key ProveGroth16 keeps, so the
// next call reloads them. See ReleaseCaches.
func releaseProverCache() {
	globalMutex.Lock()
	defer globalMutex.Unlock()
	globalR1cs, globalR1csDataDir = nil, ""
	globalR1csMeta, globalR1csMetaDataDir = nil, ""
	globalPk, globalPkDataDir = nil, ""
}
