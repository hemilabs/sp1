package main

/*
#include "./koalabear.h"
#include <stdint.h>
#include <stdlib.h>

typedef struct {
	char *PublicInputs[5];
	char *EncodedProof;
	char *RawProof;
} C_PlonkBn254Proof;

typedef struct {
	char *PublicInputs[5];
	char *EncodedProof;
	char *RawProof;
} C_Groth16Bn254Proof;
*/
import "C"
import (
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"runtime/debug"
	"sync"
	"unsafe"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/backend/plonk"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	"github.com/consensys/gnark/frontend/cs/scs"
	"github.com/consensys/gnark/test/unsafekzg"
	"github.com/succinctlabs/sp1-recursion-gnark/sp1"
	"github.com/succinctlabs/sp1-recursion-gnark/sp1/koalabear"
	"github.com/succinctlabs/sp1-recursion-gnark/sp1/poseidon2"
)

func main() {}

//export ProvePlonkBn254
func ProvePlonkBn254(dataDir *C.char, witnessPath *C.char) *C.C_PlonkBn254Proof {
	// The circuit and proving key read below are tens of GB of garbage afterwards. Return them to
	// the host rather than leave a long-lived prover that much larger: the next proof's GPU PLONK
	// helper (~100 GiB of host memory) did not fit beside them, and was OOM-killed.
	defer freeOSMemory()
	dataDirString := C.GoString(dataDir)
	witnessPathString := C.GoString(witnessPath)

	sp1PlonkBn254Proof := sp1.ProvePlonk(dataDirString, witnessPathString)

	ms := C.malloc(C.sizeof_C_PlonkBn254Proof)
	if ms == nil {
		return nil
	}

	structPtr := (*C.C_PlonkBn254Proof)(ms)
	structPtr.PublicInputs[0] = C.CString(sp1PlonkBn254Proof.PublicInputs[0])
	structPtr.PublicInputs[1] = C.CString(sp1PlonkBn254Proof.PublicInputs[1])
	structPtr.PublicInputs[2] = C.CString(sp1PlonkBn254Proof.PublicInputs[2])
	structPtr.PublicInputs[3] = C.CString(sp1PlonkBn254Proof.PublicInputs[3])
	structPtr.PublicInputs[4] = C.CString(sp1PlonkBn254Proof.PublicInputs[4])
	structPtr.EncodedProof = C.CString(sp1PlonkBn254Proof.EncodedProof)
	structPtr.RawProof = C.CString(sp1PlonkBn254Proof.RawProof)
	return structPtr
}

//export FreePlonkBn254Proof
func FreePlonkBn254Proof(proof *C.C_PlonkBn254Proof) {
	C.free(unsafe.Pointer(proof.EncodedProof))
	C.free(unsafe.Pointer(proof.RawProof))
	C.free(unsafe.Pointer(proof.PublicInputs[0]))
	C.free(unsafe.Pointer(proof.PublicInputs[1]))
	C.free(unsafe.Pointer(proof.PublicInputs[2]))
	C.free(unsafe.Pointer(proof.PublicInputs[3]))
	C.free(unsafe.Pointer(proof.PublicInputs[4]))
	C.free(unsafe.Pointer(proof))
}

//export BuildPlonkBn254
func BuildPlonkBn254(dataDir *C.char) {
	// Sanity check the required arguments have been provided.
	dataDirString := C.GoString(dataDir)

	sp1.BuildPlonk(dataDirString)
}

//export VerifyPlonkBn254
func VerifyPlonkBn254(dataDir *C.char, proof *C.char, vkeyHash *C.char, committedValuesDigest *C.char, exitCode *C.char, vkRoot *C.char, proofNonce *C.char) *C.char {
	dataDirString := C.GoString(dataDir)
	proofString := C.GoString(proof)
	vkeyHashString := C.GoString(vkeyHash)
	committedValuesDigestString := C.GoString(committedValuesDigest)
	exitCodeString := C.GoString(exitCode)
	vkRootString := C.GoString(vkRoot)
	proofNonceString := C.GoString(proofNonce)
	err := sp1.VerifyPlonk(dataDirString, proofString, vkeyHashString, committedValuesDigestString, exitCodeString, vkRootString, proofNonceString)
	if err != nil {
		return C.CString(err.Error())
	}
	return nil
}

var testMutex = &sync.Mutex{}

//export TestPlonkBn254
func TestPlonkBn254(witnessPath *C.char, constraintsJson *C.char) *C.char {
	// Because of the global env variables used here, we need to lock this function
	testMutex.Lock()
	witnessPathString := C.GoString(witnessPath)
	constraintsJsonString := C.GoString(constraintsJson)
	os.Setenv("WITNESS_JSON", witnessPathString)
	os.Setenv("CONSTRAINTS_JSON", constraintsJsonString)
	err := TestMain()
	testMutex.Unlock()
	if err != nil {
		return C.CString(err.Error())
	}
	return nil
}

//export ProveGroth16Bn254
func ProveGroth16Bn254(dataDir *C.char, witnessPath *C.char) *C.C_Groth16Bn254Proof {
	dataDirString := C.GoString(dataDir)
	witnessPathString := C.GoString(witnessPath)

	sp1Groth16Bn254Proof := sp1.ProveGroth16(dataDirString, witnessPathString)
	return groth16ProofToC(sp1Groth16Bn254Proof)
}

// ProveGroth16Bn254WithR1cs returns nil and sets *r1csErr when the R1CS cannot be read; see
// sp1.ProveGroth16WithR1cs.
//
//export ProveGroth16Bn254WithR1cs
func ProveGroth16Bn254WithR1cs(dataDir *C.char, r1csPath *C.char, witnessPath *C.char, r1csErr **C.char) *C.C_Groth16Bn254Proof {
	proof, err := sp1.ProveGroth16WithR1cs(C.GoString(dataDir), C.GoString(r1csPath), C.GoString(witnessPath))
	if err != nil {
		*r1csErr = C.CString(err.Error())
		return nil
	}
	return groth16ProofToC(proof)
}

//export ReleaseGroth16Caches
func ReleaseGroth16Caches() {
	sp1.ReleaseCaches()
}

//export TrimGroth16Caches
func TrimGroth16Caches() {
	sp1.TrimCaches()
}

//export ExportGroth16StrippedR1cs
func ExportGroth16StrippedR1cs(dataDir *C.char, outputPath *C.char) (errStr *C.char) {
	// The full circuit read below is ~9 GB of garbage afterwards, whether or not the export
	// succeeds. The CPU helper proves right after this in the same process, so return it to the
	// host before that prove's own peak. Deferred first, so it runs after the recover below.
	defer freeOSMemory()
	defer func() {
		if r := recover(); r != nil {
			errStr = C.CString(fmt.Sprintf("[groth16-strip] panic: %v", r))
		}
	}()
	sp1.ExportGroth16StrippedR1cs(C.GoString(dataDir), C.GoString(outputPath))
	return nil
}

// freeOSMemory collects the garbage and returns the freed memory to the host now, rather than
// as Go's background scavenger gets to it.
func freeOSMemory() {
	runtime.GC()
	debug.FreeOSMemory()
}

func groth16ProofToC(proof sp1.Proof) *C.C_Groth16Bn254Proof {
	ms := C.malloc(C.sizeof_C_Groth16Bn254Proof)
	if ms == nil {
		return nil
	}

	structPtr := (*C.C_Groth16Bn254Proof)(ms)
	structPtr.PublicInputs[0] = C.CString(proof.PublicInputs[0])
	structPtr.PublicInputs[1] = C.CString(proof.PublicInputs[1])
	structPtr.PublicInputs[2] = C.CString(proof.PublicInputs[2])
	structPtr.PublicInputs[3] = C.CString(proof.PublicInputs[3])
	structPtr.PublicInputs[4] = C.CString(proof.PublicInputs[4])
	structPtr.EncodedProof = C.CString(proof.EncodedProof)
	structPtr.RawProof = C.CString(proof.RawProof)
	return structPtr
}

// SetGroth16GpuHooks registers the host's GPU entry points for the Groth16 prover (C function
// pointers, see sp1/groth16_gpu.go; zeros unregister them) and the GPU's total memory in bytes (0
// if unknown).
//
//export SetGroth16GpuHooks
func SetGroth16GpuHooks(computeH C.uintptr_t, msmG1 C.uintptr_t, msmG2 C.uintptr_t, totalMemory C.uint64_t) {
	sp1.SetGpuHooks(uintptr(computeH), uintptr(msmG1), uintptr(msmG2), uint64(totalMemory))
}

// SetGroth16G2MsmOnGpu records whether the host's GPU runs the Groth16 prover's G2 MSM faster
// than the CPU (onGpu != 0), in place of the default (see sp1/groth16_gpu.go: g2MsmOnCpu).
//
//export SetGroth16G2MsmOnGpu
func SetGroth16G2MsmOnGpu(onGpu C.int) {
	sp1.SetG2MsmOnGpu(onGpu != 0)
}

// SetGroth16GpuR1csSolver registers the host's GPU R1CS solver for the Groth16 prover (C function
// pointers for sp1_r1cs_solver_create, _solve and _destroy; zeros unregister it). It solves where
// SetGroth16R1csPrepDir names its circuit data.
//
//export SetGroth16GpuR1csSolver
func SetGroth16GpuR1csSolver(create C.uintptr_t, solve C.uintptr_t, destroy C.uintptr_t) {
	sp1.SetGpuR1csSolver(uintptr(create), uintptr(solve), uintptr(destroy))
}

// ReleaseGroth16GpuR1csSolver destroys the GPU R1CS solver kept between Groth16 proofs, if any,
// giving its device memory back.
//
//export ReleaseGroth16GpuR1csSolver
func ReleaseGroth16GpuR1csSolver() {
	sp1.ReleaseGpuR1csSolver()
}

// SetGroth16R1csPrepDir names the GPU R1CS solver's circuit data (`r1cs_solve_plan
// prep-circuit-prod`'s output) for the Groth16 proofs of the circuit in dataDir; empty for none.
//
//export SetGroth16R1csPrepDir
func SetGroth16R1csPrepDir(dataDir *C.char, dir *C.char) {
	sp1.SetGpuR1csPrepDir(C.GoString(dataDir), C.GoString(dir))
}

//export FreeGroth16Bn254Proof
func FreeGroth16Bn254Proof(proof *C.C_Groth16Bn254Proof) {
	C.free(unsafe.Pointer(proof.EncodedProof))
	C.free(unsafe.Pointer(proof.RawProof))
	C.free(unsafe.Pointer(proof.PublicInputs[0]))
	C.free(unsafe.Pointer(proof.PublicInputs[1]))
	C.free(unsafe.Pointer(proof.PublicInputs[2]))
	C.free(unsafe.Pointer(proof.PublicInputs[3]))
	C.free(unsafe.Pointer(proof.PublicInputs[4]))
	C.free(unsafe.Pointer(proof))
}

//export ExportGroth16GpuData
func ExportGroth16GpuData(dataDir *C.char, outputDir *C.char) (errStr *C.char) {
	// The export reads the full circuit and proving key (~9 GB of garbage afterwards); return it
	// to the host either way rather than leave a long-lived prover that much larger.
	defer freeOSMemory()
	defer func() {
		if r := recover(); r != nil {
			errStr = C.CString(fmt.Sprintf("[groth16-export] panic: %v", r))
		}
	}()
	sp1.ExportGroth16GpuData(C.GoString(dataDir), C.GoString(outputDir))
	return nil
}

//export ExportGroth16GpuWitness
func ExportGroth16GpuWitness(dataDir *C.char, witnessPath *C.char, pkDir *C.char, outputDir *C.char) (errStr *C.char) {
	defer func() {
		if r := recover(); r != nil {
			errStr = C.CString(fmt.Sprintf("[groth16-witness] panic: %v", r))
		}
	}()
	sp1.ExportGroth16GpuWitness(C.GoString(dataDir), C.GoString(witnessPath), C.GoString(pkDir), C.GoString(outputDir))
	return nil
}

//export PrefetchGroth16GpuWitnessCircuit
func PrefetchGroth16GpuWitnessCircuit(dataDir *C.char, pkDir *C.char) (errStr *C.char) {
	defer func() {
		if r := recover(); r != nil {
			errStr = C.CString(fmt.Sprintf("[groth16-witness] prefetch panic: %v", r))
		}
	}()
	sp1.PrefetchGroth16GpuWitnessCircuit(C.GoString(dataDir), C.GoString(pkDir))
	return nil
}

//export ExportPlonkGpuData
func ExportPlonkGpuData(dataDir *C.char, outputDir *C.char) (errStr *C.char) {
	// As in ProvePlonkBn254, whether or not the export succeeds. Deferred first, so it runs after
	// the recover below.
	defer freeOSMemory()
	defer func() {
		if r := recover(); r != nil {
			errStr = C.CString(fmt.Sprintf("[plonk-export] panic: %v", r))
		}
	}()
	sp1.ExportPlonkData(C.GoString(dataDir), C.GoString(outputDir))
	return nil
}

//export ExportPlonkGpuWitness
func ExportPlonkGpuWitness(dataDir *C.char, witnessPath *C.char, outputDir *C.char) (errStr *C.char) {
	// As in ExportPlonkGpuData: the GPU PLONK helper starts right after this, and needs the room.
	defer freeOSMemory()
	defer func() {
		if r := recover(); r != nil {
			errStr = C.CString(fmt.Sprintf("[plonk-witness] panic: %v", r))
		}
	}()
	sp1.ExportSolvedWitness(C.GoString(dataDir), C.GoString(witnessPath), C.GoString(outputDir))
	return nil
}

//export BuildGroth16Bn254
func BuildGroth16Bn254(dataDir *C.char) {
	// Sanity check the required arguments have been provided.
	dataDirString := C.GoString(dataDir)

	sp1.BuildGroth16(dataDirString)
}

//export VerifyGroth16Bn254
func VerifyGroth16Bn254(dataDir *C.char, proof *C.char, vkeyHash *C.char, committedValuesDigest *C.char, exitCode *C.char, vkRoot *C.char, proofNonce *C.char) *C.char {
	dataDirString := C.GoString(dataDir)
	proofString := C.GoString(proof)
	vkeyHashString := C.GoString(vkeyHash)
	committedValuesDigestString := C.GoString(committedValuesDigest)
	exitCodeString := C.GoString(exitCode)
	vkRootString := C.GoString(vkRoot)
	proofNonceString := C.GoString(proofNonce)
	err := sp1.VerifyGroth16(dataDirString, proofString, vkeyHashString, committedValuesDigestString, exitCodeString, vkRootString, proofNonceString)
	if err != nil {
		return C.CString(err.Error())
	}
	return nil
}

//export TestGroth16Bn254
func TestGroth16Bn254(witnessJson *C.char, constraintsJson *C.char) *C.char {
	// Because of the global env variables used here, we need to lock this function
	testMutex.Lock()
	witnessPathString := C.GoString(witnessJson)
	constraintsJsonString := C.GoString(constraintsJson)
	os.Setenv("WITNESS_JSON", witnessPathString)
	os.Setenv("CONSTRAINTS_JSON", constraintsJsonString)
	os.Setenv("GROTH16", "1")
	err := TestMain()
	testMutex.Unlock()
	if err != nil {
		return C.CString(err.Error())
	}
	return nil
}

func TestMain() error {
	// Get the file name from an environment variable.
	fileName := os.Getenv("WITNESS_JSON")
	if fileName == "" {
		fileName = "plonk_witness.json"
	}

	// Read the file.
	data, err := os.ReadFile(fileName)
	if err != nil {
		return err
	}

	// Deserialize the JSON data into a slice of Instruction structs
	var inputs sp1.WitnessInput
	err = json.Unmarshal(data, &inputs)
	if err != nil {
		return err
	}

	// Compile the circuit.
	circuit := sp1.NewCircuit(inputs)
	scs, err := frontend.Compile(ecc.BN254.ScalarField(), scs.NewBuilder, &circuit)
	if err != nil {
		return err
	}
	fmt.Println("[sp1] gnark verifier constraints:", scs.GetNbConstraints())

	// Run the dummy setup.
	srs, srsLagrange, err := unsafekzg.NewSRS(scs)
	if err != nil {
		return err
	}
	var pk plonk.ProvingKey
	pk, _, err = plonk.Setup(scs, srs, srsLagrange)
	if err != nil {
		return err
	}

	// Generate witness.
	assignment := sp1.NewCircuit(inputs)
	witness, err := frontend.NewWitness(&assignment, ecc.BN254.ScalarField())
	if err != nil {
		return err
	}

	// Generate the proof.
	_, err = plonk.Prove(scs, pk, witness)
	if err != nil {
		return err
	}

	return nil
}

//export TestPoseidonKoalaBear2
func TestPoseidonKoalaBear2() *C.char {
	input := [poseidon2.KOALABEAR_WIDTH]koalabear.Variable{
		koalabear.NewFConst("0"),
		koalabear.NewFConst("0"),
		koalabear.NewFConst("0"),
		koalabear.NewFConst("0"),
		koalabear.NewFConst("0"),
		koalabear.NewFConst("0"),
		koalabear.NewFConst("0"),
		koalabear.NewFConst("0"),
		koalabear.NewFConst("0"),
		koalabear.NewFConst("0"),
		koalabear.NewFConst("0"),
		koalabear.NewFConst("0"),
		koalabear.NewFConst("0"),
		koalabear.NewFConst("0"),
		koalabear.NewFConst("0"),
		koalabear.NewFConst("0"),
	}

	expectedOutput := [poseidon2.KOALABEAR_WIDTH]koalabear.Variable{
		koalabear.NewFConst("145589356"),
		koalabear.NewFConst("1876041682"),
		koalabear.NewFConst("1734203622"),
		koalabear.NewFConst("499355069"),
		koalabear.NewFConst("673349476"),
		koalabear.NewFConst("595701365"),
		koalabear.NewFConst("270340205"),
		koalabear.NewFConst("131707822"),
		koalabear.NewFConst("1236787881"),
		koalabear.NewFConst("1085405948"),
		koalabear.NewFConst("2065733208"),
		koalabear.NewFConst("1999012278"),
		koalabear.NewFConst("2062318124"),
		koalabear.NewFConst("1616707536"),
		koalabear.NewFConst("324813015"),
		koalabear.NewFConst("749520722"),
	}

	circuit := sp1.TestPoseidon2KoalaBearCircuit{Input: input, ExpectedOutput: expectedOutput}
	assignment := sp1.TestPoseidon2KoalaBearCircuit{Input: input, ExpectedOutput: expectedOutput}

	r1cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &circuit)
	if err != nil {
		return C.CString(err.Error())
	}

	var pk groth16.ProvingKey
	pk, err = groth16.DummySetup(r1cs)
	if err != nil {
		return C.CString(err.Error())
	}

	// Generate witness.
	witness, err := frontend.NewWitness(&assignment, ecc.BN254.ScalarField())
	if err != nil {
		return C.CString(err.Error())
	}

	// Generate the proof.
	_, err = groth16.Prove(r1cs, pk, witness)
	if err != nil {
		return C.CString(err.Error())
	}

	return nil
}

//export FreeString
func FreeString(s *C.char) {
	C.free(unsafe.Pointer(s))
}
