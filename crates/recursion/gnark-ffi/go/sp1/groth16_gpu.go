package sp1

// The Groth16 prover with its quotient polynomial and multi-scalar multiplications on the GPU.
//
// The host process (the SP1 GPU prover server) registers its GPU entry points with
// SetGroth16GpuHooks; proveGroth16Gpu is then gnark's bn254 prover (backend/groth16/bn254/prove.go,
// Apache-2.0, at the version go.mod pins) with computeH and the five MultiExps calling them. The
// solving, the commitments and the proof's assembly are gnark's own, so the proof is the one gnark
// would make, and proveGroth16OnGpu checks it against the verifying key before it is used.

/*
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

typedef int32_t (*sp1_g16_msm_fn)(void* out, const void* points, const void* scalars, size_t n);
typedef int32_t (*sp1_g16_h_fn)(void* h, const void* a, const void* b, const void* c, size_t n_in,
                                uint32_t lg, const void* consts);

static int32_t sp1_g16_call_msm(uintptr_t f, void* out, const void* points, const void* scalars,
                                size_t n) {
	return ((sp1_g16_msm_fn)f)(out, points, scalars, n);
}

static int32_t sp1_g16_call_h(uintptr_t f, void* h, const void* a, const void* b, const void* c,
                              size_t n_in, uint32_t lg, const void* consts) {
	return ((sp1_g16_h_fn)f)(h, a, b, c, n_in, lg, consts);
}

// The GPU R1CS solver (sp1-gpu's sp1_r1cs_solver_create, _solve and _destroy).
typedef void* (*sp1_r1cs_create_fn)(const char* prep_dir, uint64_t* n_wires,
                                    uint64_t* n_constraints);
typedef int (*sp1_r1cs_solve_fn)(void* h, const void* wires_initial, void* wires_out, void* a,
                                 void* b, void* c);
typedef void (*sp1_r1cs_destroy_fn)(void* h);

static void* sp1_g16_call_r1cs_create(uintptr_t f, const char* prep_dir, uint64_t* n_wires,
                                      uint64_t* n_constraints) {
	return ((sp1_r1cs_create_fn)f)(prep_dir, n_wires, n_constraints);
}

static int sp1_g16_call_r1cs_solve(uintptr_t f, void* h, const void* wires_initial,
                                   void* wires_out, void* a, void* b, void* c) {
	return ((sp1_r1cs_solve_fn)f)(h, wires_initial, wires_out, a, b, c);
}

static void sp1_g16_call_r1cs_destroy(uintptr_t f, void* h) {
	((sp1_r1cs_destroy_fn)f)(h);
}
*/
import "C"

import (
	"fmt"
	"math/big"
	"math/bits"
	"os"
	"runtime"
	"sort"
	"sync"
	"time"
	"unsafe"

	"github.com/consensys/gnark-crypto/ecc"
	curve "github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr/fft"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr/hash_to_field"
	"github.com/consensys/gnark/backend"
	"github.com/consensys/gnark/backend/groth16"
	groth16_bn254 "github.com/consensys/gnark/backend/groth16/bn254"
	"github.com/consensys/gnark/backend/witness"
	"github.com/consensys/gnark/constraint"
	cs "github.com/consensys/gnark/constraint/bn254"
	"github.com/consensys/gnark/constraint/solver"
	fcs "github.com/consensys/gnark/frontend/cs"
)

var gpuHooks struct {
	computeH, msmG1, msmG2 uintptr
	totalMemory            uint64
	// g2OnGpu is the host's choice for the G2 MSM (SetG2MsmOnGpu): -1 for none, else 0 or 1.
	g2OnGpu int32
}

func init() {
	gpuHooks.g2OnGpu = -1
}

// SetG2MsmOnGpu records where the host's GPU is known to run the G2 MSM faster (onGpu) or slower
// than the CPU does alongside the G1 MSMs, in place of g2MsmOnCpu's default.
func SetG2MsmOnGpu(onGpu bool) {
	if onGpu {
		gpuHooks.g2OnGpu = 1
	} else {
		gpuHooks.g2OnGpu = 0
	}
}

// SetGpuHooks registers the GPU's computeH and MSM entry points (C function pointers; 0 unsets),
// and the GPU's total memory in bytes (0 if unknown).
func SetGpuHooks(computeH, msmG1, msmG2 uintptr, totalMemory uint64) {
	gpuHooks.computeH = computeH
	gpuHooks.msmG1 = msmG1
	gpuHooks.msmG2 = msmG2
	gpuHooks.totalMemory = totalMemory
}

// g2MsmOnCpu: whether the G2 MSM runs on the CPU, alongside the G1 MSMs on the GPU, rather than
// on the GPU after them. SP1_GROTH16_G2_ON_CPU=0 runs it on the GPU and any other value on the
// CPU. By default it runs on the CPU, where that took about as long as the G1 MSMs (~3 s each on
// an Arc Pro B70, whose G2 MSM takes ~2.7 s more), except on cards of 64 GiB or more and on
// hosts with fewer than 16 CPUs: on an MI300X with 13 vCPUs the CPU G2 MSM took 6.7-6.9 s and the
// GPU's 0.58 s (Groth16 proofs 16-18% faster on the GPU). The host's choice (SetG2MsmOnGpu), if
// any, replaces that default.
func g2MsmOnCpu() bool {
	if v, ok := os.LookupEnv("SP1_GROTH16_G2_ON_CPU"); ok {
		return v != "0"
	}
	if gpuHooks.g2OnGpu >= 0 {
		return gpuHooks.g2OnGpu == 0
	}
	return gpuHooks.totalMemory < 64<<30 && runtime.NumCPU() >= 16
}

// gpuR1cs is the GPU R1CS solver the host registered (SetGpuR1csSolver) and, for each circuit's
// data directory, the directory of the circuit data the solver reads for it (SetGpuR1csPrepDir,
// `r1cs_solve_plan prep-circuit-prod`'s output). Per data directory, so that a proof of one
// circuit never solves with another's data, whatever the host proved before or proves meanwhile.
var gpuR1cs struct {
	sync.Mutex
	create, solve, destroy uintptr
	prepDirs               map[string]string
}

// SetGpuR1csSolver registers the GPU R1CS solver's entry points (C function pointers; 0 unsets).
// The solver kept from earlier proofs goes: destroyed, unless this unregisters the solver, which
// gnark-ffi does after a device reset, when its memory is gone with the device's state.
func SetGpuR1csSolver(create, solve, destroy uintptr) {
	gpuR1cs.Lock()
	gpuR1cs.create, gpuR1cs.solve, gpuR1cs.destroy = create, solve, destroy
	gpuR1cs.Unlock()
	if create == 0 {
		gpuR1csKept.Lock()
		gpuR1csKept.h, gpuR1csKept.key = nil, ""
		gpuR1csKept.Unlock()
		return
	}
	ReleaseGpuR1csSolver()
}

// gpuR1csKept is the solver gpuSolveR1cs created, with its circuit data (~3 GB on the GPU) and
// buffers (~2 GB), kept for the next proofs where keepGpuR1csSolver says so: creating it took
// 2.0-2.3 s of each proof on an MI300X. key names the circuit data it read (gpuR1csPrepKey), and
// nW and nC are its numbers of wires and constraints, which size what a solve writes back: it is
// used only for a circuit of the same sizes. Its
// device memory comes from the prover's allocator, out of the device memory arena where there is
// one, so it does not take the arena's headroom, which kernel scratch needs. The lock is held
// through each solve: the solver has one set of buffers.
var gpuR1csKept struct {
	sync.Mutex
	h       unsafe.Pointer
	destroy uintptr
	key     string
	nW, nC  int
}

// gpuR1csKeptFits: whether the kept solver (gpuR1csKept, locked by the caller) can solve the
// circuit with the circuit data key names (gpuR1csPrepKey; "" when the solver is not to be kept),
// with nbWires wires and nbConstraints constraints, for the registered destroy. A solver for
// other sizes would write past the end of the vectors sized for this circuit.
func gpuR1csKeptFits(key string, destroy uintptr, nbWires, nbConstraints int) bool {
	return key != "" && key == gpuR1csKept.key && destroy == gpuR1csKept.destroy &&
		nbWires == gpuR1csKept.nW && nbConstraints == gpuR1csKept.nC
}

// ReleaseGpuR1csSolver destroys the solver kept between proofs, if any, giving its device memory
// back (with the other device caches the host empties).
func ReleaseGpuR1csSolver() {
	gpuR1csKept.Lock()
	defer gpuR1csKept.Unlock()
	if gpuR1csKept.h != nil {
		C.sp1_g16_call_r1cs_destroy(C.uintptr_t(gpuR1csKept.destroy), gpuR1csKept.h)
	}
	gpuR1csKept.h, gpuR1csKept.key = nil, ""
}

// keepGpuR1csSolver: whether the GPU R1CS solver stays on the GPU between proofs.
// SP1_GPU_R1CS_KEEP=0 destroys it after each solve and any other value keeps it; by default it is
// kept on cards of 64 GiB or more, where its ~5 GB are a small part of the arena.
func keepGpuR1csSolver() bool {
	if v, ok := os.LookupEnv("SP1_GPU_R1CS_KEEP"); ok {
		return v != "0"
	}
	return gpuHooks.totalMemory >= 64<<30
}

// gpuR1csPrepKey identifies the circuit data in prepDir: the directory and the size and
// modification time of each file the solver reads, so that data rebuilt in place, for another
// circuit, is read again. "" if a file cannot be read (the solver is then not kept).
func gpuR1csPrepKey(prepDir string) string {
	key := prepDir
	for _, name := range []string{"circuit_meta.txt", "coeffs.bin", "layers_descs.bin",
		"layers_terms.bin", "layers.idx", "hints.idx", "layers_hints.bin", "hint_in_les.bin",
		"desc_decl_idx.bin"} {
		info, err := os.Stat(prepDir + "/" + name)
		if err != nil {
			return ""
		}
		key += fmt.Sprintf("|%s:%d:%d", name, info.Size(), info.ModTime().UnixNano())
	}
	return key
}

// SetGpuR1csPrepDir names the GPU R1CS solver's circuit data for the proofs of the circuit in
// dataDir ("" for none: gnark then solves them on the CPU).
func SetGpuR1csPrepDir(dataDir, dir string) {
	gpuR1cs.Lock()
	defer gpuR1cs.Unlock()
	if dir == "" {
		delete(gpuR1cs.prepDirs, dataDir)
		return
	}
	if gpuR1cs.prepDirs == nil {
		gpuR1cs.prepDirs = make(map[string]string)
	}
	gpuR1cs.prepDirs[dataDir] = dir
}

// gpuSolveBufs holds the GPU R1CS solve's host buffers between proofs (about 2.5 GB for the
// SP1 circuit): fresh Go heap pages cost proofs 2+ a page fault each on the first write, as no
// GC runs between proofs to recycle the last proof's. The initial wires are only ever written
// in their first nbInputs entries, so the rest stays zero for the next proof.
var gpuSolveBufs struct {
	sync.Mutex
	nbInputs   int
	wires      []fr.Element
	w, a, b, c fr.Vector
}

// takeGpuSolveBufs returns the initial wires (ONE, then vec, then zeros) and the vectors for the
// solution, the kept ones where their sizes match and fresh ones otherwise.
func takeGpuSolveBufs(vec fr.Vector, nbWires, nbConstraints int) ([]fr.Element, *cs.R1CSSolution) {
	nbInputs := len(vec) + 1
	gpuSolveBufs.Lock()
	wires := gpuSolveBufs.wires
	if len(wires) != nbWires || gpuSolveBufs.nbInputs != nbInputs {
		wires = make([]fr.Element, nbWires)
	}
	solution := &cs.R1CSSolution{W: gpuSolveBufs.w, A: gpuSolveBufs.a, B: gpuSolveBufs.b, C: gpuSolveBufs.c}
	gpuSolveBufs.wires, gpuSolveBufs.w, gpuSolveBufs.a, gpuSolveBufs.b, gpuSolveBufs.c = nil, nil, nil, nil, nil
	gpuSolveBufs.Unlock()
	if len(solution.W) != nbWires {
		solution.W = make(fr.Vector, nbWires)
	}
	for _, v := range []*fr.Vector{&solution.A, &solution.B, &solution.C} {
		if len(*v) != nbConstraints {
			*v = make(fr.Vector, nbConstraints)
		}
	}
	wires[0].SetOne()
	copy(wires[1:], vec)
	return wires, solution
}

// putGpuSolveBufs keeps the buffers takeGpuSolveBufs returned for the next proof. Nothing may
// use them after.
func putGpuSolveBufs(wires []fr.Element, nbInputs int, w, a, b, c fr.Vector) {
	gpuSolveBufs.Lock()
	defer gpuSolveBufs.Unlock()
	gpuSolveBufs.nbInputs = nbInputs
	gpuSolveBufs.wires, gpuSolveBufs.w, gpuSolveBufs.a, gpuSolveBufs.b, gpuSolveBufs.c = wires, w, a, b, c
}

// dropGpuSolveBufs lets the GC have the kept buffers (ReleaseCaches, TrimCaches), so a process
// that hands its Groth16 memory back to the host does not keep them.
func dropGpuSolveBufs() {
	gpuSolveBufs.Lock()
	defer gpuSolveBufs.Unlock()
	gpuSolveBufs.nbInputs = 0
	gpuSolveBufs.wires, gpuSolveBufs.w, gpuSolveBufs.a, gpuSolveBufs.b, gpuSolveBufs.c = nil, nil, nil, nil, nil
}

// useGpuProver: hooks registered, and not turned off with SP1_GROTH16_GPU=0.
func useGpuProver() bool {
	if gpuHooks.computeH == 0 || gpuHooks.msmG1 == 0 || gpuHooks.msmG2 == 0 {
		return false
	}
	return os.Getenv("SP1_GROTH16_GPU") != "0"
}

// gpuR1csSolverArmed: whether gpuSolveR1cs will solve the proofs of the circuit in dataDir on
// the GPU R1CS solver, as far as can be told before a proof: the GPU prover is used
// (useGpuProver), the host registered the solver and named the circuit data for dataDir, all of
// whose files are there (gpuR1csPrepKey), and SP1_GPU_R1CS_SOLVER is not cpu. ProveGroth16 then
// reads only the R1CS's metadata (readR1csMeta), not the R1CS, which only gnark's CPU solver
// needs. SP1_GROTH16_FULL_R1CS=1 has it read the R1CS anyway, as it did before.
func gpuR1csSolverArmed(dataDir string) bool {
	if !useGpuProver() || os.Getenv("SP1_GPU_R1CS_SOLVER") == "cpu" ||
		os.Getenv("SP1_GROTH16_FULL_R1CS") == "1" {
		return false
	}
	gpuR1cs.Lock()
	create, solve, destroy := gpuR1cs.create, gpuR1cs.solve, gpuR1cs.destroy
	prepDir := gpuR1cs.prepDirs[dataDir]
	gpuR1cs.Unlock()
	if create == 0 || solve == 0 || destroy == 0 || prepDir == "" {
		return false
	}
	return gpuR1csPrepKey(prepDir) != ""
}

// checkGpu: SP1_GROTH16_GPU_CHECK=1 compares every GPU result with gnark's CPU computation.
func checkGpu() bool { return os.Getenv("SP1_GROTH16_GPU_CHECK") == "1" }

func checkedMsmG1(name string, points []curve.G1Affine, scalars []fr.Element) (curve.G1Jac, error) {
	res, err := gpuMsmG1(points, scalars)
	if err != nil || !checkGpu() {
		return res, err
	}
	var cpu curve.G1Jac
	if _, err := cpu.MultiExp(points, scalars, ecc.MultiExpConfig{}); err != nil {
		return res, err
	}
	if !cpu.Equal(&res) {
		return res, fmt.Errorf("gpu msm %s (%d) differs from the CPU's", name, len(points))
	}
	fmt.Printf("GPU msm %s (%d) matches the CPU\n", name, len(points))
	return res, nil
}

func checkedMsmG2(name string, points []curve.G2Affine, scalars []fr.Element) (curve.G2Jac, error) {
	res, err := gpuMsmG2(points, scalars)
	if err != nil || !checkGpu() {
		return res, err
	}
	var cpu curve.G2Jac
	if _, err := cpu.MultiExp(points, scalars, ecc.MultiExpConfig{}); err != nil {
		return res, err
	}
	if !cpu.Equal(&res) {
		return res, fmt.Errorf("gpu msm %s (%d) differs from the CPU's", name, len(points))
	}
	fmt.Printf("GPU msm %s (%d) matches the CPU\n", name, len(points))
	return res, nil
}

// cpuComputeH is gnark's computeH, for the check.
func cpuComputeH(a, b, c []fr.Element, domain *fft.Domain) []fr.Element {
	n := len(a)
	padding := make([]fr.Element, int(domain.Cardinality)-n)
	a = append(append([]fr.Element{}, a...), padding...)
	b = append(append([]fr.Element{}, b...), padding...)
	c = append(append([]fr.Element{}, c...), padding...)
	n = len(a)
	domain.FFTInverse(a, fft.DIF)
	domain.FFTInverse(b, fft.DIF)
	domain.FFTInverse(c, fft.DIF)
	domain.FFT(a, fft.DIT, fft.OnCoset())
	domain.FFT(b, fft.DIT, fft.OnCoset())
	domain.FFT(c, fft.DIT, fft.OnCoset())
	var den, one fr.Element
	one.SetOne()
	den.Exp(domain.FrMultiplicativeGen, big.NewInt(int64(domain.Cardinality)))
	den.Sub(&den, &one).Inverse(&den)
	for i := 0; i < n; i++ {
		a[i].Mul(&a[i], &b[i]).Sub(&a[i], &c[i]).Mul(&a[i], &den)
	}
	domain.FFTInverse(a, fft.DIF, fft.OnCoset())
	return a
}

func gpuMsmG1(points []curve.G1Affine, scalars []fr.Element) (curve.G1Jac, error) {
	var res curve.G1Jac
	if len(points) != len(scalars) {
		return res, fmt.Errorf("gpu msm G1: %d points, %d scalars", len(points), len(scalars))
	}
	var out curve.G1Affine
	if len(points) > 0 {
		rc := C.sp1_g16_call_msm(C.uintptr_t(gpuHooks.msmG1), unsafe.Pointer(&out),
			unsafe.Pointer(&points[0]), unsafe.Pointer(&scalars[0]), C.size_t(len(points)))
		if rc != 0 {
			return res, fmt.Errorf("gpu msm G1 failed (%d)", int(rc))
		}
	}
	res.FromAffine(&out)
	return res, nil
}

func gpuMsmG2(points []curve.G2Affine, scalars []fr.Element) (curve.G2Jac, error) {
	var res curve.G2Jac
	if len(points) != len(scalars) {
		return res, fmt.Errorf("gpu msm G2: %d points, %d scalars", len(points), len(scalars))
	}
	var out curve.G2Affine
	if len(points) > 0 {
		rc := C.sp1_g16_call_msm(C.uintptr_t(gpuHooks.msmG2), unsafe.Pointer(&out),
			unsafe.Pointer(&points[0]), unsafe.Pointer(&scalars[0]), C.size_t(len(points)))
		if rc != 0 {
			return res, fmt.Errorf("gpu msm G2 failed (%d)", int(rc))
		}
	}
	res.FromAffine(&out)
	return res, nil
}

// gpuComputeH is gnark's computeH (see the C entry point sp1_groth16_compute_h).
func gpuComputeH(a, b, c []fr.Element, domain *fft.Domain) ([]fr.Element, error) {
	n := int(domain.Cardinality)
	if len(a) != len(b) || len(a) != len(c) || len(a) > n {
		return nil, fmt.Errorf("gpu computeH: lengths %d %d %d, domain %d", len(a), len(b), len(c), n)
	}
	lg := uint32(bits.TrailingZeros64(domain.Cardinality))
	var den, one fr.Element
	one.SetOne()
	den.Exp(domain.FrMultiplicativeGen, big.NewInt(int64(domain.Cardinality)))
	den.Sub(&den, &one).Inverse(&den)
	consts := [6]fr.Element{
		domain.Generator, domain.GeneratorInv, domain.CardinalityInv,
		domain.FrMultiplicativeGen, domain.FrMultiplicativeGenInv, den,
	}
	h := make([]fr.Element, n)
	rc := C.sp1_g16_call_h(C.uintptr_t(gpuHooks.computeH), unsafe.Pointer(&h[0]),
		unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&c[0]), C.size_t(len(a)),
		C.uint32_t(lg), unsafe.Pointer(&consts[0]))
	if rc != 0 {
		return nil, fmt.Errorf("gpu computeH failed (%d)", int(rc))
	}
	return h, nil
}

// groth16Circuit is the circuit ProveGroth16 proves: its metadata (wire counts, commitment
// info), and the parsed R1CS, which only gnark's CPU solver needs. Where the GPU R1CS solver is
// armed (gpuR1csSolverArmed) ProveGroth16 reads only the metadata, and the R1CS is parsed
// (loadR1cs) only if a proof falls back to the CPU solver after all.
type groth16Circuit struct {
	dataDir string
	meta    *r1csMeta
	r1cs    *cs.R1CS
	// loadR1cs parses the R1CS (when r1cs is nil).
	loadR1cs func() (*cs.R1CS, error)
}

// fullR1cs is the parsed R1CS, parsed now if it was not, and checked against the metadata.
func (c *groth16Circuit) fullR1cs() (*cs.R1CS, error) {
	if c.r1cs != nil {
		return c.r1cs, nil
	}
	if c.loadR1cs == nil {
		return nil, fmt.Errorf("no R1CS for %s", c.dataDir)
	}
	start := time.Now()
	r1cs, err := c.loadR1cs()
	if err != nil {
		return nil, err
	}
	meta, err := r1csMetaOf(r1cs)
	if err != nil {
		return nil, err
	}
	if !meta.equal(c.meta) {
		return nil, fmt.Errorf("the R1CS read for %s (%+v) is not the one proved with (%+v)", c.dataDir, *meta, *c.meta)
	}
	fmt.Printf("Reading R1CS for the CPU solver took %s\n", time.Since(start))
	c.r1cs = r1cs
	return r1cs, nil
}

// proveGroth16Gpu is gnark's groth16_bn254.Prove with the GPU doing computeH and the MSMs, for
// the circuit in dataDir.
func proveGroth16Gpu(circuit *groth16Circuit, pk *groth16_bn254.ProvingKey, fullWitness witness.Witness, opts ...backend.ProverOption) (*groth16_bn254.Proof, error) {
	dataDir, meta := circuit.dataDir, circuit.meta
	opt, err := backend.NewProverConfig(opts...)
	if err != nil {
		return nil, fmt.Errorf("new prover config: %w", err)
	}
	if opt.HashToFieldFn == nil {
		opt.HashToFieldFn = hash_to_field.New([]byte(constraint.CommitmentDst))
	}

	commitmentInfo := meta.commitments

	proof := &groth16_bn254.Proof{Commitments: make([]curve.G1Affine, len(commitmentInfo))}

	solverOpts := opt.SolverOpts[:len(opt.SolverOpts):len(opt.SolverOpts)]

	privateCommittedValues := make([][]fr.Element, len(commitmentInfo))

	// override hints
	bsb22ID := solver.GetHintID(fcs.Bsb22CommitmentComputePlaceholder)
	solverOpts = append(solverOpts, solver.OverrideHint(bsb22ID, func(_ *big.Int, in []*big.Int, out []*big.Int) error {
		i := int(in[0].Int64())
		in = in[1:]
		privateCommittedValues[i] = make([]fr.Element, len(commitmentInfo[i].PrivateCommitted))
		hashed := in[:len(commitmentInfo[i].PublicAndCommitmentCommitted)]
		committed := in[+len(hashed):]
		for j, inJ := range committed {
			privateCommittedValues[i][j].SetBigInt(inJ)
		}

		var err error
		if proof.Commitments[i], err = pk.CommitmentKeys[i].Commit(privateCommittedValues[i]); err != nil {
			return err
		}

		opt.HashToFieldFn.Write(constraint.SerializeCommitment(proof.Commitments[i].Marshal(), hashed, (fr.Bits-1)/8+1))
		hashBts := opt.HashToFieldFn.Sum(nil)
		opt.HashToFieldFn.Reset()
		nbBuf := fr.Bytes
		if opt.HashToFieldFn.Size() < fr.Bytes {
			nbBuf = opt.HashToFieldFn.Size()
		}
		var res fr.Element
		res.SetBytes(hashBts[:nbBuf])
		res.BigInt(out[0])
		return nil
	}))

	start := time.Now()
	// On the GPU where the host registered its solver, but not with BSB22 commitments, whose hint
	// (above) it does not run.
	var solution *cs.R1CSSolution
	var releaseSolution func()
	if len(commitmentInfo) == 0 {
		solution, releaseSolution = gpuSolveR1cs(dataDir, meta, fullWitness)
	}
	if solution == nil {
		// gnark's solver, on the parsed R1CS: read now if ProveGroth16 read only the metadata.
		r1cs, err := circuit.fullR1cs()
		if err != nil {
			return nil, err
		}
		_solution, err := r1cs.Solve(fullWitness, solverOpts...)
		if err != nil {
			return nil, err
		}
		solution = _solution.(*cs.R1CSSolution)
	}
	fmt.Printf("Solving the constraint system took %s\n", time.Since(start))

	wireValues := []fr.Element(solution.W)

	start = time.Now()
	poks := make([]curve.G1Affine, len(pk.CommitmentKeys))

	for i := range pk.CommitmentKeys {
		var err error
		if poks[i], err = pk.CommitmentKeys[i].ProveKnowledge(privateCommittedValues[i]); err != nil {
			return nil, err
		}
	}
	// compute challenge for folding the PoKs from the commitments
	commitmentsSerialized := make([]byte, fr.Bytes*len(commitmentInfo))
	for i := range commitmentInfo {
		copy(commitmentsSerialized[fr.Bytes*i:], wireValues[commitmentInfo[i].CommitmentIndex].Marshal())
	}
	challenge, err := fr.Hash(commitmentsSerialized, []byte("G16-BSB22"), 1)
	if err != nil {
		return nil, err
	}
	if _, err = proof.CommitmentPok.Fold(poks, challenge[0], ecc.MultiExpConfig{NbTasks: 1}); err != nil {
		return nil, err
	}

	// We need to copy and filter the wireValues for each multi exp, as pk.G1.A, pk.G1.B and
	// pk.G2.B may have (a significant) number of points at infinity. This runs on the CPU while
	// the GPU computes H.
	var wireValuesA, wireValuesB []fr.Element
	chWireValuesA, chWireValuesB := make(chan struct{}, 1), make(chan struct{}, 1)

	go func() {
		wireValuesA = make([]fr.Element, len(wireValues)-int(pk.NbInfinityA))
		for i, j := 0, 0; j < len(wireValuesA); i++ {
			if pk.InfinityA[i] {
				continue
			}
			wireValuesA[j] = wireValues[i]
			j++
		}
		close(chWireValuesA)
	}()
	go func() {
		wireValuesB = make([]fr.Element, len(wireValues)-int(pk.NbInfinityB))
		for i, j := 0, 0; j < len(wireValuesB); i++ {
			if pk.InfinityB[i] {
				continue
			}
			wireValuesB[j] = wireValues[i]
			j++
		}
		close(chWireValuesB)
	}()
	var kWireValues []fr.Element
	chKWireValues := make(chan struct{}, 1)
	go func() {
		toRemove := commitmentInfo.GetPrivateCommitted()
		toRemove = append(toRemove, commitmentInfo.CommitmentIndexes())
		kWireValues = filterSorted(wireValues[meta.nbPublic:], meta.nbPublic, concatAll(toRemove...))
		close(chKWireValues)
	}()

	// H (witness reduction / FFT part)
	tH := time.Now()
	h, err := gpuComputeH(solution.A, solution.B, solution.C, &pk.Domain)
	if err != nil {
		return nil, err
	}
	if checkGpu() {
		cpu := cpuComputeH(solution.A, solution.B, solution.C, &pk.Domain)
		for i := range cpu {
			if !cpu[i].Equal(&h[i]) {
				return nil, fmt.Errorf("gpu computeH differs from the CPU's at %d of %d", i, len(cpu))
			}
		}
		fmt.Printf("GPU computeH matches the CPU\n")
	}
	solution.A = nil
	solution.B = nil
	solution.C = nil
	fmt.Printf("GPU computeH took %s\n", time.Since(tH))

	// sample random r and s
	var r, s big.Int
	var _r, _s, _kr fr.Element
	if _, err := _r.SetRandom(); err != nil {
		return nil, err
	}
	if _, err := _s.SetRandom(); err != nil {
		return nil, err
	}
	_kr.Mul(&_r, &_s).Neg(&_kr)

	_r.BigInt(&r)
	_s.BigInt(&s)

	// computes r[δ], s[δ], kr[δ]
	deltas := curve.BatchScalarMultiplicationG1(&pk.G1.Delta, []fr.Element{_r, _s, _kr})

	// The G2 MSM on the CPU while the GPU does the G1 ones, or on the GPU after them (see
	// g2MsmOnCpu).
	g2OnCpu := g2MsmOnCpu()
	var Bs curve.G2Jac
	chBs := make(chan error, 1)
	if g2OnCpu {
		go func() {
			<-chWireValuesB
			t := time.Now()
			_, err := Bs.MultiExp(pk.G2.B, wireValuesB, ecc.MultiExpConfig{NbTasks: runtime.NumCPU()})
			fmt.Printf("CPU msm B2 (%d) took %s\n", len(wireValuesB), time.Since(t))
			chBs <- err
		}()
	}

	// The G1 MSMs, one after the other on the GPU.
	tMsm := time.Now()
	<-chWireValuesA
	ar, err := checkedMsmG1("A", pk.G1.A, wireValuesA)
	if err != nil {
		return nil, err
	}
	ar.AddMixed(&pk.G1.Alpha)
	ar.AddMixed(&deltas[0])
	proof.Ar.FromJacobian(&ar)

	<-chWireValuesB
	bs1, err := checkedMsmG1("B1", pk.G1.B, wireValuesB)
	if err != nil {
		return nil, err
	}
	bs1.AddMixed(&pk.G1.Beta)
	bs1.AddMixed(&deltas[1])

	sizeH := int(pk.Domain.Cardinality - 1) // comes from the fact the deg(H)=(n-1)+(n-1)-n=n-2
	krs2, err := checkedMsmG1("Z", pk.G1.Z, h[:sizeH])
	if err != nil {
		return nil, err
	}
	<-chKWireValues
	krs, err := checkedMsmG1("K", pk.G1.K, kWireValues)
	if err != nil {
		return nil, err
	}
	krs.AddMixed(&deltas[2])
	krs.AddAssign(&krs2)
	var p1 curve.G1Jac
	p1.ScalarMultiplication(&ar, &s)
	krs.AddAssign(&p1)
	p1.ScalarMultiplication(&bs1, &r)
	krs.AddAssign(&p1)
	proof.Krs.FromJacobian(&krs)
	fmt.Printf("GPU G1 MSMs took %s\n", time.Since(tMsm))

	if g2OnCpu {
		if err := <-chBs; err != nil {
			return nil, err
		}
	} else {
		if Bs, err = checkedMsmG2("B2", pk.G2.B, wireValuesB); err != nil {
			return nil, err
		}
	}
	var deltaS curve.G2Jac
	deltaS.FromAffine(&pk.G2.Delta)
	deltaS.ScalarMultiplication(&deltaS, &s)
	Bs.AddAssign(&deltaS)
	Bs.AddMixed(&pk.G2.Beta)
	proof.Bs.FromJacobian(&Bs)
	fmt.Printf("MSMs took %s\n", time.Since(tMsm))

	fmt.Printf("GPU prover (after solving) took %s\n", time.Since(start))
	// Everything that read the wire values is done (the goroutines above were waited for).
	if releaseSolution != nil {
		releaseSolution()
	}
	return proof, nil
}

// gpuSolveR1cs is r1cs.Solve, for the R1CS meta describes, on the GPU R1CS solver the host
// registered, reading the circuit data SetGpuR1csPrepDir named for dataDir: the wire values, and
// A, B and C, all in gnark's Montgomery form.
// nil, after saying why, where there is no solver or data, or the solve fails; the caller then
// solves with gnark. The solver checks the constraints it does not solve for, and the proof is
// checked against the verifying key before it is used (proveGroth16OnGpu). The solution's
// vectors are kept for the next proof by the func returned with it, once nothing uses them.
func gpuSolveR1cs(dataDir string, meta *r1csMeta, fullWitness witness.Witness) (*cs.R1CSSolution, func()) {
	gpuR1cs.Lock()
	create, solve, destroy := gpuR1cs.create, gpuR1cs.solve, gpuR1cs.destroy
	prepDir := gpuR1cs.prepDirs[dataDir]
	gpuR1cs.Unlock()
	if create == 0 || solve == 0 || destroy == 0 || prepDir == "" ||
		os.Getenv("SP1_GPU_R1CS_SOLVER") == "cpu" {
		return nil, nil
	}
	if os.Getenv("SP1_GPU_R1CS_SOLVER") == "fail" {
		// For testing the fallback: armed (gpuR1csSolverArmed), then failing as a solve can.
		fmt.Printf("GPU R1CS solver: failing as SP1_GPU_R1CS_SOLVER=fail says, solving on the CPU\n")
		return nil, nil
	}
	vec, ok := fullWitness.Vector().(fr.Vector)
	if !ok {
		return nil, nil
	}
	// The wires as gnark's solver starts them: the constant ONE, the witness, then the rest
	// zero, for the solver to fill in.
	nbInputs := meta.nbPublic + meta.nbSecret
	nbWires := meta.nbWires()
	nbConstraints := meta.nbConstraints
	if len(vec) != nbInputs-1 || nbConstraints == 0 {
		fmt.Printf("GPU R1CS solver: witness of %d for %d inputs, solving on the CPU\n", len(vec), nbInputs)
		return nil, nil
	}

	start := time.Now()
	keep := keepGpuR1csSolver()
	key := ""
	if keep {
		key = gpuR1csPrepKey(prepDir)
	}
	gpuR1csKept.Lock()
	defer gpuR1csKept.Unlock()
	h := gpuR1csKept.h
	reused := h != nil
	if h != nil && !gpuR1csKeptFits(key, destroy, nbWires, nbConstraints) {
		// Another circuit's data or sizes, or not kept any more.
		C.sp1_g16_call_r1cs_destroy(C.uintptr_t(gpuR1csKept.destroy), h)
		h, reused = nil, false
	}
	gpuR1csKept.h, gpuR1csKept.key = nil, ""
	if h == nil {
		cDir := C.CString(prepDir)
		defer C.free(unsafe.Pointer(cDir))
		var nW, nC C.uint64_t
		h = C.sp1_g16_call_r1cs_create(C.uintptr_t(create), cDir, &nW, &nC)
		if h == nil {
			fmt.Printf("GPU R1CS solver: no solver for %s, solving on the CPU\n", prepDir)
			return nil, nil
		}
		if int(nW) != nbWires || int(nC) != nbConstraints {
			C.sp1_g16_call_r1cs_destroy(C.uintptr_t(destroy), h)
			fmt.Printf("GPU R1CS solver: %s has %d wires and %d constraints, the circuit %d and %d; solving on the CPU\n",
				prepDir, uint64(nW), uint64(nC), nbWires, nbConstraints)
			return nil, nil
		}
	}
	// Kept for the next proof only once it has solved this one. Otherwise its circuit data and
	// buffers go back as soon as the solve is done, for the MSMs.
	solved := false
	defer func() {
		if solved && keep && key != "" {
			gpuR1csKept.h, gpuR1csKept.destroy, gpuR1csKept.key = h, destroy, key
			gpuR1csKept.nW, gpuR1csKept.nC = nbWires, nbConstraints
		} else {
			C.sp1_g16_call_r1cs_destroy(C.uintptr_t(destroy), h)
		}
	}()
	setup := time.Since(start)

	wires, solution := takeGpuSolveBufs(vec, nbWires, nbConstraints)
	w, a, b, c := solution.W, solution.A, solution.B, solution.C
	release := func() { putGpuSolveBufs(wires, nbInputs, w, a, b, c) }
	rc := C.sp1_g16_call_r1cs_solve(C.uintptr_t(solve), h, unsafe.Pointer(&wires[0]),
		unsafe.Pointer(&solution.W[0]), unsafe.Pointer(&solution.A[0]),
		unsafe.Pointer(&solution.B[0]), unsafe.Pointer(&solution.C[0]))
	if rc != 0 {
		fmt.Printf("GPU R1CS solver failed (%d), solving on the CPU\n", int(rc))
		release()
		return nil, nil
	}
	solved = true
	// The solver gives A, B and C in canonical form.
	toMontgomery(solution.A, solution.B, solution.C)
	if reused {
		fmt.Printf("GPU R1CS solve took %s (setup %s, solver kept from an earlier proof)\n",
			time.Since(start), setup)
	} else {
		fmt.Printf("GPU R1CS solve took %s (setup %s)\n", time.Since(start), setup)
	}
	return solution, release
}

// toMontgomery converts the canonical values in the vectors to gnark's Montgomery form, in
// place, on all cores: x·(R² mod q)·R⁻¹ = x·R.
func toMontgomery(vectors ...fr.Vector) {
	// The element whose value is R = 2^256 mod q holds R² mod q.
	var rSquare fr.Element
	rSquare.SetBigInt(new(big.Int).Lsh(big.NewInt(1), 256))
	nbTasks := runtime.NumCPU()
	var wg sync.WaitGroup
	for _, v := range vectors {
		chunk := (len(v) + nbTasks - 1) / nbTasks
		for lo := 0; lo < len(v); lo += chunk {
			hi := min(lo+chunk, len(v))
			wg.Add(1)
			go func(part []fr.Element) {
				defer wg.Done()
				for i := range part {
					part[i].Mul(&part[i], &rSquare)
				}
			}(v[lo:hi])
		}
	}
	wg.Wait()
}

// concatAll concatenates slices (gnark's internal.ConcatAll).
func concatAll(slices ...[]int) []int {
	n := 0
	for _, s := range slices {
		n += len(s)
	}
	res := make([]int, 0, n)
	for _, s := range slices {
		res = append(res, s...)
	}
	return res
}

// filterSorted returns slice without the indexes in toRemove, the first value in slice being at
// index sliceFirstIndex (gnark's filterHeap). It modifies toRemove.
func filterSorted(slice []fr.Element, sliceFirstIndex int, toRemove []int) []fr.Element {
	if len(toRemove) == 0 {
		return slice
	}
	sort.Ints(toRemove)
	r := make([]fr.Element, 0, len(slice))
	k := 0
	for i := 0; i < len(slice); i++ {
		if k < len(toRemove) && i+sliceFirstIndex == toRemove[k] {
			for k < len(toRemove) && i+sliceFirstIndex == toRemove[k] {
				k++
			}
			continue
		}
		r = append(r, slice[i])
	}
	return r
}

// proveGroth16OnGpu proves with proveGroth16Gpu (with the circuit and key ProveGroth16 read under
// its lock) and verifies the proof against the verifying key in the circuit's data directory;
// nil if either fails (the caller then proves on the CPU, with the parsed R1CS).
func proveGroth16OnGpu(
	circuit *groth16Circuit,
	pkIface groth16.ProvingKey,
	fullWitness witness.Witness,
) groth16.Proof {
	dataDir := circuit.dataDir
	pk, ok := pkIface.(*groth16_bn254.ProvingKey)
	if !ok {
		return nil
	}
	proof, err := proveGroth16Gpu(circuit, pk, fullWitness)
	if err != nil {
		fmt.Printf("GPU Groth16 prover failed, proving on the CPU: %v\n", err)
		return nil
	}
	start := time.Now()
	vkFile, err := os.Open(dataDir + "/" + groth16VkPath)
	if err != nil {
		fmt.Printf("GPU Groth16 proof not checked (%v), proving on the CPU\n", err)
		return nil
	}
	defer vkFile.Close()
	vk := groth16.NewVerifyingKey(ecc.BN254)
	if _, err := vk.ReadFrom(vkFile); err != nil {
		fmt.Printf("GPU Groth16 proof not checked (%v), proving on the CPU\n", err)
		return nil
	}
	publicWitness, err := fullWitness.Public()
	if err != nil {
		fmt.Printf("GPU Groth16 proof not checked (%v), proving on the CPU\n", err)
		return nil
	}
	if err := groth16.Verify(proof, vk, publicWitness); err != nil {
		fmt.Printf("GPU Groth16 proof does not verify (%v), proving on the CPU\n", err)
		// The next proof creates the GPU R1CS solver anew rather than trust the one kept.
		ReleaseGpuR1csSolver()
		return nil
	}
	fmt.Printf("Verifying the GPU proof took %s\n", time.Since(start))
	return proof
}
