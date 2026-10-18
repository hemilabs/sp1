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
*/
import "C"

import (
	"fmt"
	"math/big"
	"math/bits"
	"os"
	"runtime"
	"sort"
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
}

// SetGpuHooks registers the GPU's computeH and MSM entry points (C function pointers; 0 unsets).
func SetGpuHooks(computeH, msmG1, msmG2 uintptr) {
	gpuHooks.computeH = computeH
	gpuHooks.msmG1 = msmG1
	gpuHooks.msmG2 = msmG2
}

// useGpuProver: hooks registered, and not turned off with SP1_GROTH16_GPU=0.
func useGpuProver() bool {
	if gpuHooks.computeH == 0 || gpuHooks.msmG1 == 0 || gpuHooks.msmG2 == 0 {
		return false
	}
	return os.Getenv("SP1_GROTH16_GPU") != "0"
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

// proveGroth16Gpu is gnark's groth16_bn254.Prove with the GPU doing computeH and the MSMs.
func proveGroth16Gpu(r1cs *cs.R1CS, pk *groth16_bn254.ProvingKey, fullWitness witness.Witness, opts ...backend.ProverOption) (*groth16_bn254.Proof, error) {
	opt, err := backend.NewProverConfig(opts...)
	if err != nil {
		return nil, fmt.Errorf("new prover config: %w", err)
	}
	if opt.HashToFieldFn == nil {
		opt.HashToFieldFn = hash_to_field.New([]byte(constraint.CommitmentDst))
	}

	commitmentInfo := r1cs.CommitmentInfo.(constraint.Groth16Commitments)

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
	_solution, err := r1cs.Solve(fullWitness, solverOpts...)
	if err != nil {
		return nil, err
	}
	fmt.Printf("Solving the constraint system took %s\n", time.Since(start))

	solution := _solution.(*cs.R1CSSolution)
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
		kWireValues = filterSorted(wireValues[r1cs.GetNbPublicVariables():], r1cs.GetNbPublicVariables(), concatAll(toRemove...))
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

	// The G2 MSM on the CPU while the GPU does the G1 ones: about as long as those on a Radeon
	// RX 9070 XT or an Arc Pro B70 (~3 s each), where the G2 MSM on the GPU after them takes
	// ~2.7 s more. SP1_GROTH16_G2_ON_CPU=0 runs it on the GPU.
	g2OnCpu := os.Getenv("SP1_GROTH16_G2_ON_CPU") != "0"
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
	return proof, nil
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

// proveGroth16OnGpu proves with proveGroth16Gpu (with the R1CS and key ProveGroth16 read under
// its lock) and verifies the proof against the verifying key in
// dataDir; nil if either fails (the caller then proves on the CPU).
func proveGroth16OnGpu(
	dataDir string,
	r1csIface constraint.ConstraintSystem,
	pkIface groth16.ProvingKey,
	fullWitness witness.Witness,
) groth16.Proof {
	r1cs, ok := r1csIface.(*cs.R1CS)
	if !ok {
		return nil
	}
	pk, ok := pkIface.(*groth16_bn254.ProvingKey)
	if !ok {
		return nil
	}
	proof, err := proveGroth16Gpu(r1cs, pk, fullWitness)
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
		return nil
	}
	fmt.Printf("Verifying the GPU proof took %s\n", time.Since(start))
	return proof
}
