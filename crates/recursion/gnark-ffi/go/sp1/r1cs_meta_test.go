package sp1

import (
	"bufio"
	"os"
	"path/filepath"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	groth16_bn254 "github.com/consensys/gnark/backend/groth16/bn254"
	"github.com/consensys/gnark/constraint"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	"github.com/consensys/gnark/frontend/cs/scs"
)

// metaTestCircuit: x³ + x + 5 = y with public y, a few secrets, and with commit a BSB22
// commitment to some of them.
type metaTestCircuit struct {
	X, A, B frontend.Variable
	Y       frontend.Variable `gnark:",public"`
	Z       frontend.Variable `gnark:",public"`
	commit  bool
}

func (c *metaTestCircuit) Define(api frontend.API) error {
	x3 := api.Mul(c.X, c.X, c.X)
	api.AssertIsEqual(c.Y, api.Add(x3, c.X, 5))
	api.AssertIsEqual(c.Z, api.Mul(c.A, c.B))
	if c.commit {
		committer, ok := api.(frontend.Committer)
		if !ok {
			panic("no committer")
		}
		cm, err := committer.Commit(c.A, c.B)
		if err != nil {
			return err
		}
		api.AssertIsDifferent(cm, 0)
	}
	return nil
}

func compileMetaTestCircuit(t *testing.T, commit bool) constraint.ConstraintSystem {
	t.Helper()
	ccs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &metaTestCircuit{commit: commit})
	if err != nil {
		t.Fatal(err)
	}
	return ccs
}

func writeR1csFile(t *testing.T, path string, ccs constraint.ConstraintSystem) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	bw := bufio.NewWriter(f)
	if _, err := ccs.WriteTo(bw); err != nil {
		t.Fatal(err)
	}
	if err := bw.Flush(); err != nil {
		t.Fatal(err)
	}
}

// readR1csMeta reads from the file what the fully parsed R1CS says, with and without BSB22
// commitments, from the full and the stripped circuit (writeStrippedR1cs).
func TestReadR1csMetaMatchesFullParse(t *testing.T) {
	for _, commit := range []bool{false, true} {
		ccs := compileMetaTestCircuit(t, commit)
		dir := t.TempDir()
		full := filepath.Join(dir, "full.bin")
		writeR1csFile(t, full, ccs)
		parsed, err := readR1cs(full)
		if err != nil {
			t.Fatal(err)
		}
		want, err := r1csMetaOf(parsed)
		if err != nil {
			t.Fatal(err)
		}
		compiled, err := r1csMetaOf(ccs)
		if err != nil {
			t.Fatal(err)
		}
		if !want.equal(compiled) {
			t.Fatalf("parsed %+v, compiled %+v", *want, *compiled)
		}
		if commit && len(want.commitments) != 1 {
			t.Fatalf("expected 1 commitment, got %d", len(want.commitments))
		}
		if !commit && len(want.commitments) != 0 {
			t.Fatalf("expected no commitment, got %d", len(want.commitments))
		}
		if want.nbPublic != 3 || want.nbSecret != 3 {
			t.Fatalf("public %d, secret %d", want.nbPublic, want.nbSecret)
		}

		stripped := filepath.Join(dir, "stripped.bin")
		writeStrippedR1cs(parsed, stripped)
		for _, path := range []string{full, stripped} {
			got, err := readR1csMeta(path)
			if err != nil {
				t.Fatalf("commit %v, %s: %v", commit, path, err)
			}
			if !got.equal(want) {
				t.Fatalf("commit %v, %s: metadata %+v, the parsed R1CS %+v", commit, path, *got, *want)
			}
			if commit {
				g, w := got.commitments[0], want.commitments[0]
				if g.CommitmentIndex != w.CommitmentIndex || len(g.PrivateCommitted) != len(w.PrivateCommitted) {
					t.Fatalf("commitment %+v, want %+v", g, w)
				}
			}
		}
	}
}

// A file that is not a whole BN254 R1CS is an error, never a panic or wrong counts.
func TestReadR1csMetaRejectsBadFiles(t *testing.T) {
	dir := t.TempDir()
	full := filepath.Join(dir, "full.bin")
	writeR1csFile(t, full, compileMetaTestCircuit(t, false))
	data, err := os.ReadFile(full)
	if err != nil {
		t.Fatal(err)
	}
	cases := map[string][]byte{
		"empty":       {},
		"header only": data[:40],
		"truncated":   data[:len(data)-1],
		"extended":    append(append([]byte{}, data...), 0),
	}
	// The body's first byte (the CBOR map's head) changed to an array's.
	body := append([]byte{}, data...)
	var off uint64 = 64
	for i := 0; i < 3; i++ {
		var l uint64
		for k := 7; k >= 0; k-- {
			l = l<<8 | uint64(data[32+8*i+k])
		}
		off += l
	}
	body[off] = 0x80 | (body[off] & 31)
	cases["not a map"] = body
	for name, bad := range cases {
		path := filepath.Join(dir, "bad.bin")
		if err := os.WriteFile(path, bad, 0o644); err != nil {
			t.Fatal(err)
		}
		if meta, err := readR1csMeta(path); err == nil {
			t.Errorf("%s: no error, metadata %+v", name, *meta)
		}
	}
	// A PLONK circuit.
	sccs, err := frontend.Compile(ecc.BN254.ScalarField(), scs.NewBuilder, &metaTestCircuit{})
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "scs.bin")
	writeR1csFile(t, path, sccs)
	if meta, err := readR1csMeta(path); err == nil {
		t.Errorf("SCS: no error, metadata %+v", *meta)
	}
	if _, err := readR1csMeta(filepath.Join(dir, "missing.bin")); err == nil {
		t.Error("missing file: no error")
	}
}

// The real circuit, when GROTH16_DATA_DIR names it: its metadata against its stripped copy
// parsed (GROTH16_STRIPPED_R1CS; the full one takes ~19 s to parse).
func TestReadR1csMetaRealCircuit(t *testing.T) {
	dataDir, stripped := os.Getenv("GROTH16_DATA_DIR"), os.Getenv("GROTH16_STRIPPED_R1CS")
	if dataDir == "" || stripped == "" {
		t.Skip("set GROTH16_DATA_DIR and GROTH16_STRIPPED_R1CS to run this test")
	}
	got, err := readR1csMeta(filepath.Join(dataDir, groth16CircuitPath))
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := readR1cs(stripped)
	if err != nil {
		t.Fatal(err)
	}
	want, err := r1csMetaOf(parsed)
	if err != nil {
		t.Fatal(err)
	}
	if !got.equal(want) {
		t.Fatalf("metadata %+v, the parsed stripped R1CS %+v", *got, *want)
	}
	t.Logf("%+v (%d wires)", *got, got.nbWires())
}

// setFakeGpuSolver registers non-zero GPU hooks and solver entry points (never called by the
// tests that use it) and a prep directory with all the solver's files for dataDir.
func setFakeGpuSolver(t *testing.T, dataDir string) string {
	t.Helper()
	savedHooks := gpuHooks
	gpuR1cs.Lock()
	savedCreate, savedSolve, savedDestroy, savedDirs := gpuR1cs.create, gpuR1cs.solve, gpuR1cs.destroy, gpuR1cs.prepDirs
	gpuR1cs.create, gpuR1cs.solve, gpuR1cs.destroy, gpuR1cs.prepDirs = 1, 2, 3, nil
	gpuR1cs.Unlock()
	t.Cleanup(func() {
		gpuHooks = savedHooks
		gpuR1cs.Lock()
		gpuR1cs.create, gpuR1cs.solve, gpuR1cs.destroy, gpuR1cs.prepDirs = savedCreate, savedSolve, savedDestroy, savedDirs
		gpuR1cs.Unlock()
	})
	gpuHooks.computeH, gpuHooks.msmG1, gpuHooks.msmG2 = 1, 2, 3
	prepDir := t.TempDir()
	for _, name := range []string{"circuit_meta.txt", "coeffs.bin", "layers_descs.bin",
		"layers_terms.bin", "layers.idx", "hints.idx", "layers_hints.bin", "hint_in_les.bin",
		"desc_decl_idx.bin"} {
		if err := os.WriteFile(filepath.Join(prepDir, name), []byte("data"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	SetGpuR1csPrepDir(dataDir, prepDir)
	for _, env := range []string{"SP1_GROTH16_GPU", "SP1_GPU_R1CS_SOLVER", "SP1_GROTH16_FULL_R1CS"} {
		t.Setenv(env, "")
		os.Unsetenv(env)
	}
	return prepDir
}

// ProveGroth16 skips the R1CS parse only where the GPU R1CS solver will be tried.
func TestGpuR1csSolverArmed(t *testing.T) {
	dataDir := "/data/groth16"
	prepDir := setFakeGpuSolver(t, dataDir)
	if !gpuR1csSolverArmed(dataDir) {
		t.Fatal("not armed with the solver and its data")
	}
	if gpuR1csSolverArmed("/data/other") {
		t.Fatal("armed for a circuit without solver data")
	}
	for env, want := range map[string]map[string]bool{
		"SP1_GROTH16_GPU":       {"0": false, "1": true},
		"SP1_GPU_R1CS_SOLVER":   {"cpu": false, "fail": true, "gpu": true},
		"SP1_GROTH16_FULL_R1CS": {"1": false, "0": true},
	} {
		for value, armed := range want {
			t.Setenv(env, value)
			if got := gpuR1csSolverArmed(dataDir); got != armed {
				t.Errorf("%s=%s: armed %v, want %v", env, value, got, armed)
			}
			os.Unsetenv(env)
		}
	}
	// A solver file missing.
	if err := os.Remove(filepath.Join(prepDir, "layers_terms.bin")); err != nil {
		t.Fatal(err)
	}
	if gpuR1csSolverArmed(dataDir) {
		t.Fatal("armed with a solver file missing")
	}
	if err := os.WriteFile(filepath.Join(prepDir, "layers_terms.bin"), nil, 0o644); err != nil {
		t.Fatal(err)
	}
	// No GPU prover, or no solver.
	gpuHooks.msmG2 = 0
	if gpuR1csSolverArmed(dataDir) {
		t.Fatal("armed without the GPU prover")
	}
	gpuHooks.msmG2 = 3
	SetGpuR1csPrepDir(dataDir, "")
	if gpuR1csSolverArmed(dataDir) {
		t.Fatal("armed after the prep directory was unset")
	}
	SetGpuR1csPrepDir(dataDir, prepDir)
	gpuR1cs.Lock()
	gpuR1cs.solve = 0
	gpuR1cs.Unlock()
	if gpuR1csSolverArmed(dataDir) {
		t.Fatal("armed without a registered solver")
	}
}

func TestNeedFullR1cs(t *testing.T) {
	meta := &r1csMeta{nbPublic: 2, nbSecret: 3, nbInternal: 5, nbConstraints: 7}
	pk := &groth16_bn254.ProvingKey{InfinityA: make([]bool, 10), InfinityB: make([]bool, 10)}
	if full, why := needFullR1cs(true, meta, nil, pk); full {
		t.Fatalf("full R1CS with the solver armed and fitting metadata: %s", why)
	}
	withCommitment := *meta
	withCommitment.commitments = constraint.Groth16Commitments{{CommitmentIndex: 4}}
	short := &groth16_bn254.ProvingKey{InfinityA: make([]bool, 9), InfinityB: make([]bool, 9)}
	for name, c := range map[string]struct {
		armed bool
		meta  *r1csMeta
		err   error
		pk    groth16.ProvingKey
	}{
		"not armed":           {false, meta, nil, pk},
		"no metadata":         {true, nil, os.ErrNotExist, pk},
		"commitments":         {true, &withCommitment, nil, pk},
		"another proving key": {true, meta, nil, short},
		"another curve's key": {true, meta, nil, groth16.NewProvingKey(ecc.BLS12_381)},
	} {
		if full, _ := needFullR1cs(c.armed, c.meta, c.err, c.pk); !full {
			t.Errorf("%s: proves without the full R1CS", name)
		}
	}
}

// writeGroth16TestData writes the small circuit's R1CS, proving key and verifying key in the
// layout ProveGroth16 reads.
func writeGroth16TestData(t *testing.T, ccs constraint.ConstraintSystem) string {
	t.Helper()
	dir := t.TempDir()
	writeR1csFile(t, filepath.Join(dir, groth16CircuitPath), ccs)
	pk, vk, err := groth16.Setup(ccs)
	if err != nil {
		t.Fatal(err)
	}
	pkFile, err := os.Create(filepath.Join(dir, groth16PkPath))
	if err != nil {
		t.Fatal(err)
	}
	if err := pk.WriteDump(pkFile); err != nil {
		t.Fatal(err)
	}
	pkFile.Close()
	vkFile, err := os.Create(filepath.Join(dir, groth16VkPath))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := vk.WriteTo(vkFile); err != nil {
		t.Fatal(err)
	}
	vkFile.Close()
	return dir
}

// With the GPU solver armed, loadGroth16Circuit reads only the metadata, and the CPU fallback
// (after a GPU solve, prover or check that failed) parses the R1CS then and proves correctly;
// not armed, or with commitments, it parses the R1CS up front. Errors are returned and not
// cached.
func TestLoadGroth16CircuitFallback(t *testing.T) {
	releaseProverCache()
	t.Cleanup(releaseProverCache)
	for _, commit := range []bool{false, true} {
		ccs := compileMetaTestCircuit(t, commit)
		dataDir := writeGroth16TestData(t, ccs)
		for _, armed := range []bool{true, false} {
			releaseProverCache()
			circuit, pk, err := loadGroth16Circuit(dataDir, armed)
			if err != nil {
				t.Fatal(err)
			}
			lazy := armed && !commit
			if (circuit.r1cs == nil) != lazy {
				t.Fatalf("commit %v armed %v: R1CS parsed %v", commit, armed, circuit.r1cs != nil)
			}
			want, _ := r1csMetaOf(ccs)
			if !circuit.meta.equal(want) {
				t.Fatalf("metadata %+v, want %+v", *circuit.meta, *want)
			}
			// The next proof uses the cache.
			again, _, err := loadGroth16Circuit(dataDir, armed)
			if err != nil || !again.meta.equal(circuit.meta) || again.r1cs != circuit.r1cs || (lazy && again.meta != circuit.meta) {
				t.Fatalf("commit %v armed %v: not cached (%v)", commit, armed, err)
			}

			// The CPU fallback, as ProveGroth16 runs it.
			r1cs, err := circuit.fullR1cs()
			if err != nil {
				t.Fatal(err)
			}
			assignment := &metaTestCircuit{X: 3, Y: 35, A: 4, B: 5, Z: 20}
			w, err := frontend.NewWitness(assignment, ecc.BN254.ScalarField())
			if err != nil {
				t.Fatal(err)
			}
			proof, err := groth16.Prove(r1cs, pk, w)
			if err != nil {
				t.Fatal(err)
			}
			vk := groth16.NewVerifyingKey(ecc.BN254)
			vkFile, err := os.Open(filepath.Join(dataDir, groth16VkPath))
			if err != nil {
				t.Fatal(err)
			}
			if _, err := vk.ReadFrom(vkFile); err != nil {
				t.Fatal(err)
			}
			vkFile.Close()
			pub, _ := w.Public()
			if err := groth16.Verify(proof, vk, pub); err != nil {
				t.Fatalf("commit %v armed %v: %v", commit, armed, err)
			}
			// Parsed once, then cached for the next proof's fallback.
			if r2, err := loadGroth16R1cs(dataDir); err != nil || r2 != r1cs {
				t.Fatalf("R1CS not cached after the fallback (%v)", err)
			}
		}
	}

	// The lazily parsed R1CS must be the one the metadata describes.
	dataDir := writeGroth16TestData(t, compileMetaTestCircuit(t, false))
	releaseProverCache()
	circuit, _, err := loadGroth16Circuit(dataDir, true)
	if err != nil {
		t.Fatal(err)
	}
	other := *circuit.meta
	other.nbInternal++
	circuit.meta = &other
	if _, err := circuit.fullR1cs(); err == nil {
		t.Fatal("an R1CS that does not match the metadata was used")
	}

	// Unreadable files are errors, and nothing is cached from them.
	releaseProverCache()
	if err := os.WriteFile(filepath.Join(dataDir, groth16PkPath), []byte("bad"), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, armed := range []bool{true, false} {
		if _, _, err := loadGroth16Circuit(dataDir, armed); err == nil {
			t.Fatalf("armed %v: no error for a corrupt proving key", armed)
		}
		if globalPk != nil {
			t.Fatal("a corrupt proving key was cached")
		}
	}
	releaseProverCache()
	empty := t.TempDir()
	if _, _, err := loadGroth16Circuit(empty, false); err == nil {
		t.Fatal("no error without the files")
	}
	if globalR1cs != nil || globalPk != nil {
		t.Fatal("cached something from a missing directory")
	}
}

// SP1_GPU_R1CS_SOLVER=fail: the solver stays armed and the solve fails, for testing the
// fallback on a GPU host; the solver's entry points are not called.
func TestGpuSolveR1csForcedFailure(t *testing.T) {
	dataDir := "/data/groth16"
	setFakeGpuSolver(t, dataDir)
	t.Setenv("SP1_GPU_R1CS_SOLVER", "fail")
	if !gpuR1csSolverArmed(dataDir) {
		t.Fatal("not armed")
	}
	w, err := frontend.NewWitness(&metaTestCircuit{X: 3, Y: 35, A: 4, B: 5, Z: 20}, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatal(err)
	}
	meta := &r1csMeta{nbPublic: 3, nbSecret: 3, nbInternal: 4, nbConstraints: 4}
	if solution, _ := gpuSolveR1cs(dataDir, meta, w); solution != nil {
		t.Fatal("solved")
	}
}
