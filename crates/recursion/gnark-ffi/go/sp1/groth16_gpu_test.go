package sp1

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// The kept GPU R1CS solver is read again when its circuit data changes: another directory, a
// file rebuilt in place, or a file missing (then it is not kept at all).
func TestGpuR1csPrepKey(t *testing.T) {
	dir := t.TempDir()
	names := []string{"circuit_meta.txt", "coeffs.bin", "layers_descs.bin", "layers_terms.bin",
		"layers.idx", "hints.idx", "layers_hints.bin", "hint_in_les.bin", "desc_decl_idx.bin"}
	for _, name := range names[1:] {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("data"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if key := gpuR1csPrepKey(dir); key != "" {
		t.Fatalf("key %q without circuit_meta.txt", key)
	}
	meta := filepath.Join(dir, "circuit_meta.txt")
	if err := os.WriteFile(meta, []byte("n_wires=1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	key := gpuR1csPrepKey(dir)
	if key == "" || key != gpuR1csPrepKey(dir) {
		t.Fatalf("key %q is not stable", key)
	}
	if other := gpuR1csPrepKey(dir + "/."); other == key {
		t.Fatalf("same key for another directory name")
	}
	// Rebuilt in place: another size, then the same size with a later modification time.
	if err := os.WriteFile(meta, []byte("n_wires=22\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	resized := gpuR1csPrepKey(dir)
	if resized == key {
		t.Fatalf("same key after circuit_meta.txt changed size")
	}
	later := time.Now().Add(time.Hour)
	if err := os.Chtimes(meta, later, later); err != nil {
		t.Fatal(err)
	}
	if gpuR1csPrepKey(dir) == resized {
		t.Fatalf("same key after circuit_meta.txt was rewritten")
	}
}

func TestKeepGpuR1csSolver(t *testing.T) {
	saved := gpuHooks.totalMemory
	defer func() { gpuHooks.totalMemory = saved }()
	t.Setenv("SP1_GPU_R1CS_KEEP", "")
	os.Unsetenv("SP1_GPU_R1CS_KEEP")
	gpuHooks.totalMemory = 192 << 30
	if !keepGpuR1csSolver() {
		t.Fatal("not kept on a 192 GiB card")
	}
	gpuHooks.totalMemory = 16 << 30
	if keepGpuR1csSolver() {
		t.Fatal("kept on a 16 GiB card")
	}
	t.Setenv("SP1_GPU_R1CS_KEEP", "1")
	if !keepGpuR1csSolver() {
		t.Fatal("SP1_GPU_R1CS_KEEP=1 does not keep it")
	}
	gpuHooks.totalMemory = 192 << 30
	t.Setenv("SP1_GPU_R1CS_KEEP", "0")
	if keepGpuR1csSolver() {
		t.Fatal("SP1_GPU_R1CS_KEEP=0 keeps it")
	}
}

// The kept GPU R1CS solver is used again only for the same circuit data, solver and sizes: one
// kept for another circuit, also one with the same circuit data key (the data directory proved
// before), would write its sizes' wires and constraints into vectors sized for this one.
func TestGpuR1csKeptFits(t *testing.T) {
	gpuR1csKept.Lock()
	defer gpuR1csKept.Unlock()
	h, destroy, key, nW, nC := gpuR1csKept.h, gpuR1csKept.destroy, gpuR1csKept.key,
		gpuR1csKept.nW, gpuR1csKept.nC
	defer func() {
		gpuR1csKept.h, gpuR1csKept.destroy, gpuR1csKept.key = h, destroy, key
		gpuR1csKept.nW, gpuR1csKept.nC = nW, nC
	}()
	gpuR1csKept.destroy, gpuR1csKept.key, gpuR1csKept.nW, gpuR1csKept.nC = 7, "prep", 100, 90
	if !gpuR1csKeptFits("prep", 7, 100, 90) {
		t.Fatal("not used again for the same circuit")
	}
	for _, c := range []struct {
		name          string
		key           string
		destroy       uintptr
		nbWires, nbCs int
	}{
		{"fewer wires", "prep", 7, 99, 90},
		{"more wires", "prep", 7, 101, 90},
		{"fewer constraints", "prep", 7, 100, 89},
		{"more constraints", "prep", 7, 100, 91},
		{"other circuit data", "other", 7, 100, 90},
		{"not kept", "", 7, 100, 90},
		{"other solver", "prep", 8, 100, 90},
	} {
		if gpuR1csKeptFits(c.key, c.destroy, c.nbWires, c.nbCs) {
			t.Fatalf("used again with %s", c.name)
		}
	}
}

// The GPU R1CS solver's circuit data is per data directory: a proof of a circuit whose directory
// has none named solves on the CPU, whatever another circuit's proofs named.
func TestGpuR1csPrepDirPerCircuit(t *testing.T) {
	SetGpuR1csPrepDir("/circuits/a", "/prep/a")
	SetGpuR1csPrepDir("/circuits/b", "/prep/b")
	defer SetGpuR1csPrepDir("/circuits/a", "")
	defer SetGpuR1csPrepDir("/circuits/b", "")
	get := func(dataDir string) string {
		gpuR1cs.Lock()
		defer gpuR1cs.Unlock()
		return gpuR1cs.prepDirs[dataDir]
	}
	if get("/circuits/a") != "/prep/a" || get("/circuits/b") != "/prep/b" {
		t.Fatal("prep dirs are not kept per data directory")
	}
	if get("/circuits/c") != "" {
		t.Fatal("a circuit with no prep dir named has one")
	}
	SetGpuR1csPrepDir("/circuits/a", "")
	if get("/circuits/a") != "" || get("/circuits/b") != "/prep/b" {
		t.Fatal("clearing one circuit's prep dir changed another's")
	}
}
