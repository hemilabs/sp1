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
