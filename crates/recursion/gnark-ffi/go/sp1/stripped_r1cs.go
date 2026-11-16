package sp1

import "sync"

// strippedR1cs names, per data directory, a copy of its Groth16 R1CS without the debug information
// the prover does not use (SetGroth16StrippedR1cs), which ProveGroth16 reads instead of the full
// one: ~2 s against ~9 s for the SP1 circuit on a B70 host.
var strippedR1cs sync.Map

// SetGroth16StrippedR1cs names the stripped copy of the Groth16 R1CS in dataDir ("" for none).
func SetGroth16StrippedR1cs(dataDir, path string) {
	if path == "" {
		strippedR1cs.Delete(dataDir)
		return
	}
	strippedR1cs.Store(dataDir, path)
}

// strippedR1csPath is the stripped copy SetGroth16StrippedR1cs named for dataDir, or "".
func strippedR1csPath(dataDir string) string {
	if path, ok := strippedR1cs.Load(dataDir); ok {
		return path.(string)
	}
	return ""
}
