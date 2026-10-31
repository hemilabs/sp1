package sp1

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/frontend"
)

// The fast witness equals gnark's on the installed circuit's sample witness
// (SP1_GROTH16_WITNESS_JSON overrides the path).
func TestNewWitnessMatchesGnark(t *testing.T) {
	path := os.Getenv("SP1_GROTH16_WITNESS_JSON")
	if path == "" {
		home, _ := os.UserHomeDir()
		matches, _ := filepath.Glob(filepath.Join(home, ".sp1/circuits/groth16/*/groth16_witness.json"))
		if len(matches) == 0 {
			t.Skip("no groth16_witness.json installed")
		}
		path = matches[len(matches)-1]
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var in WitnessInput
	if err := json.Unmarshal(data, &in); err != nil {
		t.Fatal(err)
	}
	t0 := time.Now()
	assignment := NewCircuit(in)
	want, err := frontend.NewWitness(&assignment, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatal(err)
	}
	t1 := time.Now()
	got, err := newWitness(in)
	if err != nil {
		t.Fatal(err)
	}
	t2 := time.Now()
	t.Logf("frontend.NewWitness %s, newWitness %s", t1.Sub(t0), t2.Sub(t1))
	wb, _ := want.MarshalBinary()
	gb, _ := got.MarshalBinary()
	if !bytes.Equal(wb, gb) {
		t.Fatalf("witnesses differ (%d vs %d bytes)", len(wb), len(gb))
	}
	wp, _ := want.Public()
	gp, _ := got.Public()
	wpb, _ := wp.MarshalBinary()
	gpb, _ := gp.MarshalBinary()
	if !bytes.Equal(wpb, gpb) {
		t.Fatal("public witnesses differ")
	}
}
