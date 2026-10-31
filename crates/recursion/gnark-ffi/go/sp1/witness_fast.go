package sp1

import (
	"fmt"
	"os"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/witness"
	"github.com/consensys/gnark/frontend"
)

// newWitness returns the full witness that frontend.NewWitness(&NewCircuit(in), BN254) builds,
// without its reflection: that walks the Circuit struct three times and hands every value over
// an unbuffered channel, which took 0.69-0.93 s for the ~65 K values of the v6.1.0 circuit on a
// 13-vCPU MI300X host ("Generating witness took", every Groth16 proof).
//
// The order is the one gnark's schema walk gives: the public fields of Circuit in declaration
// order (VkeyHash, CommittedValuesDigest, ExitCode, VkRoot, ProofNonce), then the secret leaves
// in declaration order: Vars, then each Felt's Value, then each Ext's four Values.
// (koalabear.Variable's UpperBound is a *big.Int, not a frontend.Variable, so it is no leaf.)
// TestNewWitnessMatchesGnark checks this against frontend.NewWitness.
//
// SP1_GNARK_REFLECT_WITNESS=1 uses frontend.NewWitness instead.
func newWitness(in WitnessInput) (witness.Witness, error) {
	if os.Getenv("SP1_GNARK_REFLECT_WITNESS") == "1" {
		assignment := NewCircuit(in)
		return frontend.NewWitness(&assignment, ecc.BN254.ScalarField())
	}
	public := []string{in.VkeyHash, in.CommittedValuesDigest, in.ExitCode, in.VkRoot, in.ProofNonce}
	nbSecret := len(in.Vars) + len(in.Felts) + 4*len(in.Exts)
	values := make(chan any, len(public)+nbSecret)
	for _, v := range public {
		values <- v
	}
	for _, v := range in.Vars {
		values <- v
	}
	for _, v := range in.Felts {
		values <- v
	}
	for i, e := range in.Exts {
		if len(e) < 4 {
			return nil, fmt.Errorf("ext %d has %d limbs, expected 4", i, len(e))
		}
		for j := 0; j < 4; j++ {
			values <- e[j]
		}
	}
	close(values)
	w, err := witness.New(ecc.BN254.ScalarField())
	if err != nil {
		return nil, err
	}
	if err := w.Fill(len(public), nbSecret, values); err != nil {
		return nil, err
	}
	return w, nil
}
