package sp1

import (
	"testing"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

// The GPU R1CS solve's buffers are reused across proofs, and the initial wires are ONE, the
// witness, then zeros every time, also after a longer witness.
func TestGpuSolveBufsReused(t *testing.T) {
	const nbWires, nbConstraints = 64, 80
	witness := func(n int, v uint64) fr.Vector {
		vec := make(fr.Vector, n)
		for i := range vec {
			vec[i].SetUint64(v + uint64(i))
		}
		return vec
	}
	check := func(wires []fr.Element, vec fr.Vector) {
		t.Helper()
		var one fr.Element
		one.SetOne()
		if len(wires) != nbWires || !wires[0].Equal(&one) {
			t.Fatalf("wires[0] is not ONE")
		}
		for i := 1; i < nbWires; i++ {
			want := fr.Element{}
			if i <= len(vec) {
				want = vec[i-1]
			}
			if !wires[i].Equal(&want) {
				t.Fatalf("wires[%d] = %s, want %s", i, wires[i].String(), want.String())
			}
		}
	}

	vec := witness(10, 7)
	wires, s := takeGpuSolveBufs(vec, nbWires, nbConstraints)
	check(wires, vec)
	if len(s.W) != nbWires || len(s.A) != nbConstraints || len(s.B) != nbConstraints || len(s.C) != nbConstraints {
		t.Fatalf("solution sizes %d %d %d %d", len(s.W), len(s.A), len(s.B), len(s.C))
	}
	s.W[nbWires-1].SetUint64(99)
	putGpuSolveBufs(wires, len(vec)+1, s.W, s.A, s.B, s.C)

	vec2 := witness(10, 1000)
	wires2, s2 := takeGpuSolveBufs(vec2, nbWires, nbConstraints)
	check(wires2, vec2)
	if &wires2[0] != &wires[0] || &s2.W[0] != &s.W[0] || &s2.A[0] != &s.A[0] ||
		&s2.B[0] != &s.B[0] || &s2.C[0] != &s.C[0] {
		t.Fatalf("buffers were not reused")
	}
	// Taken buffers are not handed out twice.
	wires3, s3 := takeGpuSolveBufs(vec2, nbWires, nbConstraints)
	if &wires3[0] == &wires2[0] || &s3.W[0] == &s2.W[0] {
		t.Fatalf("buffers handed out twice")
	}
	putGpuSolveBufs(wires2, len(vec2)+1, s2.W, s2.A, s2.B, s2.C)

	// A different number of inputs gets fresh initial wires, with zeros after the witness.
	vec4 := witness(5, 3)
	wires4, _ := takeGpuSolveBufs(vec4, nbWires, nbConstraints)
	check(wires4, vec4)
	if &wires4[0] == &wires2[0] {
		t.Fatalf("initial wires reused across a different number of inputs")
	}
}
