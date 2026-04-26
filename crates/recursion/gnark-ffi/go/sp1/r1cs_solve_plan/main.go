// r1cs_solve_plan: Phase 1 of the GPU R1CS solver implementation plan.
//
// Three subcommands:
//
//   emit       <build_dir> <out_solve_plan.bin>
//     Walk the production R1CS, emit a flat binary "solve plan" the
//     GPU dispatcher will eventually consume. The plan stores raw R1Cs
//     and hint instructions in declaration order plus a coefficient
//     table. No layering / dedup yet — that's Phase 2 work.
//
//   interpret  <build_dir> <solve_plan.bin> <witness.json> <out_wire_values.bin>
//     Load the plan + a witness JSON, walk the instruction stream,
//     and produce a wire vector — same job as gnark's Solve, but
//     driven entirely by the binary plan. This is the CPU reference
//     the GPU kernel will be diffed against.
//
//   roundtrip  <build_dir> <witness.json>
//     Convenience: emit, interpret, then run gnark Solve and assert
//     wire vectors match byte-for-byte. The Phase 1 success gate.
//
// The solve plan format is intentionally minimal — see "Solve plan
// format v1" below for the spec.
//
// Usage:  go run ./sp1/r1cs_solve_plan emit       <build_dir> <out>
//         go run ./sp1/r1cs_solve_plan interpret  <build_dir> <plan> <witness> <out>
//         go run ./sp1/r1cs_solve_plan roundtrip  <build_dir> <witness>
package main

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math/big"
	"os"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/constraint"
	cs "github.com/consensys/gnark/constraint/bn254"
	"github.com/consensys/gnark/constraint/solver"
	"github.com/consensys/gnark/frontend"

	sp1 "github.com/succinctlabs/sp1-recursion-gnark/sp1"
)

// ---------------- Solve plan format ----------------
//
// Two versions are supported:
//   v1 (Phase 1): minimal CPU-friendly. R1Cs stored as raw L/R/O term
//                 lists; loc determined at interpret time.
//   v2 (Phase 2): adds per-R1C GPU-friendly metadata (loc, out_wire,
//                 out_coeff_idx) determined by a dry-run at emit time.
//                 The GPU kernel reads loc directly without scanning.
//
// Header (28 bytes, little-endian throughout):
//   magic[4]            "SPS1"
//   version[u32]        1 or 2
//   nbPublic[u32]       includes the ONE wire
//   nbSecret[u32]
//   nbInternal[u32]
//   nbCoefficients[u32]
//   nbInstructions[u32] total number of records that follow
//
// Coefficient table: nbCoefficients × 32 bytes
//   Each entry is the canonical Mongtomery representation of a fr.Element
//   (fr.Element is exactly [4]uint64 / 32 bytes; we serialize the bytes
//   verbatim — same byte pattern gnark uses internally).
//
// Instruction stream:
//   for each instruction:
//     kind[u8]  0 = R1C, 1 = HINT
//
//     R1C (v1):
//       L_count[u32], then L_count × (CID[u32], VID[u32])
//       R_count[u32], then R_count × (CID[u32], VID[u32])
//       O_count[u32], then O_count × (CID[u32], VID[u32])
//
//     R1C (v2): same as v1 followed by:
//       loc[u8]            0 = verify-only, 1 = L, 2 = R, 3 = O
//       out_coeff_idx[u32] coeff of the unset term (for div-by-coeff);
//                          unused if loc == 0
//       out_wire_id[u32]   wire to compute; unused if loc == 0
//
//     HINT (both versions identical):
//       hintID[u32]
//       nInputs[u32]
//       per input: count[u32], then count × (CID[u32], VID[u32])
//       outStart[u32]
//       outEnd[u32]
//
// Wire IDs and coeff IDs reference the existing gnark namespaces:
//   - Wire 0       = ONE wire (constant 1).
//   - Wires [1, 1+nbPublic+nbSecret) = witness inputs (public then secret).
//   - Wires [1+nbPublic+nbSecret, ...) = internal, computed by the solver.
//   - CoeffIDs index into the coefficient table emitted in the header.

const (
	planMagic   = "SPS1"
	planV1      = 1
	planV2      = 2
	kindR1C     = 0
	kindHint    = 1
)

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	switch os.Args[1] {
	case "emit":
		if len(os.Args) != 4 {
			usage()
		}
		emit(os.Args[2], os.Args[3], planV1)
	case "emit-v2":
		if len(os.Args) != 4 {
			usage()
		}
		emit(os.Args[2], os.Args[3], planV2)
	case "emit-layers":
		if len(os.Args) != 4 {
			usage()
		}
		emitLayers(os.Args[2], os.Args[3])
	case "interpret":
		if len(os.Args) != 6 {
			usage()
		}
		interpret(os.Args[2], os.Args[3], os.Args[4], os.Args[5])
	case "roundtrip":
		if len(os.Args) != 4 {
			usage()
		}
		roundtrip(os.Args[2], os.Args[3])
	case "roundtrip-v2":
		if len(os.Args) != 4 {
			usage()
		}
		roundtripV2(os.Args[2], os.Args[3])
	default:
		usage()
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "Usage:")
	fmt.Fprintln(os.Stderr, "  r1cs_solve_plan emit          <build_dir> <out_solve_plan.bin>")
	fmt.Fprintln(os.Stderr, "  r1cs_solve_plan emit-v2       <build_dir> <out_solve_plan.bin>")
	fmt.Fprintln(os.Stderr, "  r1cs_solve_plan emit-layers   <build_dir> <out_layers.bin>")
	fmt.Fprintln(os.Stderr, "  r1cs_solve_plan interpret     <build_dir> <solve_plan.bin> <witness.json> <out_wire_values.bin>")
	fmt.Fprintln(os.Stderr, "  r1cs_solve_plan roundtrip     <build_dir> <witness.json>")
	fmt.Fprintln(os.Stderr, "  r1cs_solve_plan roundtrip-v2  <build_dir> <witness.json>")
	os.Exit(1)
}

// ---------------- emit ----------------

func loadR1CS(buildDir string) *cs.R1CS {
	os.Setenv("CONSTRAINTS_JSON", buildDir+"/constraints.json")
	os.Setenv("GROTH16", "1")

	t0 := time.Now()
	r1csObj := groth16.NewCS(ecc.BN254)
	f, err := os.Open(buildDir + "/groth16_circuit.bin")
	if err != nil {
		fail("open R1CS: %v", err)
	}
	if _, err := r1csObj.ReadFrom(bufio.NewReaderSize(f, 1<<20)); err != nil {
		fail("read R1CS: %v", err)
	}
	f.Close()
	r := r1csObj.(*cs.R1CS)
	fmt.Fprintf(os.Stderr, "[plan] R1CS loaded in %s (constraints=%d wires=%d)\n",
		time.Since(t0), r.GetNbConstraints(),
		r.NbInternalVariables+r.GetNbPublicVariables()+r.GetNbSecretVariables())
	return r
}

func emit(buildDir, outPath string, version uint32) {
	r := loadR1CS(buildDir)

	out, err := os.Create(outPath)
	if err != nil {
		fail("create output: %v", err)
	}
	w := bufio.NewWriterSize(out, 1<<20)
	defer func() {
		if err := w.Flush(); err != nil {
			fail("flush: %v", err)
		}
		out.Close()
	}()

	// Locate BlueprintGenericHint (only blueprint kind we expect besides R1C).
	hintBpID, found := findHintBlueprint(r)

	// Header
	must(w.Write([]byte(planMagic)))
	mustU32(w, version)
	mustU32(w, uint32(r.GetNbPublicVariables()))
	mustU32(w, uint32(r.GetNbSecretVariables()))
	mustU32(w, uint32(r.NbInternalVariables))
	mustU32(w, uint32(len(r.Coefficients)))

	nbInstr := r.GetNbInstructions()
	mustU32(w, uint32(nbInstr))

	// Coefficient table.
	for i := range r.Coefficients {
		var buf [32]byte
		for limb := 0; limb < 4; limb++ {
			binary.LittleEndian.PutUint64(buf[limb*8:], r.Coefficients[i][limb])
		}
		must(w.Write(buf[:]))
	}

	// For v2, dry-run the solver to determine per-constraint loc +
	// out_wire metadata. This walks the same instruction stream, marks
	// hint outputs as "defined", and for each R1C records which of its
	// L/R/O linear expressions contains the unset wire (if any).
	var nbWires int
	var solved []bool
	if version >= planV2 {
		nbWires = r.NbInternalVariables + r.GetNbPublicVariables() + r.GetNbSecretVariables()
		solved = make([]bool, nbWires)
		// Wire 0 (ONE) + public + secret are pre-defined.
		nbInputs := r.GetNbPublicVariables() + r.GetNbSecretVariables()
		for i := 0; i < nbInputs; i++ {
			solved[i] = true
		}
	}

	t0 := time.Now()
	hintBP, _ := r.Blueprints[hintBpID].(constraint.BlueprintHint)
	var hm constraint.HintMapping

	emittedR1C, emittedHint := 0, 0
	verifyOnly := 0
	locL, locR, locO := 0, 0, 0
	for i := 0; i < nbInstr; i++ {
		pi := r.Instructions[i]
		inst := pi.Unpack(&r.System)
		bp := r.Blueprints[pi.BlueprintID]

		if found && pi.BlueprintID == hintBpID {
			hm.Inputs = hm.Inputs[:0]
			hintBP.DecompressHint(&hm, inst)
			must1(w.WriteByte(kindHint))
			mustU32(w, uint32(hm.HintID))
			mustU32(w, uint32(len(hm.Inputs)))
			for _, le := range hm.Inputs {
				mustU32(w, uint32(len(le)))
				for _, t := range le {
					mustU32(w, uint32(t.CoeffID()))
					mustU32(w, uint32(t.WireID()))
				}
			}
			mustU32(w, hm.OutputRange.Start)
			mustU32(w, hm.OutputRange.End)
			if solved != nil {
				for w := hm.OutputRange.Start; w < hm.OutputRange.End; w++ {
					solved[w] = true
				}
			}
			emittedHint++
			continue
		}

		r1c, ok := bp.(constraint.BlueprintR1C)
		if !ok {
			fail("instruction %d uses unexpected blueprint %T (id=%d)", i, bp, pi.BlueprintID)
		}
		var c constraint.R1C
		r1c.DecompressR1C(&c, inst)
		must1(w.WriteByte(kindR1C))
		writeLE(w, c.L)
		writeLE(w, c.R)
		writeLE(w, c.O)
		emittedR1C++

		if solved == nil {
			continue
		}

		// v2: scan L/R/O for the (at most one) unset wire and emit loc + out info.
		var (
			loc          uint8 = 0
			outCoeffIdx  uint32
			outWireID    uint32
		)
		findUnset := func(le constraint.LinearExpression, candidate uint8) {
			for _, t := range le {
				vid := uint32(t.WireID())
				if t.IsConstant() || int(vid) >= len(solved) {
					continue
				}
				if !solved[vid] {
					if loc != 0 {
						fail("R1C #%d has multiple unsolved wires (vid=%d after loc=%d wire=%d)",
							emittedR1C, vid, loc, outWireID)
					}
					loc = candidate
					outCoeffIdx = uint32(t.CoeffID())
					outWireID = vid
				}
			}
		}
		findUnset(c.L, 1)
		findUnset(c.R, 2)
		findUnset(c.O, 3)

		must1(w.WriteByte(loc))
		mustU32(w, outCoeffIdx)
		mustU32(w, outWireID)

		if loc != 0 {
			solved[outWireID] = true
		}

		switch loc {
		case 0:
			verifyOnly++
		case 1:
			locL++
		case 2:
			locR++
		case 3:
			locO++
		}
	}
	if version >= planV2 {
		fmt.Fprintf(os.Stderr, "[plan] v2 loc distribution: L=%d R=%d O=%d verify-only=%d\n",
			locL, locR, locO, verifyOnly)
	}
	fmt.Fprintf(os.Stderr, "[plan] emitted %d R1Cs + %d hints in %s (version=%d)\n",
		emittedR1C, emittedHint, time.Since(t0), version)
}

func writeLE(w io.Writer, le constraint.LinearExpression) {
	mustU32(w, uint32(len(le)))
	for _, t := range le {
		mustU32(w, uint32(t.CoeffID()))
		mustU32(w, uint32(t.WireID()))
	}
}

// ---------------- emit-layers ----------------
//
// Sidecar file giving each instruction a topological layer ID so the
// Rust loader can group instructions by layer for the GPU dispatcher.
//
// File format:
//   magic[4]       "LAYR"
//   version[u32]   1
//   nbInstr[u32]   matches the v2 plan's nbInstructions
//   nbLayers[u32]  max(layerID) + 1
//   layerID[u32 × nbInstr]
//
// Computation:
//   - Wire 0..nbInputs-1 have depth 0 (witness inputs).
//   - For each instruction in declaration order:
//     - depth = max(input wire depths) + 1
//     - assign output wires that depth
//   - layerID = depth (so layer 0 = constraints/hints whose only inputs
//     are witness wires; layer N = furthest from inputs).

func emitLayers(buildDir, outPath string) {
	r := loadR1CS(buildDir)

	nbWires := r.NbInternalVariables + r.GetNbPublicVariables() + r.GetNbSecretVariables()
	wireDepth := make([]int32, nbWires)
	for i := range wireDepth {
		wireDepth[i] = -1
	}
	nbInputs := r.GetNbPublicVariables() + r.GetNbSecretVariables()
	for i := 0; i < nbInputs; i++ {
		wireDepth[i] = 0
	}

	hintBpID, found := findHintBlueprint(r)
	hintBP, _ := r.Blueprints[hintBpID].(constraint.BlueprintHint)
	var hm constraint.HintMapping

	nbInstr := r.GetNbInstructions()
	layerID := make([]uint32, nbInstr)
	maxDepth := int32(0)

	t0 := time.Now()
	for i := 0; i < nbInstr; i++ {
		pi := r.Instructions[i]
		inst := pi.Unpack(&r.System)
		var d int32 = 0

		processVID := func(vid uint32) {
			if int(vid) >= len(wireDepth) || wireDepth[vid] < 0 {
				return
			}
			if wireDepth[vid] > d {
				d = wireDepth[vid]
			}
		}

		if found && pi.BlueprintID == hintBpID {
			hm.Inputs = hm.Inputs[:0]
			hintBP.DecompressHint(&hm, inst)
			for _, le := range hm.Inputs {
				for _, t := range le {
					if !t.IsConstant() {
						processVID(uint32(t.WireID()))
					}
				}
			}
			thisDepth := d + 1
			for w := hm.OutputRange.Start; w < hm.OutputRange.End; w++ {
				if int(w) < len(wireDepth) {
					wireDepth[w] = thisDepth
				}
			}
			layerID[i] = uint32(thisDepth)
			if thisDepth > maxDepth {
				maxDepth = thisDepth
			}
			continue
		}

		bp := r.Blueprints[pi.BlueprintID]
		r1c, ok := bp.(constraint.BlueprintR1C)
		if !ok {
			fail("inst %d: unexpected blueprint %T", i, bp)
		}
		var c constraint.R1C
		r1c.DecompressR1C(&c, inst)
		var newWire int32 = -1
		processLE := func(le constraint.LinearExpression, allowNew bool) {
			for _, t := range le {
				vid := int32(t.WireID())
				if vid < 0 || t.IsConstant() || int(vid) >= len(wireDepth) {
					continue
				}
				if wireDepth[vid] < 0 {
					if allowNew {
						newWire = vid
					}
					continue
				}
				if wireDepth[vid] > d {
					d = wireDepth[vid]
				}
			}
		}
		processLE(c.L, false)
		processLE(c.R, false)
		processLE(c.O, true)
		thisDepth := d + 1
		if newWire >= 0 {
			wireDepth[newWire] = thisDepth
		}
		layerID[i] = uint32(thisDepth)
		if thisDepth > maxDepth {
			maxDepth = thisDepth
		}
	}

	out, err := os.Create(outPath)
	if err != nil {
		fail("create layers out: %v", err)
	}
	w := bufio.NewWriterSize(out, 1<<20)
	must(w.Write([]byte("LAYR")))
	mustU32(w, 1)
	mustU32(w, uint32(nbInstr))
	mustU32(w, uint32(maxDepth+1))
	for _, l := range layerID {
		mustU32(w, l)
	}
	w.Flush()
	out.Close()

	// Histogram
	width := make([]int, maxDepth+1)
	for _, l := range layerID {
		width[l]++
	}
	wide := 0
	wideWork := 0
	for _, n := range width {
		if n >= 10000 {
			wide++
			wideWork += n
		}
	}
	fmt.Fprintf(os.Stderr, "[plan] emitted %d layer-IDs in %s; nbLayers=%d max-width=%d wide-layers(>=10k)=%d covering %d/%d insts (%.1f%%)\n",
		nbInstr, time.Since(t0), maxDepth+1, max(width), wide, wideWork, nbInstr,
		100*float64(wideWork)/float64(nbInstr))
}

func max(xs []int) int {
	m := 0
	for _, x := range xs {
		if x > m {
			m = x
		}
	}
	return m
}

// ---------------- interpret ----------------

func interpret(buildDir, planPath, witnessPath, outWirePath string) {
	r := loadR1CS(buildDir) // for hint registry + coefficient parity check
	wires := interpretPlan(r, planPath, witnessPath)

	// Write wire vector as raw bytes (4 × u64 LE per element, same layout
	// as the coefficient table).
	out, err := os.Create(outWirePath)
	if err != nil {
		fail("create wire-values out: %v", err)
	}
	w := bufio.NewWriterSize(out, 1<<20)
	for i := range wires {
		var buf [32]byte
		for limb := 0; limb < 4; limb++ {
			binary.LittleEndian.PutUint64(buf[limb*8:], wires[i][limb])
		}
		if _, err := w.Write(buf[:]); err != nil {
			fail("write wire-values: %v", err)
		}
	}
	w.Flush()
	out.Close()
	fmt.Fprintf(os.Stderr, "[plan] wrote %d wire values to %s\n", len(wires), outWirePath)
}

// loadPlanHeader reads the header + coefficient table.
type planHeader struct {
	version        uint32
	nbPublic       uint32
	nbSecret       uint32
	nbInternal     uint32
	nbCoefficients uint32
	nbInstructions uint32
	coeffs         []fr.Element // canonical Montgomery limbs
}

func readPlanHeader(planPath string) (*planHeader, *bufio.Reader, *os.File) {
	pf, err := os.Open(planPath)
	if err != nil {
		fail("open plan: %v", err)
	}
	r := bufio.NewReaderSize(pf, 1<<20)

	magic := make([]byte, 4)
	if _, err := io.ReadFull(r, magic); err != nil {
		fail("read magic: %v", err)
	}
	if string(magic) != planMagic {
		fail("bad plan magic %q", magic)
	}
	version := readU32(r)
	if version != planV1 && version != planV2 {
		fail("unsupported plan version %d", version)
	}
	h := &planHeader{
		version:        version,
		nbPublic:       readU32(r),
		nbSecret:       readU32(r),
		nbInternal:     readU32(r),
		nbCoefficients: readU32(r),
		nbInstructions: readU32(r),
	}
	h.coeffs = make([]fr.Element, h.nbCoefficients)
	var buf [32]byte
	for i := range h.coeffs {
		if _, err := io.ReadFull(r, buf[:]); err != nil {
			fail("read coeff[%d]: %v", i, err)
		}
		for limb := 0; limb < 4; limb++ {
			h.coeffs[i][limb] = binary.LittleEndian.Uint64(buf[limb*8:])
		}
	}
	return h, r, pf
}

func interpretPlan(r1cs *cs.R1CS, planPath, witnessPath string) []fr.Element {
	h, br, pf := readPlanHeader(planPath)
	defer pf.Close()

	nbWires := int(h.nbPublic + h.nbSecret + h.nbInternal)
	values := make([]fr.Element, nbWires)
	solved := make([]bool, nbWires)

	// Wire 0 = ONE
	values[0].SetOne()
	solved[0] = true

	// Load witness inputs into wires [1, nbPublic + nbSecret).
	witnessInputs := loadWitnessAsFrVector(witnessPath)
	expectedSize := int(h.nbPublic-1) + int(h.nbSecret)
	if len(witnessInputs) != expectedSize {
		fail("witness size mismatch: got %d, expected %d", len(witnessInputs), expectedSize)
	}
	for i := range witnessInputs {
		values[i+1] = witnessInputs[i]
		solved[i+1] = true
	}

	// Walk instructions.
	t0 := time.Now()

	// Constants for the gnark "fast path" coefficient IDs.
	// CoeffIdZero=0, CoeffIdOne=1, CoeffIdTwo=2, CoeffIdMinusOne=3
	type term struct{ cid, vid uint32 }
	readTerm := func() term {
		return term{cid: readU32(br), vid: readU32(br)}
	}

	// Evaluate sum(coeff*value) over a flat term list, treating any
	// term whose wire is unset as the "to be solved" term — record it.
	var unset term
	var unsetLoc uint8 // 1=L, 2=R, 3=O
	evalLE := func(loc uint8) (fr.Element, bool) {
		var acc fr.Element
		n := readU32(br)
		hasUnset := false
		for i := uint32(0); i < n; i++ {
			t := readTerm()
			if !solved[t.vid] {
				if hasUnset || unsetLoc != 0 {
					fail("more than one unsolved wire in same R1C: vid=%d (also %d)", t.vid, unset.vid)
				}
				hasUnset = true
				unset = t
				unsetLoc = loc
				continue
			}
			accumulate(&acc, &h.coeffs[t.cid], t.vid, values)
		}
		return acc, hasUnset
	}

	for instIdx := uint32(0); instIdx < h.nbInstructions; instIdx++ {
		kind, err := br.ReadByte()
		if err != nil {
			fail("read kind: %v", err)
		}
		switch kind {
		case kindR1C:
			unsetLoc = 0
			a, _ := evalLE(1)
			b, _ := evalLE(2)
			c, _ := evalLE(3)

			// v2 records loc + out info after the LRO terms; for the
			// interpreter we recompute live, so cross-check against the
			// recorded values when present (cheap regression check).
			if h.version >= planV2 {
				recLoc, err := br.ReadByte()
				if err != nil {
					fail("read v2 loc: %v", err)
				}
				recCoeff := readU32(br)
				recWire := readU32(br)
				if recLoc != unsetLoc {
					fail("R1C %d: v2 loc=%d but interpret loc=%d", instIdx, recLoc, unsetLoc)
				}
				if unsetLoc != 0 {
					if recCoeff != unset.cid || recWire != unset.vid {
						fail("R1C %d: v2 (cid=%d vid=%d) != interpret (cid=%d vid=%d)",
							instIdx, recCoeff, recWire, unset.cid, unset.vid)
					}
				}
			}

			if unsetLoc == 0 {
				// All wires solved — verify a*b == c.
				var check fr.Element
				if !check.Mul(&a, &b).Equal(&c) {
					fail("R1C %d unsatisfied: %s * %s != %s",
						instIdx, a.String(), b.String(), c.String())
				}
				continue
			}

			// Solve for the unset wire. Mirrors gnark's solveR1C.
			var wire fr.Element
			switch unsetLoc {
			case 1:
				if !b.IsZero() {
					wire.Div(&c, &b).Sub(&wire, &a)
				}
			case 2:
				if !a.IsZero() {
					wire.Div(&c, &a).Sub(&wire, &b)
				}
			case 3:
				wire.Mul(&a, &b).Sub(&wire, &c)
			}

			// Divide out the term's coefficient (gnark stores value, not term value).
			divByCoeff(&wire, unset.cid, h.coeffs)
			values[unset.vid] = wire
			solved[unset.vid] = true

		case kindHint:
			hintID := solver.HintID(readU32(br))
			nIn := readU32(br)
			inputs := make([]*big.Int, nIn)
			for i := uint32(0); i < nIn; i++ {
				cnt := readU32(br)
				var acc fr.Element
				for j := uint32(0); j < cnt; j++ {
					t := readTerm()
					if !solved[t.vid] {
						fail("hint %d input %d references unsolved wire %d", instIdx, i, t.vid)
					}
					accumulate(&acc, &h.coeffs[t.cid], t.vid, values)
				}
				inputs[i] = new(big.Int)
				acc.BigInt(inputs[i])
			}
			outStart := readU32(br)
			outEnd := readU32(br)
			nOut := int(outEnd - outStart)

			fn := solver.GetRegisteredHint(hintID)
			if fn == nil {
				fail("hint %d (id=%d) is not registered", instIdx, hintID)
			}
			outputs := make([]*big.Int, nOut)
			for i := range outputs {
				outputs[i] = new(big.Int)
			}
			if err := fn(fr.Modulus(), inputs, outputs); err != nil {
				fail("hint %d (id=%d) failed: %v", instIdx, hintID, err)
			}
			for i, o := range outputs {
				w := outStart + uint32(i)
				values[w].SetBigInt(o)
				solved[w] = true
			}

		default:
			fail("instruction %d: unknown kind %d", instIdx, kind)
		}
	}
	fmt.Fprintf(os.Stderr, "[plan] interpreted %d instructions in %s\n",
		h.nbInstructions, time.Since(t0))
	return values
}

// accumulate: r += coeff * values[vid], using the gnark fast paths for
// the common coefficient IDs.
func accumulate(r *fr.Element, coeff *fr.Element, vid uint32, values []fr.Element) {
	switch *coeff {
	// We can't easily switch on coeff IDs here since we passed the value;
	// fall through to the general path. The fast paths are an optimization
	// only — gnark uses them for kernel speed but the general path is
	// always correct.
	}
	var tmp fr.Element
	tmp.Mul(coeff, &values[vid])
	r.Add(r, &tmp)
}

func divByCoeff(wire *fr.Element, cid uint32, coeffs []fr.Element) {
	switch cid {
	case 0: // CoeffIdZero
		// shouldn't happen — division by zero
	case 1: // CoeffIdOne
		// nop
	case 2: // CoeffIdTwo
		var two fr.Element
		two.SetUint64(2)
		var inv fr.Element
		inv.Inverse(&two)
		wire.Mul(wire, &inv)
	case 3: // CoeffIdMinusOne
		wire.Neg(wire)
	default:
		var inv fr.Element
		inv.Inverse(&coeffs[cid])
		wire.Mul(wire, &inv)
	}
}

// ---------------- roundtrip ----------------

func roundtrip(buildDir, witnessPath string) {
	roundtripVersion(buildDir, witnessPath, planV1)
}

func roundtripV2(buildDir, witnessPath string) {
	roundtripVersion(buildDir, witnessPath, planV2)
}

func roundtripVersion(buildDir, witnessPath string, version uint32) {
	tmp, err := os.CreateTemp("", "solve_plan-*.bin")
	if err != nil {
		fail("temp: %v", err)
	}
	tmp.Close()
	defer os.Remove(tmp.Name())

	emit(buildDir, tmp.Name(), version)

	r := loadR1CS(buildDir)

	t0 := time.Now()
	mine := interpretPlan(r, tmp.Name(), witnessPath)
	tInterp := time.Since(t0)

	t0 = time.Now()
	ref := gnarkSolveReference(r, witnessPath)
	tRef := time.Since(t0)

	fmt.Fprintf(os.Stderr, "[plan] interpret: %s   gnark Solve: %s\n", tInterp, tRef)

	if len(mine) != len(ref) {
		fail("wire count mismatch: mine=%d, gnark=%d", len(mine), len(ref))
	}

	mismatches := 0
	firstMismatch := -1
	for i := range mine {
		if !mine[i].Equal(&ref[i]) {
			if firstMismatch == -1 {
				firstMismatch = i
			}
			mismatches++
		}
	}
	if mismatches > 0 {
		fmt.Fprintf(os.Stderr, "[plan] FAIL — %d wire mismatches; first at %d:\n", mismatches, firstMismatch)
		fmt.Fprintf(os.Stderr, "  mine[%d] = %s\n", firstMismatch, mine[firstMismatch].String())
		fmt.Fprintf(os.Stderr, "  gnrk[%d] = %s\n", firstMismatch, ref[firstMismatch].String())
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "[plan] PASS — %d wires match byte-for-byte\n", len(mine))
}

func gnarkSolveReference(r *cs.R1CS, witnessPath string) []fr.Element {
	witnessFile, err := os.ReadFile(witnessPath)
	if err != nil {
		fail("read witness: %v", err)
	}
	var witnessInput sp1.WitnessInput
	if err := json.Unmarshal(witnessFile, &witnessInput); err != nil {
		fail("unmarshal witness: %v", err)
	}
	assignment := sp1.NewCircuit(witnessInput)
	witness, err := frontend.NewWitness(&assignment, ecc.BN254.ScalarField())
	if err != nil {
		fail("new witness: %v", err)
	}
	sol, err := r.Solve(witness)
	if err != nil {
		fail("gnark solve: %v", err)
	}
	return sol.(*cs.R1CSSolution).W
}

// loadWitnessAsFrVector takes the production witness JSON (sp1.WitnessInput)
// and returns the wire values for [1, 1+nbPublic+nbSecret) — i.e. the
// flattened public + secret inputs in the order gnark expects.
func loadWitnessAsFrVector(witnessPath string) []fr.Element {
	witnessFile, err := os.ReadFile(witnessPath)
	if err != nil {
		fail("read witness: %v", err)
	}
	var witnessInput sp1.WitnessInput
	if err := json.Unmarshal(witnessFile, &witnessInput); err != nil {
		fail("unmarshal witness: %v", err)
	}
	assignment := sp1.NewCircuit(witnessInput)
	w, err := frontend.NewWitness(&assignment, ecc.BN254.ScalarField())
	if err != nil {
		fail("new witness: %v", err)
	}
	v := w.Vector()
	vec, ok := v.(fr.Vector)
	if !ok {
		fail("witness vector type %T", v)
	}
	out := make([]fr.Element, len(vec))
	copy(out, vec)
	return out
}

// ---------------- low-level helpers ----------------

func findHintBlueprint(r *cs.R1CS) (constraint.BlueprintID, bool) {
	for i, b := range r.Blueprints {
		if _, ok := b.(*constraint.BlueprintGenericHint); ok {
			return constraint.BlueprintID(i), true
		}
	}
	return 0, false
}

func mustU32(w io.Writer, v uint32) {
	var buf [4]byte
	binary.LittleEndian.PutUint32(buf[:], v)
	if _, err := w.Write(buf[:]); err != nil {
		fail("write u32: %v", err)
	}
}

func readU32(r io.Reader) uint32 {
	var buf [4]byte
	if _, err := io.ReadFull(r, buf[:]); err != nil {
		fail("read u32: %v", err)
	}
	return binary.LittleEndian.Uint32(buf[:])
}

func must1(err error) {
	if err != nil {
		fail("write byte: %v", err)
	}
}

func must(_ int, err error) {
	if err != nil {
		fail("write: %v", err)
	}
}

func fail(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "[plan] "+format+"\n", args...)
	os.Exit(2)
}
