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
	case "prep-full":
		if len(os.Args) != 5 {
			usage()
		}
		prepFull(os.Args[2], os.Args[3], os.Args[4])
	case "prep-layer":
		if len(os.Args) < 5 || len(os.Args) > 6 {
			usage()
		}
		layerID := int32(-1) // -1 = pick widest
		if len(os.Args) == 6 {
			var x int
			fmt.Sscanf(os.Args[5], "%d", &x)
			layerID = int32(x)
		}
		prepLayer(os.Args[2], os.Args[3], os.Args[4], layerID)
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
	fmt.Fprintln(os.Stderr, "  r1cs_solve_plan prep-layer    <build_dir> <witness.json> <out_dir> [layer_id|widest]")
	fmt.Fprintln(os.Stderr, "  r1cs_solve_plan prep-full     <build_dir> <witness.json> <out_dir>")
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

// ---------------- prep-full ----------------
//
// Dumps everything the standalone HIP driver needs to perform a full
// layered solve (Phase 4 prototype):
//
//   coeffs.bin              all coefficients
//   wires_initial.bin       wires at start: ONE + witness + all hint outputs
//                           (R1C-defined wires are zero)
//   wires_expected.bin      wires after solve (Phase 1 interpreter output)
//   layers.idx              per-layer (n_descs, descs_off, terms_off)
//   layers_descs.bin        all per-layer R1C descriptors concatenated
//   layers_terms.bin        all per-layer terms concatenated
//   meta.txt                summary
//
// The C++ driver loops over layers: read the (descs_off, n_descs) for
// layer i, launch the eval_constraints kernel, advance.

func prepFull(buildDir, witnessPath, outDir string) {
	r := loadR1CS(buildDir)
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		fail("mkdir out: %v", err)
	}

	t0 := time.Now()
	tmp, _ := os.CreateTemp("", "solve_plan-*.bin")
	tmp.Close()
	defer os.Remove(tmp.Name())
	emit(buildDir, tmp.Name(), planV2)
	wires := interpretPlan(r, tmp.Name(), witnessPath)
	fmt.Fprintf(os.Stderr, "[plan] reference solve in %s\n", time.Since(t0))

	nbWires := r.NbInternalVariables + r.GetNbPublicVariables() + r.GetNbSecretVariables()
	wireDepth := make([]int32, nbWires)
	for i := range wireDepth {
		wireDepth[i] = -1
	}
	nbInputs := r.GetNbPublicVariables() + r.GetNbSecretVariables()
	for i := 0; i < nbInputs; i++ {
		wireDepth[i] = 0
	}
	hintBpID, hintFound := findHintBlueprint(r)
	hintBP, _ := r.Blueprints[hintBpID].(constraint.BlueprintHint)
	var hm constraint.HintMapping

	type r1cMeta struct {
		layer int32
		L, R, O constraint.LinearExpression
		loc uint8
		outCoeffID uint32
		outWireID  uint32
	}

	nbInstr := r.GetNbInstructions()
	allR1Cs := make([]*r1cMeta, 0, nbInstr)
	hintOutputWires := make(map[uint32]bool) // wires defined by hints
	r1cOutputWires := make(map[uint32]bool)  // wires defined by R1Cs (loc != 0)

	solved := make([]bool, nbWires)
	for i := 0; i < nbInputs; i++ {
		solved[i] = true
	}
	maxLayer := int32(0)

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
		if hintFound && pi.BlueprintID == hintBpID {
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
					solved[w] = true
					hintOutputWires[w] = true
				}
			}
			if thisDepth > maxLayer {
				maxLayer = thisDepth
			}
			continue
		}
		bp := r.Blueprints[pi.BlueprintID]
		r1c, _ := bp.(constraint.BlueprintR1C)
		var c constraint.R1C
		r1c.DecompressR1C(&c, inst)
		var newWire int32 = -1
		var loc uint8 = 0
		var outCoeff uint32 = 0
		processLE := func(le constraint.LinearExpression, locCandidate uint8) {
			for _, t := range le {
				vid := int32(t.WireID())
				if vid < 0 || t.IsConstant() || int(vid) >= len(wireDepth) {
					continue
				}
				if !solved[vid] {
					if loc != 0 {
						continue
					}
					loc = locCandidate
					newWire = vid
					outCoeff = uint32(t.CoeffID())
				} else if wireDepth[vid] > d {
					d = wireDepth[vid]
				}
			}
		}
		processLE(c.L, 1)
		processLE(c.R, 2)
		processLE(c.O, 3)
		thisDepth := d + 1
		if newWire >= 0 {
			wireDepth[newWire] = thisDepth
			solved[newWire] = true
			r1cOutputWires[uint32(newWire)] = true
		}
		if thisDepth > maxLayer {
			maxLayer = thisDepth
		}
		m := &r1cMeta{
			layer:      thisDepth,
			L:          append(constraint.LinearExpression{}, c.L...),
			R:          append(constraint.LinearExpression{}, c.R...),
			O:          append(constraint.LinearExpression{}, c.O...),
			loc:        loc,
			outCoeffID: outCoeff,
		}
		if newWire >= 0 {
			m.outWireID = uint32(newWire)
		}
		allR1Cs = append(allR1Cs, m)
	}

	nbLayers := int(maxLayer) + 1
	fmt.Fprintf(os.Stderr, "[plan] %d layers; %d hint-output wires; %d R1C-output wires\n",
		nbLayers, len(hintOutputWires), len(r1cOutputWires))

	// Group R1Cs by layer
	byLayer := make([][]*r1cMeta, nbLayers)
	for _, m := range allR1Cs {
		byLayer[m.layer] = append(byLayer[m.layer], m)
	}

	// Coefficients
	{
		f, _ := os.Create(outDir + "/coeffs.bin")
		bw := bufio.NewWriterSize(f, 1<<20)
		var buf [32]byte
		for i := range r.Coefficients {
			for limb := 0; limb < 4; limb++ {
				binary.LittleEndian.PutUint64(buf[limb*8:], r.Coefficients[i][limb])
			}
			bw.Write(buf[:])
		}
		bw.Flush()
		f.Close()
	}

	// Initial wires (witness + all hint outputs; R1C-defined wires zeroed).
	// Expected wires.
	{
		ef, _ := os.Create(outDir + "/wires_expected.bin")
		ew := bufio.NewWriterSize(ef, 1<<20)
		ifc, _ := os.Create(outDir + "/wires_initial.bin")
		iw := bufio.NewWriterSize(ifc, 1<<20)
		var buf [32]byte
		for i := range wires {
			for limb := 0; limb < 4; limb++ {
				binary.LittleEndian.PutUint64(buf[limb*8:], wires[i][limb])
			}
			ew.Write(buf[:])
			if r1cOutputWires[uint32(i)] {
				var zero [32]byte
				iw.Write(zero[:])
			} else {
				iw.Write(buf[:])
			}
		}
		ew.Flush()
		iw.Flush()
		ef.Close()
		ifc.Close()
	}

	// Per-layer descs + terms, concatenated.
	df, _ := os.Create(outDir + "/layers_descs.bin")
	dbw := bufio.NewWriterSize(df, 1<<20)
	tf, _ := os.Create(outDir + "/layers_terms.bin")
	tbw := bufio.NewWriterSize(tf, 1<<20)
	idxF, _ := os.Create(outDir + "/layers.idx")
	ibw := bufio.NewWriterSize(idxF, 1<<20)
	mustU32(ibw, uint32(nbLayers))

	var totalDescs uint64 = 0
	var totalTerms uint64 = 0
	for _, ms := range byLayer {
		// idx entry: n_descs, descs_off, terms_off
		mustU32(ibw, uint32(len(ms)))
		mustU64(ibw, totalDescs)
		mustU64(ibw, totalTerms)
		for _, m := range ms {
			lOff := uint32(totalTerms)
			lCnt := uint32(len(m.L))
			for _, t := range m.L {
				mustU32(tbw, uint32(t.CoeffID()))
				mustU32(tbw, uint32(t.WireID()))
			}
			totalTerms += uint64(lCnt)
			rOff := uint32(totalTerms)
			rCnt := uint32(len(m.R))
			for _, t := range m.R {
				mustU32(tbw, uint32(t.CoeffID()))
				mustU32(tbw, uint32(t.WireID()))
			}
			totalTerms += uint64(rCnt)
			oOff := uint32(totalTerms)
			oCnt := uint32(len(m.O))
			for _, t := range m.O {
				mustU32(tbw, uint32(t.CoeffID()))
				mustU32(tbw, uint32(t.WireID()))
			}
			totalTerms += uint64(oCnt)
			mustU32(dbw, lOff)
			mustU32(dbw, lCnt)
			mustU32(dbw, rOff)
			mustU32(dbw, rCnt)
			mustU32(dbw, oOff)
			mustU32(dbw, oCnt)
			mustU32(dbw, m.outCoeffID)
			mustU32(dbw, m.outWireID)
			must1(dbw.WriteByte(m.loc))
			must1(dbw.WriteByte(0))
			must1(dbw.WriteByte(0))
			must1(dbw.WriteByte(0))
		}
		totalDescs += uint64(len(ms))
	}
	dbw.Flush()
	df.Close()
	tbw.Flush()
	tf.Close()
	ibw.Flush()
	idxF.Close()

	fmt.Fprintf(os.Stderr, "[plan] %d total descs across %d layers, %d total terms\n",
		totalDescs, nbLayers, totalTerms)

	// Meta
	mf, _ := os.Create(outDir + "/meta.txt")
	fmt.Fprintf(mf, "build_dir=%s\n", buildDir)
	fmt.Fprintf(mf, "witness=%s\n", witnessPath)
	fmt.Fprintf(mf, "n_wires=%d\n", len(wires))
	fmt.Fprintf(mf, "n_coefficients=%d\n", len(r.Coefficients))
	fmt.Fprintf(mf, "n_layers=%d\n", nbLayers)
	fmt.Fprintf(mf, "n_descs=%d\n", totalDescs)
	fmt.Fprintf(mf, "n_terms=%d\n", totalTerms)
	mf.Close()
	fmt.Fprintf(os.Stderr, "[plan] prep-full done in %s; outDir=%s\n",
		time.Since(t0), outDir)
}

func mustU64(w io.Writer, v uint64) {
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], v)
	if _, err := w.Write(buf[:]); err != nil {
		fail("write u64: %v", err)
	}
}

// ---------------- prep-layer ----------------
//
// Dumps the inputs + expected outputs for a single layer's R1C-only
// kernel test, into a directory. Picks the widest R1C-only layer
// (i.e. the layer with the most loc != 0 R1Cs) so the kernel sees a
// large, GPU-friendly workload.
//
// Files written:
//   coeffs.bin          all coefficients (32 B / element, little-endian Fr)
//   terms.bin           flat (cid u32, vid u32) pairs for L||R||O of test layer
//   descs.bin           per-R1C descriptors (28 bytes — see C++ side)
//   wires_blank.bin     full wire vector with the test layer's outputs zeroed
//   wires_expected.bin  full wire vector from the CPU interpreter (gold)
//   meta.txt            human-readable summary (n_descs, n_terms, n_wires, …)

func prepLayer(buildDir, witnessPath, outDir string, requestedLayer int32) {
	r := loadR1CS(buildDir)
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		fail("mkdir out: %v", err)
	}

	// Run the Phase 1 interpreter on the production witness to get the
	// reference wire vector and -- as a side effect -- the wire-defined
	// state we need to choose a layer with computable R1Cs.
	t0 := time.Now()
	tmp, _ := os.CreateTemp("", "solve_plan-*.bin")
	tmp.Close()
	defer os.Remove(tmp.Name())
	emit(buildDir, tmp.Name(), planV2)
	wires := interpretPlan(r, tmp.Name(), witnessPath)
	fmt.Fprintf(os.Stderr, "[plan] reference solve in %s\n", time.Since(t0))

	// Build per-instruction layer ID by mirroring emit-layers in memory.
	nbWires := r.NbInternalVariables + r.GetNbPublicVariables() + r.GetNbSecretVariables()
	wireDepth := make([]int32, nbWires)
	for i := range wireDepth {
		wireDepth[i] = -1
	}
	nbInputs := r.GetNbPublicVariables() + r.GetNbSecretVariables()
	for i := 0; i < nbInputs; i++ {
		wireDepth[i] = 0
	}
	hintBpID, hintFound := findHintBlueprint(r)
	hintBP, _ := r.Blueprints[hintBpID].(constraint.BlueprintHint)
	var hm constraint.HintMapping

	type r1cMeta struct {
		layer int32
		L, R, O constraint.LinearExpression
		loc uint8
		outCoeffID uint32
		outWireID  uint32
		instIdx    int
	}

	nbInstr := r.GetNbInstructions()
	layerCount := make(map[int32]int) // counts only R1Cs with loc != 0

	// First pass: walk to compute layer + loc per R1C, populate layerCount.
	allR1Cs := make([]*r1cMeta, 0, nbInstr)
	solved := make([]bool, nbWires)
	for i := 0; i < nbInputs; i++ {
		solved[i] = true
	}

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

		if hintFound && pi.BlueprintID == hintBpID {
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
					solved[w] = true
				}
			}
			continue
		}

		bp := r.Blueprints[pi.BlueprintID]
		r1c, _ := bp.(constraint.BlueprintR1C)
		var c constraint.R1C
		r1c.DecompressR1C(&c, inst)

		var newWire int32 = -1
		var loc uint8 = 0
		var outCoeff uint32 = 0
		processLE := func(le constraint.LinearExpression, locCandidate uint8) {
			for _, t := range le {
				vid := int32(t.WireID())
				if vid < 0 || t.IsConstant() || int(vid) >= len(wireDepth) {
					continue
				}
				if !solved[vid] {
					if loc != 0 {
						continue
					}
					loc = locCandidate
					newWire = vid
					outCoeff = uint32(t.CoeffID())
				} else {
					if wireDepth[vid] > d {
						d = wireDepth[vid]
					}
				}
			}
		}
		processLE(c.L, 1)
		processLE(c.R, 2)
		processLE(c.O, 3)
		thisDepth := d + 1
		if newWire >= 0 {
			wireDepth[newWire] = thisDepth
			solved[newWire] = true
		}

		meta := &r1cMeta{
			layer:      thisDepth,
			L:          append(constraint.LinearExpression{}, c.L...),
			R:          append(constraint.LinearExpression{}, c.R...),
			O:          append(constraint.LinearExpression{}, c.O...),
			loc:        loc,
			outCoeffID: outCoeff,
			instIdx:    i,
		}
		if newWire >= 0 {
			meta.outWireID = uint32(newWire)
		}
		allR1Cs = append(allR1Cs, meta)
		if loc != 0 {
			layerCount[thisDepth]++
		}
	}

	// Pick layer: requested or widest with computable R1Cs.
	var pickLayer int32 = -1
	pickWidth := 0
	if requestedLayer >= 0 {
		pickLayer = requestedLayer
		pickWidth = layerCount[requestedLayer]
		if pickWidth == 0 {
			fmt.Fprintf(os.Stderr,
				"[plan] WARNING: requested layer %d has 0 computable R1Cs\n",
				requestedLayer)
		}
	} else {
		for l, n := range layerCount {
			if n > pickWidth {
				pickWidth = n
				pickLayer = l
			}
		}
		if pickLayer < 0 {
			fail("no layer with computable R1Cs found?")
		}
	}
	fmt.Fprintf(os.Stderr, "[plan] picked layer %d with %d computable R1Cs\n",
		pickLayer, pickWidth)

	// Gather the chosen layer's R1Cs (keep verify-only ones too — they
	// exercise the kernel's verify path).
	picked := make([]*r1cMeta, 0)
	for _, m := range allR1Cs {
		if m.layer == pickLayer {
			picked = append(picked, m)
		}
	}
	fmt.Fprintf(os.Stderr, "[plan] layer %d has %d R1Cs total (%d computable, %d verify-only)\n",
		pickLayer, len(picked), pickWidth, len(picked)-pickWidth)

	// Emit coefficients (raw bytes; little-endian uint64×4 per element)
	{
		f, _ := os.Create(outDir + "/coeffs.bin")
		bw := bufio.NewWriterSize(f, 1<<20)
		var buf [32]byte
		for i := range r.Coefficients {
			for limb := 0; limb < 4; limb++ {
				binary.LittleEndian.PutUint64(buf[limb*8:], r.Coefficients[i][limb])
			}
			bw.Write(buf[:])
		}
		bw.Flush()
		f.Close()
	}

	// Emit terms + descs.
	{
		df, _ := os.Create(outDir + "/descs.bin")
		dbw := bufio.NewWriterSize(df, 1<<20)
		tf, _ := os.Create(outDir + "/terms.bin")
		tbw := bufio.NewWriterSize(tf, 1<<20)

		var termOff uint32 = 0
		writeLEFlat := func(le constraint.LinearExpression) (uint32, uint32) {
			off := termOff
			cnt := uint32(0)
			for _, t := range le {
				mustU32(tbw, uint32(t.CoeffID()))
				mustU32(tbw, uint32(t.WireID()))
				cnt++
			}
			termOff += cnt
			return off, cnt
		}
		for _, m := range picked {
			lOff, lCnt := writeLEFlat(m.L)
			rOff, rCnt := writeLEFlat(m.R)
			oOff, oCnt := writeLEFlat(m.O)
			mustU32(dbw, lOff)
			mustU32(dbw, lCnt)
			mustU32(dbw, rOff)
			mustU32(dbw, rCnt)
			mustU32(dbw, oOff)
			mustU32(dbw, oCnt)
			mustU32(dbw, m.outCoeffID)
			mustU32(dbw, m.outWireID)
			must1(dbw.WriteByte(m.loc))
			// Pad to 4-byte boundary so descs.bin is a clean array.
			must1(dbw.WriteByte(0))
			must1(dbw.WriteByte(0))
			must1(dbw.WriteByte(0))
		}
		dbw.Flush()
		df.Close()
		tbw.Flush()
		tf.Close()
		fmt.Fprintf(os.Stderr, "[plan] wrote %d descriptors and %d terms\n",
			len(picked), termOff)
	}

	// Emit blanked wires (reference with this layer's outputs zeroed)
	// and expected wires (the reference itself).
	{
		ef, _ := os.Create(outDir + "/wires_expected.bin")
		ew := bufio.NewWriterSize(ef, 1<<20)
		bf, _ := os.Create(outDir + "/wires_blank.bin")
		bw := bufio.NewWriterSize(bf, 1<<20)

		blanked := map[uint32]bool{}
		for _, m := range picked {
			if m.loc != 0 {
				blanked[m.outWireID] = true
			}
		}

		var buf [32]byte
		for i := range wires {
			for limb := 0; limb < 4; limb++ {
				binary.LittleEndian.PutUint64(buf[limb*8:], wires[i][limb])
			}
			ew.Write(buf[:])
			if blanked[uint32(i)] {
				var zero [32]byte
				bw.Write(zero[:])
			} else {
				bw.Write(buf[:])
			}
		}
		ew.Flush()
		bw.Flush()
		ef.Close()
		bf.Close()
		fmt.Fprintf(os.Stderr, "[plan] wrote %d wires (blanked %d for layer test)\n",
			len(wires), len(blanked))
	}

	// Meta
	{
		mf, _ := os.Create(outDir + "/meta.txt")
		fmt.Fprintf(mf, "build_dir=%s\n", buildDir)
		fmt.Fprintf(mf, "witness=%s\n", witnessPath)
		fmt.Fprintf(mf, "n_wires=%d\n", len(wires))
		fmt.Fprintf(mf, "n_coefficients=%d\n", len(r.Coefficients))
		fmt.Fprintf(mf, "picked_layer=%d\n", pickLayer)
		fmt.Fprintf(mf, "n_descs=%d\n", len(picked))
		fmt.Fprintf(mf, "n_computable=%d\n", pickWidth)
		fmt.Fprintf(mf, "n_verify_only=%d\n", len(picked)-pickWidth)
		mf.Close()
	}
	fmt.Fprintf(os.Stderr, "[plan] prep-layer done in %s; outDir=%s\n",
		time.Since(t0), outDir)
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
