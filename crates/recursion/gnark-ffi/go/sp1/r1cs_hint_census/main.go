// r1cs_hint_census: catalog every hint kind invoked by the SP1 Groth16
// circuit, with call count, input/output sizes, and topological depth
// distribution.
//
// This is Phase 0 of the GPU R1CS solver implementation plan
// (crates/recursion/gnark-ffi/docs/gpu_r1cs_solver_plan.md). The output
// determines the hint-port catalog: every kind that appears here needs a
// matching GPU kernel in Phase 3, so the smaller this list, the smaller
// the project.
//
// Walks gnark's R1CS instruction stream. For each BlueprintGenericHint
// instruction, decodes the HintMapping (HintID, inputs, output range)
// and tallies. Hint names come from the R1CS's persisted
// MHintsDependencies map — no external registration needed.
//
// Per-hint depth is the max topological depth of any input wire,
// computed alongside the constraint depth. This tells us which hint
// kinds dominate the deep tail (and thus which ones are worth porting
// for the hybrid shape).
//
// Usage: go run . <build_dir>
package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"sort"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/constraint"
	cs "github.com/consensys/gnark/constraint/bn254"
	"github.com/consensys/gnark/constraint/solver"
)

type hintStats struct {
	id              solver.HintID
	name            string // empty if not in MHintsDependencies
	calls           int
	totalInputs     int64
	totalOutputs    int64
	maxInputs       int
	maxOutputs      int
	minInputs       int
	minOutputs      int
	depthSum        int64
	maxDepth        int32
	wideLayerCalls  int // calls firing at depth-0..3 (wide layers)
	deepTailCalls   int // calls firing past depth 1000
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "Usage: r1cs_hint_census <build_dir>")
		os.Exit(1)
	}
	buildDir := os.Args[1]
	r1csPath := buildDir + "/groth16_circuit.bin"

	os.Setenv("CONSTRAINTS_JSON", buildDir+"/constraints.json")
	os.Setenv("GROTH16", "1")

	t0 := time.Now()
	r1csObj := groth16.NewCS(ecc.BN254)
	f, err := os.Open(r1csPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "open R1CS: %v\n", err)
		os.Exit(2)
	}
	if _, err := r1csObj.ReadFrom(bufio.NewReaderSize(f, 1<<20)); err != nil {
		fmt.Fprintf(os.Stderr, "read R1CS: %v\n", err)
		os.Exit(2)
	}
	f.Close()
	r := r1csObj.(*cs.R1CS)
	fmt.Printf("[hint-census] R1CS loaded in %s\n", time.Since(t0))

	nbWires := r.NbInternalVariables + r.GetNbPublicVariables() + r.GetNbSecretVariables()
	nbInstr := r.GetNbInstructions()
	fmt.Printf("[hint-census] nbWires=%d  nbInstructions=%d  nbHintRegistered=%d\n",
		nbWires, nbInstr, len(r.MHintsDependencies))

	// Report BSB22 commitment usage — these are special hints that
	// involve a G1 MSM and don't appear in MHintsDependencies because
	// they are runtime-overridden during prove. If non-zero the solve
	// plan must support a "commitment" instruction kind separate from
	// the simple per-call hint kinds.
	if comm, ok := r.CommitmentInfo.(constraint.Groth16Commitments); ok {
		fmt.Printf("[hint-census] BSB22 commitments in this circuit: %d\n", len(comm))
		for i, c := range comm {
			fmt.Printf("  commitment %d: nbPrivateCommitted=%d, commitmentIndex=%d, hashCommitted=%d\n",
				i, len(c.PrivateCommitted), c.CommitmentIndex, len(c.PublicAndCommitmentCommitted))
		}
	} else {
		fmt.Printf("[hint-census] CommitmentInfo type %T (not Groth16Commitments)\n", r.CommitmentInfo)
	}

	// Pre-compute per-wire topological depth using the same algorithm as
	// r1cs_characterize. We walk constraints in declaration order, but
	// here we also walk hint instructions and assign a depth to each
	// hint output wire so that downstream constraints see the right
	// depth.
	nbInputs := r.GetNbPublicVariables() + r.GetNbSecretVariables()
	wireDepth := make([]int32, nbWires)
	for i := range wireDepth {
		wireDepth[i] = -1
	}
	for i := 0; i < nbInputs; i++ {
		wireDepth[i] = 0
	}

	// Find the BlueprintGenericHint blueprint ID. It's the only blueprint
	// kind we care about for the census (BlueprintR1CS handles regular
	// constraints).
	hintBlueprintID := constraint.BlueprintID(0)
	hintBlueprintFound := false
	for i, b := range r.Blueprints {
		if _, ok := b.(*constraint.BlueprintGenericHint); ok {
			hintBlueprintID = constraint.BlueprintID(i)
			hintBlueprintFound = true
			break
		}
	}
	if !hintBlueprintFound {
		fmt.Fprintln(os.Stderr, "[hint-census] no BlueprintGenericHint found in this R1CS — nothing to census")
		os.Exit(0)
	}
	fmt.Printf("[hint-census] BlueprintGenericHint ID = %d\n", hintBlueprintID)

	// Tally per HintID.
	stats := make(map[solver.HintID]*hintStats)
	getStats := func(id solver.HintID) *hintStats {
		s := stats[id]
		if s == nil {
			s = &hintStats{
				id:         id,
				name:       r.MHintsDependencies[id],
				minInputs:  math.MaxInt32,
				minOutputs: math.MaxInt32,
			}
			stats[id] = s
		}
		return s
	}

	t1 := time.Now()
	// We must walk hint instructions interleaved with constraint
	// instructions to keep wire depths in sync. The R1CIterator only
	// returns R1Cs (skipping hints), so we iterate raw instructions
	// instead and dispatch by blueprint.
	hintBlueprint := r.Blueprints[hintBlueprintID].(constraint.BlueprintHint)
	var hm constraint.HintMapping

	maxConstraintDepth := int32(0)
	totalConstraints := 0
	totalHintCalls := 0

	for i := 0; i < nbInstr; i++ {
		pi := r.Instructions[i]
		inst := pi.Unpack(&r.System)
		if pi.BlueprintID == hintBlueprintID {
			// Hint instruction. Decode and tally.
			hm.Inputs = hm.Inputs[:0]
			hintBlueprint.DecompressHint(&hm, inst)
			s := getStats(hm.HintID)
			s.calls++
			totalHintCalls++

			// Inputs are LinearExpressions; sum the term count to get a
			// rough complexity proxy. (One LE may contain many terms;
			// the GPU kernel cost depends on total terms.)
			nIn := 0
			var d int32 = 0
			for _, le := range hm.Inputs {
				nIn += len(le)
				for _, term := range le {
					vid := int32(term.WireID())
					if vid < 0 || vid == math.MaxInt32 || term.IsConstant() {
						continue
					}
					if int(vid) >= len(wireDepth) {
						continue
					}
					if wireDepth[vid] >= 0 && wireDepth[vid] > d {
						d = wireDepth[vid]
					}
				}
			}
			thisDepth := d + 1
			nOut := int(hm.OutputRange.End - hm.OutputRange.Start)

			s.totalInputs += int64(nIn)
			s.totalOutputs += int64(nOut)
			if nIn > s.maxInputs {
				s.maxInputs = nIn
			}
			if nIn < s.minInputs {
				s.minInputs = nIn
			}
			if nOut > s.maxOutputs {
				s.maxOutputs = nOut
			}
			if nOut < s.minOutputs {
				s.minOutputs = nOut
			}
			s.depthSum += int64(thisDepth)
			if thisDepth > s.maxDepth {
				s.maxDepth = thisDepth
			}
			if thisDepth <= 3 {
				s.wideLayerCalls++
			}
			if thisDepth > 1000 {
				s.deepTailCalls++
			}

			// Mark hint output wires with this hint's firing depth.
			for w := hm.OutputRange.Start; w < hm.OutputRange.End; w++ {
				if int(w) < len(wireDepth) {
					wireDepth[w] = thisDepth
				}
			}
			continue
		}

		// Otherwise it's a regular R1C. Compute the constraint's depth
		// and update wireDepth for whichever wire it first defines.
		if inst.Calldata == nil {
			continue
		}
		bp := r.Blueprints[pi.BlueprintID]
		if r1cBP, ok := bp.(constraint.BlueprintR1C); ok {
			var c constraint.R1C
			r1cBP.DecompressR1C(&c, inst)
			var d int32 = 0
			newWireID := int32(-1)
			processLE := func(le constraint.LinearExpression, allowNew bool) {
				for _, term := range le {
					vid := int32(term.WireID())
					if vid < 0 || vid == math.MaxInt32 || term.IsConstant() {
						continue
					}
					if int(vid) >= len(wireDepth) {
						continue
					}
					if wireDepth[vid] < 0 {
						if allowNew {
							newWireID = vid
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
			if newWireID >= 0 {
				wireDepth[newWireID] = thisDepth
			}
			if thisDepth > maxConstraintDepth {
				maxConstraintDepth = thisDepth
			}
			totalConstraints++
		}
	}

	fmt.Printf("[hint-census] walked %d instructions in %s (constraints=%d, hint-calls=%d, max-constraint-depth=%d)\n",
		nbInstr, time.Since(t1), totalConstraints, totalHintCalls, maxConstraintDepth)

	// Sort by call count descending and report.
	rows := make([]*hintStats, 0, len(stats))
	for _, s := range stats {
		rows = append(rows, s)
	}
	sort.Slice(rows, func(i, j int) bool {
		return rows[i].calls > rows[j].calls
	})

	fmt.Printf("\n=== HINT KIND CENSUS ===\n")
	fmt.Printf("%-3s  %-50s  %12s  %10s  %12s  %12s  %10s  %10s\n",
		"#", "name", "calls", "% of all", "avg-inputs", "avg-outputs", "avg-depth", "max-depth")
	fmt.Println(string([]byte("------------------------------------------------------------------------------------------------------------------------------------------")))
	for i, s := range rows {
		name := s.name
		if name == "" {
			name = fmt.Sprintf("<unregistered hint id=%d>", s.id)
		}
		if len(name) > 50 {
			name = "…" + name[len(name)-49:]
		}
		avgIn := float64(s.totalInputs) / float64(s.calls)
		avgOut := float64(s.totalOutputs) / float64(s.calls)
		avgDepth := float64(s.depthSum) / float64(s.calls)
		pct := 100 * float64(s.calls) / float64(totalHintCalls)
		fmt.Printf("%-3d  %-50s  %12d  %9.2f%%  %12.2f  %12.2f  %10.0f  %10d\n",
			i+1, name, s.calls, pct, avgIn, avgOut, avgDepth, s.maxDepth)
	}

	// Detailed view: range of inputs/outputs and "where do they fire" for
	// the top kinds.
	fmt.Printf("\n=== TOP-10 DETAIL (input/output range and layer position) ===\n")
	for i := 0; i < 10 && i < len(rows); i++ {
		s := rows[i]
		name := s.name
		if name == "" {
			name = fmt.Sprintf("<id=%d>", s.id)
		}
		fmt.Printf("\n[%d] %s  (calls=%d)\n", i+1, name, s.calls)
		fmt.Printf("    inputs : min=%d  avg=%.2f  max=%d\n",
			s.minInputs, float64(s.totalInputs)/float64(s.calls), s.maxInputs)
		fmt.Printf("    outputs: min=%d  avg=%.2f  max=%d\n",
			s.minOutputs, float64(s.totalOutputs)/float64(s.calls), s.maxOutputs)
		fmt.Printf("    depth  : avg=%.1f  max=%d\n",
			float64(s.depthSum)/float64(s.calls), s.maxDepth)
		fmt.Printf("    where  : %d calls in wide layers (depth<=3),  %d in deep tail (depth>1000)\n",
			s.wideLayerCalls, s.deepTailCalls)
	}

	// Bottom-line summary used by the implementation plan to decide
	// whether the full GPU port is reasonable scope.
	uniqueKinds := len(rows)
	registeredKinds := 0
	for _, s := range rows {
		if s.name != "" {
			registeredKinds++
		}
	}
	fmt.Printf("\n=== SUMMARY ===\n")
	fmt.Printf("unique hint kinds in this circuit : %d\n", uniqueKinds)
	fmt.Printf("kinds with registered names       : %d\n", registeredKinds)
	fmt.Printf("kinds without registered names    : %d  (would need source lookup)\n", uniqueKinds-registeredKinds)
	fmt.Printf("total hint calls                  : %d\n", totalHintCalls)
	fmt.Printf("\nport scope: write %d GPU kernels (one per kind) + cover any anonymous IDs.\n",
		uniqueKinds)
}
