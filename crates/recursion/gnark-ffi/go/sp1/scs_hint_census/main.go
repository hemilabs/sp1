// scs_hint_census: catalog every hint kind, blueprint kind, BSB22
// commitment, and topological-layer width distribution for the SP1 PLONK
// SparseR1CS.
//
// This is Phase A of the GPU PLONK solver implementation plan
// (sp1-gpu/crates/plonk/docs/gpu_plonk_solver_plan.md). It mirrors the
// Groth16 R1CS census tool in the sibling directory r1cs_hint_census/.
//
// Critical STOP-GATE: if BlueprintLogDerivLookup is encountered, the
// program prints a STOP message and exits non-zero. Such blueprints are
// stateful (they cache resolved table entries across Solve invocations)
// and would require a 7th category of GPU dispatch — re-scope required.
//
// Output (stdout, also captured by docs/gpu_scs_solver_status.md):
//   - Blueprint kind census: count + total payload bytes for each of the
//     4 PLONK blueprints + any unexpected blueprint kinds.
//   - Hint kind census: per-HintID call count, total inputs, total outputs.
//   - BSB22 commitment census: list of PlonkCommitments with constraint
//     index and Committed length.
//   - Layer width histogram: total layers, max/p50/p95/p99/min widths,
//     log-spaced bucket distribution.
//
// Usage: scs_hint_census <build_dir>
//   build_dir must contain plonk_circuit.bin and constraints.json.
package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"reflect"
	"sort"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/plonk"
	"github.com/consensys/gnark/constraint"
	cs "github.com/consensys/gnark/constraint/bn254"
	"github.com/consensys/gnark/constraint/solver"
)

const plonkCircuitPath = "plonk_circuit.bin"

type blueprintStats struct {
	id           constraint.BlueprintID
	typeName     string
	calls        int
	payloadWords int64 // total uint32 words consumed across all calls
}

type hintStats struct {
	id           solver.HintID
	name         string // empty if not in MHintsDependencies
	calls        int
	totalInputs  int64 // sum of LinearExpression term counts
	totalOutputs int64
	maxInputs    int
	maxOutputs   int
	minInputs    int
	minOutputs   int
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "Usage: scs_hint_census <build_dir>")
		os.Exit(1)
	}
	buildDir := os.Args[1]
	scsPath := buildDir + "/" + plonkCircuitPath

	os.Setenv("CONSTRAINTS_JSON", buildDir+"/constraints.json")
	os.Setenv("PLONK", "1")

	t0 := time.Now()
	scsObj := plonk.NewCS(ecc.BN254)
	f, err := os.Open(scsPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "open SCS: %v\n", err)
		os.Exit(2)
	}
	if _, err := scsObj.ReadFrom(bufio.NewReaderSize(f, 1<<20)); err != nil {
		fmt.Fprintf(os.Stderr, "read SCS: %v\n", err)
		os.Exit(2)
	}
	f.Close()
	spr := scsObj.(*cs.SparseR1CS)
	fmt.Printf("[scs-census] SCS loaded in %s\n", time.Since(t0))

	nbPub := spr.GetNbPublicVariables()
	nbSec := spr.GetNbSecretVariables()
	nbInternal := spr.NbInternalVariables
	nbWires := nbInternal + nbPub + nbSec
	nbInstr := spr.GetNbInstructions()
	nbConstraints := spr.GetNbConstraints()
	fmt.Printf("[scs-census] nbConstraints=%d  nbInstructions=%d  nbWires=%d (pub=%d, sec=%d, internal=%d)\n",
		nbConstraints, nbInstr, nbWires, nbPub, nbSec, nbInternal)
	fmt.Printf("[scs-census] nbBlueprints=%d  nbHintsRegistered=%d  nbLevels=%d\n",
		len(spr.Blueprints), len(spr.MHintsDependencies), len(spr.Levels))

	// ---------------------------------------------------------------
	// STOP-GATE: scan blueprints for BlueprintLogDerivLookup (a.k.a.
	// BlueprintLookupHint in gnark v0.14.0). If present anywhere in the
	// blueprint table, we cannot proceed with the planned phase B/D
	// kernel design — re-scope required.
	// ---------------------------------------------------------------
	logDerivPresent := false
	logDerivBpIDs := []constraint.BlueprintID{}
	for i, b := range spr.Blueprints {
		// Match by reflect type name; the concrete type is
		// constraint.BlueprintLookupHint[E] (generic), and this matches
		// either the U32 or U64 instantiation.
		tname := reflect.TypeOf(b).String()
		if containsAny(tname, "BlueprintLookupHint", "BlueprintLogDerivLookup") {
			logDerivPresent = true
			logDerivBpIDs = append(logDerivBpIDs, constraint.BlueprintID(i))
		}
	}
	// Also check if any actual instruction uses such a blueprint, since
	// presence in the table without instructions is harmless.
	logDerivInstUsed := 0
	if logDerivPresent {
		idSet := map[constraint.BlueprintID]struct{}{}
		for _, id := range logDerivBpIDs {
			idSet[id] = struct{}{}
		}
		for i := 0; i < nbInstr; i++ {
			if _, ok := idSet[spr.Instructions[i].BlueprintID]; ok {
				logDerivInstUsed++
			}
		}
	}

	// ---------------------------------------------------------------
	// Blueprint kind census: walk all instructions, tally per blueprint
	// id, identify each blueprint by reflect type name, and sum the
	// calldata words consumed.
	// ---------------------------------------------------------------
	bpStats := make([]*blueprintStats, len(spr.Blueprints))
	for i, b := range spr.Blueprints {
		bpStats[i] = &blueprintStats{
			id:       constraint.BlueprintID(i),
			typeName: reflect.TypeOf(b).String(),
		}
	}
	hintBlueprintIDs := map[constraint.BlueprintID]constraint.BlueprintHint{}
	for i, b := range spr.Blueprints {
		if hb, ok := b.(constraint.BlueprintHint); ok {
			hintBlueprintIDs[constraint.BlueprintID(i)] = hb
		}
	}

	stats := make(map[solver.HintID]*hintStats)
	getStats := func(id solver.HintID) *hintStats {
		s := stats[id]
		if s == nil {
			s = &hintStats{
				id:         id,
				name:       spr.MHintsDependencies[id],
				minInputs:  math.MaxInt32,
				minOutputs: math.MaxInt32,
			}
			stats[id] = s
		}
		return s
	}

	t1 := time.Now()
	totalHintCalls := 0
	var hm constraint.HintMapping
	for i := 0; i < nbInstr; i++ {
		pi := spr.Instructions[i]
		bs := bpStats[pi.BlueprintID]
		bs.calls++
		bp := spr.Blueprints[pi.BlueprintID]
		// Compute per-instruction calldata size. CalldataSize() returns -1
		// for variable-size blueprints (e.g. BlueprintLookupHint), in
		// which case the first calldata word holds the actual count.
		cSize := bp.CalldataSize()
		if cSize < 0 {
			cSize = int(spr.CallData[pi.StartCallData])
		}
		bs.payloadWords += int64(cSize)

		// Hint-blueprint specialisation: collect hint-kind statistics.
		if hb, ok := hintBlueprintIDs[pi.BlueprintID]; ok {
			inst := pi.Unpack(&spr.System)
			hm.Inputs = hm.Inputs[:0]
			hb.DecompressHint(&hm, inst)
			s := getStats(hm.HintID)
			s.calls++
			totalHintCalls++
			nIn := 0
			for _, le := range hm.Inputs {
				nIn += len(le)
			}
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
		}
	}
	fmt.Printf("[scs-census] walked %d instructions in %s (hint-calls=%d)\n",
		nbInstr, time.Since(t1), totalHintCalls)

	// ---------------------------------------------------------------
	// Layer width histogram.
	// ---------------------------------------------------------------
	nLayers := len(spr.Levels)
	widths := make([]int, nLayers)
	totalRows := int64(0)
	maxW, minW := 0, math.MaxInt32
	for i, lvl := range spr.Levels {
		w := len(lvl)
		widths[i] = w
		totalRows += int64(w)
		if w > maxW {
			maxW = w
		}
		if w < minW {
			minW = w
		}
	}
	if nLayers == 0 {
		minW = 0
	}
	sortedWidths := make([]int, nLayers)
	copy(sortedWidths, widths)
	sort.Ints(sortedWidths)
	pct := func(p float64) int {
		if nLayers == 0 {
			return 0
		}
		idx := int(math.Floor(p * float64(nLayers-1)))
		if idx < 0 {
			idx = 0
		}
		if idx >= nLayers {
			idx = nLayers - 1
		}
		return sortedWidths[idx]
	}
	p50 := pct(0.50)
	p95 := pct(0.95)
	p99 := pct(0.99)
	p999 := pct(0.999)

	// Log-spaced bucket histogram: 1, 2-3, 4-7, 8-15, ..., 2^k..2^(k+1)-1.
	const maxBucket = 28
	buckets := make([]int, maxBucket+1)
	bucketRows := make([]int64, maxBucket+1)
	for _, w := range widths {
		if w <= 0 {
			buckets[0]++
			continue
		}
		b := int(math.Floor(math.Log2(float64(w))))
		if b > maxBucket {
			b = maxBucket
		}
		buckets[b]++
		bucketRows[b] += int64(w)
	}

	// ---------------------------------------------------------------
	// BSB22 commitment census.
	// ---------------------------------------------------------------
	var commInfo constraint.PlonkCommitments
	var hasComm bool
	if pc, ok := spr.CommitmentInfo.(constraint.PlonkCommitments); ok {
		commInfo = pc
		hasComm = true
	}

	// =========================================================
	// Print census report.
	// =========================================================

	// Sort blueprints by call count descending.
	sortedBP := make([]*blueprintStats, 0, len(bpStats))
	for _, b := range bpStats {
		sortedBP = append(sortedBP, b)
	}
	sort.Slice(sortedBP, func(i, j int) bool {
		return sortedBP[i].calls > sortedBP[j].calls
	})

	fmt.Printf("\n=== BLUEPRINT KIND CENSUS ===\n")
	fmt.Printf("%-3s  %-60s  %12s  %14s  %12s\n",
		"id", "type", "calls", "payload-words", "%-of-instr")
	fmt.Println(string([]byte("------------------------------------------------------------------------------------------------------------------")))
	for _, b := range sortedBP {
		if b.calls == 0 {
			continue
		}
		typeName := b.typeName
		if len(typeName) > 60 {
			typeName = "…" + typeName[len(typeName)-59:]
		}
		pctOf := 100 * float64(b.calls) / float64(nbInstr)
		fmt.Printf("%-3d  %-60s  %12d  %14d  %11.3f%%\n",
			b.id, typeName, b.calls, b.payloadWords, pctOf)
	}

	// Sort hints by call count descending.
	hintRows := make([]*hintStats, 0, len(stats))
	for _, s := range stats {
		hintRows = append(hintRows, s)
	}
	sort.Slice(hintRows, func(i, j int) bool {
		return hintRows[i].calls > hintRows[j].calls
	})

	fmt.Printf("\n=== HINT KIND CENSUS ===\n")
	fmt.Printf("%-3s  %-60s  %12s  %10s  %12s  %12s  %10s  %10s\n",
		"#", "name", "calls", "% of all", "avg-inputs", "avg-outputs", "max-inputs", "max-outputs")
	fmt.Println(string([]byte("--------------------------------------------------------------------------------------------------------------------------------------------")))
	for i, s := range hintRows {
		name := s.name
		if name == "" {
			name = fmt.Sprintf("<unregistered hint id=%d>", s.id)
		}
		if len(name) > 60 {
			name = "…" + name[len(name)-59:]
		}
		avgIn := float64(s.totalInputs) / float64(s.calls)
		avgOut := float64(s.totalOutputs) / float64(s.calls)
		pctOf := 100 * float64(s.calls) / float64(totalHintCalls)
		fmt.Printf("%-3d  %-60s  %12d  %9.3f%%  %12.2f  %12.2f  %10d  %10d\n",
			i+1, name, s.calls, pctOf, avgIn, avgOut, s.maxInputs, s.maxOutputs)
	}

	fmt.Printf("\n=== BSB22 COMMITMENT CENSUS ===\n")
	if !hasComm {
		fmt.Printf("CommitmentInfo type %T (NOT PlonkCommitments)\n", spr.CommitmentInfo)
	} else if len(commInfo) == 0 {
		fmt.Printf("PlonkCommitments present but empty (no BSB22 commitments).\n")
	} else {
		fmt.Printf("Found %d BSB22 commitment(s).\n", len(commInfo))
		fmt.Printf("%-3s  %20s  %20s\n", "i", "commitmentIndex", "len(Committed)")
		fmt.Println("---------------------------------------------------")
		for i, c := range commInfo {
			fmt.Printf("%-3d  %20d  %20d\n", i, c.CommitmentIndex, len(c.Committed))
		}
	}

	fmt.Printf("\n=== LAYER WIDTH HISTOGRAM ===\n")
	fmt.Printf("total layers       : %d\n", nLayers)
	fmt.Printf("total rows in levels: %d\n", totalRows)
	fmt.Printf("min width          : %d\n", minW)
	fmt.Printf("p50 width          : %d\n", p50)
	fmt.Printf("p95 width          : %d\n", p95)
	fmt.Printf("p99 width          : %d\n", p99)
	fmt.Printf("p99.9 width        : %d\n", p999)
	fmt.Printf("max width          : %d\n", maxW)
	fmt.Printf("\nLog-spaced buckets (width range -> #layers, #rows, %%-of-rows):\n")
	fmt.Printf("%-20s  %12s  %14s  %12s\n", "width-range", "#layers", "#rows", "%-of-rows")
	fmt.Println("------------------------------------------------------------------------")
	for b := 0; b <= maxBucket; b++ {
		if buckets[b] == 0 {
			continue
		}
		lo := 1 << b
		hi := (1 << (b + 1)) - 1
		var rng string
		if b == 0 {
			rng = "1"
		} else {
			rng = fmt.Sprintf("%d..%d", lo, hi)
		}
		pctRows := 100 * float64(bucketRows[b]) / float64(totalRows)
		fmt.Printf("%-20s  %12d  %14d  %11.3f%%\n", rng, buckets[b], bucketRows[b], pctRows)
	}

	// ---------------------------------------------------------------
	// Summary.
	// ---------------------------------------------------------------
	fmt.Printf("\n=== SUMMARY ===\n")
	usedBPs := 0
	for _, b := range bpStats {
		if b.calls > 0 {
			usedBPs++
		}
	}
	uniqueHintKinds := len(hintRows)
	registeredHintKinds := 0
	for _, s := range hintRows {
		if s.name != "" {
			registeredHintKinds++
		}
	}
	fmt.Printf("blueprint kinds in table         : %d\n", len(spr.Blueprints))
	fmt.Printf("blueprint kinds actually used    : %d\n", usedBPs)
	fmt.Printf("unique hint kinds in this circuit: %d\n", uniqueHintKinds)
	fmt.Printf("hint kinds with registered names : %d\n", registeredHintKinds)
	fmt.Printf("hint kinds without registered names: %d  (would need source lookup)\n", uniqueHintKinds-registeredHintKinds)
	fmt.Printf("total hint calls                 : %d\n", totalHintCalls)
	fmt.Printf("total instructions               : %d\n", nbInstr)
	fmt.Printf("total layers                     : %d\n", nLayers)
	fmt.Printf("max layer width                  : %d\n", maxW)

	// ---------------------------------------------------------------
	// STOP-GATE final result.
	// ---------------------------------------------------------------
	fmt.Printf("\n=== STOP-GATE: BlueprintLogDerivLookup ===\n")
	if logDerivPresent && logDerivInstUsed > 0 {
		fmt.Printf("[STOP-GATE TRIPPED] BlueprintLogDerivLookup found in PLONK SCS — re-scope required\n")
		fmt.Printf("  blueprint id(s) registered: %v\n", logDerivBpIDs)
		fmt.Printf("  instructions using it     : %d\n", logDerivInstUsed)
		os.Exit(3)
	}
	if logDerivPresent {
		fmt.Printf("BlueprintLogDerivLookup type registered in blueprint table but NOT instantiated by any instruction. Safe to proceed.\n")
		fmt.Printf("  blueprint id(s): %v\n", logDerivBpIDs)
	} else {
		fmt.Printf("BlueprintLogDerivLookup ABSENT from blueprint table and instruction stream. Phase B can proceed as planned.\n")
	}
}

// containsAny returns true if s contains any of the given substrings.
func containsAny(s string, subs ...string) bool {
	for _, sub := range subs {
		if indexOf(s, sub) >= 0 {
			return true
		}
	}
	return false
}

func indexOf(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}
