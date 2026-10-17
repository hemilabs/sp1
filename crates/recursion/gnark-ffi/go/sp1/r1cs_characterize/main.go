// r1cs_characterize: dependency-graph characterization of a gnark BN254 R1CS.
//
// Loads the SP1 Groth16 R1CS, walks every R1C instruction, and computes the
// topological depth of each constraint based on the constraint that first
// "computes" each wire (the canonical definition gnark uses). Reports the
// layer distribution so we can decide whether a layered GPU R1CS solver
// could plausibly scale to SP1's 16M-constraint circuit.
//
// Output: layer count, max layer width, layer-width distribution percentiles,
// and total dependency depth — the key metrics for GPU layered-solve viability.
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
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "Usage: r1cs_characterize <build_dir>")
		os.Exit(1)
	}
	buildDir := os.Args[1]
	r1csPath := buildDir + "/groth16_circuit.bin"

	os.Setenv("CONSTRAINTS_JSON", buildDir+"/constraints.json")
	os.Setenv("GROTH16", "1")

	// Load R1CS
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
	fmt.Printf("[char] R1CS loaded in %s\n", time.Since(t0))

	nbWires := r.NbInternalVariables + r.GetNbPublicVariables() + r.GetNbSecretVariables()
	nbConstraints := r.GetNbConstraints()
	nbInstr := r.GetNbInstructions()
	fmt.Printf("[char] nbWires=%d  nbInternal=%d  nbPublic=%d  nbSecret=%d\n",
		nbWires, r.NbInternalVariables, r.GetNbPublicVariables(), r.GetNbSecretVariables())
	fmt.Printf("[char] nbConstraints=%d  nbInstructions=%d\n", nbConstraints, nbInstr)

	// wireDepth[w] = depth at which wire w is first defined; -1 = not yet defined
	// Public + secret wires are inputs (depth 0).
	nbInputs := r.GetNbPublicVariables() + r.GetNbSecretVariables()
	wireDepth := make([]int32, nbWires)
	for i := range wireDepth {
		wireDepth[i] = -1
	}
	for i := 0; i < nbInputs; i++ {
		wireDepth[i] = 0
	}

	// Walk constraints in order, computing depth = max(input_depth) + 1.
	// We approximate the "computed wire" as the highest-VID wire in O whose
	// depth is still -1 (matches gnark's solver behavior for the common
	// arithmetic constraint shape `a*b = c` where c is a fresh internal wire).
	// For commitment / hint-driven constraints, the wire may be defined by
	// an earlier hint instruction; we treat such constraints as depth-only
	// dependents (no new wire).
	t1 := time.Now()
	layerHist := make(map[int32]int)
	maxDepth := int32(0)
	depthSum := int64(0)
	noNewWire := 0
	multiNewWire := 0
	hintLikeBlueprint := 0

	it := r.GetR1CIterator()
	idx := 0
	for c := it.Next(); c != nil; c = it.Next() {
		// Compute max depth over L, R, O input wires.
		var d int32 = 0
		newWireID := int32(-1)
		newWireCount := 0
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
						newWireCount++
						newWireID = vid
					}
					// else: undefined input — likely a hint output computed
					// before this constraint by an earlier hint instruction.
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
		} else {
			noNewWire++
		}
		if newWireCount > 1 {
			multiNewWire++
		}

		layerHist[thisDepth]++
		if thisDepth > maxDepth {
			maxDepth = thisDepth
		}
		depthSum += int64(thisDepth)
		idx++
		if idx%1000000 == 0 {
			fmt.Printf("[char] walked %d/%d constraints, max depth so far %d\n",
				idx, nbConstraints, maxDepth)
		}
	}
	_ = hintLikeBlueprint
	fmt.Printf("[char] dependency walk took %s\n", time.Since(t1))

	// Layer width distribution
	layers := make([]int, maxDepth+1)
	for d, n := range layerHist {
		layers[d] = n
	}
	widths := make([]int, 0, len(layers))
	for _, w := range layers {
		if w > 0 {
			widths = append(widths, w)
		}
	}
	sort.Ints(widths)

	pct := func(p float64) int {
		if len(widths) == 0 {
			return 0
		}
		i := int(math.Round(p * float64(len(widths)-1) / 100.0))
		return widths[i]
	}
	maxW := 0
	if len(widths) > 0 {
		maxW = widths[len(widths)-1]
	}

	// Define "wide enough for GPU efficiency" as ≥10K constraints/layer
	// (rough rule: a ~10K-thread launch on RDNA3 fills ~half the CUs at
	// minimum for compute-bound work).
	wideCount := 0
	wideTotalWork := 0
	for _, w := range widths {
		if w >= 10000 {
			wideCount++
			wideTotalWork += w
		}
	}

	fmt.Printf("\n=== R1CS DEPENDENCY GRAPH SUMMARY ===\n")
	fmt.Printf("constraints           : %d\n", nbConstraints)
	fmt.Printf("layers (max depth+1)  : %d\n", len(layers))
	fmt.Printf("avg depth             : %.1f\n", float64(depthSum)/float64(nbConstraints))
	fmt.Printf("constraints with no new output wire (commitment/hint sinks): %d (%.2f%%)\n",
		noNewWire, 100*float64(noNewWire)/float64(nbConstraints))
	fmt.Printf("constraints with >1 new output wire (unusual)              : %d\n", multiNewWire)
	fmt.Printf("\nlayer-width distribution (constraints per layer):\n")
	fmt.Printf("  max  : %d\n", maxW)
	fmt.Printf("  p99  : %d\n", pct(99))
	fmt.Printf("  p95  : %d\n", pct(95))
	fmt.Printf("  p90  : %d\n", pct(90))
	fmt.Printf("  p50  : %d\n", pct(50))
	fmt.Printf("  p10  : %d\n", pct(10))
	fmt.Printf("  min  : %d\n", widths[0])
	fmt.Printf("\nGPU-friendly layers (≥10K constraints): %d / %d (%.1f%%)\n",
		wideCount, len(layers), 100*float64(wideCount)/float64(len(layers)))
	fmt.Printf("Constraints in those layers           : %d / %d (%.1f%%)\n",
		wideTotalWork, nbConstraints, 100*float64(wideTotalWork)/float64(nbConstraints))

	// Estimate GPU layered-solve cost from these stats. Best-case kernel
	// launch overhead is ~5 µs on HIP, ~3 µs on CUDA (lower with graphs
	// but graphs cost building time on each prove). Per-constraint GPU
	// work is ~50 ns even at peak (BN254 Fr ops on RDNA3).
	const launchUsHIP = 5.0
	const launchUsCUDA = 3.0
	const perConstraintNs = 50.0
	totalLaunchHIPus := float64(len(layers)) * launchUsHIP
	totalLaunchCUDAus := float64(len(layers)) * launchUsCUDA
	totalComputeMs := float64(nbConstraints) * perConstraintNs / 1e6
	cpuBaselineSec := 4.8
	fmt.Printf("\n=== GPU LAYERED-SOLVE PROJECTION ===\n")
	fmt.Printf("kernel launch overhead alone (HIP, %dµs/launch) : %.1f ms\n",
		int(launchUsHIP), totalLaunchHIPus/1000)
	fmt.Printf("kernel launch overhead alone (CUDA, %dµs/launch): %.1f ms\n",
		int(launchUsCUDA), totalLaunchCUDAus/1000)
	fmt.Printf("compute @50 ns/constraint (lower bound)         : %.1f ms\n",
		totalComputeMs)
	fmt.Printf("projected GPU layered-solve total (HIP)         : %.2f s\n",
		(totalLaunchHIPus/1000+totalComputeMs)/1000)
	fmt.Printf("projected GPU layered-solve total (CUDA)        : %.2f s\n",
		(totalLaunchCUDAus/1000+totalComputeMs)/1000)
	fmt.Printf("CPU baseline (gnark Solve)                      : %.2f s\n",
		cpuBaselineSec)

	// First few + last few layer widths to see if there's structure.
	fmt.Printf("\nlayer widths, first 20: ")
	for i := 0; i < 20 && i < len(layers); i++ {
		fmt.Printf("%d ", layers[i])
	}
	fmt.Printf("\nlayer widths, last 20:  ")
	start := len(layers) - 20
	if start < 0 {
		start = 0
	}
	for i := start; i < len(layers); i++ {
		fmt.Printf("%d ", layers[i])
	}
	fmt.Println()
}
