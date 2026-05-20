package sp1

import (
	"bufio"
	"crypto/rand"
	"fmt"
	"math/big"
	"os"
	"runtime"
	"testing"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark/backend/plonk"
	plonk_bn254 "github.com/consensys/gnark/backend/plonk/bn254"
)

// TestMSMTiming benchmarks gnark-crypto's BN254 G1 MSM at various sizes.
// Uses the actual SRS points from the PLONK proving key as the base points.
func TestMSMTiming(t *testing.T) {
	dataDir := getPlonkDataDir(t)

	t.Logf("System: %d CPUs, GOMAXPROCS=%d", runtime.NumCPU(), runtime.GOMAXPROCS(0))

	// Load proving key to get real SRS points
	start := time.Now()
	pkFile, err := os.Open(dataDir + "/" + plonkPkPath)
	if err != nil {
		t.Fatal(err)
	}
	pk := plonk.NewProvingKey(ecc.BN254)
	bufReader := bufio.NewReaderSize(pkFile, 1024*1024)
	pk.UnsafeReadFrom(bufReader)
	pkFile.Close()
	t.Logf("Loaded PK in %s", time.Since(start))

	pkBn254 := pk.(*plonk_bn254.ProvingKey)
	srsPoints := pkBn254.KzgLagrange.G1
	t.Logf("SRS has %d points", len(srsPoints))

	// Test MSM at different sizes
	sizes := []int{
		1 << 20, // 2^20 = ~1M
		1 << 22, // 2^22 = ~4M
		1 << 24, // 2^24 = ~16M
		1 << 25, // 2^25 = ~33M (full circuit size)
	}

	for _, n := range sizes {
		if n > len(srsPoints) {
			t.Logf("Skipping N=%d (only %d SRS points available)", n, len(srsPoints))
			continue
		}

		t.Run(fmt.Sprintf("N=%d_2^%d", n, log2(uint64(n))), func(t *testing.T) {
			points := srsPoints[:n]

			// Generate random scalars
			scalars := make([]fr.Element, n)
			for i := range scalars {
				b := make([]byte, 32)
				rand.Read(b)
				scalars[i].SetBigInt(new(big.Int).SetBytes(b))
			}

			config := ecc.MultiExpConfig{}

			// Warm up
			if n <= (1 << 20) {
				var warmup bn254.G1Affine
				warmup.MultiExp(points, scalars, config)
			}

			// Time the MSM
			runtime.GC()
			var result bn254.G1Affine

			msmStart := time.Now()
			result.MultiExp(points, scalars, config)
			msmElapsed := time.Since(msmStart)

			// Avoid dead code elimination
			_ = result

			throughput := float64(n) / msmElapsed.Seconds()
			t.Logf("MSM(2^%d = %d points): %s  (%.1f Mpoints/s)",
				log2(uint64(n)), n, msmElapsed, throughput/1e6)

			fmt.Printf("[MSM] N=2^%d (%d points): %s (%.1f Mpts/s) on %d cores\n",
				log2(uint64(n)), n, msmElapsed, throughput/1e6, runtime.NumCPU())
		})
	}
}

// TestMSMRepeated runs MSM at 2^25 multiple times to get a stable average.
func TestMSMRepeated(t *testing.T) {
	dataDir := getPlonkDataDir(t)

	t.Logf("System: %d CPUs, GOMAXPROCS=%d", runtime.NumCPU(), runtime.GOMAXPROCS(0))

	// Load proving key
	start := time.Now()
	pkFile, err := os.Open(dataDir + "/" + plonkPkPath)
	if err != nil {
		t.Fatal(err)
	}
	pk := plonk.NewProvingKey(ecc.BN254)
	bufReader := bufio.NewReaderSize(pkFile, 1024*1024)
	pk.UnsafeReadFrom(bufReader)
	pkFile.Close()
	t.Logf("Loaded PK in %s", time.Since(start))

	pkBn254 := pk.(*plonk_bn254.ProvingKey)
	n := len(pkBn254.KzgLagrange.G1) // 2^25
	points := pkBn254.KzgLagrange.G1

	// Generate random scalars
	scalars := make([]fr.Element, n)
	for i := range scalars {
		b := make([]byte, 32)
		rand.Read(b)
		scalars[i].SetBigInt(new(big.Int).SetBytes(b))
	}

	config := ecc.MultiExpConfig{}
	numRuns := 3

	t.Logf("Running %d MSMs at N=2^%d = %d points...", numRuns, log2(uint64(n)), n)

	var totalElapsed time.Duration
	for i := 0; i < numRuns; i++ {
		// Re-randomize scalars each time
		for j := range scalars {
			b := make([]byte, 32)
			rand.Read(b)
			scalars[j].SetBigInt(new(big.Int).SetBytes(b))
		}

		runtime.GC()
		var result bn254.G1Affine
		msmStart := time.Now()
		result.MultiExp(points, scalars, config)
		elapsed := time.Since(msmStart)
		_ = result

		totalElapsed += elapsed
		t.Logf("  Run %d: %s", i+1, elapsed)
	}

	avg := totalElapsed / time.Duration(numRuns)
	t.Logf("")
	t.Logf("=== MSM RESULTS (N=2^%d = %d) ===", log2(uint64(n)), n)
	t.Logf("Average: %s over %d runs", avg, numRuns)
	t.Logf("Throughput: %.1f Mpoints/s", float64(n)/avg.Seconds()/1e6)
	t.Logf("Compared to GPU MSM at ~8.4s: gnark CPU is %.1fx %s",
		func() float64 {
			if avg.Seconds() > 8.4 {
				return avg.Seconds() / 8.4
			}
			return 8.4 / avg.Seconds()
		}(),
		func() string {
			if avg.Seconds() > 8.4 {
				return "SLOWER"
			}
			return "FASTER"
		}())

	fmt.Printf("\n[RESULT] gnark CPU MSM at 2^%d: avg %s over %d runs on %d cores\n",
		log2(uint64(n)), avg, numRuns, runtime.NumCPU())
}
