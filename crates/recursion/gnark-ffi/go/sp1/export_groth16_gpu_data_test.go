package sp1

import (
	"os"
	"testing"
)

func TestExportGroth16GpuWitness(t *testing.T) {
	dataDir := os.Getenv("GROTH16_DATA_DIR")
	if dataDir == "" {
		dataDir = "/home/max/.sp1/circuits/groth16/v6.0.0"
	}
	witnessPath := os.Getenv("GROTH16_WITNESS_PATH")
	if witnessPath == "" {
		t.Skip("Set GROTH16_WITNESS_PATH to run this test")
	}
	if _, err := os.Stat(dataDir + "/" + groth16CircuitPath); os.IsNotExist(err) {
		t.Skipf("Groth16 circuit data not found at %s", dataDir)
	}

	outputDir := "/tmp/groth16_gpu_exported"
	ExportGroth16GpuWitness(dataDir, witnessPath, outputDir)

	// Verify files exist
	files := []string{"wire_values.bin", "solution_a.bin", "solution_b.bin", "solution_c.bin",
		"commitments.bin", "commitment_pok.bin"}
	for _, f := range files {
		path := outputDir + "/" + f
		info, err := os.Stat(path)
		if err != nil {
			t.Errorf("Missing file: %s", f)
		} else {
			t.Logf("%s: %d bytes", f, info.Size())
		}
	}
}

func TestExportGroth16GpuData(t *testing.T) {
	dataDir := os.Getenv("GROTH16_DATA_DIR")
	if dataDir == "" {
		dataDir = "/home/max/.sp1/circuits/groth16/v6.0.0"
	}

	// Check if data exists
	if _, err := os.Stat(dataDir + "/" + groth16CircuitPath); os.IsNotExist(err) {
		t.Skipf("Groth16 circuit data not found at %s", dataDir)
	}

	outputDir := "/tmp/groth16_gpu_exported"
	os.RemoveAll(outputDir)

	ExportGroth16GpuData(dataDir, outputDir)

	// Verify files exist
	files := []string{
		"groth16_metadata.bin",
		"pk_g1_a.bin",
		"pk_g1_b.bin",
		"pk_g1_z.bin",
		"pk_g1_k.bin",
		"pk_g2_b.bin",
		"pk_g1_alpha.bin",
		"pk_g1_beta.bin",
		"pk_g1_delta.bin",
		"pk_g2_beta.bin",
		"pk_g2_delta.bin",
	}
	for _, f := range files {
		path := outputDir + "/" + f
		info, err := os.Stat(path)
		if err != nil {
			t.Errorf("Missing file: %s", f)
		} else {
			t.Logf("%s: %d bytes", f, info.Size())
		}
	}
}
