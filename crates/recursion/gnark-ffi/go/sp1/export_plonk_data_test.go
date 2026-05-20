package sp1

import (
	"os"
	"testing"
)

func getPlonkDataDir(t *testing.T) string {
	dataDir := os.Getenv("SP1_PLONK_DATA_DIR")
	if dataDir == "" {
		dataDir = os.Getenv("HOME") + "/.sp1/circuits/plonk/v6.0.0"
	}
	if _, err := os.Stat(dataDir + "/" + plonkPkPath); os.IsNotExist(err) {
		t.Skipf("PLONK artifacts not found at %s (set SP1_PLONK_DATA_DIR)", dataDir)
	}
	return dataDir
}

func TestExportPlonkData(t *testing.T) {
	dataDir := getPlonkDataDir(t)

	outputDir := "/tmp/plonk_exported"
	os.MkdirAll(outputDir, 0755)

	t.Logf("Exporting PLONK data from %s to %s", dataDir, outputDir)
	ExportPlonkData(dataDir, outputDir)

	// Verify output files exist
	expectedFiles := []string{
		"srs_g1_lagrange.bin",
		"srs_g1_canonical.bin",
		"trace_ql.bin",
		"trace_qr.bin",
		"trace_qm.bin",
		"trace_qo.bin",
		"trace_qk.bin",
		"trace_s1.bin",
		"trace_s2.bin",
		"trace_s3.bin",
		"plonk_domain_info.bin",
	}

	for _, f := range expectedFiles {
		path := outputDir + "/" + f
		info, err := os.Stat(path)
		if err != nil {
			t.Errorf("Missing output file: %s", f)
		} else {
			t.Logf("  %s: %d bytes", f, info.Size())
		}
	}
}

func TestExportSolvedWitness(t *testing.T) {
	dataDir := getPlonkDataDir(t)

	witnessPath := dataDir + "/plonk_witness.json"
	if _, err := os.Stat(witnessPath); os.IsNotExist(err) {
		t.Skipf("PLONK witness not found at %s", witnessPath)
	}

	outputDir := "/tmp/plonk_exported"
	os.MkdirAll(outputDir, 0755)

	t.Logf("Exporting solved witness from %s to %s", dataDir, outputDir)
	ExportSolvedWitness(dataDir, witnessPath, outputDir)

	// Verify witness output files exist
	expectedFiles := []string{
		"witness_l.bin",
		"witness_r.bin",
		"witness_o.bin",
		"witness_info.bin",
	}

	for _, f := range expectedFiles {
		path := outputDir + "/" + f
		info, err := os.Stat(path)
		if err != nil {
			t.Errorf("Missing output file: %s", f)
		} else {
			t.Logf("  %s: %d bytes", f, info.Size())
		}
	}

	// Check for BSB22 files (SP1 has exactly 1)
	if info, err := os.Stat(outputDir + "/bsb22_poly_0.bin"); err == nil {
		t.Logf("  bsb22_poly_0.bin: %d bytes", info.Size())
	} else {
		t.Error("Missing bsb22_poly_0.bin")
	}
	if info, err := os.Stat(outputDir + "/bsb22_commitment_0.bin"); err == nil {
		t.Logf("  bsb22_commitment_0.bin: %d bytes", info.Size())
	} else {
		t.Error("Missing bsb22_commitment_0.bin")
	}
}

func TestExportAll(t *testing.T) {
	dataDir := getPlonkDataDir(t)

	witnessPath := dataDir + "/plonk_witness.json"
	if _, err := os.Stat(witnessPath); os.IsNotExist(err) {
		t.Skipf("PLONK witness not found at %s", witnessPath)
	}

	outputDir := "/tmp/plonk_exported"
	os.MkdirAll(outputDir, 0755)

	t.Log("Step 1: Exporting circuit-static proving data...")
	ExportPlonkData(dataDir, outputDir)

	t.Log("Step 2: Exporting solved witness data...")
	ExportSolvedWitness(dataDir, witnessPath, outputDir)

	t.Log("Export complete. Files at:", outputDir)

	// List all exported files with sizes
	entries, _ := os.ReadDir(outputDir)
	for _, e := range entries {
		info, _ := e.Info()
		t.Logf("  %s: %d bytes (%.1f MB)", e.Name(), info.Size(), float64(info.Size())/1024/1024)
	}
}
