// scs_solve_plan: Phase B of the GPU PLONK SparseR1CS solver.
//
// Mirrors crates/recursion/gnark-ffi/go/sp1/r1cs_solve_plan/main.go but
// targets the production PLONK SparseR1CS at ~/.sp1/circuits/plonk/v6.0.0/.
//
// Subcommands:
//
//	emit-v2 <build_dir> <out_dir>
//	  Walk the production PLONK SCS and dump its structure as a packed
//	  binary "solve plan". Files written under out_dir:
//	    coeffs.bin
//	    layers.idx
//	    layers_descs.bin
//	    hints.idx
//	    layers_hints.bin
//	    hint_in_les.bin
//	    bsb22_meta.bin
//	    pubinput_layout.bin
//	    circuit_meta.txt
//
//	interpret-v2 <build_dir> <plan_dir> <witness.json> <out_wires>
//	  Pure-Go CPU re-solve from the binary plan (no gnark coupling). Walks
//	  layers in order; for each layer, runs blueprint Solve dispatch + hint
//	  calls, mirroring gnark/constraint/blueprint_scs.go and
//	  gnark/constraint/bn254/solver.go semantics. Writes the solved wire
//	  vector to out_wires.
//
//	roundtrip-v2 <build_dir> <witness.json>
//	  emit + interpret + run gnark spr.Solve, byte-diff L/R/O. PASS gate.
//
//	prep-circuit-prod <build_dir> <out_dir>
//	  Alias for emit-v2 (cached per-circuit form for Phase H).
//
// Plan format:
//
//   coeffs.bin
//     32 bytes per Fr coefficient (Montgomery limbs little-endian uint64×4).
//
//   layers.idx (u32-counted, little-endian)
//     u32 nbLayers
//     for each layer:
//       u32 n_descs
//       u32 n_hints
//       u64 descs_off    cumulative count of descs before this layer
//       u64 hints_off    cumulative count of hints before this layer
//
//   layers_descs.bin
//     for each row, in topological + within-layer order:
//       u8  blueprint_kind   0=Generic, 1=Mul, 2=Add, 3=Bool
//       u8  loc              0=verify-only, 1=XA, 2=XB, 3=XC
//       u16 _pad
//       u32 xa, xb, xc        wire IDs (raw uint32; never MaxUint32)
//       u32 ql, qr, qo, qm, qc  coefficient indices into coeffs.bin
//                               (0 means CoeffIdZero — the kernel skips zero terms)
//       u32 commitment        constraint.CommitmentConstraint (NOT/COMMITTED/COMMITMENT)
//       u32 decl_idx          declaration order index of this constraint
//                             (used by L/R/O scatter)
//     Total: 44 bytes/row.
//
//   hints.idx
//     u32 nbLayers   (must match layers.idx)
//     for each layer:
//       u32 n_hints
//       u64 calls_off       cumulative LE-input-record count
//
//   layers_hints.bin
//     for each hint call:
//       u8  kind             see hintNameToKind below
//       u8  _pad[3]
//       u32 n_inputs         number of LE inputs
//       u32 n_outputs        outEnd - outStart
//       u32 first_input_le_off  index into hint_in_les.bin
//       u32 out_wire_first    outStart
//       u32 hint_id           solver.HintID — used by interpreter to find fn
//
//   hint_in_les.bin
//     per LE: u32 cnt, then cnt × { u32 cid, u32 vid }.
//     LE entries are addressed by index, not byte offset.
//
//   bsb22_meta.bin
//     u32 nbBsb22
//     for each BSB22 commitment i:
//       u32 commitment_constraint_idx  spr.CommitmentInfo[i].CommitmentIndex
//       u32 layer_id                   layer that contains the BSB22 hint call
//       u32 hint_call_global_idx       global index of the call in layers_hints.bin
//       u32 n_committed                len(spr.CommitmentInfo[i].Committed)
//       n_committed × u32              committed_constraint_indexes
//
//   pubinput_layout.bin
//     u32 nb_public                    spr.GetNbPublicVariables()
//     u32 nb_public_inputs             len(spr.Public)
//     nb_public_inputs × u32           Public[i] (the index of each public input wire)
//
//   circuit_meta.txt
//     human-readable key=value pairs for cache validation and helper init.
//
// Wire layout (matches gnark SparseR1CS):
//   wires[0 .. nbPublic)                 = public inputs (witness)
//   wires[nbPublic .. nbPublic+nbSecret) = secret inputs (witness)
//   wires[nbPublic+nbSecret .. )         = internal (computed by solver)
//
// Note: unlike R1CS, SparseR1CS has NO ONE wire — the witness is loaded
// directly into wires [0 .. nbPublic+nbSecret). Constants are instead
// expressed via the QC field of each SparseR1C.
package main

import (
	"bufio"
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash"
	"io"
	"math"
	"math/big"
	"os"
	"reflect"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr/hash_to_field"
	"github.com/consensys/gnark-crypto/ecc/bn254/kzg"
	"github.com/consensys/gnark/backend/plonk"
	plonk_bn254 "github.com/consensys/gnark/backend/plonk/bn254"
	"github.com/consensys/gnark/constraint"
	cs "github.com/consensys/gnark/constraint/bn254"
	"github.com/consensys/gnark/constraint/solver"
	"github.com/consensys/gnark/frontend"
	fcs "github.com/consensys/gnark/frontend/cs"

	sp1 "github.com/succinctlabs/sp1-recursion-gnark/sp1"
)

const (
	plonkCircuitPath = "plonk_circuit.bin"
	plonkPkPath      = "plonk_pk.bin"
	plonkWitnessPath = "plonk_witness.json"
)

// Blueprint kind tags written to layers_descs.bin.
const (
	bpGeneric uint8 = 0
	bpMul     uint8 = 1
	bpAdd     uint8 = 2
	bpBool    uint8 = 3
)

// Sentinel for "absent" coefficient index. NOT WRITTEN — we use coeff index
// 0 (CoeffIdZero) for zero coefficients, which is what gnark already stores.
// A separate sentinel is unnecessary because Coefficients[0] is always zero
// and the kernel can treat it as a no-op.
//
// (The task spec mentioned 0xFFFFFFFF; we explicitly chose CoeffIdZero
// instead because it costs no kernel branch — multiplying by 0 is a noop.)

// Hint kind table (matches Phase A census).
//
// 0..5 mirror the Groth16 hint_kernels.cu numbering; 6..7 are PLONK-only.
// 8 = the Bsb22 placeholder hint, treated specially by interpret-v2.
var hintNameToKind = map[string]uint8{
	"github.com/consensys/gnark/std/math/bits.nBits":                            0,
	"github.com/consensys/gnark/constraint/solver.InvZeroHint":                  1,
	"github.com/succinctlabs/sp1-recursion-gnark/sp1/koalabear.SplitLimbsHint":  2,
	"github.com/succinctlabs/sp1-recursion-gnark/sp1/koalabear.ReduceHint":      3,
	"github.com/succinctlabs/sp1-recursion-gnark/sp1/koalabear.InvFHint":        4,
	"github.com/succinctlabs/sp1-recursion-gnark/sp1/koalabear.InvEHint":        5,
	"github.com/consensys/gnark/std/rangecheck.DecomposeHint":                   6,
	"github.com/consensys/gnark/std/internal/logderivarg.countHint":             7,
	"github.com/consensys/gnark/frontend/cs.Bsb22CommitmentComputePlaceholder":  8,
}

const (
	hintKindBsb22 uint8 = 8
)

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	switch os.Args[1] {
	case "emit-v2":
		if len(os.Args) != 4 {
			usage()
		}
		emitV2(os.Args[2], os.Args[3])
	case "prep-circuit-prod":
		if len(os.Args) != 4 {
			usage()
		}
		emitV2(os.Args[2], os.Args[3])
	case "interpret-v2":
		if len(os.Args) != 6 {
			usage()
		}
		interpretV2(os.Args[2], os.Args[3], os.Args[4], os.Args[5])
	case "dump-fixtures-c":
		// Phase C only: emit GPU prototype fixtures.
		//   wires_initial.bin   = nbWires × 32 B Mont; witness + hint outputs +
		//                         BSB22 outputs populated; SparseR1C-derived
		//                         wires zeroed (the kernel will fill these).
		//   wires_expected.bin  = full post-solve gold reference (== output of
		//                         interpret-v2). Byte-comparable to GPU output.
		// Usage:
		//   scs_solve_plan dump-fixtures-c <build_dir> <plan_dir> <witness.json> <out_dir>
		if len(os.Args) != 6 {
			usage()
		}
		dumpFixturesC(os.Args[2], os.Args[3], os.Args[4], os.Args[5])
	case "roundtrip-v2":
		if len(os.Args) != 4 {
			usage()
		}
		roundtripV2(os.Args[2], os.Args[3])
	case "dump-lro-reference":
		// Phase G only: run gnark spr.Solve() + evaluateLROSmallDomain and
		// dump the L/R/O reference vectors for byte-comparison against the
		// GPU kernel output.
		//
		// Usage:
		//   scs_solve_plan dump-lro-reference <build_dir> <plan_dir> <witness.json> <out_dir>
		//
		// <plan_dir> is the output of `emit-v2`. If it contains existing
		// `bsb22_blinding_<i>.bin` files, they will be REUSED so the L/R/O
		// reference matches an existing fixture (e.g. `wires_initial.bin`
		// from `dump-fixtures-c`). Otherwise gnark draws fresh
		// crypto/rand blindings AND writes them back into the plan dir for
		// downstream consistency.
		//
		// Writes:
		//   <out_dir>/lro_l_ref.bin    (N × 32 B Fr Mont)
		//   <out_dir>/lro_r_ref.bin
		//   <out_dir>/lro_o_ref.bin
		if len(os.Args) != 6 {
			usage()
		}
		dumpLroReference(os.Args[2], os.Args[3], os.Args[4], os.Args[5])
	case "dump-bsb22-seed":
		// Phase F only: emit the per-prove BSB22 PRF seed + its derived
		// blindings + the per-input wire-id list used to gather committedValues.
		//
		// Usage:
		//   scs_solve_plan dump-bsb22-seed <plan_dir> <out_seed_path> <out_blinding_dir>
		//
		// If <out_seed_path> equals "RANDOM" a fresh 32-byte seed is drawn
		// from crypto/rand. Otherwise the file at the given path is read
		// (must be exactly 32 bytes).
		//
		// On success writes:
		//   <out_blinding_dir>/bsb22_blinding_<i>.bin        (per BSB22, 64 B = 2 × 32 B Fr Mont LE)
		//   <out_blinding_dir>/bsb22_input_wires_<i>.bin     (per BSB22, n_committed × 4 B u32 wire IDs;
		//                                                    derived from layers_hints + hint_in_les)
		//   <out_seed_path>                                  (32 B seed; only written if "RANDOM")
		if len(os.Args) != 5 {
			usage()
		}
		dumpBsb22Seed(os.Args[2], os.Args[3], os.Args[4])
	case "make-witness-init":
		// Phase H subcommand: per-prove input materialisation for the
		// production GPU PLONK SCS solver. Combines what
		// `dump-bsb22-seed RANDOM` + `dump-fixtures-c` do for the
		// prototype validation flow, but writes ONLY the files the GPU
		// helper consumes (no wires_expected.bin, no LRO ref).
		//
		// Usage:
		//   scs_solve_plan make-witness-init <build_dir> <plan_dir> <witness.json> <out_dir>
		//
		// Writes:
		//   <out_dir>/bsb22_seed.bin              32 B fresh PRF seed
		//   <out_dir>/bsb22_blinding_<i>.bin      64 B (2 × 32 B Fr Mont)
		//   <out_dir>/bsb22_input_terms_<i>.bin   n_committed × 8 B (cid, vid)
		//   <out_dir>/bsb22_solve_meta_<i>.bin    32 B header
		//   <out_dir>/wires_initial.bin           nbWires × 32 B Mont (witness +
		//                                         non-BSB22 hints + BSB22-baked;
		//                                         SparseR1C-derived wires zeroed)
		//
		// The wires_initial.bin's BSB22-output wire is technically pre-baked
		// to the CPU's hash-to-field result for backward compatibility, but
		// the GPU sub-pipeline overwrites it from the Mont-form blindings —
		// the seed determinism guarantees the same value.
		if len(os.Args) != 6 {
			usage()
		}
		makeWitnessInit(os.Args[2], os.Args[3], os.Args[4], os.Args[5])
	case "make-witness-init-worker":
		// Long-lived worker that caches the SCS plan + PLONK PK in memory and
		// services per-prove witness-init requests over stdin/stdout. Drops
		// per-prove cost from ~30 s (cold PK reload + walk + dump) to ~1-2 s
		// (only the per-witness walk + dump).
		//
		// Usage:
		//   scs_solve_plan make-witness-init-worker <build_dir> <plan_dir>
		//
		// Protocol (newline-delimited JSON, one request per line on stdin,
		//           one response per line on stdout):
		//   request:  {"witness_path":"...","out_dir":"..."}
		//             or  {"command":"shutdown"}
		//             or  {"command":"ping"}
		//   response: {"status":"ok","wires_path":"...","seed_path":"...","elapsed_ms":N}
		//             or {"status":"error","msg":"..."}
		//             or {"status":"ready"}                       (after init)
		//             or {"status":"pong"}                        (ping reply)
		//
		// Logs to stderr only (stdout is reserved for the JSON protocol).
		if len(os.Args) != 4 {
			usage()
		}
		makeWitnessInitWorker(os.Args[2], os.Args[3])
	default:
		usage()
	}
}

// makeWitnessInit is the Phase H per-prove wrapper. Generates the BSB22 seed
// + per-input metadata, then runs `dump-fixtures-c` to produce wires_initial.bin
// (inheriting the just-written blindings via the bsb22_blinding_<i>.bin file).
//
// Layout note: dumpFixturesC reads bsb22_blinding_<i>.bin from <plan_dir>;
// dumpBsb22Seed writes them to <out_blinding_dir>. To bridge them without
// touching the plan_dir cache (which is shared across proves), we copy the
// freshly-emitted blindings into the plan_dir before dumping the fixtures,
// then optionally restore. For the production flow the plan_dir is a
// per-vk cache and the per-prove blindings ARE different on every call —
// the file is always overwritten. dumpFixturesC also writes wires_expected.bin
// which is harmless extra disk write (~1 GB) but irrelevant to the helper.
func makeWitnessInit(buildDir, planDir, witnessPath, outDir string) {
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		fail("mkdir out: %v", err)
	}
	// 1. Generate BSB22 seed + blindings + per-input terms + solve meta.
	dumpBsb22Seed(planDir, "RANDOM", outDir)

	// 2. Bridge: copy freshly-emitted bsb22_blinding_<i>.bin from outDir
	//    to planDir so dumpFixturesC sees them.
	{
		entries, err := os.ReadDir(outDir)
		if err != nil {
			fail("readdir %s: %v", outDir, err)
		}
		for _, e := range entries {
			name := e.Name()
			if len(name) >= len("bsb22_blinding_") &&
				name[:len("bsb22_blinding_")] == "bsb22_blinding_" {
				src := outDir + "/" + name
				dst := planDir + "/" + name
				data, err := os.ReadFile(src)
				if err != nil {
					fail("read %s: %v", src, err)
				}
				if err := os.WriteFile(dst, data, 0o644); err != nil {
					fail("write %s: %v", dst, err)
				}
			}
		}
	}

	// 3. Run dumpFixturesC into outDir. This produces wires_initial.bin
	//    AND wires_expected.bin; we only need the former.
	dumpFixturesC(buildDir, planDir, witnessPath, outDir)
}

// ---------------- make-witness-init-worker ----------------

// workerRequest is the per-prove request format on stdin (one JSON object per
// line). All paths are absolute.
type workerRequest struct {
	WitnessPath string `json:"witness_path,omitempty"`
	OutDir      string `json:"out_dir,omitempty"`
	Command     string `json:"command,omitempty"` // "shutdown" | "ping" | ""
}

// workerResponse is the per-prove response format on stdout (one JSON object
// per line).
type workerResponse struct {
	Status    string `json:"status"` // "ok" | "error" | "ready" | "pong" | "shutdown"
	Msg       string `json:"msg,omitempty"`
	WiresPath string `json:"wires_path,omitempty"`
	SeedPath  string `json:"seed_path,omitempty"`
	ElapsedMs int64  `json:"elapsed_ms,omitempty"`
}

// workerState caches the SCS plan + PLONK PK across requests.
type workerState struct {
	plan    *loadedPlan
	pk      *plonk_bn254.ProvingKey // nil if no BSB22 commitments
	domSize uint64
}

// makeWitnessInitWorker implements the long-lived worker subcommand. Reads
// JSON requests from stdin, writes JSON responses to stdout, logs to stderr.
//
// One worker process per (build_dir, plan_dir) pair. The worker holds the
// parsed plan + PK in memory, so per-request cost is dominated by the
// per-witness solver walk (~1-2 s) rather than the cold ~30 s reload path.
func makeWitnessInitWorker(buildDir, planDir string) {
	t0 := time.Now()
	st := &workerState{}

	// Load plan once.
	st.plan = loadPlan(planDir)
	tPlan := time.Since(t0)
	fmt.Fprintf(os.Stderr,
		"[scs-plan-worker] plan loaded in %s (layers=%d rows=%d hints=%d bsb22=%d)\n",
		tPlan, st.plan.nbLayers, len(st.plan.rows), len(st.plan.hints), len(st.plan.bsb22))

	// Load PK once if any BSB22 commitments exist.
	if len(st.plan.bsb22) > 0 {
		t1 := time.Now()
		pkFile, err := os.Open(buildDir + "/" + plonkPkPath)
		if err != nil {
			workerEmit(workerResponse{Status: "error",
				Msg: fmt.Sprintf("open PK for BSB22: %v", err)})
			return
		}
		pk := plonk.NewProvingKey(ecc.BN254)
		bufR := bufio.NewReaderSize(pkFile, 1<<20)
		if _, err := pk.UnsafeReadFrom(bufR); err != nil {
			pkFile.Close()
			workerEmit(workerResponse{Status: "error",
				Msg: fmt.Sprintf("read PK: %v", err)})
			return
		}
		pkFile.Close()
		st.pk = pk.(*plonk_bn254.ProvingKey)
		st.domSize = ecc.NextPowerOfTwo(uint64(st.plan.nbConstraints + st.plan.nbPublic))
		fmt.Fprintf(os.Stderr,
			"[scs-plan-worker] PK loaded in %s\n", time.Since(t1))
	}

	fmt.Fprintf(os.Stderr,
		"[scs-plan-worker] ready in %s (per-prove cost will be ~walk + dump only)\n",
		time.Since(t0))
	workerEmit(workerResponse{Status: "ready"})

	// Request loop.
	br := bufio.NewReaderSize(os.Stdin, 1<<16)
	for {
		line, err := br.ReadBytes('\n')
		if err == io.EOF {
			fmt.Fprintf(os.Stderr,
				"[scs-plan-worker] stdin EOF, exiting cleanly\n")
			return
		}
		if err != nil {
			fmt.Fprintf(os.Stderr,
				"[scs-plan-worker] stdin read error: %v, exiting\n", err)
			return
		}
		line = trimNewline(line)
		if len(line) == 0 {
			continue
		}
		var req workerRequest
		if err := json.Unmarshal(line, &req); err != nil {
			workerEmit(workerResponse{Status: "error",
				Msg: fmt.Sprintf("invalid JSON request: %v", err)})
			continue
		}
		switch req.Command {
		case "shutdown":
			workerEmit(workerResponse{Status: "shutdown"})
			fmt.Fprintf(os.Stderr,
				"[scs-plan-worker] shutdown requested\n")
			return
		case "ping":
			workerEmit(workerResponse{Status: "pong"})
			continue
		case "":
			// Normal solve request — fall through.
		default:
			workerEmit(workerResponse{Status: "error",
				Msg: fmt.Sprintf("unknown command: %q", req.Command)})
			continue
		}
		if req.WitnessPath == "" || req.OutDir == "" {
			workerEmit(workerResponse{Status: "error",
				Msg: "request missing witness_path or out_dir"})
			continue
		}
		t1 := time.Now()
		if err := workerHandleSolve(st, req.WitnessPath, req.OutDir); err != nil {
			workerEmit(workerResponse{Status: "error",
				Msg: err.Error()})
			continue
		}
		elapsed := time.Since(t1)
		fmt.Fprintf(os.Stderr,
			"[scs-plan-worker] solve completed in %s\n", elapsed)
		workerEmit(workerResponse{
			Status:    "ok",
			WiresPath: req.OutDir + "/wires_initial.bin",
			SeedPath:  req.OutDir + "/bsb22_seed.bin",
			ElapsedMs: elapsed.Milliseconds(),
		})
	}
}

// workerEmit writes a JSON response + newline to stdout and flushes.
// Stdout is line-buffered by default for terminal but pipe-buffered for
// pipes; we explicitly flush via os.Stdout.Sync (no-op on pipes; the
// json.Encoder write is the actual transport).
func workerEmit(resp workerResponse) {
	enc := json.NewEncoder(os.Stdout)
	if err := enc.Encode(&resp); err != nil {
		fmt.Fprintf(os.Stderr,
			"[scs-plan-worker] failed to write response: %v\n", err)
	}
}

func trimNewline(b []byte) []byte {
	for len(b) > 0 && (b[len(b)-1] == '\n' || b[len(b)-1] == '\r') {
		b = b[:len(b)-1]
	}
	return b
}

// workerHandleSolve runs the per-prove witness-init pipeline using the cached
// SCS plan + PK. Bypasses the file-bridge that the one-shot path uses (the
// plan + blindings stay in memory). Writes:
//   <outDir>/bsb22_seed.bin              (32 B)
//   <outDir>/bsb22_blinding_<i>.bin      (64 B)  — optional, debug-friendly
//   <outDir>/bsb22_input_terms_<i>.bin   — for first-time prep-cache promotion
//   <outDir>/bsb22_solve_meta_<i>.bin    — for first-time prep-cache promotion
//   <outDir>/wires_initial.bin           (n_wires × 32 B Mont)
//
// SKIPS the ~1 GB wires_expected.bin write that the one-shot dumpFixturesC
// performs.
func workerHandleSolve(st *workerState, witnessPath, outDir string) error {
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return fmt.Errorf("mkdir out: %w", err)
	}

	// 1. Generate fresh BSB22 seed.
	var seed [32]byte
	if _, err := rand.Read(seed[:]); err != nil {
		return fmt.Errorf("rand seed: %w", err)
	}
	seedPath := outDir + "/bsb22_seed.bin"
	if err := os.WriteFile(seedPath, seed[:], 0o644); err != nil {
		return fmt.Errorf("write seed: %w", err)
	}

	// 2. Derive per-bsb22 blindings + (re)write the per-bsb22 metadata files.
	//    bsb22_solve_meta_<i>.bin and bsb22_input_terms_<i>.bin are static
	//    per circuit but we still write them every call; the dispatcher
	//    promotes them into the prep-cache on the first prove. Cheap.
	p := st.plan
	bsb22Blindings := make([][2]fr.Element, len(p.bsb22))
	for i, lb := range p.bsb22 {
		// Walk the hint inputs to recover (cid, vid) pairs (skipping the
		// depth input). Same logic as dumpBsb22Seed.
		hintGlob := int(lb.hintGlobalIdx)
		if hintGlob >= len(p.hints) {
			return fmt.Errorf("bsb22[%d]: hint global idx %d out of range", i, hintGlob)
		}
		h := p.hints[hintGlob]
		nCommitted := int(h.nIn) - 1
		if nCommitted != int(lb.nCommitted()) {
			return fmt.Errorf("bsb22[%d]: input count mismatch", i)
		}

		// Derive blindings from seed.
		for k := 0; k < 2; k++ {
			hh := newSha256()
			hh.Write(seed[:])
			hh.Write([]byte{byte(k)})
			out := hh.Sum(nil)
			bsb22Blindings[i][k].SetBytes(out)
		}

		// Persist blinding (debug-friendly; not consumed by the helper).
		blindPath := fmt.Sprintf("%s/bsb22_blinding_%d.bin", outDir, i)
		var bbuf [64]byte
		for k := 0; k < 2; k++ {
			for limb := 0; limb < 4; limb++ {
				binary.LittleEndian.PutUint64(bbuf[k*32+limb*8:], bsb22Blindings[i][k][limb])
			}
		}
		if err := os.WriteFile(blindPath, bbuf[:], 0o644); err != nil {
			return fmt.Errorf("write %s: %w", blindPath, err)
		}

		// Per-input wire terms (cid, vid) — the C ABI helper consumes
		// bsb22_input_terms_<i>.bin from prep_dir; we emit it into outDir so
		// the dispatcher can promote it on first prove.
		wirePath := fmt.Sprintf("%s/bsb22_input_terms_%d.bin", outDir, i)
		{
			f, err := os.Create(wirePath)
			if err != nil {
				return fmt.Errorf("create %s: %w", wirePath, err)
			}
			bw := bufio.NewWriterSize(f, 1<<20)
			for j := 0; j < nCommitted; j++ {
				le := h.inputs[1+j]
				if len(le) != 1 {
					f.Close()
					return fmt.Errorf("bsb22[%d] input %d: expected single-term LE",
						i, j)
				}
				t := le[0]
				mustU32(bw, t.cid)
				mustU32(bw, t.vid)
			}
			bw.Flush()
			f.Close()
		}

		// Compact per-bsb22 metadata.
		metaPath := fmt.Sprintf("%s/bsb22_solve_meta_%d.bin", outDir, i)
		{
			domainSize := nextPow2U32(uint32(p.nbConstraints + p.nbPublic))
			f, err := os.Create(metaPath)
			if err != nil {
				return fmt.Errorf("create %s: %w", metaPath, err)
			}
			bw := bufio.NewWriterSize(f, 1<<20)
			mustU32(bw, lb.layerID)
			mustU32(bw, h.outStart)
			mustU32(bw, lb.commitmentConstraintIdx)
			mustU32(bw, uint32(p.nbConstraints))
			mustU32(bw, uint32(p.nbPublic))
			mustU32(bw, uint32(nCommitted))
			mustU32(bw, domainSize)
			mustU32(bw, 0)
			bw.Flush()
			f.Close()
		}
	}

	// 3. Solver walk (per-witness work).
	const (
		srcUnset   uint8 = 0
		srcWitness uint8 = 1
		srcHint    uint8 = 2
		srcRow     uint8 = 3
	)
	wires := make([]fr.Element, p.nbWires)
	solved := make([]bool, p.nbWires)
	source := make([]uint8, p.nbWires)

	// Pre-validate witness path before handing to loadWitnessAsFrVector,
	// which calls os.Exit on failure (would kill the worker). The Rust
	// dispatcher will see stdout EOF and respawn, but we can avoid that
	// by catching the most common failure modes here first.
	if _, err := os.Stat(witnessPath); err != nil {
		return fmt.Errorf("witness file not accessible: %w", err)
	}
	witnessVec := loadWitnessAsFrVector(witnessPath)
	expected := p.nbPublic + p.nbSecret
	if len(witnessVec) != expected {
		return fmt.Errorf("witness size %d != expected %d", len(witnessVec), expected)
	}
	for i := range witnessVec {
		wires[i] = witnessVec[i]
		solved[i] = true
		source[i] = srcWitness
	}

	for layerID := 0; layerID < p.nbLayers; layerID++ {
		hStart := p.layerHintOff[layerID]
		hEnd := p.layerHintOff[layerID+1]
		for hi := hStart; hi < hEnd; hi++ {
			h := p.hints[hi]
			runHint(h, p.coeffs, wires, solved, p, layerID, hi, st.pk,
				st.domSize, bsb22Blindings, p.nbConstraints)
			for w := h.outStart; w < h.outStart+h.nOut; w++ {
				if int(w) < len(source) && source[w] == srcUnset {
					source[w] = srcHint
				}
			}
		}
		dStart := p.layerDescOff[layerID]
		dEnd := p.layerDescOff[layerID+1]
		for di := dStart; di < dEnd; di++ {
			r := p.rows[di]
			var w uint32
			has := true
			switch r.bpKind {
			case bpMul, bpAdd:
				w = r.xc
			case bpGeneric:
				if r.commitment == uint32(constraint.NOT) && r.loc != 0 {
					switch r.loc {
					case 1:
						w = r.xa
					case 2:
						w = r.xb
					case 3:
						w = r.xc
					}
				} else {
					has = false
				}
			default:
				has = false
			}
			runRow(r, p.coeffs, wires, solved)
			if has && int(w) < len(source) && source[w] == srcUnset {
				source[w] = srcRow
			}
		}
	}

	// 4. Write wires_initial.bin (zero out SparseR1C-derived slots).
	initPath := outDir + "/wires_initial.bin"
	f, err := os.Create(initPath)
	if err != nil {
		return fmt.Errorf("create %s: %w", initPath, err)
	}
	bw := bufio.NewWriterSize(f, 1<<20)
	var zero32 [32]byte
	var buf [32]byte
	for i := range wires {
		if source[i] == srcRow {
			bw.Write(zero32[:])
		} else {
			for limb := 0; limb < 4; limb++ {
				binary.LittleEndian.PutUint64(buf[limb*8:], wires[i][limb])
			}
			bw.Write(buf[:])
		}
	}
	if err := bw.Flush(); err != nil {
		f.Close()
		return fmt.Errorf("flush %s: %w", initPath, err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("close %s: %w", initPath, err)
	}
	return nil
}

func usage() {
	fmt.Fprintln(os.Stderr, "Usage:")
	fmt.Fprintln(os.Stderr, "  scs_solve_plan emit-v2          <build_dir> <out_dir>")
	fmt.Fprintln(os.Stderr, "  scs_solve_plan prep-circuit-prod <build_dir> <out_dir>")
	fmt.Fprintln(os.Stderr, "  scs_solve_plan interpret-v2     <build_dir> <plan_dir> <witness.json> <out_wires.bin>")
	fmt.Fprintln(os.Stderr, "  scs_solve_plan dump-fixtures-c  <build_dir> <plan_dir> <witness.json> <out_dir>")
	fmt.Fprintln(os.Stderr, "  scs_solve_plan roundtrip-v2     <build_dir> <witness.json>")
	fmt.Fprintln(os.Stderr, "  scs_solve_plan dump-lro-reference <build_dir> <plan_dir> <witness.json> <out_dir>")
	fmt.Fprintln(os.Stderr, "  scs_solve_plan dump-bsb22-seed  <plan_dir> <seed_path|RANDOM> <out_blinding_dir>")
	fmt.Fprintln(os.Stderr, "  scs_solve_plan make-witness-init <build_dir> <plan_dir> <witness.json> <out_dir>")
	fmt.Fprintln(os.Stderr, "  scs_solve_plan make-witness-init-worker <build_dir> <plan_dir>")
	os.Exit(1)
}

// ---------------- SCS load ----------------

func loadSCS(buildDir string) *cs.SparseR1CS {
	os.Setenv("CONSTRAINTS_JSON", buildDir+"/constraints.json")
	// Make sure GROTH16 is NOT set (NewCircuit checks this and would emit
	// a different circuit shape).
	os.Unsetenv("GROTH16")

	t0 := time.Now()
	scsObj := plonk.NewCS(ecc.BN254)
	f, err := os.Open(buildDir + "/" + plonkCircuitPath)
	if err != nil {
		fail("open SCS: %v", err)
	}
	if _, err := scsObj.ReadFrom(bufio.NewReaderSize(f, 1<<20)); err != nil {
		fail("read SCS: %v", err)
	}
	f.Close()
	spr := scsObj.(*cs.SparseR1CS)
	fmt.Fprintf(os.Stderr,
		"[scs-plan] SCS loaded in %s (constraints=%d wires=%d levels=%d)\n",
		time.Since(t0), spr.GetNbConstraints(),
		spr.NbInternalVariables+spr.GetNbPublicVariables()+spr.GetNbSecretVariables(),
		len(spr.Levels))
	return spr
}

// ---------------- emit-v2 ----------------

// rowMeta is the in-memory representation of a single SparseR1C row plus
// the loc/declIdx metadata determined by the dry-run.
type rowMeta struct {
	bpKind     uint8
	loc        uint8
	declIdx    uint32
	c          constraint.SparseR1C
	commitment constraint.CommitmentConstraint
}

// hintMeta is the in-memory representation of a single hint call.
type hintMeta struct {
	kind     uint8
	hintID   solver.HintID
	inputs   []constraint.LinearExpression
	outStart uint32
	outEnd   uint32
}

// layerEntry holds one layer's rows + hint calls in order.
type layerEntry struct {
	rows  []*rowMeta
	hints []*hintMeta
}

func emitV2(buildDir, outDir string) {
	spr := loadSCS(buildDir)
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		fail("mkdir out: %v", err)
	}

	t0 := time.Now()

	// Build per-layer instruction lists from spr.Levels. Each Level entry
	// is an instruction index. We classify each instruction as either a
	// SparseR1C row (one of 4 blueprints) or a hint call.
	nbLayers := len(spr.Levels)
	layers := make([]*layerEntry, nbLayers)
	for i := range layers {
		layers[i] = &layerEntry{}
	}

	nbWires := spr.NbInternalVariables + spr.GetNbPublicVariables() + spr.GetNbSecretVariables()
	nbInputs := spr.GetNbPublicVariables() + spr.GetNbSecretVariables()
	solved := make([]bool, nbWires)
	for i := 0; i < nbInputs; i++ {
		solved[i] = true
	}

	// Identify blueprints by reflect kind. We support exactly the 4
	// SparseR1C kinds + BlueprintGenericHint.
	bpKinds := classifyBlueprints(spr)

	var hm constraint.HintMapping
	var sr1c constraint.SparseR1C

	declIdx := uint32(0)
	totalRows := 0
	totalHints := 0
	unmappedKinds := map[string]int{}
	locL, locR, locO, locVerify := 0, 0, 0, 0

	for layerID, level := range spr.Levels {
		entry := layers[layerID]
		for _, instIdx := range level {
			pi := spr.Instructions[instIdx]
			inst := pi.Unpack(&spr.System)
			bp := spr.Blueprints[pi.BlueprintID]

			// Hint blueprint? -> record hint call, mark outputs solved.
			if hb, ok := bp.(constraint.BlueprintHint); ok {
				hm.Inputs = hm.Inputs[:0]
				hb.DecompressHint(&hm, inst)
				name := spr.MHintsDependencies[hm.HintID]
				kind, ok := hintNameToKind[name]
				if !ok {
					unmappedKinds[name]++
					kind = 255
				}
				cpInputs := make([]constraint.LinearExpression, len(hm.Inputs))
				for j, le := range hm.Inputs {
					cpInputs[j] = append(constraint.LinearExpression{}, le...)
				}
				entry.hints = append(entry.hints, &hintMeta{
					kind:     kind,
					hintID:   hm.HintID,
					inputs:   cpInputs,
					outStart: hm.OutputRange.Start,
					outEnd:   hm.OutputRange.End,
				})
				for w := hm.OutputRange.Start; w < hm.OutputRange.End; w++ {
					if int(w) < len(solved) {
						solved[w] = true
					}
				}
				totalHints++
				continue
			}

			// SparseR1C blueprint? -> decompress and classify loc.
			sbp, ok := bp.(constraint.BlueprintSparseR1C)
			if !ok {
				fail("instruction %d uses unexpected blueprint %T (id=%d)",
					instIdx, bp, pi.BlueprintID)
			}
			kind, ok := bpKinds[pi.BlueprintID]
			if !ok {
				fail("blueprint id=%d type=%T has no SparseR1C kind mapping",
					pi.BlueprintID, bp)
			}
			sbp.DecompressSparseR1C(&sr1c, inst)

			// Determine loc by checking which of XA/XB/XC is unsolved at
			// this point. SparseR1C wires have IsConstant() == false (no
			// MaxUint32 sentinel); a wire that "doesn't matter" for a
			// blueprint will have its corresponding coeff == 0, but the
			// wire ID is still a real index. We must treat such wires as
			// "solved (no-op)" for loc purposes.
			loc := uint8(0)
			// Generic: any of the 3 may be unsolved.
			// Mul/Add: always solve XC (per blueprint Solve methods).
			// Bool: assertion-only — no wire write.
			switch kind {
			case bpGeneric:
				if sr1c.Commitment == constraint.NOT {
					// Replicate gnark's check order: XA -> XB -> XC.
					if !solved[sr1c.XA] {
						loc = 1
					} else if !solved[sr1c.XB] {
						loc = 2
					} else if !solved[sr1c.XC] {
						loc = 3
					}
				}
			case bpMul, bpAdd:
				// Always sets XC.
				loc = 3
			case bpBool:
				loc = 0
			}

			if loc != 0 {
				var w uint32
				switch loc {
				case 1:
					w = sr1c.XA
				case 2:
					w = sr1c.XB
				case 3:
					w = sr1c.XC
				}
				if int(w) < len(solved) {
					solved[w] = true
				}
			}

			row := &rowMeta{
				bpKind:     kind,
				loc:        loc,
				declIdx:    declIdx,
				c:          sr1c,
				commitment: sr1c.Commitment,
			}
			entry.rows = append(entry.rows, row)
			totalRows++
			declIdx++

			switch loc {
			case 0:
				locVerify++
			case 1:
				locL++
			case 2:
				locR++
			case 3:
				locO++
			}
		}
	}

	if len(unmappedKinds) > 0 {
		for n, c := range unmappedKinds {
			fmt.Fprintf(os.Stderr, "[scs-plan] WARNING: unmapped hint kind %q (%d calls)\n", n, c)
		}
		fail("unmapped hint kinds present; add to hintNameToKind")
	}

	fmt.Fprintf(os.Stderr,
		"[scs-plan] classified %d rows (%d hints) across %d layers in %s\n"+
			"[scs-plan] loc distribution: L=%d R=%d O=%d verify-only=%d\n",
		totalRows, totalHints, nbLayers, time.Since(t0),
		locL, locR, locO, locVerify)

	// ---------- write coeffs.bin ----------
	{
		f, err := os.Create(outDir + "/coeffs.bin")
		if err != nil {
			fail("create coeffs.bin: %v", err)
		}
		bw := bufio.NewWriterSize(f, 1<<20)
		var buf [32]byte
		for i := range spr.Coefficients {
			for limb := 0; limb < 4; limb++ {
				binary.LittleEndian.PutUint64(buf[limb*8:], spr.Coefficients[i][limb])
			}
			bw.Write(buf[:])
		}
		bw.Flush()
		f.Close()
	}

	// ---------- write layers.idx, layers_descs.bin ----------
	{
		idxF, _ := os.Create(outDir + "/layers.idx")
		idxBw := bufio.NewWriterSize(idxF, 1<<20)
		mustU32(idxBw, uint32(nbLayers))

		descF, _ := os.Create(outDir + "/layers_descs.bin")
		descBw := bufio.NewWriterSize(descF, 1<<20)

		var descsOff uint64 = 0
		var hintsOff uint64 = 0
		for _, le := range layers {
			mustU32(idxBw, uint32(len(le.rows)))
			mustU32(idxBw, uint32(len(le.hints)))
			mustU64(idxBw, descsOff)
			mustU64(idxBw, hintsOff)
			for _, r := range le.rows {
				writeDesc(descBw, r)
			}
			descsOff += uint64(len(le.rows))
			hintsOff += uint64(len(le.hints))
		}
		descBw.Flush()
		descF.Close()
		idxBw.Flush()
		idxF.Close()
	}

	// ---------- write layer_kinds.bin (Phase D addition) ----------
	//
	// One u8 per layer classifying its dominant blueprint mix. The Phase D
	// dispatcher reads this to pick the right kernel: the "fast" path for
	// layers that are pure Add/Mul (single-thread-per-row coalesced), and
	// the "slow" warp-cooperative path for layers containing any Generic
	// row (one warp per row, lanes split term loads + Fermat inverse).
	//
	// Code values:
	//   0 = mixed / unknown (defaults to slow path)
	//   1 = Mul-only
	//   2 = Add-only
	//   3 = Mul+Add (no Generic, no Bool)
	//   4 = has-Generic (slow path required)
	//   5 = Bool-only (skip layer at solve time)
	{
		kindsF, _ := os.Create(outDir + "/layer_kinds.bin")
		kindsBw := bufio.NewWriterSize(kindsF, 1<<20)
		mustU32(kindsBw, uint32(nbLayers))
		for _, le := range layers {
			var hasGeneric, hasMul, hasAdd, hasBool, hasOther bool
			for _, r := range le.rows {
				switch r.bpKind {
				case bpGeneric:
					hasGeneric = true
				case bpMul:
					hasMul = true
				case bpAdd:
					hasAdd = true
				case bpBool:
					hasBool = true
				default:
					hasOther = true
				}
			}
			var kind uint8
			switch {
			case hasOther:
				kind = 0
			case hasGeneric:
				// Generic forces the slow path even if mixed with Mul/Add/Bool.
				kind = 4
			case hasMul && !hasAdd && !hasBool:
				kind = 1
			case hasAdd && !hasMul && !hasBool:
				kind = 2
			case (hasMul || hasAdd) && !hasBool:
				kind = 3
			case hasBool && !hasMul && !hasAdd:
				kind = 5
			default:
				// e.g. Bool + Mul/Add — treat as fast (Bool is no-op).
				kind = 3
			}
			must1(kindsBw.WriteByte(kind))
		}
		kindsBw.Flush()
		kindsF.Close()
	}

	// ---------- write hints.idx, layers_hints.bin, hint_in_les.bin ----------
	bsb22ID := solver.GetHintID(fcs.Bsb22CommitmentComputePlaceholder)
	type bsb22Locator struct {
		layerID         uint32
		hintGlobalIndex uint32
	}
	var bsb22Locations []bsb22Locator
	{
		hxF, _ := os.Create(outDir + "/hints.idx")
		hxBw := bufio.NewWriterSize(hxF, 1<<20)
		mustU32(hxBw, uint32(nbLayers))

		chF, _ := os.Create(outDir + "/layers_hints.bin")
		chBw := bufio.NewWriterSize(chF, 1<<20)

		leF, _ := os.Create(outDir + "/hint_in_les.bin")
		leBw := bufio.NewWriterSize(leF, 1<<20)

		var totalCalls uint64 = 0
		var totalLEs uint32 = 0
		for layerID, le := range layers {
			mustU32(hxBw, uint32(len(le.hints)))
			mustU64(hxBw, totalCalls)
			for _, h := range le.hints {
				if h.hintID == bsb22ID {
					bsb22Locations = append(bsb22Locations, bsb22Locator{
						layerID:         uint32(layerID),
						hintGlobalIndex: uint32(totalCalls) + uint32(0),
					})
				}
				must1(chBw.WriteByte(h.kind))
				must1(chBw.WriteByte(0))
				must1(chBw.WriteByte(0))
				must1(chBw.WriteByte(0))
				mustU32(chBw, uint32(len(h.inputs)))
				mustU32(chBw, h.outEnd-h.outStart)
				mustU32(chBw, totalLEs)
				mustU32(chBw, h.outStart)
				mustU32(chBw, uint32(h.hintID))
				for _, expr := range h.inputs {
					mustU32(leBw, uint32(len(expr)))
					for _, t := range expr {
						mustU32(leBw, uint32(t.CoeffID()))
						mustU32(leBw, uint32(t.WireID()))
					}
					totalLEs++
				}
				totalCalls++
			}
		}
		hxBw.Flush()
		hxF.Close()
		chBw.Flush()
		chF.Close()
		leBw.Flush()
		leF.Close()

		fmt.Fprintf(os.Stderr,
			"[scs-plan] %d hint calls, %d input LEs, %d BSB22 commitments\n",
			totalCalls, totalLEs, len(bsb22Locations))
	}

	// ---------- bsb22_meta.bin ----------
	var commInfo constraint.PlonkCommitments
	if pc, ok := spr.CommitmentInfo.(constraint.PlonkCommitments); ok {
		commInfo = pc
	}
	{
		f, _ := os.Create(outDir + "/bsb22_meta.bin")
		bw := bufio.NewWriterSize(f, 1<<20)
		mustU32(bw, uint32(len(commInfo)))
		for i, ci := range commInfo {
			mustU32(bw, uint32(ci.CommitmentIndex))
			if i < len(bsb22Locations) {
				mustU32(bw, bsb22Locations[i].layerID)
				mustU32(bw, bsb22Locations[i].hintGlobalIndex)
			} else {
				// No matching hint located — emit zero (will fail interp).
				mustU32(bw, 0)
				mustU32(bw, 0)
			}
			mustU32(bw, uint32(len(ci.Committed)))
			for _, idx := range ci.Committed {
				mustU32(bw, uint32(idx))
			}
		}
		bw.Flush()
		f.Close()
	}

	// ---------- lro_layout.bin (Phase G addition) ----------
	//
	// Per-constraint (XA, XB, XC) wire-id triple in INSTRUCTION DECLARATION
	// ORDER (NOT topological/declIdx order). gnark's
	// `evaluateLROSmallDomain` walks `spr.Instructions` in declaration
	// order and assigns L[off+j] = solution[XA(j)], so the GPU output
	// MUST follow the same ordering.
	//
	// Header (N, nb_public, nb_constraints) lets the GPU kernel produce
	// the L/R/O vectors in one pass:
	//
	//   for i in [0, nbPublic): L[i] = solution[i], R[i] = O[i] = solution[0]
	//   for j in [0, nbConstraints): L[off+j] = solution[xa[j]], etc.
	//   for i in [off+nbConstraints, N): L[i] = R[i] = O[i] = solution[0]
	//
	// Total bytes ≈ 16 + 12 * nbConstraints. For the production SCS
	// (~27.6 M constraints) that's ~330 MB.
	{
		f, err := os.Create(outDir + "/lro_layout.bin")
		if err != nil {
			fail("create lro_layout.bin: %v", err)
		}
		bw := bufio.NewWriterSize(f, 1<<20)
		nbPublic := uint32(spr.GetNbPublicVariables())
		nbConstraints := uint32(spr.GetNbConstraints())
		N := uint32(ecc.NextPowerOfTwo(uint64(spr.GetNbConstraints() + spr.GetNbPublicVariables())))
		mustU32(bw, N)
		mustU32(bw, nbPublic)
		mustU32(bw, nbConstraints)
		mustU32(bw, 0) // pad

		// Walk spr.Instructions in declaration order — this matches gnark's
		// own evaluateLROSmallDomain loop and produces the same j index.
		var sr1c2 constraint.SparseR1C
		written := uint32(0)
		for _, pi := range spr.Instructions {
			bp := spr.Blueprints[pi.BlueprintID]
			sbp, ok := bp.(constraint.BlueprintSparseR1C)
			if !ok {
				continue
			}
			sbp.DecompressSparseR1C(&sr1c2, pi.Unpack(&spr.System))
			mustU32(bw, sr1c2.XA)
			mustU32(bw, sr1c2.XB)
			mustU32(bw, sr1c2.XC)
			written++
		}
		if written != nbConstraints {
			fail("lro_layout: wrote %d triples, expected nbConstraints %d",
				written, nbConstraints)
		}
		bw.Flush()
		f.Close()
	}

	// ---------- pubinput_layout.bin ----------
	{
		f, _ := os.Create(outDir + "/pubinput_layout.bin")
		bw := bufio.NewWriterSize(f, 1<<20)
		mustU32(bw, uint32(spr.GetNbPublicVariables()))
		mustU32(bw, uint32(len(spr.Public)))
		for _, name := range spr.Public {
			// spr.Public is a []string of public-input names; the wire
			// position is implicit (wires [0..nbPublic) hold the public
			// inputs in declaration order). We emit a stable fingerprint
			// — the index — for every entry.
			_ = name
		}
		// Body: emit sequential indices for now. This format is a stub —
		// future per-name lookup work would extend it. The kernel doesn't
		// yet need this for correctness but Phase H/I may.
		for i := 0; i < len(spr.Public); i++ {
			mustU32(bw, uint32(i))
		}
		bw.Flush()
		f.Close()
	}

	// ---------- circuit_meta.txt ----------
	{
		mf, _ := os.Create(outDir + "/circuit_meta.txt")
		fmt.Fprintf(mf, "n_wires=%d\n", nbWires)
		fmt.Fprintf(mf, "n_public=%d\n", spr.GetNbPublicVariables())
		fmt.Fprintf(mf, "n_secret=%d\n", spr.GetNbSecretVariables())
		fmt.Fprintf(mf, "n_internal=%d\n", spr.NbInternalVariables)
		fmt.Fprintf(mf, "n_inputs=%d\n", nbInputs)
		fmt.Fprintf(mf, "n_coefficients=%d\n", len(spr.Coefficients))
		fmt.Fprintf(mf, "n_layers=%d\n", nbLayers)
		fmt.Fprintf(mf, "n_constraints=%d\n", spr.GetNbConstraints())
		fmt.Fprintf(mf, "n_descs=%d\n", totalRows)
		fmt.Fprintf(mf, "n_hints=%d\n", totalHints)
		fmt.Fprintf(mf, "n_bsb22=%d\n", len(commInfo))
		fmt.Fprintf(mf, "loc_l=%d\n", locL)
		fmt.Fprintf(mf, "loc_r=%d\n", locR)
		fmt.Fprintf(mf, "loc_o=%d\n", locO)
		fmt.Fprintf(mf, "loc_verify=%d\n", locVerify)
		fmt.Fprintf(mf, "production=true\n")
		mf.Close()

		// Mirror into build_dir for fast-path consumers.
		if src, err := os.ReadFile(outDir + "/circuit_meta.txt"); err == nil {
			_ = os.WriteFile(buildDir+"/_scs_circuit_meta.txt", src, 0o644)
		}
	}

	fmt.Fprintf(os.Stderr, "[scs-plan] emit-v2 done in %s; outDir=%s\n",
		time.Since(t0), outDir)
}

// classifyBlueprints inspects the blueprint table and returns a
// blueprintID -> uint8 kind mapping for the four SparseR1C blueprints.
// It does NOT include the BlueprintGenericHint (handled separately).
func classifyBlueprints(spr *cs.SparseR1CS) map[constraint.BlueprintID]uint8 {
	m := make(map[constraint.BlueprintID]uint8)
	for i, b := range spr.Blueprints {
		// Only SparseR1C blueprints; skip hint and any unknown.
		if _, ok := b.(constraint.BlueprintSparseR1C); !ok {
			continue
		}
		tname := reflect.TypeOf(b).String()
		var kind uint8
		switch {
		case containsAny(tname, "BlueprintGenericSparseR1C"):
			kind = bpGeneric
		case containsAny(tname, "BlueprintSparseR1CMul"):
			kind = bpMul
		case containsAny(tname, "BlueprintSparseR1CAdd"):
			kind = bpAdd
		case containsAny(tname, "BlueprintSparseR1CBool"):
			kind = bpBool
		default:
			fail("unknown SparseR1C blueprint type %s (id=%d)", tname, i)
		}
		m[constraint.BlueprintID(i)] = kind
	}
	return m
}

// writeDesc writes a single 44-byte row descriptor.
func writeDesc(w io.Writer, r *rowMeta) {
	must1Wb(w, r.bpKind)
	must1Wb(w, r.loc)
	mustU16(w, 0) // pad
	mustU32(w, r.c.XA)
	mustU32(w, r.c.XB)
	mustU32(w, r.c.XC)
	mustU32(w, r.c.QL)
	mustU32(w, r.c.QR)
	mustU32(w, r.c.QO)
	mustU32(w, r.c.QM)
	mustU32(w, r.c.QC)
	mustU32(w, uint32(r.commitment))
	mustU32(w, r.declIdx)
}

// ---------------- interpret-v2 ----------------

// loadedPlan holds in-memory representations of all binary plan files,
// re-shaped for fast layered interpretation.
type loadedPlan struct {
	coeffs    []fr.Element
	nbLayers  int
	nbWires   int
	nbPublic  int
	nbSecret  int
	nbInternal int
	nbConstraints int
	rows      []*loadedRow      // total rows; layer ranges via layerDescOff
	hints     []*loadedHint     // total hint calls; layer ranges via layerHintOff
	layerDescOff []int          // [nbLayers+1] cumulative
	layerHintOff []int          // [nbLayers+1] cumulative
	bsb22     []*loadedBsb22
	commLayer map[uint32]uint32 // bsb22 commitment_constraint_idx -> layer_id
}

type loadedRow struct {
	bpKind     uint8
	loc        uint8
	xa, xb, xc uint32
	ql, qr, qo, qm, qc uint32
	commitment uint32
	declIdx    uint32
}

type loadedHint struct {
	kind     uint8
	nIn      uint32
	nOut     uint32
	leOff    uint32
	outStart uint32
	hintID   uint32
	// Inputs decoded as a slice of LEs (each LE is a slice of (cid,vid)).
	inputs [][]termCV
}

type termCV struct {
	cid uint32
	vid uint32
}

type loadedBsb22 struct {
	commitmentConstraintIdx uint32
	layerID                 uint32
	hintGlobalIdx           uint32
	committed               []uint32 // committed constraint indexes
}

func loadPlan(planDir string) *loadedPlan {
	p := &loadedPlan{}

	// circuit_meta.txt
	{
		b, err := os.ReadFile(planDir + "/circuit_meta.txt")
		if err != nil {
			fail("read circuit_meta.txt: %v", err)
		}
		p.nbWires = readKey(string(b), "n_wires")
		p.nbPublic = readKey(string(b), "n_public")
		p.nbSecret = readKey(string(b), "n_secret")
		p.nbInternal = readKey(string(b), "n_internal")
		p.nbConstraints = readKey(string(b), "n_constraints")
	}

	// coeffs.bin
	{
		b, err := os.ReadFile(planDir + "/coeffs.bin")
		if err != nil {
			fail("read coeffs.bin: %v", err)
		}
		if len(b)%32 != 0 {
			fail("coeffs.bin size %d not multiple of 32", len(b))
		}
		n := len(b) / 32
		p.coeffs = make([]fr.Element, n)
		for i := 0; i < n; i++ {
			for limb := 0; limb < 4; limb++ {
				p.coeffs[i][limb] = binary.LittleEndian.Uint64(b[i*32+limb*8:])
			}
		}
	}

	// layers.idx + layers_descs.bin
	{
		idxB, err := os.ReadFile(planDir + "/layers.idx")
		if err != nil {
			fail("read layers.idx: %v", err)
		}
		idxR := newByteReader(idxB)
		nbLayers := int(idxR.u32())
		p.nbLayers = nbLayers
		p.layerDescOff = make([]int, nbLayers+1)
		p.layerHintOff = make([]int, nbLayers+1)
		layerNRows := make([]uint32, nbLayers)
		layerNHints := make([]uint32, nbLayers)
		for i := 0; i < nbLayers; i++ {
			layerNRows[i] = idxR.u32()
			layerNHints[i] = idxR.u32()
			_ = idxR.u64() // descs_off (we recompute)
			_ = idxR.u64() // hints_off
		}
		var descsOff, hintsOff int
		for i := 0; i < nbLayers; i++ {
			p.layerDescOff[i] = descsOff
			p.layerHintOff[i] = hintsOff
			descsOff += int(layerNRows[i])
			hintsOff += int(layerNHints[i])
		}
		p.layerDescOff[nbLayers] = descsOff
		p.layerHintOff[nbLayers] = hintsOff

		descB, err := os.ReadFile(planDir + "/layers_descs.bin")
		if err != nil {
			fail("read layers_descs.bin: %v", err)
		}
		const descSize = 1 + 1 + 2 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4
		if len(descB) != descsOff*descSize {
			fail("layers_descs.bin size %d != n_descs*%d (n_descs=%d)",
				len(descB), descSize, descsOff)
		}
		p.rows = make([]*loadedRow, descsOff)
		dr := newByteReader(descB)
		for i := 0; i < descsOff; i++ {
			r := &loadedRow{}
			r.bpKind = dr.u8()
			r.loc = dr.u8()
			_ = dr.u16() // pad
			r.xa = dr.u32()
			r.xb = dr.u32()
			r.xc = dr.u32()
			r.ql = dr.u32()
			r.qr = dr.u32()
			r.qo = dr.u32()
			r.qm = dr.u32()
			r.qc = dr.u32()
			r.commitment = dr.u32()
			r.declIdx = dr.u32()
			p.rows[i] = r
		}
	}

	// hints.idx + layers_hints.bin + hint_in_les.bin
	{
		idxB, err := os.ReadFile(planDir + "/hints.idx")
		if err != nil {
			fail("read hints.idx: %v", err)
		}
		idxR := newByteReader(idxB)
		nbLayers := int(idxR.u32())
		if nbLayers != p.nbLayers {
			fail("hints.idx nbLayers=%d != layers.idx nbLayers=%d", nbLayers, p.nbLayers)
		}
		// We've already filled layerHintOff from layers.idx; just consume.
		for i := 0; i < nbLayers; i++ {
			_ = idxR.u32() // n_hints
			_ = idxR.u64() // calls_off
		}
		nbHints := p.layerHintOff[nbLayers]

		chB, err := os.ReadFile(planDir + "/layers_hints.bin")
		if err != nil {
			fail("read layers_hints.bin: %v", err)
		}
		const hintRecSize = 1 + 3 + 4 + 4 + 4 + 4 + 4
		if len(chB) != nbHints*hintRecSize {
			fail("layers_hints.bin size %d != n_hints*%d (n_hints=%d)",
				len(chB), hintRecSize, nbHints)
		}
		p.hints = make([]*loadedHint, nbHints)
		hr := newByteReader(chB)
		for i := 0; i < nbHints; i++ {
			h := &loadedHint{}
			h.kind = hr.u8()
			_ = hr.u8()
			_ = hr.u8()
			_ = hr.u8()
			h.nIn = hr.u32()
			h.nOut = hr.u32()
			h.leOff = hr.u32()
			h.outStart = hr.u32()
			h.hintID = hr.u32()
			p.hints[i] = h
		}

		leB, err := os.ReadFile(planDir + "/hint_in_les.bin")
		if err != nil {
			fail("read hint_in_les.bin: %v", err)
		}
		// Decode all LEs. We need them indexed by leOff (LE index, not byte
		// offset). First pass: decode flat list.
		ler := newByteReader(leB)
		var allLEs [][]termCV
		for ler.pos < len(leB) {
			cnt := ler.u32()
			le := make([]termCV, cnt)
			for j := uint32(0); j < cnt; j++ {
				le[j] = termCV{cid: ler.u32(), vid: ler.u32()}
			}
			allLEs = append(allLEs, le)
		}
		// Attach to hints.
		for _, h := range p.hints {
			h.inputs = make([][]termCV, h.nIn)
			for j := uint32(0); j < h.nIn; j++ {
				idx := int(h.leOff + j)
				if idx >= len(allLEs) {
					fail("hint LE index %d out of range (have %d)", idx, len(allLEs))
				}
				h.inputs[j] = allLEs[idx]
			}
		}
	}

	// bsb22_meta.bin
	{
		b, err := os.ReadFile(planDir + "/bsb22_meta.bin")
		if err != nil {
			fail("read bsb22_meta.bin: %v", err)
		}
		br := newByteReader(b)
		nb := int(br.u32())
		p.bsb22 = make([]*loadedBsb22, nb)
		p.commLayer = make(map[uint32]uint32, nb)
		for i := 0; i < nb; i++ {
			lb := &loadedBsb22{}
			lb.commitmentConstraintIdx = br.u32()
			lb.layerID = br.u32()
			lb.hintGlobalIdx = br.u32()
			nc := int(br.u32())
			lb.committed = make([]uint32, nc)
			for j := 0; j < nc; j++ {
				lb.committed[j] = br.u32()
			}
			p.bsb22[i] = lb
			p.commLayer[lb.commitmentConstraintIdx] = lb.layerID
		}
	}

	return p
}

// interpretV2 walks the loaded plan layer-by-layer and re-solves all wires.
func interpretV2(buildDir, planDir, witnessPath, outWires string) {
	t0 := time.Now()
	p := loadPlan(planDir)
	tLoad := time.Since(t0)
	fmt.Fprintf(os.Stderr,
		"[scs-plan] loaded plan: layers=%d rows=%d hints=%d bsb22=%d in %s\n",
		p.nbLayers, len(p.rows), len(p.hints), len(p.bsb22), tLoad)

	// Initialize wires.
	wires := make([]fr.Element, p.nbWires)
	solved := make([]bool, p.nbWires)
	witnessVec := loadWitnessAsFrVector(witnessPath)
	expected := p.nbPublic + p.nbSecret
	if len(witnessVec) != expected {
		fail("witness size %d != expected %d", len(witnessVec), expected)
	}
	for i := range witnessVec {
		wires[i] = witnessVec[i]
		solved[i] = true
	}

	// Optional KZG SRS for BSB22 — only loaded if any BSB22 hint is present.
	var pkBn254 *plonk_bn254.ProvingKey
	var nbConstraints int
	if len(p.bsb22) > 0 {
		t1 := time.Now()
		pkFile, err := os.Open(buildDir + "/" + plonkPkPath)
		if err != nil {
			fail("open PK for BSB22: %v", err)
		}
		pk := plonk.NewProvingKey(ecc.BN254)
		bufR := bufio.NewReaderSize(pkFile, 1<<20)
		if _, err := pk.UnsafeReadFrom(bufR); err != nil {
			fail("read PK: %v", err)
		}
		pkFile.Close()
		pkBn254 = pk.(*plonk_bn254.ProvingKey)
		nbConstraints = p.nbConstraints
		fmt.Fprintf(os.Stderr, "[scs-plan] loaded PK for BSB22 in %s\n", time.Since(t1))
	}

	// Domain size for BSB22 committed polynomial.
	domainSize := uint64(0)
	if len(p.bsb22) > 0 {
		domainSize = ecc.NextPowerOfTwo(uint64(p.nbConstraints + p.nbPublic))
	}

	// Pre-load any BSB22 blinding scalars (one file per commitment); if
	// missing, emit fresh randomness from crypto/rand. Each file holds
	// 2 × 32 bytes Fr (Mont).
	bsb22Blindings := make([][2]fr.Element, len(p.bsb22))
	for i := range bsb22Blindings {
		path := fmt.Sprintf("%s/bsb22_blinding_%d.bin", planDir, i)
		if data, err := os.ReadFile(path); err == nil && len(data) == 64 {
			for k := 0; k < 2; k++ {
				for limb := 0; limb < 4; limb++ {
					bsb22Blindings[i][k][limb] = binary.LittleEndian.Uint64(data[k*32+limb*8:])
				}
			}
		} else {
			// Generate via crypto/rand so behaviour matches gnark's
			// SetRandom() in expectation (won't match byte-for-byte unless
			// the same seed is used, which is fine for roundtrip-v2 since
			// we override the BSB22 hint there).
			for k := 0; k < 2; k++ {
				if _, err := bsb22Blindings[i][k].SetRandom(); err != nil {
					fail("rand: %v", err)
				}
			}
		}
	}
	_ = rand.Reader // (Fr.SetRandom uses crypto/rand internally)

	// Precompute coeff fast-path classification.
	t1 := time.Now()
	for layerID := 0; layerID < p.nbLayers; layerID++ {
		// Hint dispatch first (gnark runs them in instruction order, but
		// since hints depend only on prior-layer wires they can be run in
		// any order within a layer).
		hStart := p.layerHintOff[layerID]
		hEnd := p.layerHintOff[layerID+1]
		for hi := hStart; hi < hEnd; hi++ {
			h := p.hints[hi]
			runHint(h, p.coeffs, wires, solved, p, layerID, hi, pkBn254, domainSize,
				bsb22Blindings, nbConstraints)
		}

		dStart := p.layerDescOff[layerID]
		dEnd := p.layerDescOff[layerID+1]
		for di := dStart; di < dEnd; di++ {
			r := p.rows[di]
			runRow(r, p.coeffs, wires, solved)
		}
	}
	tInterp := time.Since(t1)
	fmt.Fprintf(os.Stderr,
		"[scs-plan] interpret-v2: %d layers in %s (load %s, total %s)\n",
		p.nbLayers, tInterp, tLoad, time.Since(t0))

	// Sanity: every wire should be solved.
	unsolved := 0
	for i := range solved {
		if !solved[i] {
			unsolved++
		}
	}
	if unsolved > 0 {
		fmt.Fprintf(os.Stderr, "[scs-plan] WARNING: %d wires unsolved after walk\n", unsolved)
	}

	// Write wire values out.
	{
		f, err := os.Create(outWires)
		if err != nil {
			fail("create %s: %v", outWires, err)
		}
		bw := bufio.NewWriterSize(f, 1<<20)
		var buf [32]byte
		for i := range wires {
			for limb := 0; limb < 4; limb++ {
				binary.LittleEndian.PutUint64(buf[limb*8:], wires[i][limb])
			}
			bw.Write(buf[:])
		}
		bw.Flush()
		f.Close()
	}
	fmt.Fprintf(os.Stderr, "[scs-plan] wrote %d wires to %s\n", len(wires), outWires)
}

// dumpFixturesC is a Phase-C-only helper. It runs the same layered solve
// as interpret-v2 but tracks the source of each wire write (witness, hint,
// BSB22, or SparseR1C row), and emits two files into outDir:
//
//	wires_initial.bin   nbWires × 32 B Fr (Mont); contains witness + hint
//	                    outputs + BSB22 outputs. Wires written by SparseR1C
//	                    rows during the solve are zeroed — the GPU kernel
//	                    will fill them.
//	wires_expected.bin  the full post-solve gold reference (identical to
//	                    interpret-v2's <out_wires>).
//
// These two files mirror the Groth16 prototype harness contract
// (`r1cs_solve_plan prep-full` produces wires_initial + wires_expected).
//
// In Phase C we pre-bake hints + BSB22 on the CPU. Phase D and onwards
// will compute these on the GPU and `wires_initial.bin` will then carry
// only the witness vector.
func dumpFixturesC(buildDir, planDir, witnessPath, outDir string) {
	t0 := time.Now()
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		fail("mkdir out: %v", err)
	}
	p := loadPlan(planDir)
	tLoad := time.Since(t0)
	fmt.Fprintf(os.Stderr,
		"[scs-plan] dump-fixtures-c: loaded plan in %s (layers=%d rows=%d hints=%d bsb22=%d)\n",
		tLoad, p.nbLayers, len(p.rows), len(p.hints), len(p.bsb22))

	// Wire source tags for the post-walk projection.
	const (
		srcUnset   uint8 = 0
		srcWitness uint8 = 1
		srcHint    uint8 = 2 // includes BSB22 outputs
		srcRow     uint8 = 3
	)
	wires := make([]fr.Element, p.nbWires)
	solved := make([]bool, p.nbWires)
	source := make([]uint8, p.nbWires)

	witnessVec := loadWitnessAsFrVector(witnessPath)
	expected := p.nbPublic + p.nbSecret
	if len(witnessVec) != expected {
		fail("witness size %d != expected %d", len(witnessVec), expected)
	}
	for i := range witnessVec {
		wires[i] = witnessVec[i]
		solved[i] = true
		source[i] = srcWitness
	}

	// Optional KZG SRS for BSB22 (same path as interpretV2).
	var pkBn254 *plonk_bn254.ProvingKey
	var nbConstraints int
	if len(p.bsb22) > 0 {
		t1 := time.Now()
		pkFile, err := os.Open(buildDir + "/" + plonkPkPath)
		if err != nil {
			fail("open PK for BSB22: %v", err)
		}
		pk := plonk.NewProvingKey(ecc.BN254)
		bufR := bufio.NewReaderSize(pkFile, 1<<20)
		if _, err := pk.UnsafeReadFrom(bufR); err != nil {
			fail("read PK: %v", err)
		}
		pkFile.Close()
		pkBn254 = pk.(*plonk_bn254.ProvingKey)
		nbConstraints = p.nbConstraints
		fmt.Fprintf(os.Stderr,
			"[scs-plan] dump-fixtures-c: loaded PK for BSB22 in %s\n",
			time.Since(t1))
	}
	domainSize := uint64(0)
	if len(p.bsb22) > 0 {
		domainSize = ecc.NextPowerOfTwo(uint64(p.nbConstraints + p.nbPublic))
	}

	// Optional pre-loaded blindings (same convention as interpret-v2).
	bsb22Blindings := make([][2]fr.Element, len(p.bsb22))
	for i := range bsb22Blindings {
		path := fmt.Sprintf("%s/bsb22_blinding_%d.bin", planDir, i)
		if data, err := os.ReadFile(path); err == nil && len(data) == 64 {
			for k := 0; k < 2; k++ {
				for limb := 0; limb < 4; limb++ {
					bsb22Blindings[i][k][limb] = binary.LittleEndian.Uint64(data[k*32+limb*8:])
				}
			}
		} else {
			for k := 0; k < 2; k++ {
				if _, err := bsb22Blindings[i][k].SetRandom(); err != nil {
					fail("rand: %v", err)
				}
			}
		}
	}

	// Layered walk: hints first, then rows. We tag wire sources as we go.
	t1 := time.Now()
	rowWrites := 0
	hintWrites := 0
	for layerID := 0; layerID < p.nbLayers; layerID++ {
		hStart := p.layerHintOff[layerID]
		hEnd := p.layerHintOff[layerID+1]
		for hi := hStart; hi < hEnd; hi++ {
			h := p.hints[hi]
			runHint(h, p.coeffs, wires, solved, p, layerID, hi, pkBn254, domainSize,
				bsb22Blindings, nbConstraints)
			for w := h.outStart; w < h.outStart+h.nOut; w++ {
				if int(w) < len(source) && source[w] == srcUnset {
					source[w] = srcHint
					hintWrites++
				}
			}
		}
		dStart := p.layerDescOff[layerID]
		dEnd := p.layerDescOff[layerID+1]
		for di := dStart; di < dEnd; di++ {
			r := p.rows[di]
			// Snapshot the wire that will be written, if any.
			var w uint32
			has := true
			switch r.bpKind {
			case bpMul, bpAdd:
				w = r.xc
			case bpGeneric:
				if r.commitment == uint32(constraint.NOT) && r.loc != 0 {
					switch r.loc {
					case 1:
						w = r.xa
					case 2:
						w = r.xb
					case 3:
						w = r.xc
					}
				} else {
					has = false
				}
			default:
				has = false
			}
			runRow(r, p.coeffs, wires, solved)
			if has && int(w) < len(source) && source[w] == srcUnset {
				source[w] = srcRow
				rowWrites++
			}
		}
	}
	tInterp := time.Since(t1)

	// Sanity: every wire should be set after the walk.
	unsolved := 0
	for i := range solved {
		if !solved[i] {
			unsolved++
		}
	}
	if unsolved > 0 {
		fmt.Fprintf(os.Stderr,
			"[scs-plan] dump-fixtures-c: WARNING: %d wires unsolved after walk\n", unsolved)
	}

	// Write wires_expected.bin (the gold reference).
	{
		expPath := outDir + "/wires_expected.bin"
		f, err := os.Create(expPath)
		if err != nil {
			fail("create %s: %v", expPath, err)
		}
		bw := bufio.NewWriterSize(f, 1<<20)
		var buf [32]byte
		for i := range wires {
			for limb := 0; limb < 4; limb++ {
				binary.LittleEndian.PutUint64(buf[limb*8:], wires[i][limb])
			}
			bw.Write(buf[:])
		}
		bw.Flush()
		f.Close()
	}

	// Write wires_initial.bin: same as wires, but with all SparseR1C-derived
	// wires zeroed.
	zeroed := 0
	{
		initPath := outDir + "/wires_initial.bin"
		f, err := os.Create(initPath)
		if err != nil {
			fail("create %s: %v", initPath, err)
		}
		bw := bufio.NewWriterSize(f, 1<<20)
		var zero32 [32]byte
		var buf [32]byte
		for i := range wires {
			if source[i] == srcRow {
				bw.Write(zero32[:])
				zeroed++
			} else {
				for limb := 0; limb < 4; limb++ {
					binary.LittleEndian.PutUint64(buf[limb*8:], wires[i][limb])
				}
				bw.Write(buf[:])
			}
		}
		bw.Flush()
		f.Close()
	}

	fmt.Fprintf(os.Stderr,
		"[scs-plan] dump-fixtures-c: walk %s (load %s, total %s); "+
			"sources: hint=%d row=%d zeroed=%d total=%d\n",
		tInterp, tLoad, time.Since(t0),
		hintWrites, rowWrites, zeroed, p.nbWires)
	fmt.Fprintf(os.Stderr,
		"[scs-plan] dump-fixtures-c: wrote %s/{wires_initial,wires_expected}.bin\n",
		outDir)
}

// runRow re-implements the 4 SparseR1C blueprint Solve methods on a wire
// vector. Mirrors gnark/constraint/blueprint_scs.go exactly.
func runRow(r *loadedRow, coeffs []fr.Element, wires []fr.Element, solved []bool) {
	switch r.bpKind {
	case bpMul:
		// Mul: m0 = QM * wires[XA]; m1 = wires[XB]; XC = m0 * m1.
		var m0, m1, out fr.Element
		computeTerm(&m0, r.qm, r.xa, coeffs, wires)
		m1 = wires[r.xb]
		out.Mul(&m0, &m1)
		wires[r.xc] = out
		solved[r.xc] = true
	case bpAdd:
		// Add: a = QL * wires[XA]; b = QR * wires[XB]; k = QC; XC = a + b + k.
		var a, b, k, sum fr.Element
		computeTerm(&a, r.ql, r.xa, coeffs, wires)
		computeTerm(&b, r.qr, r.xb, coeffs, wires)
		k = coeffs[r.qc]
		sum.Add(&a, &b)
		sum.Add(&sum, &k)
		wires[r.xc] = sum
		solved[r.xc] = true
	case bpBool:
		// Bool: assertion v + (-v*v) == 0. No wire write. Skip in solver.
	case bpGeneric:
		runGeneric(r, coeffs, wires, solved)
	default:
		fail("unknown blueprint kind %d", r.bpKind)
	}
}

// runGeneric mirrors BlueprintGenericSparseR1C.Solve for all 3 loc cases.
func runGeneric(r *loadedRow, coeffs []fr.Element, wires []fr.Element, solved []bool) {
	if r.commitment != uint32(constraint.NOT) {
		// commitment-related row; gnark skips it during solve.
		return
	}
	switch r.loc {
	case 1:
		// Solve XA. den = QM*XB + QL; num = -(QR*XB + QO*XC + QC) / den.
		var den, u1, qmXB, v1, v2, num, sum, qC fr.Element
		u1 = coeffs[r.ql]
		computeTerm(&qmXB, r.qm, r.xb, coeffs, wires)
		den.Add(&qmXB, &u1)
		var ok bool
		denInv, ok2 := safeInverse(&den)
		if !ok2 {
			ok = false
		} else {
			ok = true
		}
		_ = ok
		computeTerm(&v1, r.qr, r.xb, coeffs, wires)
		computeTerm(&v2, r.qo, r.xc, coeffs, wires)
		qC = coeffs[r.qc]
		sum.Add(&v1, &v2)
		sum.Add(&sum, &qC)
		num.Mul(&sum, &denInv)
		num.Neg(&num)
		wires[r.xa] = num
		solved[r.xa] = true
	case 2:
		// Solve XB. den = QM*XA + QR.
		var u2, qmXA, den, v1, v2, qC, sum, num fr.Element
		u2 = coeffs[r.qr]
		computeTerm(&qmXA, r.qm, r.xa, coeffs, wires)
		den.Add(&qmXA, &u2)
		denInv, ok := safeInverse(&den)
		_ = ok
		computeTerm(&v1, r.ql, r.xa, coeffs, wires)
		computeTerm(&v2, r.qo, r.xc, coeffs, wires)
		qC = coeffs[r.qc]
		sum.Add(&v1, &v2)
		sum.Add(&sum, &qC)
		num.Mul(&sum, &denInv)
		num.Neg(&num)
		wires[r.xb] = num
		solved[r.xb] = true
	case 3:
		// Solve XC. o = -((QM*XA*XB) + QL*XA + QR*XB + QC) / QO.
		var l, rv, m0, m1, qC, o, denInv fr.Element
		computeTerm(&l, r.ql, r.xa, coeffs, wires)
		computeTerm(&rv, r.qr, r.xb, coeffs, wires)
		computeTerm(&m0, r.qm, r.xa, coeffs, wires)
		// gnark uses GetValue(CoeffIdOne, XB) — i.e. wires[XB].
		m1 = wires[r.xb]
		qC = coeffs[r.qc]
		o.Mul(&m0, &m1)
		o.Add(&o, &l)
		o.Add(&o, &rv)
		o.Add(&o, &qC)
		var qO fr.Element
		qO = coeffs[r.qo]
		denInv, _ = safeInverse(&qO)
		o.Mul(&o, &denInv)
		o.Neg(&o)
		wires[r.xc] = o
		solved[r.xc] = true
	case 0:
		// All wires solved or assertion-only — skip.
	}
}

// computeTerm: r = coeffs[cid] * wires[vid], with gnark's fast paths.
func computeTerm(out *fr.Element, cid, vid uint32, coeffs []fr.Element, wires []fr.Element) {
	switch cid {
	case 0: // CoeffIdZero
		out.SetZero()
	case 1: // CoeffIdOne
		*out = wires[vid]
	case 2: // CoeffIdTwo
		out.Double(&wires[vid])
	case 3: // CoeffIdMinusOne
		out.Neg(&wires[vid])
	default:
		out.Mul(&coeffs[cid], &wires[vid])
	}
}

// safeInverse returns inv(x) and whether the inversion succeeded (x != 0).
func safeInverse(x *fr.Element) (fr.Element, bool) {
	var inv fr.Element
	if x.IsZero() {
		return inv, false
	}
	inv.Inverse(x)
	return inv, true
}

// runHint dispatches on hint kind. For non-BSB22 hints, we look up the
// hint function from gnark's solver registry (matching what gnark would
// do internally) — this is portable and exact.
//
// For BSB22, we run the gnark-equivalent customBsb22Hint inline,
// substituting our pre-loaded blinding scalars when available.
func runHint(h *loadedHint, coeffs []fr.Element, wires []fr.Element,
	solved []bool, p *loadedPlan, layerID, hintGlobalIdx int,
	pkBn254 *plonk_bn254.ProvingKey, domainSize uint64,
	bsb22Blindings [][2]fr.Element, nbConstraints int,
) {
	// Evaluate inputs into big.Ints. Hint LE terms may be constants,
	// signalled by vid == MaxUint32 (constraint.Term.IsConstant); for
	// those we add coefficient[cid] directly.
	inputs := make([]*big.Int, h.nIn)
	for i := uint32(0); i < h.nIn; i++ {
		var acc fr.Element
		for _, t := range h.inputs[i] {
			if t.vid == math.MaxUint32 {
				// constant term: acc += coeffs[cid]
				acc.Add(&acc, &coeffs[t.cid])
				continue
			}
			if !solved[t.vid] {
				fail("hint (kind=%d hint_id=%d) input %d references unsolved wire %d",
					h.kind, h.hintID, i, t.vid)
			}
			var tmp fr.Element
			computeTerm(&tmp, t.cid, t.vid, coeffs, wires)
			acc.Add(&acc, &tmp)
		}
		inputs[i] = new(big.Int)
		acc.BigInt(inputs[i])
	}

	// BSB22? Special path that needs PK + domainSize + nbConstraints.
	if h.kind == hintKindBsb22 {
		if pkBn254 == nil {
			fail("BSB22 hint encountered but PK not loaded")
		}
		// Locate which BSB22 commitment this is by looking up via layer
		// + hintGlobalIdx in p.bsb22.
		idx := -1
		for i, lb := range p.bsb22 {
			if int(lb.layerID) == layerID && int(lb.hintGlobalIdx) == hintGlobalIdx {
				idx = i
				break
			}
		}
		if idx < 0 {
			fail("BSB22 hint at layer=%d global=%d not found in bsb22_meta",
				layerID, hintGlobalIdx)
		}
		nOut := int(h.nOut)
		outs := make([]*big.Int, nOut)
		for i := range outs {
			outs[i] = new(big.Int)
		}
		if err := customBsb22Hint(p, idx, inputs, outs, pkBn254, domainSize,
			bsb22Blindings, nbConstraints); err != nil {
			fail("BSB22 hint: %v", err)
		}
		for i, o := range outs {
			w := h.outStart + uint32(i)
			wires[w].SetBigInt(o)
			solved[w] = true
		}
		return
	}

	// Generic path: dispatch via gnark's hint registry. This is portable
	// — every hint kind in our census has a registered function.
	fn := solver.GetRegisteredHint(solver.HintID(h.hintID))
	if fn == nil {
		fail("hint id=%d (kind=%d) not registered in gnark solver", h.hintID, h.kind)
	}
	nOut := int(h.nOut)
	outs := make([]*big.Int, nOut)
	for i := range outs {
		outs[i] = new(big.Int)
	}
	if err := fn(fr.Modulus(), inputs, outs); err != nil {
		fail("hint id=%d failed: %v", h.hintID, err)
	}
	for i, o := range outs {
		w := h.outStart + uint32(i)
		wires[w].SetBigInt(o)
		solved[w] = true
	}
}

// customBsb22Hint mirrors gnark's bsb22Hint (prove.go:280-315) and the
// SP1 export_solved_witness.go variant. Replaces gnark's SetRandom()
// blindings with our pre-loaded values when available so the hint is
// deterministic across runs.
func customBsb22Hint(p *loadedPlan, bsbIdx int, ins, outs []*big.Int,
	pk *plonk_bn254.ProvingKey, domainSize uint64,
	blindings [][2]fr.Element, nbConstraints int,
) error {
	commDepth := int(ins[0].Int64())
	ins = ins[1:]
	if commDepth != bsbIdx {
		// The first input from gnark is the commitment depth — same as our
		// bsbIdx (we maintain order). Just sanity-check.
		fmt.Fprintf(os.Stderr,
			"[scs-plan] WARN: BSB22 commDepth=%d != located bsbIdx=%d\n",
			commDepth, bsbIdx)
	}
	lb := p.bsb22[commDepth]

	committedValues := make([]fr.Element, domainSize)
	offset := p.nbPublic
	for i := range ins {
		var v fr.Element
		v.SetBigInt(ins[i])
		committedValues[offset+int(lb.committed[i])] = v
	}
	committedValues[offset+int(lb.commitmentConstraintIdx)] = blindings[commDepth][0]
	committedValues[offset+nbConstraints-1] = blindings[commDepth][1]

	digest, err := kzg.Commit(committedValues, pk.KzgLagrange)
	if err != nil {
		return fmt.Errorf("BSB22 KZG commit: %w", err)
	}

	// Phase F: optionally dump the BSB22 commitment digest to a file so
	// the GPU prototype can use it as a precomputed bypass when its own
	// MSM linkage misbehaves.
	if dumpPath := os.Getenv("SCS_DUMP_BSB22_COMMIT_PATH"); dumpPath != "" {
		marshaled := digest.Marshal()
		path := fmt.Sprintf("%s/bsb22_commitment_%d.bin", dumpPath, bsbIdx)
		if err := os.WriteFile(path, marshaled, 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "[scs-plan] WARN: write %s: %v\n", path, err)
		} else {
			fmt.Fprintf(os.Stderr,
				"[scs-plan] dumped BSB22 commit %d → %s (%d B)\n",
				bsbIdx, path, len(marshaled))
		}
	}

	htf := hash_to_field.New([]byte("BSB22-Plonk"))
	htf.Write(digest.Marshal())
	hashBts := htf.Sum(nil)
	htf.Reset()

	nbBuf := fr.Bytes
	if htf.Size() < fr.Bytes {
		nbBuf = htf.Size()
	}
	var res fr.Element
	res.SetBytes(hashBts[:nbBuf])
	res.BigInt(outs[0])

	// Stash digest back so the verifier can use it, but for roundtrip-v2
	// we don't need to surface it.
	_ = bn254.G1Affine(digest)
	return nil
}

// ---------------- roundtrip-v2 ----------------

func roundtripV2(buildDir, witnessPath string) {
	tmpDir, err := os.MkdirTemp("", "scs-plan-rt-*")
	if err != nil {
		fail("mkdir tmp: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// 1) Emit plan.
	t0 := time.Now()
	emitV2(buildDir, tmpDir)
	tEmit := time.Since(t0)

	// Compute plan size (sum of files written).
	planSize := dirSizeBytes(tmpDir)
	fmt.Fprintf(os.Stderr,
		"[scs-plan] plan emitted in %s, total size %.1f MB\n",
		tEmit, float64(planSize)/(1024*1024))

	// 2) Run gnark with the same BSB22 override so the blindings used
	//    match what we'll feed our interpreter. Capture them.
	spr := loadSCS(buildDir)
	t0 = time.Now()
	wsr1c, capturedBlindings, lRef, rRef, oRef := gnarkSolveAndCapture(spr, witnessPath)
	tGnark := time.Since(t0)
	_ = wsr1c
	fmt.Fprintf(os.Stderr,
		"[scs-plan] gnark spr.Solve in %s (L=%d R=%d O=%d, captured %d BSB22 blindings)\n",
		tGnark, len(lRef), len(rRef), len(oRef), len(capturedBlindings))

	// 3) Drop captured blindings to disk so interpret-v2 picks them up.
	for i, pair := range capturedBlindings {
		path := fmt.Sprintf("%s/bsb22_blinding_%d.bin", tmpDir, i)
		var buf [64]byte
		for k := 0; k < 2; k++ {
			for limb := 0; limb < 4; limb++ {
				binary.LittleEndian.PutUint64(buf[k*32+limb*8:], pair[k][limb])
			}
		}
		_ = os.WriteFile(path, buf[:], 0o644)
	}

	// 4) Interpret.
	outWires := tmpDir + "/wires_out.bin"
	t0 = time.Now()
	interpretV2(buildDir, tmpDir, witnessPath, outWires)
	tInterp := time.Since(t0)

	// 5) Re-derive L/R/O from solved wires using the same logic gnark uses
	//    (evaluateLROSmallDomain). This is the byte-comparable artifact.
	mineWires := readFrFile(outWires)
	lMine, rMine, oMine := evaluateLROSmallDomain(spr, mineWires)

	// 6) Compare byte-for-byte.
	mismatch := func(name string, mine, ref []fr.Element) bool {
		if len(mine) != len(ref) {
			fmt.Fprintf(os.Stderr, "[scs-plan] FAIL: %s len mismatch mine=%d ref=%d\n",
				name, len(mine), len(ref))
			return true
		}
		first := -1
		count := 0
		for i := range mine {
			if !mine[i].Equal(&ref[i]) {
				if first == -1 {
					first = i
				}
				count++
			}
		}
		if count > 0 {
			fmt.Fprintf(os.Stderr,
				"[scs-plan] FAIL: %s has %d mismatches; first at %d:\n  mine = %s\n  ref  = %s\n",
				name, count, first, mine[first].String(), ref[first].String())
			return true
		}
		return false
	}
	bad := false
	bad = mismatch("L", lMine, lRef) || bad
	bad = mismatch("R", rMine, rRef) || bad
	bad = mismatch("O", oMine, oRef) || bad

	fmt.Fprintf(os.Stderr,
		"[scs-plan] timings: emit=%s gnark=%s interpret=%s plan_size=%.1f MB\n",
		tEmit, tGnark, tInterp, float64(planSize)/(1024*1024))

	if bad {
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr,
		"[scs-plan] PASS — L/R/O match byte-for-byte (%d × 3 elements)\n",
		len(lMine))
}

// gnarkSolveAndCapture runs spr.Solve with a custom BSB22 hint that:
//   1. Captures the random blindings gnark generates.
//   2. Returns them so we can replay them in our interpreter.
//
// Returns (solution, blindings, L, R, O).
func gnarkSolveAndCapture(spr *cs.SparseR1CS, witnessPath string,
) (*cs.SparseR1CSSolution, [][2]fr.Element, []fr.Element, []fr.Element, []fr.Element) {
	return gnarkSolveAndCaptureWithBlindings(spr, witnessPath, nil, nil)
}

// gnarkSolveAndCaptureWithBlindings is the workhorse: if `pre[i]` is
// non-nil and `valid[i]` is true, it pins the BSB22 blindings for
// commitment i to the supplied values; otherwise it draws fresh
// crypto/rand blindings.
func gnarkSolveAndCaptureWithBlindings(
	spr *cs.SparseR1CS, witnessPath string,
	pre [][2]fr.Element, valid []bool,
) (*cs.SparseR1CSSolution, [][2]fr.Element, []fr.Element, []fr.Element, []fr.Element) {
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

	commitmentInfo, _ := spr.CommitmentInfo.(constraint.PlonkCommitments)
	nbCommitments := len(commitmentInfo)
	captured := make([][2]fr.Element, nbCommitments)

	// Load PK so we can KZG.commit (matches what gnark's prover does).
	// We need this because the captured blinding scalars can come from
	// gnark's own SetRandom() — which we override below.
	pkFile, err := os.Open(scsBuildDirOf(witnessPath) + "/" + plonkPkPath)
	if err == nil {
		// fall through — only used by export_solved_witness for export;
		// here we just need the BSB22 hint to run successfully so gnark's
		// L/R/O is well-defined.
	}
	defer func() {
		if pkFile != nil {
			pkFile.Close()
		}
	}()

	// Domain size + nbPublic from spr.
	domainSize := ecc.NextPowerOfTwo(uint64(spr.GetNbConstraints() + len(spr.Public)))
	nbPublic := spr.GetNbPublicVariables()

	// Replicate gnark's bsb22Hint but capture blindings. We need pkBn254
	// for kzg.Commit; load it just-in-time.
	buildDir := scsBuildDirOf(witnessPath)
	pkObj := plonk.NewProvingKey(ecc.BN254)
	{
		pf, err := os.Open(buildDir + "/" + plonkPkPath)
		if err != nil {
			fail("open PK for capture: %v", err)
		}
		bufR := bufio.NewReaderSize(pf, 1<<20)
		if _, err := pkObj.UnsafeReadFrom(bufR); err != nil {
			fail("read PK: %v", err)
		}
		pf.Close()
	}
	pkBn254 := pkObj.(*plonk_bn254.ProvingKey)

	captureBsb22 := func(_ *big.Int, ins, outs []*big.Int) error {
		commDepth := int(ins[0].Int64())
		ins = ins[1:]
		ci := commitmentInfo[commDepth]
		committedValues := make([]fr.Element, domainSize)
		offset := nbPublic
		for i := range ins {
			committedValues[offset+ci.Committed[i]].SetBigInt(ins[i])
		}
		// Generate two blindings via fr.SetRandom() OR pin them from the
		// caller-supplied `pre` slice (used by Phase G's
		// dump-lro-reference to deterministically match an existing
		// fixture).
		if pre != nil && commDepth < len(valid) && valid[commDepth] {
			captured[commDepth][0] = pre[commDepth][0]
			captured[commDepth][1] = pre[commDepth][1]
		} else {
			if _, err := captured[commDepth][0].SetRandom(); err != nil {
				return err
			}
			if _, err := captured[commDepth][1].SetRandom(); err != nil {
				return err
			}
		}
		committedValues[offset+ci.CommitmentIndex] = captured[commDepth][0]
		committedValues[offset+spr.GetNbConstraints()-1] = captured[commDepth][1]

		digest, err := kzg.Commit(committedValues, pkBn254.KzgLagrange)
		if err != nil {
			return err
		}
		htf := hash_to_field.New([]byte("BSB22-Plonk"))
		htf.Write(digest.Marshal())
		hashBts := htf.Sum(nil)
		htf.Reset()
		nbBuf := fr.Bytes
		if htf.Size() < fr.Bytes {
			nbBuf = htf.Size()
		}
		var res fr.Element
		res.SetBytes(hashBts[:nbBuf])
		res.BigInt(outs[0])
		return nil
	}

	bsb22ID := solver.GetHintID(fcs.Bsb22CommitmentComputePlaceholder)
	sol, err := spr.Solve(witness, solver.OverrideHint(bsb22ID, captureBsb22))
	if err != nil {
		fail("gnark spr.Solve: %v", err)
	}
	rsol := sol.(*cs.SparseR1CSSolution)
	return rsol, captured, []fr.Element(rsol.L), []fr.Element(rsol.R), []fr.Element(rsol.O)
}

// scsBuildDirOf returns the build dir of an SCS — given the witness path,
// uses the heuristic that build_dir is the witness's containing directory.
// This works for the production layout ~/.sp1/circuits/plonk/v6.0.0/.
func scsBuildDirOf(witnessPath string) string {
	for i := len(witnessPath) - 1; i >= 0; i-- {
		if witnessPath[i] == '/' {
			return witnessPath[:i]
		}
	}
	return "."
}

// dumpLroReference runs gnark spr.Solve() (with the same BSB22 capture
// override as roundtrip-v2) on the production circuit + witness, then runs
// `evaluateLROSmallDomain` and dumps the three L/R/O vectors to disk in
// the same Mont-LE wire format as wires_initial.bin. This is the byte
// reference Phase G's GPU kernel must reproduce.
//
// If <plan_dir> contains existing `bsb22_blinding_<i>.bin` files, they are
// passed to gnark via the BSB22 override hint so the LRO ref deterministically
// matches an existing wires_initial.bin / wires_expected.bin fixture from
// `dump-fixtures-c`. Otherwise fresh crypto/rand blindings are drawn AND
// written back into <plan_dir> so subsequent fixture/lro emissions stay in
// sync.
func dumpLroReference(buildDir, planDir, witnessPath, outDir string) {
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		fail("mkdir out: %v", err)
	}
	t0 := time.Now()
	spr := loadSCS(buildDir)
	tLoad := time.Since(t0)

	// Look up CommitmentInfo to size the blinding slice; load/seed
	// per-commitment blindings.
	commInfo, _ := spr.CommitmentInfo.(constraint.PlonkCommitments)
	nbCommitments := len(commInfo)
	preBlindings := make([][2]fr.Element, nbCommitments)
	preBlindingsValid := make([]bool, nbCommitments)
	for i := 0; i < nbCommitments; i++ {
		path := fmt.Sprintf("%s/bsb22_blinding_%d.bin", planDir, i)
		data, err := os.ReadFile(path)
		if err == nil && len(data) == 64 {
			for k := 0; k < 2; k++ {
				for limb := 0; limb < 4; limb++ {
					preBlindings[i][k][limb] = binary.LittleEndian.Uint64(data[k*32+limb*8:])
				}
			}
			preBlindingsValid[i] = true
		}
	}

	t0 = time.Now()
	_, captured, lRef, rRef, oRef := gnarkSolveAndCaptureWithBlindings(
		spr, witnessPath, preBlindings, preBlindingsValid)
	tSolve := time.Since(t0)

	// If any blindings were freshly generated, persist them back into the
	// plan dir so downstream fixture emissions can match.
	for i := 0; i < nbCommitments; i++ {
		if preBlindingsValid[i] {
			continue
		}
		path := fmt.Sprintf("%s/bsb22_blinding_%d.bin", planDir, i)
		var buf [64]byte
		for k := 0; k < 2; k++ {
			for limb := 0; limb < 4; limb++ {
				binary.LittleEndian.PutUint64(buf[k*32+limb*8:], captured[i][k][limb])
			}
		}
		_ = os.WriteFile(path, buf[:], 0o644)
		fmt.Fprintf(os.Stderr,
			"[scs-plan] dump-lro-reference: wrote fresh blinding to %s\n", path)
	}

	writeFr := func(path string, v []fr.Element) {
		f, err := os.Create(path)
		if err != nil {
			fail("create %s: %v", path, err)
		}
		bw := bufio.NewWriterSize(f, 1<<20)
		var buf [32]byte
		for i := range v {
			for limb := 0; limb < 4; limb++ {
				binary.LittleEndian.PutUint64(buf[limb*8:], v[i][limb])
			}
			bw.Write(buf[:])
		}
		bw.Flush()
		f.Close()
	}

	writeFr(outDir+"/lro_l_ref.bin", lRef)
	writeFr(outDir+"/lro_r_ref.bin", rRef)
	writeFr(outDir+"/lro_o_ref.bin", oRef)

	fmt.Fprintf(os.Stderr,
		"[scs-plan] dump-lro-reference: load %s, solve %s; "+
			"wrote %s/{lro_l_ref,lro_r_ref,lro_o_ref}.bin (N=%d each)\n",
		tLoad, tSolve, outDir, len(lRef))
}

// evaluateLROSmallDomain mirrors gnark's evaluateLROSmallDomain
// (constraint/bn254/system.go:163) bit-for-bit. We re-implement here
// because the gnark function is unexported.
func evaluateLROSmallDomain(spr *cs.SparseR1CS, solution []fr.Element) (
	[]fr.Element, []fr.Element, []fr.Element,
) {
	s := spr.GetNbConstraints() + len(spr.Public)
	s = int(ecc.NextPowerOfTwo(uint64(s)))

	l := make([]fr.Element, s, s+4)
	r := make([]fr.Element, s, s+4)
	o := make([]fr.Element, s, s+4)
	s0 := solution[0]

	for i := 0; i < len(spr.Public); i++ {
		l[i] = solution[i]
		r[i] = s0
		o[i] = s0
	}
	offset := len(spr.Public)
	nbConstraints := spr.GetNbConstraints()

	var sr1c constraint.SparseR1C
	j := 0
	for _, inst := range spr.Instructions {
		blueprint := spr.Blueprints[inst.BlueprintID]
		if bc, ok := blueprint.(constraint.BlueprintSparseR1C); ok {
			bc.DecompressSparseR1C(&sr1c, inst.Unpack(&spr.System))
			l[offset+j] = solution[sr1c.XA]
			r[offset+j] = solution[sr1c.XB]
			o[offset+j] = solution[sr1c.XC]
			j++
		}
	}

	offset += nbConstraints
	for i := 0; i < s-offset; i++ {
		l[offset+i] = s0
		r[offset+i] = s0
		o[offset+i] = s0
	}
	return l, r, o
}

// ---------------- helpers ----------------

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

func readFrFile(path string) []fr.Element {
	b, err := os.ReadFile(path)
	if err != nil {
		fail("read %s: %v", path, err)
	}
	if len(b)%32 != 0 {
		fail("%s size %d not multiple of 32", path, len(b))
	}
	n := len(b) / 32
	out := make([]fr.Element, n)
	for i := 0; i < n; i++ {
		for limb := 0; limb < 4; limb++ {
			out[i][limb] = binary.LittleEndian.Uint64(b[i*32+limb*8:])
		}
	}
	return out
}

func dirSizeBytes(dir string) int64 {
	var total int64
	entries, err := os.ReadDir(dir)
	if err != nil {
		return 0
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		fi, err := e.Info()
		if err != nil {
			continue
		}
		total += fi.Size()
	}
	return total
}

func readKey(text, key string) int {
	for i := 0; i < len(text); i++ {
		// find start of line
		ls := i
		for i < len(text) && text[i] != '\n' {
			i++
		}
		line := text[ls:i]
		// look for key=
		if len(line) <= len(key)+1 {
			continue
		}
		if line[:len(key)] == key && line[len(key)] == '=' {
			var v int
			fmt.Sscanf(line[len(key)+1:], "%d", &v)
			return v
		}
	}
	return 0
}

// byteReader: a tiny helper for sequential decoding.
type byteReader struct {
	b   []byte
	pos int
}

func newByteReader(b []byte) *byteReader { return &byteReader{b: b, pos: 0} }
func (r *byteReader) u8() uint8 {
	v := r.b[r.pos]
	r.pos++
	return v
}
func (r *byteReader) u16() uint16 {
	v := binary.LittleEndian.Uint16(r.b[r.pos:])
	r.pos += 2
	return v
}
func (r *byteReader) u32() uint32 {
	v := binary.LittleEndian.Uint32(r.b[r.pos:])
	r.pos += 4
	return v
}
func (r *byteReader) u64() uint64 {
	v := binary.LittleEndian.Uint64(r.b[r.pos:])
	r.pos += 8
	return v
}

func mustU16(w io.Writer, v uint16) {
	var buf [2]byte
	binary.LittleEndian.PutUint16(buf[:], v)
	if _, err := w.Write(buf[:]); err != nil {
		fail("write u16: %v", err)
	}
}

func mustU32(w io.Writer, v uint32) {
	var buf [4]byte
	binary.LittleEndian.PutUint32(buf[:], v)
	if _, err := w.Write(buf[:]); err != nil {
		fail("write u32: %v", err)
	}
}

func mustU64(w io.Writer, v uint64) {
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], v)
	if _, err := w.Write(buf[:]); err != nil {
		fail("write u64: %v", err)
	}
}

func must1(err error) {
	if err != nil {
		fail("write byte: %v", err)
	}
}

// must1Wb writes a single byte to a generic io.Writer. (bufio.Writer
// has its own WriteByte method but we want to support raw writers too.)
func must1Wb(w io.Writer, b uint8) {
	var buf [1]byte
	buf[0] = b
	if _, err := w.Write(buf[:]); err != nil {
		fail("write byte: %v", err)
	}
}

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

// ---------------- Phase F: dump-bsb22-seed ----------------
//
// Produces the per-prove BSB22 PRF seed and the matching blinding scalars.
// Also emits the per-input wire-id list (gather indices) the GPU sub-pipeline
// will use to assemble committedValues from the wires array.
//
// The PRF is intentionally simple so both the Go side (here) and the GPU
// kernel (full_solve_phasef.{cu,hip.cu}) compute identical blindings:
//
//     blinding[k] = fr.SetBytes(SHA-256(seed || k_byte))   for k in {0,1}
//
// fr.SetBytes interprets 32 BE bytes as an integer in [0, 2^256) and
// returns its image mod r — the same reduction we will run on GPU.
func dumpBsb22Seed(planDir, seedPathArg, outBlindDir string) {
	if err := os.MkdirAll(outBlindDir, 0o755); err != nil {
		fail("mkdir blinding-dir: %v", err)
	}

	// Resolve / generate the seed.
	var seed [32]byte
	if seedPathArg == "RANDOM" {
		if _, err := rand.Read(seed[:]); err != nil {
			fail("rand: %v", err)
		}
		// Persist the freshly generated seed to bsb22_seed.bin in the
		// blinding dir so the GPU kernel and the Go reference share input.
		seedPath := outBlindDir + "/bsb22_seed.bin"
		if err := os.WriteFile(seedPath, seed[:], 0o644); err != nil {
			fail("write seed: %v", err)
		}
		fmt.Fprintf(os.Stderr,
			"[scs-plan] dump-bsb22-seed: generated random seed → %s\n", seedPath)
	} else {
		data, err := os.ReadFile(seedPathArg)
		if err != nil {
			fail("read seed: %v", err)
		}
		if len(data) != 32 {
			fail("seed file %s: want 32 bytes, got %d", seedPathArg, len(data))
		}
		copy(seed[:], data)
	}

	// Load BSB22 metadata + walk hint LEs to materialise per-input wire IDs.
	p := loadPlan(planDir)
	if len(p.bsb22) == 0 {
		fmt.Fprintf(os.Stderr,
			"[scs-plan] dump-bsb22-seed: no BSB22 commitments — nothing to do\n")
		return
	}

	// Locate each BSB22 hint by (layerID, hintGlobalIdx) → walk its inputs.
	// Hint inputs for BSB22 are [depth, committed_value_0, committed_value_1, ...].
	// We skip the depth input and emit the wire IDs for the rest.
	for i, lb := range p.bsb22 {
		// Find the hint record matching this BSB22.
		hintGlob := int(lb.hintGlobalIdx)
		if hintGlob >= len(p.hints) {
			fail("bsb22[%d]: hint global idx %d out of range (%d hints)",
				i, hintGlob, len(p.hints))
		}
		h := p.hints[hintGlob]
		if int(h.nIn) < 1 {
			fail("bsb22[%d]: hint has no inputs", i)
		}
		// Skip the depth input. Each remaining input must be a single-LE
		// term {cid=1, vid=W}; we emit W for each.
		nCommitted := int(h.nIn) - 1
		if nCommitted != int(lb.nCommitted()) {
			fail("bsb22[%d]: input count %d != metadata committed count %d",
				i, nCommitted, lb.nCommitted())
		}
		// Each BSB22 input LE in the production circuit is single-term
		// (verified empirically — see Phase F notes). The term may use a
		// non-identity coefficient (cid=78 was seen for ~50% of inputs in
		// the production fixture). We emit (cid, vid) pairs so the GPU
		// gather kernel can compute coeffs[cid] * wires[vid] per input.
		wirePath := fmt.Sprintf("%s/bsb22_input_terms_%d.bin", outBlindDir, i)
		{
			f, err := os.Create(wirePath)
			if err != nil {
				fail("create %s: %v", wirePath, err)
			}
			bw := bufio.NewWriterSize(f, 1<<20)
			for j := 0; j < nCommitted; j++ {
				le := h.inputs[1+j]
				if len(le) != 1 {
					fail("bsb22[%d] input %d: expected single-term LE, got %d terms",
						i, j, len(le))
				}
				t := le[0]
				mustU32(bw, t.cid)
				mustU32(bw, t.vid)
			}
			bw.Flush()
			f.Close()
		}

		// Derive the 2 blinding scalars from the seed via SHA-256(seed || k).
		// fr.SetBytes interprets 32 BE bytes mod r — identical to the GPU
		// kernel's reduction.
		var blindings [2]fr.Element
		for k := 0; k < 2; k++ {
			hh := newSha256()
			hh.Write(seed[:])
			hh.Write([]byte{byte(k)})
			out := hh.Sum(nil)
			blindings[k].SetBytes(out)
		}
		blindPath := fmt.Sprintf("%s/bsb22_blinding_%d.bin", outBlindDir, i)
		{
			var buf [64]byte
			for k := 0; k < 2; k++ {
				for limb := 0; limb < 4; limb++ {
					binary.LittleEndian.PutUint64(buf[k*32+limb*8:], blindings[k][limb])
				}
			}
			if err := os.WriteFile(blindPath, buf[:], 0o644); err != nil {
				fail("write %s: %v", blindPath, err)
			}
		}
		// Compact per-bsb22 metadata the GPU sub-pipeline needs:
		//   u32 layer_id
		//   u32 hint_output_wire_id     (h.outStart, where the hash result lands)
		//   u32 commitment_constraint_idx
		//   u32 nb_constraints           (last-slot index = nbConstraints-1)
		//   u32 nb_public                (offset for committedValues[])
		//   u32 n_committed
		//   u32 domain_size              (next power of 2 of nbConstraints + nbPublic)
		//   u32 _pad
		metaPath := fmt.Sprintf("%s/bsb22_solve_meta_%d.bin", outBlindDir, i)
		{
			domainSize := nextPow2U32(uint32(p.nbConstraints + p.nbPublic))
			f, err := os.Create(metaPath)
			if err != nil {
				fail("create %s: %v", metaPath, err)
			}
			bw := bufio.NewWriterSize(f, 1<<20)
			mustU32(bw, lb.layerID)
			mustU32(bw, h.outStart)
			mustU32(bw, lb.commitmentConstraintIdx)
			mustU32(bw, uint32(p.nbConstraints))
			mustU32(bw, uint32(p.nbPublic))
			mustU32(bw, uint32(nCommitted))
			mustU32(bw, domainSize)
			mustU32(bw, 0)
			bw.Flush()
			f.Close()
		}

		fmt.Fprintf(os.Stderr,
			"[scs-plan] dump-bsb22-seed: bsb22[%d] n_committed=%d (cmt_idx=%d, last_idx=%d, layer=%d, out_wire=%d) → %s, %s, %s\n",
			i, nCommitted, lb.commitmentConstraintIdx, p.nbConstraints-1,
			lb.layerID, h.outStart, wirePath, blindPath, metaPath)
	}
}

// nCommitted is a small accessor; loadedBsb22's field is unexported to
// the helper package but we read it via this helper to keep the rest of
// dumpBsb22Seed tidy.
func (lb *loadedBsb22) nCommitted() int { return len(lb.committed) }

// nextPow2U32 returns the smallest power of 2 >= n. Mirrors
// gnark-crypto's ecc.NextPowerOfTwo for u32 inputs.
func nextPow2U32(n uint32) uint32 {
	if n == 0 {
		return 1
	}
	p := uint32(1)
	for p < n {
		p <<= 1
	}
	return p
}

// newSha256 is split out so the import diff for the new subcommand stays
// localised; the standard library SHA-256 is matched bit-for-bit by the
// GPU kernel (full_solve_phasef sha256.cuh).
func newSha256() hash.Hash { return sha256.New() }

func fail(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "[scs-plan] "+format+"\n", args...)
	os.Exit(2)
}

