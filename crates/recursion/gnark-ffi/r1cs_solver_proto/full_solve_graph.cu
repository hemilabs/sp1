// Phase 5 prototype: full layered solve using HIP Graphs to amortize
// the 135K per-launch overhead from the Phase 4 driver.
//
// Captures the entire per-layer kernel launch sequence into a HIP graph
// once at startup, then launches the graph in a single hipGraphLaunch.
// Per the spike (and confirmed by the Phase 4 measurement) the bare
// per-launch overhead on RDNA3 is ~28 µs — multiplied by 135K layers
// that's nearly 4 s of pure dispatch overhead. Graphs should drop this
// dramatically.
//
// Build:
//   hipcc -std=c++20 -O3 -I .../sys/include --offload-arch=gfx1100 \
//         -DUSE_HIP full_solve_graph.cu -o build/full_solve_graph

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>
#include <hip/hip_runtime.h>

#include "fields/bn254_t.cuh"

struct R1CDesc {
    uint32_t L_off, L_cnt;
    uint32_t R_off, R_cnt;
    uint32_t O_off, O_cnt;
    uint32_t out_coeff_idx;
    uint32_t out_wire_id;
    uint8_t  loc;
    uint8_t  pad[3];
};
struct Term { uint32_t cid, vid; };

#define HIP_CHECK(call)                                                     \
    do {                                                                    \
        hipError_t e = (call);                                              \
        if (e != hipSuccess) {                                              \
            fprintf(stderr, "HIP error %s at %s:%d: %s\n",                  \
                    hipGetErrorName(e), __FILE__, __LINE__,                 \
                    hipGetErrorString(e));                                  \
            std::exit(2);                                                   \
        }                                                                   \
    } while (0)

__device__ __forceinline__ void
accumulate_LE(bn254_t& acc,
              const Term* terms, uint32_t off, uint32_t cnt,
              const bn254_t* coeffs, const bn254_t* wires,
              bool is_unsolved_le, uint32_t unset_wire) {
    for (uint32_t i = 0; i < cnt; ++i) {
        Term t = terms[off + i];
        if (is_unsolved_le && t.vid == unset_wire) continue;
        bn254_t prod = coeffs[t.cid] * wires[t.vid];
        acc = acc + prod;
    }
}

__global__ void eval_constraints_kernel(
    const R1CDesc* descs,
    const Term*    terms,
    const bn254_t* coeffs,
    bn254_t*       wires,
    uint32_t       n_descs,
    int*           error_flag) {

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_descs) return;

    R1CDesc d = descs[tid];

    bn254_t a = bn254_t::zero();
    bn254_t b = bn254_t::zero();
    bn254_t c = bn254_t::zero();
    accumulate_LE(a, terms, d.L_off, d.L_cnt, coeffs, wires, d.loc == 1, d.out_wire_id);
    accumulate_LE(b, terms, d.R_off, d.R_cnt, coeffs, wires, d.loc == 2, d.out_wire_id);
    accumulate_LE(c, terms, d.O_off, d.O_cnt, coeffs, wires, d.loc == 3, d.out_wire_id);

    if (d.loc == 0) {
        bn254_t lhs = a * b;
        bool eq = true;
        for (int i = 0; i < bn254_t::N; ++i) {
            if (lhs.data[i] != c.data[i]) { eq = false; break; }
        }
        if (!eq) atomicCAS(error_flag, 0, (int)tid + 1);
        return;
    }
    bn254_t wire;
    switch (d.loc) {
    case 1: { bn254_t binv = b.inv(); wire = c * binv; wire = wire - a; break; }
    case 2: { bn254_t ainv = a.inv(); wire = c * ainv; wire = wire - b; break; }
    case 3: { wire = a * b; wire = wire - c; break; }
    default: return;
    }
    if (d.out_coeff_idx == 1) {}
    else if (d.out_coeff_idx == 3) { wire = -wire; }
    else { bn254_t inv = coeffs[d.out_coeff_idx].inv(); wire = wire * inv; }
    wires[d.out_wire_id] = wire;
}

static std::vector<uint8_t> read_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "open %s\n", path.c_str()); std::exit(2); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(sz);
    if (fread(buf.data(), 1, sz, f) != (size_t)sz) std::exit(2);
    fclose(f);
    return buf;
}

struct LayerEntry {
    uint32_t n_descs;
    uint64_t descs_off;
    uint64_t terms_off;
};

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <prep_full_dir>\n", argv[0]);
        return 1;
    }
    std::string dir = argv[1];

    auto coeffs_bytes   = read_file(dir + "/coeffs.bin");
    auto initial_bytes  = read_file(dir + "/wires_initial.bin");
    auto expected_bytes = read_file(dir + "/wires_expected.bin");
    auto descs_bytes    = read_file(dir + "/layers_descs.bin");
    auto terms_bytes    = read_file(dir + "/layers_terms.bin");
    auto idx_bytes      = read_file(dir + "/layers.idx");

    size_t n_wires  = initial_bytes.size() / sizeof(bn254_t);
    uint32_t nb_layers = *(const uint32_t*)idx_bytes.data();
    std::vector<LayerEntry> layers(nb_layers);
    {
        const uint8_t* p = idx_bytes.data() + 4;
        for (uint32_t i = 0; i < nb_layers; ++i) {
            layers[i].n_descs   = *(const uint32_t*)(p + 0);
            layers[i].descs_off = *(const uint64_t*)(p + 4);
            layers[i].terms_off = *(const uint64_t*)(p + 12);
            p += 20;
        }
    }
    fprintf(stderr, "[graph] %u layers, %zu wires\n", nb_layers, n_wires);

    bn254_t *d_coeffs, *d_wires;
    Term *d_terms;
    R1CDesc *d_descs;
    int *d_err;
    HIP_CHECK(hipMalloc(&d_coeffs, coeffs_bytes.size()));
    HIP_CHECK(hipMalloc(&d_terms,  terms_bytes.size()));
    HIP_CHECK(hipMalloc(&d_descs,  descs_bytes.size()));
    HIP_CHECK(hipMalloc(&d_wires,  initial_bytes.size()));
    HIP_CHECK(hipMalloc(&d_err,    sizeof(int)));

    HIP_CHECK(hipMemcpy(d_coeffs, coeffs_bytes.data(), coeffs_bytes.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_terms,  terms_bytes.data(),  terms_bytes.size(),  hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_descs,  descs_bytes.data(),  descs_bytes.size(),  hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_wires,  initial_bytes.data(),initial_bytes.size(),hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_err, 0, sizeof(int)));
    HIP_CHECK(hipDeviceSynchronize());

    int block = 128;

    // ----- Build the graph by stream capture -----
    auto t0 = std::chrono::steady_clock::now();
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // Cap captured layers at a configurable limit. ROCm's HIP graph
    // instantiate seems unable to handle 135K nodes (segfault); empirically
    // a graph with the wide layers only is small enough to instantiate.
    const char* cap_env = getenv("MAX_GRAPH_LAYERS");
    uint32_t max_graph = cap_env ? (uint32_t)atoi(cap_env) : nb_layers;

    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    int empty_layers = 0;
    int captured = 0;
    for (uint32_t i = 0; i < nb_layers && (uint32_t)captured < max_graph; ++i) {
        uint32_t n = layers[i].n_descs;
        if (n == 0) { empty_layers++; continue; }
        int grid = (int)((n + block - 1) / block);
        eval_constraints_kernel<<<grid, block, 0, stream>>>(
            d_descs + layers[i].descs_off,
            d_terms,
            d_coeffs, d_wires,
            n, d_err);
        captured++;
    }
    hipGraph_t graph;
    HIP_CHECK(hipStreamEndCapture(stream, &graph));
    auto t1 = std::chrono::steady_clock::now();
    double cap_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    fprintf(stderr, "[graph] capture: %.1f ms (%d kernels captured, %d empty layers skipped)\n",
            cap_ms, captured, empty_layers);

    hipGraphExec_t exec;
    auto t2 = std::chrono::steady_clock::now();
    HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    auto t3 = std::chrono::steady_clock::now();
    double inst_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    fprintf(stderr, "[graph] instantiate: %.1f ms\n", inst_ms);

    // ----- Warmup launch (often the first run pays init costs) -----
    auto t4 = std::chrono::steady_clock::now();
    HIP_CHECK(hipGraphLaunch(exec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    auto t5 = std::chrono::steady_clock::now();
    double first_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();
    fprintf(stderr, "[graph] first launch: %.1f ms\n", first_ms);

    // Reset wires for second run
    HIP_CHECK(hipMemcpy(d_wires, initial_bytes.data(), initial_bytes.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_err, 0, sizeof(int)));
    HIP_CHECK(hipDeviceSynchronize());

    auto t6 = std::chrono::steady_clock::now();
    HIP_CHECK(hipGraphLaunch(exec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    auto t7 = std::chrono::steady_clock::now();
    double second_ms = std::chrono::duration<double, std::milli>(t7 - t6).count();
    fprintf(stderr, "[graph] second launch: %.1f ms\n", second_ms);

    // Pull error flag and computed wires for verification on the
    // second run only (first run output already overwritten by reset).
    int err_flag = 0;
    HIP_CHECK(hipMemcpy(&err_flag, d_err, sizeof(int), hipMemcpyDeviceToHost));
    if (err_flag != 0) fprintf(stderr, "[graph] verify-only fail: %d\n", err_flag - 1);

    std::vector<uint8_t> got(initial_bytes.size());
    HIP_CHECK(hipMemcpy(got.data(), d_wires, initial_bytes.size(), hipMemcpyDeviceToHost));
    size_t mismatches = 0;
    long first_mismatch = -1;
    for (size_t i = 0; i < n_wires; ++i) {
        if (memcmp(got.data() + i * 32, expected_bytes.data() + i * 32, 32) != 0) {
            if (first_mismatch < 0) first_mismatch = (long)i;
            mismatches++;
        }
    }
    if (mismatches > 0) {
        fprintf(stderr, "[graph] FAIL — %zu wire mismatches; first at %ld\n",
                mismatches, first_mismatch);
        return 3;
    }
    if (err_flag != 0) return 4;

    fprintf(stderr, "[graph] PASS — all %zu wires match expected\n", n_wires);
    fprintf(stderr,
            "[graph] PERF — capture %.1fms instantiate %.1fms launch1 %.1fms launch2 %.1fms; CPU 5400ms\n",
            cap_ms, inst_ms, first_ms, second_ms);

    HIP_CHECK(hipGraphExecDestroy(exec));
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipStreamDestroy(stream));
    return 0;
}
