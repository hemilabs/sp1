// CUDA port of full_solve.cu using sppark's fr_t directly.
//
// Build (sm_89 = 4090, sm_90 = Hopper, sm_120 = Blackwell/5090):
//   nvcc -std=c++17 -O3 -arch=sm_89 \
//        -I /home/max/sp1-amd/sp1/sp1-gpu/crates/sys/include \
//        -I /home/max/sp1-amd/sp1/sp1-gpu/crates/sys/sppark \
//        -DFEATURE_BN254 \
//        full_solve_cuda.cu -o build/full_solve_cuda
// (use 13.1's nvcc for sm_120)

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>
#include <cuda_runtime.h>

#include "ff/alt_bn128.hpp"
using Fr = alt_bn128::fr_mont;

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

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t e = (call);                                             \
        if (e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                 \
                    cudaGetErrorName(e), __FILE__, __LINE__,                \
                    cudaGetErrorString(e));                                 \
            std::exit(2);                                                   \
        }                                                                   \
    } while (0)

__device__ __forceinline__ Fr fr_zero() {
    Fr r;
    r.zero();
    return r;
}

// Per-thread Fermat-based inverse. sppark's mont_t::reciprocal() uses a
// warp-cooperative algorithm (shfl_xor + ct_inverse_mod_x) that requires
// every lane in a warp to call /* warp-coop: not safe per-thread; use Fermat */ concurrently with valid
// inputs — but our kernel takes different code paths per thread (loc=O
// dominates; loc=R and rare coefficient inverses are sparse), so the
// cooperative path returns garbage for the minority threads. Fermat is
// per-thread safe.
//
// Computes a^(r-2) mod r where r-2 (little-endian uint32 limbs) is:
//   exp = 0x30644e72e131a029 b85045b68181585d 2833e84879b97091 43e1f593efffffff
__device__ __forceinline__ Fr fr_inv_fermat(const Fr& a) {
    static constexpr uint32_t exp[8] = {
        0xefffffff, 0x43e1f593, 0x79b97091, 0x2833e848,
        0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72,
    };
    // r-2 has bit 253 set; start with `a` and process bits 252 .. 0.
    Fr result = a;
    for (int bit = 252; bit >= 0; --bit) {
        result = result * result;
        int limb = bit >> 5;
        int b    = bit & 31;
        if ((exp[limb] >> b) & 1u) {
            result = result * a;
        }
    }
    return result;
}

__device__ __forceinline__ void
accumulate_LE(Fr& acc,
              const Term* terms, uint32_t off, uint32_t cnt,
              const Fr* coeffs, const Fr* wires,
              bool is_unsolved_le, uint32_t unset_wire) {
    for (uint32_t i = 0; i < cnt; ++i) {
        Term t = terms[off + i];
        if (is_unsolved_le && t.vid == unset_wire) continue;
        Fr prod = coeffs[t.cid] * wires[t.vid];
        acc = acc + prod;
    }
}

__global__ void eval_constraints_kernel(
    const R1CDesc* descs,
    const Term*    terms,
    const Fr*      coeffs,
    Fr*            wires,
    uint32_t       n_descs,
    int*           error_flag) {

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_descs) return;

    R1CDesc d = descs[tid];

    Fr a = fr_zero();
    Fr b = fr_zero();
    Fr c = fr_zero();
    accumulate_LE(a, terms, d.L_off, d.L_cnt, coeffs, wires, d.loc == 1, d.out_wire_id);
    accumulate_LE(b, terms, d.R_off, d.R_cnt, coeffs, wires, d.loc == 2, d.out_wire_id);
    accumulate_LE(c, terms, d.O_off, d.O_cnt, coeffs, wires, d.loc == 3, d.out_wire_id);

    if (d.loc == 0) {
        Fr lhs = a * b;
        // Compare 8 uint32 limbs
        const uint32_t* lp = reinterpret_cast<const uint32_t*>(&lhs);
        const uint32_t* cp = reinterpret_cast<const uint32_t*>(&c);
        bool eq = true;
        for (int i = 0; i < 8; ++i) {
            if (lp[i] != cp[i]) { eq = false; break; }
        }
        if (!eq) atomicCAS(error_flag, 0, (int)tid + 1);
        return;
    }
    Fr wire;
    switch (d.loc) {
    case 1: { Fr binv = fr_inv_fermat(b); wire = c * binv; wire = wire - a; break; }
    case 2: { Fr ainv = fr_inv_fermat(a); wire = c * ainv; wire = wire - b; break; }
    case 3: { wire = a * b; wire = wire - c; break; }
    default: return;
    }
    if (d.out_coeff_idx == 1) {}
    else if (d.out_coeff_idx == 3) { wire = -wire; }
    else { Fr inv = fr_inv_fermat(coeffs[d.out_coeff_idx]); wire = wire * inv; }
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

    size_t n_wires  = initial_bytes.size() / sizeof(Fr);
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
    fprintf(stderr, "[cuda] %u layers, %zu wires\n", nb_layers, n_wires);

    Fr *d_coeffs, *d_wires;
    Term *d_terms;
    R1CDesc *d_descs;
    int *d_err;
    CUDA_CHECK(cudaMalloc(&d_coeffs, coeffs_bytes.size()));
    CUDA_CHECK(cudaMalloc(&d_terms,  terms_bytes.size()));
    CUDA_CHECK(cudaMalloc(&d_descs,  descs_bytes.size()));
    CUDA_CHECK(cudaMalloc(&d_wires,  initial_bytes.size()));
    CUDA_CHECK(cudaMalloc(&d_err,    sizeof(int)));

    auto t0 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpy(d_coeffs, coeffs_bytes.data(), coeffs_bytes.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_terms,  terms_bytes.data(),  terms_bytes.size(),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_descs,  descs_bytes.data(),  descs_bytes.size(),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wires,  initial_bytes.data(),initial_bytes.size(),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_err, 0, sizeof(int)));
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();
    fprintf(stderr, "[cuda] upload time: %.1f ms\n",
            std::chrono::duration<double, std::milli>(t1 - t0).count());

    int block = 128;

    bool use_graph = getenv("USE_GRAPH") != nullptr;
    auto solve_t0 = std::chrono::steady_clock::now();

    if (use_graph) {
        // Capture into a CUDA graph and launch as one submission.
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        for (uint32_t i = 0; i < nb_layers; ++i) {
            uint32_t n = layers[i].n_descs;
            if (n == 0) continue;
            int grid = (int)((n + block - 1) / block);
            eval_constraints_kernel<<<grid, block, 0, stream>>>(
                d_descs + layers[i].descs_off,
                d_terms, d_coeffs, d_wires,
                n, d_err);
        }
        cudaGraph_t graph;
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        auto cap_t = std::chrono::steady_clock::now();
        cudaGraphExec_t exec;
        CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        auto inst_t = std::chrono::steady_clock::now();
        fprintf(stderr,
            "[cuda] graph capture %.1f ms, instantiate %.1f ms\n",
            std::chrono::duration<double, std::milli>(cap_t - solve_t0).count(),
            std::chrono::duration<double, std::milli>(inst_t - cap_t).count());

        // Reset wires + err for the actual run we time
        CUDA_CHECK(cudaMemcpyAsync(d_wires, initial_bytes.data(),
                                   initial_bytes.size(), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemsetAsync(d_err, 0, sizeof(int), stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Time the launch only
        auto launch_t0 = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaGraphLaunch(exec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto launch_t1 = std::chrono::steady_clock::now();
        fprintf(stderr, "[cuda] graph launch: %.1f ms\n",
                std::chrono::duration<double, std::milli>(launch_t1 - launch_t0).count());

        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
    } else {
        cudaError_t first_launch_err = cudaSuccess;
        int first_bad_layer = -1;
        for (uint32_t i = 0; i < nb_layers; ++i) {
            uint32_t n = layers[i].n_descs;
            if (n == 0) continue;
            int grid = (int)((n + block - 1) / block);
            eval_constraints_kernel<<<grid, block>>>(
                d_descs + layers[i].descs_off,
                d_terms,
                d_coeffs, d_wires,
                n, d_err);
            cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess && first_launch_err == cudaSuccess) {
                first_launch_err = e;
                first_bad_layer = (int)i;
            }
        }
        if (first_launch_err != cudaSuccess) {
            fprintf(stderr, "[cuda] first launch error at layer %d: %s\n",
                    first_bad_layer, cudaGetErrorString(first_launch_err));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    auto solve_t1 = std::chrono::steady_clock::now();
    double solve_ms = std::chrono::duration<double, std::milli>(solve_t1 - solve_t0).count();
    fprintf(stderr, "[cuda] layered solve: %.1f ms (%u launches)\n", solve_ms, nb_layers);

    int err_flag = 0;
    CUDA_CHECK(cudaMemcpy(&err_flag, d_err, sizeof(int), cudaMemcpyDeviceToHost));
    if (err_flag != 0) fprintf(stderr, "[cuda] verify-only fail: %d\n", err_flag - 1);

    std::vector<uint8_t> got(initial_bytes.size());
    CUDA_CHECK(cudaMemcpy(got.data(), d_wires, initial_bytes.size(), cudaMemcpyDeviceToHost));
    size_t mismatches = 0;
    long first_mismatch = -1;
    for (size_t i = 0; i < n_wires; ++i) {
        if (memcmp(got.data() + i * 32, expected_bytes.data() + i * 32, 32) != 0) {
            if (first_mismatch < 0) first_mismatch = (long)i;
            mismatches++;
        }
    }
    if (mismatches > 0) {
        fprintf(stderr, "[cuda] FAIL — %zu wire mismatches; first at %ld\n",
                mismatches, first_mismatch);
        return 3;
    }
    if (err_flag != 0) return 4;

    fprintf(stderr, "[cuda] PASS — all %zu wires match expected\n", n_wires);
    return 0;
}
