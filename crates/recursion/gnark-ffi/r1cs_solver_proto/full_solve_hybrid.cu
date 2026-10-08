// Phase 5 prototype variant: HYBRID dispatch.
//
// Per-launch overhead on RDNA3 is ~26 µs even with HIP graphs (see
// project_groth16_amd_round*.md and full_solve_graph.cu measurements).
// With 131K small-layer launches that's the dominant cost in
// full_solve.cu (3.4 s of pure dispatch overhead).
//
// This driver fuses runs of consecutive small layers into a single
// persistent single-block kernel call:
//
//   for layer in topological order:
//     if layer.n_descs >= WIDE_THRESHOLD:
//        flush pending small-layer batch (one persistent-kernel launch)
//        launch eval_constraints_kernel for this wide layer
//     else:
//        append layer to pending small-layer batch
//   flush final pending batch
//
// The persistent kernel uses one block of THREADS_PER_BLOCK threads,
// processes layers in order with __syncthreads() between layers, and
// is bound to a single CU. It trades raw GPU parallelism (we use 1 of
// 96 CUs) for amortized launch overhead.
//
// On the SP1 100K SHA256 R1CS this should drop the deep-tail dispatch
// cost from ~3.4 s to a few hundred ms or less.
//
// Build:
//   hipcc -std=c++20 -O3 -I .../sys/include --offload-arch=gfx1100 \
//         -DUSE_HIP full_solve_hybrid.cu -o build/full_solve_hybrid

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

__device__ __forceinline__ void
process_one(const R1CDesc& d,
            const Term*    terms,
            const bn254_t* coeffs,
            bn254_t*       wires,
            int*           error_flag,
            uint32_t       desc_global_idx) {
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
        if (!eq) atomicCAS(error_flag, 0, (int)desc_global_idx + 1);
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

// Per-wide-layer kernel — one R1C per thread (same as full_solve.cu).
__global__ void eval_constraints_kernel(
    const R1CDesc* descs, const Term* terms,
    const bn254_t* coeffs, bn254_t* wires,
    uint32_t n_descs, int* error_flag) {

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_descs) return;
    process_one(descs[tid], terms, coeffs, wires, error_flag, tid);
}

// Persistent single-block kernel for a run of small layers.
// `layer_metas` is a flat array of (descs_off, n_descs) tuples for
// the layers in this batch, in topological order. The block iterates
// the metas, processes each layer's R1Cs with thread-per-R1C, and
// __syncthreads() between layers so wires written by layer N are
// visible to layer N+1.
//
// THREADS_PER_BLOCK should be at least max(n_descs in this batch).
struct LayerMeta {
    uint64_t descs_off;
    uint32_t n_descs;
    uint32_t pad;
};

template <int THREADS>
__global__ void eval_small_layers_persistent_kernel(
    const LayerMeta* metas,
    uint32_t         n_metas,
    const R1CDesc*   descs_base,
    const Term*      terms,
    const bn254_t*   coeffs,
    bn254_t*         wires,
    int*             error_flag) {

    int tid = threadIdx.x;
    for (uint32_t i = 0; i < n_metas; ++i) {
        LayerMeta m = metas[i];
        if ((uint32_t)tid < m.n_descs) {
            process_one(descs_base[m.descs_off + tid], terms, coeffs, wires,
                        error_flag, (uint32_t)m.descs_off + (uint32_t)tid);
        }
        __syncthreads();
    }
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
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s <prep_full_dir> [wide_threshold]\n", argv[0]);
        return 1;
    }
    std::string dir = argv[1];
    uint32_t wide_threshold = (argc == 3) ? (uint32_t)atoi(argv[2]) : 1024;

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
    fprintf(stderr, "[hybrid] %u layers, %zu wires, wide_threshold=%u\n",
            nb_layers, n_wires, wide_threshold);

    bn254_t *d_coeffs, *d_wires;
    Term *d_terms;
    R1CDesc *d_descs;
    int *d_err;
    LayerMeta *d_metas_pool;
    HIP_CHECK(hipMalloc(&d_coeffs, coeffs_bytes.size()));
    HIP_CHECK(hipMalloc(&d_terms,  terms_bytes.size()));
    HIP_CHECK(hipMalloc(&d_descs,  descs_bytes.size()));
    HIP_CHECK(hipMalloc(&d_wires,  initial_bytes.size()));
    HIP_CHECK(hipMalloc(&d_err,    sizeof(int)));
    HIP_CHECK(hipMalloc(&d_metas_pool, nb_layers * sizeof(LayerMeta)));

    HIP_CHECK(hipMemcpy(d_coeffs, coeffs_bytes.data(), coeffs_bytes.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_terms,  terms_bytes.data(),  terms_bytes.size(),  hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_descs,  descs_bytes.data(),  descs_bytes.size(),  hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_wires,  initial_bytes.data(),initial_bytes.size(),hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_err, 0, sizeof(int)));
    HIP_CHECK(hipDeviceSynchronize());

    // Build the dispatch plan: alternating runs of small layers and
    // single wide layers in topological order. We pre-stage all the
    // LayerMeta arrays for small batches into a single device pool and
    // remember (offset, count) per batch.
    constexpr int THREADS = 64;
    const uint32_t MAX_BATCH_LAYER_WIDTH = THREADS; // safety cap

    struct Action {
        bool is_wide;
        // for wide:
        uint32_t wide_layer_idx;
        // for small batch:
        uint32_t batch_meta_off; // index into staged_metas
        uint32_t batch_meta_cnt;
    };
    std::vector<Action> actions;
    std::vector<LayerMeta> staged_metas;

    auto flush_batch = [&](std::vector<LayerMeta>& current) {
        if (current.empty()) return;
        Action a;
        a.is_wide = false;
        a.batch_meta_off = (uint32_t)staged_metas.size();
        a.batch_meta_cnt = (uint32_t)current.size();
        for (auto& m : current) staged_metas.push_back(m);
        actions.push_back(a);
        current.clear();
    };

    {
        std::vector<LayerMeta> cur;
        int oversized_dropped = 0;
        for (uint32_t i = 0; i < nb_layers; ++i) {
            uint32_t n = layers[i].n_descs;
            if (n == 0) continue;
            if (n >= wide_threshold) {
                flush_batch(cur);
                Action a;
                a.is_wide = true;
                a.wide_layer_idx = i;
                a.batch_meta_off = a.batch_meta_cnt = 0;
                actions.push_back(a);
            } else if (n > MAX_BATCH_LAYER_WIDTH) {
                // A small layer that's still too wide for the persistent
                // single-block kernel; flush and dispatch as wide.
                flush_batch(cur);
                Action a;
                a.is_wide = true;
                a.wide_layer_idx = i;
                a.batch_meta_off = a.batch_meta_cnt = 0;
                actions.push_back(a);
                oversized_dropped++;
            } else {
                LayerMeta m;
                m.descs_off = layers[i].descs_off;
                m.n_descs   = n;
                m.pad = 0;
                cur.push_back(m);
            }
        }
        flush_batch(cur);
        fprintf(stderr,
            "[hybrid] %zu actions: %d small-batches (%zu metas total), %d wide+oversized launches\n",
            actions.size(),
            (int)std::count_if(actions.begin(), actions.end(),
                               [](const Action& a){ return !a.is_wide; }),
            staged_metas.size(),
            (int)std::count_if(actions.begin(), actions.end(),
                               [](const Action& a){ return a.is_wide; }));
        if (oversized_dropped) {
            fprintf(stderr,
                "[hybrid] %d small layers were oversized for the persistent kernel and went per-launch\n",
                oversized_dropped);
        }
    }

    // Upload staged_metas
    HIP_CHECK(hipMemcpy(d_metas_pool, staged_metas.data(),
                        staged_metas.size() * sizeof(LayerMeta),
                        hipMemcpyHostToDevice));

    // ----- SOLVE -----
    int wide_block = 128;
    auto solve_t0 = std::chrono::steady_clock::now();
    for (auto& a : actions) {
        if (a.is_wide) {
            uint32_t n = layers[a.wide_layer_idx].n_descs;
            int grid = (int)((n + wide_block - 1) / wide_block);
            eval_constraints_kernel<<<grid, wide_block>>>(
                d_descs + layers[a.wide_layer_idx].descs_off,
                d_terms, d_coeffs, d_wires,
                n, d_err);
        } else {
            eval_small_layers_persistent_kernel<THREADS><<<1, THREADS>>>(
                d_metas_pool + a.batch_meta_off,
                a.batch_meta_cnt,
                d_descs, d_terms, d_coeffs, d_wires, d_err);
        }
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto solve_t1 = std::chrono::steady_clock::now();
    double solve_ms = std::chrono::duration<double, std::milli>(solve_t1 - solve_t0).count();
    fprintf(stderr, "[hybrid] solve: %.1f ms (%zu launches)\n",
            solve_ms, actions.size());

    int err_flag = 0;
    HIP_CHECK(hipMemcpy(&err_flag, d_err, sizeof(int), hipMemcpyDeviceToHost));
    if (err_flag != 0) fprintf(stderr, "[hybrid] verify-only fail: %d\n", err_flag - 1);

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
        fprintf(stderr, "[hybrid] FAIL — %zu wire mismatches; first at %ld\n",
                mismatches, first_mismatch);
        return 3;
    }
    if (err_flag != 0) return 4;

    fprintf(stderr, "[hybrid] PASS — all %zu wires match expected\n", n_wires);
    return 0;
}

#include <algorithm>
