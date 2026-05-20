// Phase 7-v2: warp-coop with 64-bit term loads.
// Tries to push past 575ms on 5090 by halving term-load bandwidth.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "ff/alt_bn128.hpp"

namespace cg = cooperative_groups;
using Fr = alt_bn128::fr_mont;

struct R1CDesc { uint32_t L_off, L_cnt, R_off, R_cnt, O_off, O_cnt, out_coeff_idx, out_wire_id; uint8_t loc; uint8_t pad[3]; };
struct Term { uint32_t cid, vid; };
struct LayerEntry { uint32_t n_descs; uint64_t descs_off; uint64_t terms_off; };

#define CUDA_CHECK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));std::exit(2);} } while(0)

__device__ __forceinline__ Fr fr_zero() { Fr r; r.zero(); return r; }
__device__ __noinline__ Fr fr_inv_fermat(Fr a) {
    static constexpr uint32_t exp[8] = {0xefffffff,0x43e1f593,0x79b97091,0x2833e848,0x8181585d,0xb85045b6,0xe131a029,0x30644e72};
    Fr r = a;
    for (int b = 252; b >= 0; --b) { r = r*r; if ((exp[b>>5] >> (b&31)) & 1u) r = r*a; }
    return r;
}

__device__ __forceinline__ Fr warp_reduce_fr(Fr v) {
    for (int off = 16; off > 0; off >>= 1) {
        Fr o;
        uint32_t* op = reinterpret_cast<uint32_t*>(&o);
        const uint32_t* vp = reinterpret_cast<const uint32_t*>(&v);
        #pragma unroll
        for (int i = 0; i < 8; ++i) op[i] = __shfl_xor_sync(0xFFFFFFFF, vp[i], off);
        v = v + o;
    }
    return v;
}

__device__ __forceinline__ void
process_R1C_warp(int lane, const R1CDesc& d,
                 const Term* terms, const Fr* coeffs, Fr* wires,
                 int* error_flag, uint32_t global_idx) {
    Fr a_part = fr_zero(), b_part = fr_zero(), c_part = fr_zero();
    bool unsolved_L = (d.loc == 1);
    bool unsolved_R = (d.loc == 2);
    bool unsolved_O = (d.loc == 3);
    uint32_t unset = d.out_wire_id;

    const uint64_t* tL = reinterpret_cast<const uint64_t*>(terms + d.L_off);
    const uint64_t* tR = reinterpret_cast<const uint64_t*>(terms + d.R_off);
    const uint64_t* tO = reinterpret_cast<const uint64_t*>(terms + d.O_off);

    for (uint32_t i = lane; i < d.L_cnt; i += 32) {
        uint64_t t = tL[i];
        uint32_t cid = (uint32_t)t;
        uint32_t vid = (uint32_t)(t >> 32);
        if (unsolved_L && vid == unset) continue;
        a_part = a_part + coeffs[cid] * wires[vid];
    }
    for (uint32_t i = lane; i < d.R_cnt; i += 32) {
        uint64_t t = tR[i];
        uint32_t cid = (uint32_t)t;
        uint32_t vid = (uint32_t)(t >> 32);
        if (unsolved_R && vid == unset) continue;
        b_part = b_part + coeffs[cid] * wires[vid];
    }
    for (uint32_t i = lane; i < d.O_cnt; i += 32) {
        uint64_t t = tO[i];
        uint32_t cid = (uint32_t)t;
        uint32_t vid = (uint32_t)(t >> 32);
        if (unsolved_O && vid == unset) continue;
        c_part = c_part + coeffs[cid] * wires[vid];
    }

    Fr a = warp_reduce_fr(a_part);
    Fr b = warp_reduce_fr(b_part);
    Fr c = warp_reduce_fr(c_part);

    if (lane != 0) return;

    if (d.loc == 0) {
        Fr lhs = a * b;
        const uint32_t* lp = reinterpret_cast<const uint32_t*>(&lhs);
        const uint32_t* cp = reinterpret_cast<const uint32_t*>(&c);
        bool eq = true;
        for (int i = 0; i < 8; ++i) if (lp[i] != cp[i]) { eq = false; break; }
        if (!eq) atomicCAS(error_flag, 0, (int)global_idx + 1);
        return;
    }
    Fr wire;
    if (d.loc == 3) { wire = a * b; wire = wire - c; }
    else if (d.loc == 1) { Fr binv = fr_inv_fermat(b); wire = c * binv; wire = wire - a; }
    else if (d.loc == 2) { Fr ainv = fr_inv_fermat(a); wire = c * ainv; wire = wire - b; }
    else return;

    if (d.out_coeff_idx == 1) {}
    else if (d.out_coeff_idx == 3) { wire = -wire; }
    else { Fr inv = fr_inv_fermat(coeffs[d.out_coeff_idx]); wire = wire * inv; }
    wires[d.out_wire_id] = wire;
}

__global__ void persistent_solve_warp_kernel(
    const LayerEntry* layers, uint32_t n_layers,
    const R1CDesc* descs, const Term* terms, const Fr* coeffs, Fr* wires,
    int* error_flag) {
    cg::grid_group g = cg::this_grid();
    int lane = threadIdx.x & 31;
    int warp_in_block = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int warp_id = blockIdx.x * warps_per_block + warp_in_block;
    int n_warps = gridDim.x * warps_per_block;
    for (uint32_t L = 0; L < n_layers; ++L) {
        LayerEntry e = layers[L];
        for (uint32_t i = warp_id; i < e.n_descs; i += n_warps) {
            process_R1C_warp(lane, descs[e.descs_off + i], terms, coeffs, wires,
                error_flag, (uint32_t)e.descs_off + i);
        }
        g.sync();
    }
}

static std::vector<uint8_t> rd(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb");
    fseek(f, 0, SEEK_END); long s = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v(s); fread(v.data(), 1, s, f); fclose(f); return v;
}

int main(int argc, char** argv) {
    if (argc != 2) { fprintf(stderr, "Usage: %s <prep_full_dir>\n", argv[0]); return 1; }
    std::string d = argv[1];
    auto co = rd(d+"/coeffs.bin"), wi = rd(d+"/wires_initial.bin"), we = rd(d+"/wires_expected.bin"),
         de = rd(d+"/layers_descs.bin"), te = rd(d+"/layers_terms.bin"), ix = rd(d+"/layers.idx");
    size_t n_wires = wi.size() / 32;
    uint32_t nl = *(uint32_t*)ix.data();
    std::vector<LayerEntry> ls(nl);
    const uint8_t* p = ix.data() + 4;
    for (uint32_t i = 0; i < nl; ++i) {
        ls[i].n_descs = *(uint32_t*)(p+0); ls[i].descs_off = *(uint64_t*)(p+4); ls[i].terms_off = *(uint64_t*)(p+12); p += 20;
    }
    Fr *dc, *dw; Term *dt; R1CDesc *dd; int *derr; LayerEntry *dl;
    cudaMalloc(&dc, co.size()); cudaMalloc(&dt, te.size()); cudaMalloc(&dd, de.size());
    cudaMalloc(&dw, wi.size()); cudaMalloc(&derr, 4); cudaMalloc(&dl, nl*sizeof(LayerEntry));
    cudaMemcpy(dc, co.data(), co.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dt, te.data(), te.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dd, de.data(), de.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dw, wi.data(), wi.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dl, ls.data(), nl*sizeof(LayerEntry), cudaMemcpyHostToDevice);
    cudaMemset(derr, 0, 4); cudaDeviceSynchronize();

    int sm, mbpm; cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);
    int blk = 256; if (const char* e = getenv("V2_BLOCK")) blk = atoi(e);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mbpm, (const void*)persistent_solve_warp_kernel, blk, 0);
    if (const char* e = getenv("V2_BPS")) mbpm = atoi(e);
    int grid = sm * mbpm;
    fprintf(stderr,"[v2] sm=%d bps=%d grid=%d blk=%d warps=%d\n", sm, mbpm, grid, blk, (grid*blk)/32);

    void* args[] = {&dl, &nl, &dd, &dt, &dc, &dw, &derr};
    auto t0 = std::chrono::steady_clock::now();
    cudaError_t err = cudaLaunchCooperativeKernel((void*)persistent_solve_warp_kernel,
        dim3(grid), dim3(blk), args, 0, 0);
    if (err != cudaSuccess) { fprintf(stderr,"launch %s\n", cudaGetErrorString(err)); return 2; }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    fprintf(stderr,"[v2] solve: %.1f ms\n", ms);

    int ef; cudaMemcpy(&ef, derr, 4, cudaMemcpyDeviceToHost);
    std::vector<uint8_t> got(wi.size()); cudaMemcpy(got.data(), dw, wi.size(), cudaMemcpyDeviceToHost);
    size_t mm = 0; long fmm = -1;
    for (size_t i = 0; i < n_wires; ++i) if (memcmp(got.data()+i*32, we.data()+i*32, 32)) { if (fmm<0) fmm=i; mm++; }
    if (mm) { fprintf(stderr,"[v2] FAIL %zu mismatches first %ld\n", mm, fmm); return 3; }
    if (ef) return 4;
    fprintf(stderr,"[v2] PASS\n");
    return 0;
}
