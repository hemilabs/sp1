// Heartbeat variant of the warp-cooperative solver to identify hang vs slow.
// Spawns the cooperative kernel from a host thread, then polls a device-mapped
// counter every 100 ms; this lets us see whether layers are progressing.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <atomic>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include "fields/bn254_t.cuh"
namespace cg = cooperative_groups;

struct R1CDesc { uint32_t L_off,L_cnt,R_off,R_cnt,O_off,O_cnt,out_coeff_idx,out_wire_id; uint8_t loc; uint8_t pad[3]; };
struct Term { uint32_t cid, vid; };
struct LayerEntry { uint32_t n_descs; uint64_t descs_off; uint64_t terms_off; };

#define HC(call) do { hipError_t e=(call); if(e){fprintf(stderr,"HIP %s @ %d: %s\n",hipGetErrorName(e),__LINE__,hipGetErrorString(e));exit(2);} } while(0)

__device__ __attribute__((noinline)) bn254_t fr_inv(bn254_t a){return a.inv();}

__device__ __forceinline__ bn254_t warp_reduce_fr(bn254_t v) {
    for (int off=16; off>0; off>>=1) {
        bn254_t o;
        #pragma unroll
        for (int i=0;i<8;++i) o.data[i] = __shfl_xor(v.data[i], off, 32);
        v = v + o;
    }
    return v;
}

__device__ __forceinline__ void process_one(int lane, const R1CDesc& d, const Term* terms,
        const bn254_t* coeffs, bn254_t* wires, int* err, uint32_t gi) {
    bn254_t a=bn254_t::zero(), b=bn254_t::zero(), c=bn254_t::zero();
    bool uL=(d.loc==1), uR=(d.loc==2), uO=(d.loc==3);
    uint32_t un=d.out_wire_id;
    for (uint32_t i=lane;i<d.L_cnt;i+=32){Term t=terms[d.L_off+i]; if(uL && t.vid==un) continue; a=a+coeffs[t.cid]*wires[t.vid];}
    for (uint32_t i=lane;i<d.R_cnt;i+=32){Term t=terms[d.R_off+i]; if(uR && t.vid==un) continue; b=b+coeffs[t.cid]*wires[t.vid];}
    for (uint32_t i=lane;i<d.O_cnt;i+=32){Term t=terms[d.O_off+i]; if(uO && t.vid==un) continue; c=c+coeffs[t.cid]*wires[t.vid];}
    a=warp_reduce_fr(a); b=warp_reduce_fr(b); c=warp_reduce_fr(c);
    if (lane!=0) return;
    if (d.loc==0) {
        bn254_t l=a*b;
        for (int i=0;i<8;++i) if (l.data[i]!=c.data[i]) { atomicCAS(err,0,(int)gi+1); return; }
        return;
    }
    bn254_t w;
    switch(d.loc){
        case 1:{bn254_t bi=fr_inv(b); w=c*bi; w=w-a; break;}
        case 2:{bn254_t ai=fr_inv(a); w=c*ai; w=w-b; break;}
        case 3:{w=a*b; w=w-c; break;}
        default: return;
    }
    if (d.out_coeff_idx==1) {}
    else if (d.out_coeff_idx==3) w=-w;
    else { bn254_t ci=fr_inv(coeffs[d.out_coeff_idx]); w=w*ci; }
    wires[d.out_wire_id]=w;
}

__global__ void k_heartbeat(const LayerEntry* layers, uint32_t nL, const R1CDesc* descs,
        const Term* terms, const bn254_t* coeffs, bn254_t* wires, int* err,
        uint32_t* heartbeat /* device-mapped */) {
    cg::grid_group g = cg::this_grid();
    int lane = threadIdx.x & 31;
    int wpb = blockDim.x >> 5;
    int wid = blockIdx.x * wpb + (threadIdx.x >> 5);
    int nW = gridDim.x * wpb;
    for (uint32_t L=0; L<nL; ++L) {
        LayerEntry e = layers[L];
        for (uint32_t i=wid; i<e.n_descs; i+=nW)
            process_one(lane, descs[e.descs_off+i], terms, coeffs, wires, err, (uint32_t)e.descs_off+i);
        g.sync();
        if (threadIdx.x==0 && blockIdx.x==0) {
            __threadfence_system();
            heartbeat[0] = L+1;
            __threadfence_system();
        }
    }
}

static std::vector<uint8_t> rdf(const std::string&p){FILE*f=fopen(p.c_str(),"rb");if(!f)exit(2);
    fseek(f,0,SEEK_END);long s=ftell(f);fseek(f,0,SEEK_SET);std::vector<uint8_t>b(s);
    if(fread(b.data(),1,s,f)!=(size_t)s)exit(2);fclose(f);return b;}

int main(int argc, char** argv){
    if (argc!=2){fprintf(stderr,"Usage: %s <dir>\n",argv[0]);return 1;}
    std::string d=argv[1];
    auto cf=rdf(d+"/coeffs.bin"), iw=rdf(d+"/wires_initial.bin"), ew=rdf(d+"/wires_expected.bin");
    auto db=rdf(d+"/layers_descs.bin"), tb=rdf(d+"/layers_terms.bin"), ib=rdf(d+"/layers.idx");
    size_t nw=iw.size()/sizeof(bn254_t);
    uint32_t nL=*(const uint32_t*)ib.data();
    std::vector<LayerEntry> L(nL);
    {const uint8_t* p=ib.data()+4; for(uint32_t i=0;i<nL;++i){L[i].n_descs=*(const uint32_t*)(p+0);L[i].descs_off=*(const uint64_t*)(p+4);L[i].terms_off=*(const uint64_t*)(p+12);p+=20;}}
    fprintf(stderr,"[hb] nL=%u nw=%zu\n",nL,nw);

    bn254_t *dC,*dW; Term*dT; R1CDesc*dD; LayerEntry*dL; int*dE;
    HC(hipMalloc(&dC,cf.size()));HC(hipMalloc(&dT,tb.size()));HC(hipMalloc(&dD,db.size()));
    HC(hipMalloc(&dW,iw.size()));HC(hipMalloc(&dE,sizeof(int)));HC(hipMalloc(&dL,L.size()*sizeof(LayerEntry)));
    HC(hipMemcpy(dC,cf.data(),cf.size(),hipMemcpyHostToDevice));
    HC(hipMemcpy(dT,tb.data(),tb.size(),hipMemcpyHostToDevice));
    HC(hipMemcpy(dD,db.data(),db.size(),hipMemcpyHostToDevice));
    HC(hipMemcpy(dW,iw.data(),iw.size(),hipMemcpyHostToDevice));
    HC(hipMemcpy(dL,L.data(),L.size()*sizeof(LayerEntry),hipMemcpyHostToDevice));
    HC(hipMemset(dE,0,sizeof(int)));

    // Mapped (host-visible) heartbeat.
    uint32_t* hHB=nullptr;
    HC(hipHostMalloc(&hHB, sizeof(uint32_t), hipHostMallocMapped));
    *hHB = 0;
    uint32_t* dHB=nullptr;
    HC(hipHostGetDevicePointer((void**)&dHB, hHB, 0));

    int sm=0,mbpm=0;
    HC(hipDeviceGetAttribute(&sm,hipDeviceAttributeMultiprocessorCount,0));
    int block=128; if(const char*e=getenv("WARP_BLOCK")) block=atoi(e);
    HC(hipOccupancyMaxActiveBlocksPerMultiprocessor(&mbpm,(const void*)k_heartbeat,block,0));
    if(const char*e=getenv("WARP_BLOCKS_PER_SM")) mbpm=atoi(e);
    int grid=sm*mbpm; if(grid<1)grid=1;
    int nW=(grid*block)>>5;
    fprintf(stderr,"[hb] sm=%d mbpm=%d grid=%d block=%d warps=%d\n",sm,mbpm,grid,block,nW);

    void* args[] = {&dL,&nL,&dD,&dT,&dC,&dW,&dE,&dHB};

    auto t0=std::chrono::steady_clock::now();
    hipError_t le=hipLaunchCooperativeKernel((void*)k_heartbeat,dim3(grid),dim3(block),args,0,0);
    if(le){fprintf(stderr,"launch: %s\n",hipGetErrorString(le));return 2;}

    // Heartbeat poller
    std::atomic<bool> done(false);
    std::thread poll([&]{
        uint32_t prev=0;
        auto pT=std::chrono::steady_clock::now();
        while(!done.load()){
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            uint32_t cur = __atomic_load_n(hHB, __ATOMIC_RELAXED);
            auto t=std::chrono::steady_clock::now();
            double el = std::chrono::duration<double>(t-t0).count();
            if (cur != prev) {
                fprintf(stderr,"[hb] t=%.2fs L=%u (+%u in %.2fs)\n", el, cur, cur-prev,
                        std::chrono::duration<double>(t-pT).count());
                fflush(stderr);
                prev=cur; pT=t;
            } else if (el > 5.0 && (int)el % 5 == 0) {
                static int last_log = 0;
                int sec = (int)el;
                if (sec != last_log) {
                    fprintf(stderr,"[hb] t=%.1fs STALLED at L=%u\n", el, cur);
                    fflush(stderr);
                    last_log = sec;
                }
            }
        }
    });

    HC(hipDeviceSynchronize());
    done.store(true);
    poll.join();
    auto t1=std::chrono::steady_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    fprintf(stderr,"[hb] done in %.1f ms\n",ms);
    int ef=0; HC(hipMemcpy(&ef,dE,sizeof(int),hipMemcpyDeviceToHost));
    if(ef) fprintf(stderr,"[hb] err=%d\n",ef-1);
    std::vector<uint8_t> got(iw.size());
    HC(hipMemcpy(got.data(),dW,iw.size(),hipMemcpyDeviceToHost));
    size_t mm=0; long fm=-1;
    for(size_t i=0;i<nw;++i) if(memcmp(got.data()+i*32,ew.data()+i*32,32)){if(fm<0)fm=i;mm++;}
    fprintf(stderr,"[hb] mm=%zu first=%ld\n",mm,fm);
    return mm?3:0;
}
