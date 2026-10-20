// The KoalaBear NTTs (`sys/src/dft.rs`) on SYCL, in place of sppark's.
//
// Same results as the sppark wrappers (include/ntt/sppark.cuh):
// - batch_coset_dft: column c's n coefficients (natural order) give the evaluations of the
//   polynomial on the coset (3 * shift) * H_N, N = n << lg_blowup: sppark's LDE multiplies
//   coefficient i by g^i * shift^i with g = 3, its group generator. Natural order, or
//   bit-reversed with bit_rev_output.
// - batch_coset_dft_in_place: the same with each column's input at the end of its N-element
//   output region.
// - batch_lde_shift_in_place: the input is evaluations on H_n (natural order) at the end of the
//   region: they are interpolated (inverse NTT, scaled by 1/n), then extended as above.
// - batch_NTT / batch_iNTT: natural order in and out; the inverse scaled by 1/n.
// - reverse_bits_batch: a bit-reversal permutation per column.
// The roots of unity are p3's (= sppark's) powers of the 2^24-th root 0x6ac49f88.
//
// The transforms are decimation in frequency: natural-order input, bit-reversed output. Stages
// whose butterflies span at least a tile run one global kernel each; the remaining stages run in
// one kernel per tile in local memory.

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

#include "runtime/gpu_backend.cuh"
#include "fields/kb31_t.cuh"

struct RustCudaError {
    const char* message;
};
typedef struct RustCudaError rustCudaError_t;

namespace {

constexpr uint32_t P = 0x7f000001u;
constexpr int TWO_ADICITY = 24;
constexpr uint32_t ROOT_2_24 = 0x6ac49f88u;  // canonical
constexpr uint32_t GROUP_GEN = 3;
// Tile of the local-memory stages: 4096 elements (16 KiB) per work-group of 256.
constexpr int LG_TILE = 12;
constexpr uint32_t TILE = 1u << LG_TILE;
constexpr uint32_t TILE_THREADS = 256;

const rustCudaError_t SP1_SUCCESS = {nullptr};

rustCudaError_t error_message(const char* what) {
    static std::mutex m;
    static std::vector<std::string*> messages;
    std::lock_guard<std::mutex> lock(m);
    messages.push_back(new std::string(std::string("SYCL NTT error: ") + what));
    return rustCudaError_t{messages.back()->c_str()};
}

uint32_t mul_mod(uint32_t a, uint32_t b) { return (uint32_t)((uint64_t)a * b % P); }

uint32_t pow_mod(uint32_t a, uint64_t e) {
    uint32_t r = 1;
    while (e) {
        if (e & 1) r = mul_mod(r, a);
        a = mul_mod(a, a);
        e >>= 1;
    }
    return r;
}

uint32_t to_monty(uint32_t canonical) { return (uint32_t)(((uint64_t)canonical << 32) % P); }

// Twiddle tables per queue's context: w^k and w^-k (Montgomery form) for k < 2^23, w the 2^24-th
// root of unity. The root of unity of order 2^m is w^(2^(24-m)).
struct Twiddles {
    kb31_t* fwd = nullptr;
    kb31_t* inv = nullptr;
};

Twiddles& twiddles(sycl::queue& q) {
    static std::mutex m;
    static std::vector<std::pair<sycl::context, Twiddles>> per_context;
    std::lock_guard<std::mutex> lock(m);
    for (auto& [ctx, t] : per_context) {
        if (ctx == q.get_context()) return t;
    }
    const size_t half = (size_t)1 << (TWO_ADICITY - 1);
    std::vector<uint32_t> fwd(half), inv(half);
    const uint32_t w = ROOT_2_24, w_inv = pow_mod(ROOT_2_24, P - 2);
    uint32_t a = 1, b = 1;
    for (size_t k = 0; k < half; k++) {
        fwd[k] = to_monty(a);
        inv[k] = to_monty(b);
        a = mul_mod(a, w);
        b = mul_mod(b, w_inv);
    }
    Twiddles t;
    t.fwd = sycl::malloc_device<kb31_t>(half, q);
    t.inv = sycl::malloc_device<kb31_t>(half, q);
    q.memcpy(t.fwd, fwd.data(), half * sizeof(uint32_t));
    q.memcpy(t.inv, inv.data(), half * sizeof(uint32_t));
    q.wait_and_throw();
    per_context.emplace_back(q.get_context(), t);
    return per_context.back().second;
}

}  // namespace

// Times a submission for the optional kernel timing (lib/sycl/runtime.cpp).
void sp1_sycl_profile_event(const char* name, const sycl::event& e);

namespace {

inline uint32_t bit_rev(uint32_t x, int bits) {
    x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
    x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
    x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
    x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
    x = (x >> 16) | (x << 16);
    return bits ? x >> (32 - bits) : 0;
}

// One DIF stage over `cols` columns of 2^lg_n elements, butterfly half-size h = 2^lg_h.
void dif_stage(sycl::queue& q, kb31_t* data, int lg_n, int lg_h, uint32_t cols, const kb31_t* tw) {
    const size_t n = (size_t)1 << lg_n;
    const size_t pairs = (n / 2) * cols;
    const uint32_t h = 1u << lg_h;
    const int tw_shift = (TWO_ADICITY - 1) - lg_h;  // w_{2h}^j = w^(j << (23 - lg_h))
    auto e = q.parallel_for(sycl::range<1>(pairs), [=](sycl::id<1> id) {
        size_t t = id[0];
        size_t col = t >> (lg_n - 1);
        uint32_t pi = (uint32_t)(t & ((n / 2) - 1));
        uint32_t j = pi & (h - 1);
        uint32_t i0 = ((pi >> lg_h) << (lg_h + 1)) | j;
        kb31_t* x = data + col * n;
        kb31_t a = x[i0], b = x[i0 + h];
        x[i0] = a + b;
        x[i0 + h] = (a - b) * tw[(size_t)j << tw_shift];
    });
    sp1_sycl_profile_event("ntt: dif_stage", e);
}

// The DIF stages with half-size below 2^lg_top (lg_top <= LG_TILE), within tiles of 2^lg_top
// elements, in local memory.
void dif_tile(sycl::queue& q, kb31_t* data, int lg_n, int lg_top, uint32_t cols, const kb31_t* tw) {
    const uint32_t tile = 1u << lg_top;
    const size_t tiles = (((size_t)1 << lg_n) >> lg_top) * cols;
    const uint32_t threads = tile / 2 < TILE_THREADS ? std::max(tile / 2, 1u) : TILE_THREADS;
    auto e = q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<kb31_t, 1> s(sycl::range<1>(tile), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(tiles * threads, threads), [=](sycl::nd_item<1> it) {
                kb31_t* x = data + it.get_group(0) * (size_t)tile;
                const uint32_t lid = it.get_local_id(0);
                for (uint32_t i = lid; i < tile; i += threads) s[i] = x[i];
                for (int lg_h = lg_top - 1; lg_h >= 0; lg_h--) {
                    sycl::group_barrier(it.get_group());
                    const uint32_t h = 1u << lg_h;
                    const int tw_shift = (TWO_ADICITY - 1) - lg_h;
                    for (uint32_t pi = lid; pi < tile / 2; pi += threads) {
                        uint32_t j = pi & (h - 1);
                        uint32_t i0 = ((pi >> lg_h) << (lg_h + 1)) | j;
                        kb31_t a = s[i0], b = s[i0 + h];
                        s[i0] = a + b;
                        s[i0 + h] = (a - b) * tw[(size_t)j << tw_shift];
                    }
                }
                sycl::group_barrier(it.get_group());
                for (uint32_t i = lid; i < tile; i += threads) x[i] = s[i];
            });
    });
    sp1_sycl_profile_event("ntt: dif_tile", e);
}

// A full DIF transform of each column: natural order in, bit-reversed out.
void dif(sycl::queue& q, kb31_t* data, int lg_n, uint32_t cols, bool inverse) {
    if (lg_n == 0 || cols == 0) return;
    const kb31_t* tw = inverse ? twiddles(q).inv : twiddles(q).fwd;
    int lg_h = lg_n - 1;
    for (; lg_h >= LG_TILE; lg_h--) dif_stage(q, data, lg_n, lg_h, cols, tw);
    dif_tile(q, data, lg_n, lg_h + 1, cols, tw);
}

void bit_reverse_in_place(sycl::queue& q, kb31_t* data, int lg_n, uint32_t cols) {
    if (lg_n == 0 || cols == 0) return;
    const size_t n = (size_t)1 << lg_n;
    auto e = q.parallel_for(sycl::range<1>(n * cols), [=](sycl::id<1> id) {
        size_t t = id[0];
        size_t col = t >> lg_n;
        uint32_t i = (uint32_t)(t & (n - 1));
        uint32_t j = bit_rev(i, lg_n);
        if (i < j) {
            kb31_t* x = data + col * n;
            kb31_t tmp = x[i];
            x[i] = x[j];
            x[j] = tmp;
        }
    });
    sp1_sycl_profile_event("ntt: bit_reverse", e);
}

void scale(sycl::queue& q, kb31_t* data, size_t len, kb31_t factor) {
    auto e = q.parallel_for(
        sycl::range<1>(len),
        [=](sycl::id<1> id) { data[id[0]] = data[id[0]] * factor; });
    sp1_sycl_profile_event("ntt: scale", e);
}

// Column c's coefficient i (read from `in`, of stride `in_stride` per column, at position i, or
// bitrev_n(i) with `in_bit_reversed`), times factor * base^i, into position i of the N-element
// output column. Zeros are written separately (zero_tail) so that the input may lie in the tail.
void spread(
    sycl::queue& q,
    kb31_t* out,
    const kb31_t* in,
    size_t in_stride,
    int lg_n,
    int lg_blowup,
    uint32_t cols,
    kb31_t base,
    kb31_t factor,
    bool in_bit_reversed) {
    const size_t n = (size_t)1 << lg_n;
    const size_t ext = n << lg_blowup;
    auto e = q.parallel_for(sycl::range<1>(n * cols), [=](sycl::id<1> id) {
        size_t t = id[0];
        size_t col = t >> lg_n;
        uint32_t i = (uint32_t)(t & (n - 1));
        uint32_t src = in_bit_reversed ? bit_rev(i, lg_n) : i;
        kb31_t power = base ^ i;
        out[col * ext + i] = in[col * in_stride + src] * power * factor;
    });
    sp1_sycl_profile_event("ntt: spread", e);
}

void zero_tail(sycl::queue& q, kb31_t* out, int lg_n, int lg_blowup, uint32_t cols) {
    if (lg_blowup == 0) return;
    const size_t n = (size_t)1 << lg_n;
    const size_t ext = n << lg_blowup;
    const size_t tail = ext - n;
    auto e = q.parallel_for(sycl::range<1>(tail * cols), [=](sycl::id<1> id) {
        size_t t = id[0];
        size_t col = t / tail;
        size_t k = n + t % tail;
        out[col * ext + k] = kb31_t(0u);
    });
    sp1_sycl_profile_event("ntt: zero_tail", e);
}

kb31_t coset_base(kb31_t shift) { return kb31_t(to_monty(GROUP_GEN)) * shift; }

kb31_t inverse_of_size(int lg_n) { return kb31_t(to_monty(pow_mod(1u << lg_n, P - 2))); }

sycl::queue& queue_of(void* stream);

}  // namespace

// The runtime's queue for a stream handle (lib/sycl/runtime.cpp).
extern "C" void* sp1_sycl_queue_of(void* stream);

namespace {
sycl::queue& queue_of(void* stream) { return *static_cast<sycl::queue*>(sp1_sycl_queue_of(stream)); }
}  // namespace

#define NTT_TRY(body)                                                                              \
    try {                                                                                          \
        body                                                                                       \
    } catch (const sycl::exception& e) {                                                           \
        return error_message(e.what());                                                            \
    } catch (const std::exception& e) {                                                            \
        return error_message(e.what());                                                            \
    }

extern "C" {

rustCudaError_t sppark_init(void* stream) {
    NTT_TRY({
        twiddles(queue_of(stream));
        return SP1_SUCCESS;
    })
}

// The twiddle tables are built on first use, for any size (see `twiddles`): nothing to set up for
// a given largest size.
rustCudaError_t dft_init_twiddles(uint32_t, void* stream) {
    NTT_TRY({
        twiddles(queue_of(stream));
        return SP1_SUCCESS;
    })
}

// The portable coset DFT (lib/ntt/kb_ntt.cu); SP1_LEGACY_NTT=1 selects the one below.
rustCudaError_t kb_ntt_coset_dft(
    void* out,
    const void* in,
    uint32_t lg_n,
    uint32_t lg_blowup,
    uint32_t shift_monty,
    uint32_t cols,
    bool bit_rev_output,
    void* stream);

rustCudaError_t batch_coset_dft(
    kb31_t* d_out,
    kb31_t* d_in,
    uint32_t lg_domain_size,
    uint32_t lg_blowup,
    kb31_t shift,
    uint32_t poly_count,
    bool bit_rev_output,
    void* stream) {
    if (lg_domain_size == 0) return SP1_SUCCESS;
    static const bool legacy = [] {
        const char* v = std::getenv("SP1_LEGACY_NTT");
        return v && v[0] == '1';
    }();
    if (!legacy) {
        return kb_ntt_coset_dft(
            d_out, d_in, lg_domain_size, lg_blowup, shift.val, poly_count, bit_rev_output, stream);
    }
    NTT_TRY({
        auto& q = queue_of(stream);
        const int lg_ext = (int)(lg_domain_size + lg_blowup);
        const size_t n = (size_t)1 << lg_domain_size;
        spread(q, d_out, d_in, n, lg_domain_size, lg_blowup, poly_count, coset_base(shift),
               kb31_t(to_monty(1)), false);
        zero_tail(q, d_out, lg_domain_size, lg_blowup, poly_count);
        dif(q, d_out, lg_ext, poly_count, false);
        if (!bit_rev_output) bit_reverse_in_place(q, d_out, lg_ext, poly_count);
        return SP1_SUCCESS;
    })
}

rustCudaError_t batch_coset_dft_in_place(
    kb31_t* d_inout,
    uint32_t lg_domain_size,
    uint32_t lg_blowup,
    kb31_t shift,
    uint32_t poly_count,
    bool bit_rev_output,
    void* stream) {
    if (lg_domain_size == 0) return SP1_SUCCESS;
    NTT_TRY({
        auto& q = queue_of(stream);
        const int lg_ext = (int)(lg_domain_size + lg_blowup);
        const size_t n = (size_t)1 << lg_domain_size;
        const size_t ext = n << lg_blowup;
        // The input (column c's tail) and the coefficients' place (its head) are disjoint for a
        // blowup; without one they coincide and each element is scaled where it is.
        spread(q, d_inout, d_inout + (ext - n), ext, lg_domain_size, lg_blowup, poly_count,
               coset_base(shift), kb31_t(to_monty(1)), false);
        zero_tail(q, d_inout, lg_domain_size, lg_blowup, poly_count);
        dif(q, d_inout, lg_ext, poly_count, false);
        if (!bit_rev_output) bit_reverse_in_place(q, d_inout, lg_ext, poly_count);
        return SP1_SUCCESS;
    })
}

rustCudaError_t batch_lde_shift_in_place(
    kb31_t* d_inout,
    uint32_t lg_domain_size,
    uint32_t lg_blowup,
    kb31_t shift,
    uint32_t poly_count,
    bool bit_rev_output,
    void* stream) {
    if (lg_domain_size == 0) return SP1_SUCCESS;
    NTT_TRY({
        auto& q = queue_of(stream);
        const int lg_ext = (int)(lg_domain_size + lg_blowup);
        const size_t n = (size_t)1 << lg_domain_size;
        const size_t ext = n << lg_blowup;
        kb31_t* evals = d_inout + (ext - n);
        if (lg_blowup == 0) {
            // In place: interpolate, put the coefficients in natural order and scale them.
            dif(q, d_inout, lg_domain_size, poly_count, true);
            bit_reverse_in_place(q, d_inout, lg_domain_size, poly_count);
            spread(q, d_inout, d_inout, ext, lg_domain_size, 0, poly_count, coset_base(shift),
                   inverse_of_size(lg_domain_size), false);
        } else {
            // Interpolate each column's tail (whose stride is ext), leaving its coefficients
            // bit-reversed, then spread them into the head, scaled by 1/n and the coset powers.
            for (uint32_t c = 0; c < poly_count; c++)
                dif(q, evals + c * ext, lg_domain_size, 1, true);
            spread(q, d_inout, evals, ext, lg_domain_size, lg_blowup, poly_count, coset_base(shift),
                   inverse_of_size(lg_domain_size), true);
            zero_tail(q, d_inout, lg_domain_size, lg_blowup, poly_count);
        }
        dif(q, d_inout, lg_ext, poly_count, false);
        if (!bit_rev_output) bit_reverse_in_place(q, d_inout, lg_ext, poly_count);
        return SP1_SUCCESS;
    })
}

rustCudaError_t batch_NTT(kb31_t* d_inout, uint32_t lg_domain_size, uint32_t poly_count, void* stream) {
    if (lg_domain_size == 0) return SP1_SUCCESS;
    NTT_TRY({
        auto& q = queue_of(stream);
        dif(q, d_inout, lg_domain_size, poly_count, false);
        bit_reverse_in_place(q, d_inout, lg_domain_size, poly_count);
        return SP1_SUCCESS;
    })
}

rustCudaError_t batch_iNTT(kb31_t* d_inout, uint32_t lg_domain_size, uint32_t poly_count, void* stream) {
    if (lg_domain_size == 0) return SP1_SUCCESS;
    NTT_TRY({
        auto& q = queue_of(stream);
        dif(q, d_inout, lg_domain_size, poly_count, true);
        bit_reverse_in_place(q, d_inout, lg_domain_size, poly_count);
        scale(q, d_inout, ((size_t)1 << lg_domain_size) * poly_count, inverse_of_size(lg_domain_size));
        return SP1_SUCCESS;
    })
}

rustCudaError_t reverse_bits_batch(
    kb31_t* d_out,
    kb31_t* d_in,
    uint32_t lg_domain_size,
    uint32_t poly_count,
    void* stream) {
    if (lg_domain_size == 0) return SP1_SUCCESS;
    NTT_TRY({
        auto& q = queue_of(stream);
        const size_t n = (size_t)1 << lg_domain_size;
        if (d_out != d_in) q.memcpy(d_out, d_in, n * poly_count * sizeof(kb31_t));
        bit_reverse_in_place(q, d_out, lg_domain_size, poly_count);
        return SP1_SUCCESS;
    })
}

}  // extern "C"
