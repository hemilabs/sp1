
#ifndef __HIPCC__
#include <cuda.h>
#endif

#if defined(FEATURE_BLS12_381)
#include <ff/bls12-381.hpp>
#elif defined(FEATURE_BLS12_377)
#include <ff/bls12-377.hpp>
#elif defined(FEATURE_PALLAS)
#include <ff/pasta.hpp>
#elif defined(FEATURE_VESTA)
#include <ff/pasta.hpp>
#elif defined(FEATURE_BN254)
#include <ff/alt_bn128.hpp>
#elif defined(FEATURE_GOLDILOCKS)
#include <ff/goldilocks.hpp>
#elif defined(FEATURE_KOALA_BEAR)
#include <ff/koala_bear.hpp>
#else
#error "no FEATURE"
#endif

#include <ntt/ntt.cuh>
#ifdef NVIDIA_NTT
cudaError_t nvidia_ntt_init(uint32_t max_log_size, cudaStream_t stream);
cudaError_t nvidia_ntt_batch(
    fr_t* data,
    uint32_t log_size,
    uint32_t polynomial_count,
    uint32_t stride,
    bool inverse,
    cudaStream_t stream);
cudaError_t nvidia_ntt_batch_coset(
    fr_t* data,
    const fr_t* input,
    uint32_t log_size,
    uint32_t log_blowup,
    uint32_t polynomial_count,
    const fr_t* gen_powers,
    fr_t shift,
    bool perform_shift,
    cudaStream_t stream);
#endif

#include "runtime/exception.cuh"
#include "ntt/lde_fused.cuh"

#ifndef __CUDA_ARCH__

#ifdef NVIDIA_NTT
static constexpr bool LDE_INPUT_BIT_REVERSED = false;
static constexpr auto LDE_FORWARD_NTT_ORDER = NTT::InputOutputOrder::NN;
static constexpr auto LDE_INVERSE_NTT_ORDER = NTT::InputOutputOrder::NN;
#else
static constexpr bool LDE_INPUT_BIT_REVERSED = true;
static constexpr auto LDE_FORWARD_NTT_ORDER = NTT::InputOutputOrder::RN;
static constexpr auto LDE_INVERSE_NTT_ORDER = NTT::InputOutputOrder::NR;
#endif

static cudaError_t run_ntt_batch(
    fr_t* data,
    uint32_t lg_domain_size,
    uint32_t poly_count,
    uint32_t stride,
    NTT::InputOutputOrder order,
    NTT::Direction direction,
    bool reverse_result,
    const cudaStream_t stream) {
    try {
#ifdef NVIDIA_NTT
        const bool reverse_input =
            order == NTT::InputOutputOrder::RN || order == NTT::InputOutputOrder::RR;
        const bool reverse_output =
            (order == NTT::InputOutputOrder::NN || order == NTT::InputOutputOrder::RN) !=
            reverse_result;
        if (reverse_input) {
            for (uint32_t polynomial = 0; polynomial < poly_count; ++polynomial) {
                NTT::bit_rev(
                    data + polynomial * stride,
                    data + polynomial * stride,
                    lg_domain_size,
                    stream);
            }
        }
        cudaError_t error = nvidia_ntt_batch(
            data,
            lg_domain_size,
            poly_count,
            stride,
            direction == NTT::Direction::inverse,
            stream);
        if (error != cudaSuccess) {
            return error;
        }
        if (reverse_output) {
            for (uint32_t polynomial = 0; polynomial < poly_count; ++polynomial) {
                NTT::bit_rev(
                    data + polynomial * stride,
                    data + polynomial * stride,
                    lg_domain_size,
                    stream);
            }
        }
#else
        for (uint32_t polynomial = 0; polynomial < poly_count; ++polynomial) {
            NTT::Base_dev_ptr(
                stream,
                data + polynomial * stride,
                lg_domain_size,
                order,
                direction,
                NTT::Type::standard);
        }
        if (reverse_result) {
            for (uint32_t polynomial = 0; polynomial < poly_count; ++polynomial) {
                NTT::bit_rev(
                    data + polynomial * stride,
                    data + polynomial * stride,
                    lg_domain_size,
                    stream);
            }
        }
#endif
    } catch (const cuda_error& error) {
        return static_cast<cudaError_t>(-error.code());
    }
    return cudaSuccess;
}

extern "C" rustCudaError_t sppark_init(const cudaStream_t stream) {
#ifdef NVIDIA_NTT
    return CUDA_SUCCESS_CSL;
#else
    uint32_t lg_domain_size = 1;
    uint32_t domain_size = 1U << lg_domain_size;

    std::vector<fr_t> inout(domain_size);
    inout[0] = fr_t(1);
    inout[1] = fr_t(1);
    try {
        NTT::Base(
            stream,
            &inout[0],
            lg_domain_size,
            NTT::InputOutputOrder::NR,
            NTT::Direction::forward,
            NTT::Type::standard);
    } catch (const cuda_error& error) {
        CUDA_OK(static_cast<cudaError_t>(-error.code()));
    }
    return CUDA_SUCCESS_CSL;
#endif
}

extern "C" rustCudaError_t dft_init_twiddles(
    uint32_t max_log_size,
    const cudaStream_t stream) {
#ifdef NVIDIA_NTT
    CUDA_OK(nvidia_ntt_init(max_log_size, stream));
#else
    (void)max_log_size;
    (void)stream;
#endif
    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t batch_coset_dft(
    fr_t* d_out,
    fr_t* d_in,
    uint32_t lg_domain_size,
    uint32_t lg_blowup,
    fr_t shift,
    uint32_t poly_count,
    bool bit_rev_output,
    const cudaStream_t stream) {
    if (lg_domain_size == 0) {
        return CUDA_SUCCESS_CSL;
    }

    uint32_t domain_size = 1U << lg_domain_size;
    uint32_t ext_domain_size = domain_size << lg_blowup;

    try {
        const auto gen_powers = NTTParameters::all()[NTT::gpu_id()].partial_group_gen_powers;
        const bool perform_shift = shift != group_gen_inverse;
#ifdef NVIDIA_NTT
        const uint32_t lg_ext_domain_size = lg_domain_size + lg_blowup;
        if (lg_ext_domain_size >= 14) {
            CUDA_OK(nvidia_ntt_batch_coset(
                d_out,
                d_in,
                lg_ext_domain_size,
                lg_blowup,
                poly_count,
                &gen_powers[0][0],
                shift,
                perform_shift,
                stream));
            if (!bit_rev_output) {
                for (uint32_t polynomial = 0; polynomial < poly_count; ++polynomial) {
                    NTT::bit_rev(
                        d_out + polynomial * ext_domain_size,
                        d_out + polynomial * ext_domain_size,
                        lg_ext_domain_size,
                        stream);
                }
            }
            return CUDA_SUCCESS_CSL;
        }
#endif
#if !defined(NVIDIA_NTT) && !defined(SP1_PORTABLE)
        // Fused LDE + NTT (2 passes, no copy/memset) for the basefold encoder's shapes; the
        // output is bit-identical to the generic path below.
        if (!perform_shift && bit_rev_output && lde_fused::supported(lg_domain_size, lg_blowup)) {
            CUDA_OK(lde_fused::batch_lde(d_out, d_in, lg_domain_size, poly_count, stream));
            return CUDA_SUCCESS_CSL;
        }
        // Small LDEs: batch all columns into one copy/pad launch and batched GS passes.
        if (!perform_shift && bit_rev_output && lg_domain_size + lg_blowup <= 21 && poly_count > 1) {
            const uint32_t lg = lg_domain_size + lg_blowup;
            const auto& params = NTTParameters::all(false)[NTT::gpu_id()];
            for (uint32_t b0 = 0; b0 < poly_count; b0 += 1024) {
                const uint32_t nb = poly_count - b0 < 1024 ? poly_count - b0 : 1024;
                fr_t* out = d_out + (size_t)b0 * ext_domain_size;
                dim3 grid(std::max<uint32_t>(1, std::min<uint32_t>(ext_domain_size / 1024, 680)), nb);
                lde_fused::lde_copy_pad_batched<<<grid, 1024, 0, stream>>>(
                    out, d_in + (size_t)b0 * domain_size, domain_size, ext_domain_size);
                CUDA_OK(cudaGetLastError());
                GS_launcher gs{out, (int)lg, false, params, stream, nb, ext_domain_size};
                if (lg <= 10) {
                    gs.step(lg);
                } else if (lg <= 17) {
                    gs.step(lg / 2);
                    gs.step(lg / 2 + lg % 2);
                } else {
                    gs.step(lg / 3 + lg % 3);
                    gs.step(lg / 3);
                    gs.step(lg / 3);
                }
                CUDA_OK(cudaGetLastError());
            }
            return CUDA_SUCCESS_CSL;
        }
#endif
#ifdef NVIDIA_NTT
        constexpr bool direct_bit_reversed_output = false;
#else
        const bool direct_bit_reversed_output = bit_rev_output;
#endif
        const bool bit_reversed_input =
            LDE_INPUT_BIT_REVERSED && !direct_bit_reversed_output;
        for (size_t c = 0; c < poly_count; c++) {
            fr_t* domain_data = &d_in[c * domain_size];
            if (bit_reversed_input) {
                domain_data = &d_out[(c + 1) * ext_domain_size - domain_size];
                NTT::bit_rev(
                    domain_data,
                    &d_in[c * domain_size],
                    lg_domain_size,
                    stream);
            }

            NTT::LDE_launch(
                stream,
                &d_out[c * ext_domain_size],
                domain_data,
                gen_powers,
                lg_domain_size,
                lg_blowup,
                bit_reversed_input,
                perform_shift,
                shift);

        }
        CUDA_OK(run_ntt_batch(
            d_out,
            lg_domain_size + lg_blowup,
            poly_count,
            ext_domain_size,
            direct_bit_reversed_output ? NTT::InputOutputOrder::NR : LDE_FORWARD_NTT_ORDER,
            NTT::Direction::forward,
            bit_rev_output && !direct_bit_reversed_output,
            stream));
    } catch (const cuda_error& error) {
        CUDA_OK(static_cast<cudaError_t>(-error.code()));
    }

    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t batch_coset_dft_in_place(
    fr_t* d_inout,
    uint32_t lg_domain_size,
    uint32_t lg_blowup,
    fr_t shift,
    uint32_t poly_count,
    bool bit_rev_output,
    const cudaStream_t stream) {
    if (lg_domain_size == 0) {
        return CUDA_SUCCESS_CSL;
    }

    uint32_t domain_size = 1U << lg_domain_size;
    uint32_t ext_domain_size = domain_size << lg_blowup;

    try {
        const auto gen_powers = NTTParameters::all()[NTT::gpu_id()].partial_group_gen_powers;
        const bool perform_shift = shift != group_gen_inverse;
        for (size_t c = 0; c < poly_count; c++) {
            fr_t* domain_data = &d_inout[(c + 1) * ext_domain_size - domain_size];
            if constexpr (LDE_INPUT_BIT_REVERSED) {
                NTT::bit_rev(domain_data, domain_data, lg_domain_size, stream);
            }

            NTT::LDE_launch(
                stream,
                &d_inout[c * ext_domain_size],
                domain_data,
                gen_powers,
                lg_domain_size,
                lg_blowup,
                LDE_INPUT_BIT_REVERSED,
                perform_shift,
                shift);

        }
        CUDA_OK(run_ntt_batch(
            d_inout,
            lg_domain_size + lg_blowup,
            poly_count,
            ext_domain_size,
            LDE_FORWARD_NTT_ORDER,
            NTT::Direction::forward,
            bit_rev_output,
            stream));
    } catch (const cuda_error& error) {
        CUDA_OK(static_cast<cudaError_t>(-error.code()));
    }

    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t batch_lde_shift_in_place(
    fr_t* d_inout,
    uint32_t lg_domain_size,
    uint32_t lg_blowup,
    fr_t shift,
    uint32_t poly_count,
    bool bit_rev_output,
    const cudaStream_t stream) {
    if (lg_domain_size == 0) {
        return CUDA_SUCCESS_CSL;
    }

    uint32_t domain_size = 1U << lg_domain_size;
    uint32_t ext_domain_size = domain_size << lg_blowup;

    try {
        const auto gen_powers = NTTParameters::all()[NTT::gpu_id()].partial_group_gen_powers;
        const bool perform_shift = shift != group_gen_inverse;
        CUDA_OK(run_ntt_batch(
            &d_inout[ext_domain_size - domain_size],
            lg_domain_size,
            poly_count,
            ext_domain_size,
            LDE_INVERSE_NTT_ORDER,
            NTT::Direction::inverse,
            false,
            stream));
        for (size_t c = 0; c < poly_count; c++) {
            NTT::LDE_launch(
                stream,
                &d_inout[c * ext_domain_size],
                &d_inout[(c + 1) * ext_domain_size - domain_size],
                gen_powers,
                lg_domain_size,
                lg_blowup,
                LDE_INPUT_BIT_REVERSED,
                perform_shift,
                shift);

        }
        CUDA_OK(run_ntt_batch(
            d_inout,
            lg_domain_size + lg_blowup,
            poly_count,
            ext_domain_size,
            LDE_FORWARD_NTT_ORDER,
            NTT::Direction::forward,
            bit_rev_output,
            stream));
    } catch (const cuda_error& error) {
        CUDA_OK(static_cast<cudaError_t>(-error.code()));
    }

    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t
batch_NTT(fr_t* d_inout, uint32_t lg_domain_size, uint32_t poly_count, const cudaStream_t stream) {
    if (lg_domain_size == 0)
        return CUDA_SUCCESS_CSL;

    uint32_t domain_size = 1U << lg_domain_size;

    CUDA_OK(run_ntt_batch(
        d_inout,
        lg_domain_size,
        poly_count,
        domain_size,
        NTT::InputOutputOrder::NN,
        NTT::Direction::forward,
        false,
        stream));
    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t reverse_bits_batch(
    fr_t* d_out,
    fr_t* d_in,
    uint32_t lg_domain_size,
    uint32_t poly_count,
    const cudaStream_t stream) {
    if (lg_domain_size == 0)
        return CUDA_SUCCESS_CSL;

    uint32_t domain_size = 1U << lg_domain_size;

    try {
        for (size_t c = 0; c < poly_count; c++) {
            NTT::bit_rev(&d_out[c * domain_size], &d_in[c * domain_size], lg_domain_size, stream);
        }
    } catch (const cuda_error& error) {
        CUDA_OK(static_cast<cudaError_t>(-error.code()));
    }
    return CUDA_SUCCESS_CSL;
}

extern "C" rustCudaError_t
batch_iNTT(fr_t* d_inout, uint32_t lg_domain_size, uint32_t poly_count, const cudaStream_t stream) {
    if (lg_domain_size == 0)
        return CUDA_SUCCESS_CSL;

    uint32_t domain_size = 1U << lg_domain_size;

    CUDA_OK(run_ntt_batch(
        d_inout,
        lg_domain_size,
        poly_count,
        domain_size,
        NTT::InputOutputOrder::NN,
        NTT::Direction::inverse,
        false,
        stream));
    return CUDA_SUCCESS_CSL;
}

#endif
