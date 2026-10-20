#include "runtime/gpu_backend.cuh"
#include "logup_gkr/execution.cuh"
#include "logup_gkr/first_layer.cuh"
#include "logup_gkr/tracegen.cuh"
#include "config.cuh"
#include <stdio.h>

struct GkrInput {
    felt_t numerator;
    ext_t denominator;

    __device__ __forceinline__ static GkrInput padding() {
        GkrInput values;
        values.numerator = felt_t::zero();
        values.denominator = ext_t::one();
        return values;
    }
};


__device__ __forceinline__ GkrInput interactionValue(
    size_t index,
    size_t rowIdx,
    Interactions<felt_t> const interactions,
    felt_t* const preprocessed,
    felt_t* const main,
    ext_t const alpha,
    ext_t* const betas,
    size_t height) {
    // Initialize the denominator and beta powers.
    ext_t denominator = alpha;

    // Add argument index to the denominator.
    denominator += betas[0] * interactions.arg_indices[index];

    // Add the interaction values.
    for (size_t k = interactions.values_ptr[index]; k < interactions.values_ptr[index + 1]; k++) {
        felt_t acc = interactions.values_constants[k];
        for (size_t l = interactions.values_col_weights_ptr[k];
             l < interactions.values_col_weights_ptr[k + 1];
             l++) {
            acc +=
                interactions.values_col_weights[l].get(preprocessed, main, rowIdx, height);
        }
        denominator += betas[k - interactions.values_ptr[index] + 1] * acc;
    }

    // Calculate the multiplicity values.
    bool isSend = interactions.is_send[index];
    felt_t mult = interactions.mult_constants[index];

    for (size_t k = interactions.multiplicities_ptr[index];
         k < interactions.multiplicities_ptr[index + 1];
         k++) {
        mult += interactions.mult_col_weights[k].get(preprocessed, main, rowIdx, height);
    }

    if (!isSend) {
        mult = felt_t::zero() - mult;
    }

    GkrInput value;
    value.numerator = mult;
    value.denominator = denominator;

    return value;
}

// One interaction per blockIdx.x (fastest-varying, so the blocks working on the same rows run
// together and the trace columns they share stay in L2), row pairs over blockIdx.y with a
// grid-stride loop. The interaction's metadata (values, column weights, constants, betas) is
// staged once per block in shared memory instead of being chased through global memory by
// every element; interactions larger than the staging bounds use the global-memory path.
// Field arithmetic is unchanged element by element, so the output is bit-identical.
namespace {
constexpr int POP_MAXV = 16;  // values per staged interaction
constexpr int POP_MAXW = 48;  // column weights per staged interaction (values + multiplicity)
struct PopStaged {
    uint32_t staged, nv, nvw, nm;
    uint32_t vstart[POP_MAXV + 1];
    uint32_t wcol[POP_MAXW];
    uint32_t wpre[POP_MAXW];
    felt_t wgt[POP_MAXW];
    felt_t vconst[POP_MAXV];
    felt_t mconst, arg;
    uint32_t send;
    ext_t beta[POP_MAXV + 1];
};
}  // namespace

__global__ void populateLastCircuitLayer(
    Interactions<felt_t> interactions,
    const uint32_t* startIndices,
    uint32_t* colIndex,
    felt_t* numeratorValues,
    ext_t* denominatorValues,
    felt_t* const preprocessed,
    felt_t* const main,
    ext_t alpha,
    ext_t* const beta,
    size_t interactionOffset,
    size_t traceHeight,
    size_t outputHeight,
    bool is_padding) {

    size_t halfTraceHeight;
    if (traceHeight == 0) {
        halfTraceHeight = 1;
    } else {
        halfTraceHeight = (traceHeight + 1) >> 1;
    }

    const size_t j = blockIdx.x;
    if (j >= interactions.num_interactions) return;
    const size_t colIdx = j + interactionOffset;
    const size_t startIdx = (size_t)startIndices[colIdx] << 1;

    SP1_SHARED_VAR(PopStaged, s);
    if (!is_padding) {
        if (threadIdx.x == 0) {
            size_t v0 = interactions.values_ptr[j], v1 = interactions.values_ptr[j + 1];
            size_t m0 = interactions.multiplicities_ptr[j], m1 = interactions.multiplicities_ptr[j + 1];
            size_t nw = interactions.values_col_weights_ptr[v1] -
                        interactions.values_col_weights_ptr[v0] + (m1 - m0);
            s.staged = (v1 - v0 <= POP_MAXV) && (nw <= POP_MAXW) && (traceHeight % 2 == 0);
            if (s.staged) {
                s.nv = (uint32_t)(v1 - v0);
                uint32_t w = 0;
                for (size_t k = v0; k < v1; k++) {
                    s.vstart[k - v0] = w;
                    s.vconst[k - v0] = interactions.values_constants[k];
                    for (size_t l = interactions.values_col_weights_ptr[k];
                         l < interactions.values_col_weights_ptr[k + 1];
                         l++, w++) {
                        PairCol<felt_t> pc = interactions.values_col_weights[l];
                        s.wcol[w] = (uint32_t)pc.column_idx;
                        s.wpre[w] = pc.is_preprocessed;
                        s.wgt[w] = pc.weight;
                    }
                }
                s.vstart[s.nv] = w;
                s.nvw = w;
                for (size_t k = m0; k < m1; k++, w++) {
                    PairCol<felt_t> pc = interactions.mult_col_weights[k];
                    s.wcol[w] = (uint32_t)pc.column_idx;
                    s.wpre[w] = pc.is_preprocessed;
                    s.wgt[w] = pc.weight;
                }
                s.nm = w - s.nvw;
                s.mconst = interactions.mult_constants[j];
                s.arg = interactions.arg_indices[j];
                s.send = interactions.is_send[j];
            }
        }
        __syncthreads();
        if (s.staged && threadIdx.x <= s.nv) s.beta[threadIdx.x] = beta[threadIdx.x];
        __syncthreads();
    }

    for (size_t i = blockIdx.y * blockDim.x + threadIdx.x; i < halfTraceHeight;
         i += blockDim.x * gridDim.y) {
        size_t parity = i & 1;
        size_t restrictedIndex = startIdx + i;
        if (is_padding) {
            FirstLayerCircuitValues values = FirstLayerCircuitValues::paddingValues();
            values.store(numeratorValues, denominatorValues, restrictedIndex, outputHeight);
            values.store(numeratorValues, denominatorValues, restrictedIndex + 1, outputHeight);
            values.store(numeratorValues, denominatorValues, restrictedIndex + 2, outputHeight);
            values.store(numeratorValues, denominatorValues, restrictedIndex + 3, outputHeight);
            colIndex[(restrictedIndex >> 1) + 1] = colIdx;
        } else if (s.staged) {
            // Rows 2i and 2i + 1 are adjacent (traceHeight is even): one 64-bit load per weight.
            ext_t d0 = alpha + s.beta[0] * s.arg;
            ext_t d1 = d0;
            for (uint32_t v = 0; v < s.nv; v++) {
                felt_t a0 = s.vconst[v], a1 = s.vconst[v];
                for (uint32_t w = s.vstart[v]; w < s.vstart[v + 1]; w++) {
                    const felt_t* base = s.wpre[w] ? preprocessed : main;
                    uint2 p = *reinterpret_cast<const uint2*>(base + (size_t)s.wcol[w] * traceHeight + (i << 1));
                    a0 += felt_t(p.x) * s.wgt[w];
                    a1 += felt_t(p.y) * s.wgt[w];
                }
                d0 += s.beta[v + 1] * a0;
                d1 += s.beta[v + 1] * a1;
            }
            felt_t n0 = s.mconst, n1 = s.mconst;
            for (uint32_t w = s.nvw; w < s.nvw + s.nm; w++) {
                const felt_t* base = s.wpre[w] ? preprocessed : main;
                uint2 p = *reinterpret_cast<const uint2*>(base + (size_t)s.wcol[w] * traceHeight + (i << 1));
                n0 += felt_t(p.x) * s.wgt[w];
                n1 += felt_t(p.y) * s.wgt[w];
            }
            if (!s.send) {
                n0 = felt_t::zero() - n0;
                n1 = felt_t::zero() - n1;
            }
            FirstLayerCircuitValues values;
            values.numeratorZero = n0;
            values.numeratorOne = n1;
            values.denominatorZero = d0;
            values.denominatorOne = d1;
            values.store(numeratorValues, denominatorValues, restrictedIndex, outputHeight);
        } else {
            GkrInput zeroValue = interactionValue(
                j, i << 1, interactions, preprocessed, main, alpha, beta, traceHeight);
            GkrInput oneValue = interactionValue(
                j, (i << 1) + 1, interactions, preprocessed, main, alpha, beta, traceHeight);
            FirstLayerCircuitValues values;
            values.numeratorZero = zeroValue.numerator;
            values.numeratorOne = oneValue.numerator;
            values.denominatorZero = zeroValue.denominator;
            values.denominatorOne = oneValue.denominator;
            values.store(numeratorValues, denominatorValues, restrictedIndex, outputHeight);
        }

        // Write the output interaction data and dimension. Do it only once per
        // pair of points.
        if (parity == 0) {
            colIndex[restrictedIndex >> 1] = colIdx;
        }
    }
}

extern "C" void* logup_gkr_populate_last_circuit_layer() {
    return SP1_KERNEL_PTR(populateLastCircuitLayer);
}
