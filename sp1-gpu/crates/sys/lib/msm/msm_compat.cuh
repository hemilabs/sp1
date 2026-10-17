// Compatibility shim for sppark's pippenger.cuh.
// Provides CUDA_OK (throwing) and is_device_ptr<T> type trait,
// which are expected by pippenger.cuh but not defined in SP1's
// vendored sppark/util/exception.cuh (which uses CUDA_UNWRAP_SPPARK instead).
//
// This header must be included BEFORE pippenger.cuh in the MSM compilation unit.
// It does NOT modify any existing SP1 files.
#pragma once

#include <util/exception.cuh>

// CUDA_OK: throw cuda_error on failure (same semantics as CUDA_UNWRAP_SPPARK).
// pippenger.cuh uses CUDA_OK inside constructors and void methods where
// returning an error code is not possible. mult_pippenger() wraps everything
// in try/catch and converts to RustError.
#ifndef CUDA_OK
#define CUDA_OK(expr) CUDA_UNWRAP_SPPARK(expr)
#endif

// is_device_ptr<T>: type trait to detect GPU-resident pointer wrappers.
// pippenger.cuh uses this at lines 777/783 to decide whether points/scalars
// are already on the GPU (gpu_ptr_t<T>) or need to be uploaded (raw pointer).
// In SP1's FFI path, we always pass raw host pointers, so value is always false.
template<typename T>
struct is_device_ptr {
    static constexpr bool value = false;
};

// Specialize for gpu_ptr_t (defined later in gpu_t.cuh) so that if
// someone ever passes a gpu_ptr_t, it correctly reports as device.
template<typename T> class gpu_ptr_t;
template<typename T>
struct is_device_ptr<gpu_ptr_t<T>> {
    static constexpr bool value = true;
};
