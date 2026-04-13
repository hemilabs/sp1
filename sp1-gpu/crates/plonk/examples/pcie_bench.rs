//! PCIe bandwidth benchmark for AMD GPU.
//! Tests H2D, D2H, and D2D transfer rates at various sizes.

fn main() {
    use std::ffi::c_void;
    use std::time::Instant;

    // Initialize GPU
    unsafe {
        sp1_gpu_sys::runtime::cuda_device_synchronize();
    }

    let sizes: Vec<usize> = vec![
        1 << 20,      // 1 MiB
        4 << 20,      // 4 MiB
        16 << 20,     // 16 MiB
        64 << 20,     // 64 MiB
        256 << 20,    // 256 MiB
        1 << 30,      // 1 GiB
        2usize << 30, // 2 GiB
        4usize << 30, // 4 GiB
    ];

    println!("PCIe Bandwidth Benchmark");
    println!("========================");
    println!();

    // Get GPU info
    {
        let mut free: usize = 0;
        let mut total: usize = 0;
        unsafe {
            sp1_gpu_sys::runtime::cuda_mem_get_info(&mut free as *mut _, &mut total as *mut _);
        }
        println!(
            "GPU VRAM: {:.1} GiB total, {:.1} GiB free",
            total as f64 / (1 << 30) as f64,
            free as f64 / (1 << 30) as f64,
        );
        println!();
    }

    println!(
        "{:>10} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "Size", "H2D (pin)", "H2D (unpin)", "D2H (pin)", "D2H (unpin)", "D2D"
    );
    println!("{}", "-".repeat(76));

    for &size in &sizes {
        // Allocate device buffer
        let mut d_ptr: *mut c_void = std::ptr::null_mut();
        let err = unsafe { sp1_gpu_sys::runtime::cuda_malloc(&mut d_ptr as *mut _, size) };
        if err != unsafe { sp1_gpu_sys::runtime::CUDA_SUCCESS_CSL } {
            println!("{:>10} SKIP (OOM)", format_size(size));
            continue;
        }

        // Allocate second device buffer for D2D
        let mut d_ptr2: *mut c_void = std::ptr::null_mut();
        let _ = unsafe { sp1_gpu_sys::runtime::cuda_malloc(&mut d_ptr2 as *mut _, size) };

        // Allocate host buffer (page-faulted)
        let mut h_buf: Vec<u8> = vec![0u8; size];

        // --- H2D unpinned ---
        unsafe { sp1_gpu_sys::runtime::cuda_device_synchronize() };
        let t = Instant::now();
        let iters = transfer_iters(size);
        for _ in 0..iters {
            unsafe {
                sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(
                    d_ptr,
                    h_buf.as_ptr() as *const c_void,
                    size,
                );
            }
        }
        unsafe { sp1_gpu_sys::runtime::cuda_device_synchronize() };
        let h2d_unpin = size as f64 * iters as f64 / t.elapsed().as_secs_f64();

        // --- Pin the host buffer ---
        unsafe {
            sp1_gpu_sys::runtime::cuda_host_register(h_buf.as_ptr() as *const c_void, size);
        }

        // --- H2D pinned ---
        unsafe { sp1_gpu_sys::runtime::cuda_device_synchronize() };
        let t = Instant::now();
        for _ in 0..iters {
            unsafe {
                sp1_gpu_sys::runtime::cuda_mem_copy_host_to_device(
                    d_ptr,
                    h_buf.as_ptr() as *const c_void,
                    size,
                );
            }
        }
        unsafe { sp1_gpu_sys::runtime::cuda_device_synchronize() };
        let h2d_pin = size as f64 * iters as f64 / t.elapsed().as_secs_f64();

        // --- D2H pinned ---
        unsafe { sp1_gpu_sys::runtime::cuda_device_synchronize() };
        let t = Instant::now();
        for _ in 0..iters {
            unsafe {
                sp1_gpu_sys::runtime::cuda_mem_copy_device_to_host(
                    h_buf.as_mut_ptr() as *mut c_void,
                    d_ptr,
                    size,
                );
            }
        }
        unsafe { sp1_gpu_sys::runtime::cuda_device_synchronize() };
        let d2h_pin = size as f64 * iters as f64 / t.elapsed().as_secs_f64();

        // --- Unpin ---
        unsafe {
            sp1_gpu_sys::runtime::cuda_host_unregister(h_buf.as_ptr() as *const c_void);
        }

        // --- D2H unpinned ---
        unsafe { sp1_gpu_sys::runtime::cuda_device_synchronize() };
        let t = Instant::now();
        for _ in 0..iters {
            unsafe {
                sp1_gpu_sys::runtime::cuda_mem_copy_device_to_host(
                    h_buf.as_mut_ptr() as *mut c_void,
                    d_ptr,
                    size,
                );
            }
        }
        unsafe { sp1_gpu_sys::runtime::cuda_device_synchronize() };
        let d2h_unpin = size as f64 * iters as f64 / t.elapsed().as_secs_f64();

        // --- D2D ---
        let d2d = if !d_ptr2.is_null() {
            unsafe { sp1_gpu_sys::runtime::cuda_device_synchronize() };
            let t = Instant::now();
            for _ in 0..iters {
                unsafe {
                    sp1_gpu_sys::runtime::cuda_mem_copy_device_to_device(d_ptr2, d_ptr, size);
                }
            }
            unsafe { sp1_gpu_sys::runtime::cuda_device_synchronize() };
            size as f64 * iters as f64 / t.elapsed().as_secs_f64()
        } else {
            0.0
        };

        println!(
            "{:>10} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
            format_size(size),
            h2d_pin / 1e9,
            h2d_unpin / 1e9,
            d2h_pin / 1e9,
            d2h_unpin / 1e9,
            d2d / 1e9,
        );

        // Free
        unsafe {
            sp1_gpu_sys::runtime::cuda_free(d_ptr as *const c_void);
            if !d_ptr2.is_null() {
                sp1_gpu_sys::runtime::cuda_free(d_ptr2 as *const c_void);
            }
        }
    }

    println!();
    println!("All rates in GB/s (1e9 bytes/s)");
}

fn format_size(bytes: usize) -> String {
    if bytes >= (1 << 30) {
        format!("{} GiB", bytes >> 30)
    } else if bytes >= (1 << 20) {
        format!("{} MiB", bytes >> 20)
    } else {
        format!("{} KiB", bytes >> 10)
    }
}

fn transfer_iters(size: usize) -> usize {
    // Target ~1s total transfer time for accuracy
    let target_bytes = 4usize << 30; // 4 GiB
    let iters = target_bytes / size;
    iters.max(1).min(256)
}
