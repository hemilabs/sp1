#![no_main]
sp1_zkvm::entrypoint!(main);

use sp1_zkvm::syscalls::syscall_sha256_compress;

pub fn main() {
    let mut w = [1u64; 64];
    let mut state = [1u64; 8];

    for _ in 0..4 {
        syscall_sha256_compress(&mut w, &mut state);
    }

    println!("{:?}", state);

    for i in 0..2 {
        let mut buf = [1u64; 64];
        let w = buf.as_mut_ptr();
        syscall_sha256_compress(w as *mut [u64; 64], w as *mut [u64; 8]);

        let mut buf = [1u64; 64];
        let w = buf.as_mut_ptr();
        let h = unsafe { w.add(4) };
        syscall_sha256_compress(w as *mut [u64; 64], h as *mut [u64; 8]);

        let mut buf = [1u64; 64];
        let w = buf.as_mut_ptr();
        let h = unsafe { w.add(56) };
        syscall_sha256_compress(w as *mut [u64; 64], h as *mut [u64; 8]);

        let mut buf = [1u64; 68];
        let h = buf.as_mut_ptr();
        let w = unsafe { h.add(4) };
        syscall_sha256_compress(w as *mut [u64; 64], h as *mut [u64; 8]);
    }
}
