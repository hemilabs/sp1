//! Lightweight rocTX phase markers for rocprofv3 / rocprof-systems.
//!
//! Activated only when `SP1_GPU_PROFILE=1` is set. Otherwise all functions
//! are no-ops with a single atomic load. Symbols are dlopen'd at first call
//! so the prover continues to link without `libroctx64.so` at build time.
//!
//! Layered on top of existing `[T]` eprintln! timers — does NOT replace them.

use std::ffi::CString;
use std::os::raw::{c_char, c_int};
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
use std::sync::Once;

type RoctxRangePushFn = unsafe extern "C" fn(*const c_char) -> c_int;
type RoctxRangePopFn = unsafe extern "C" fn() -> c_int;
type RoctxMarkFn = unsafe extern "C" fn(*const c_char);

static INIT: Once = Once::new();
static ENABLED: AtomicBool = AtomicBool::new(false);
static PUSH_FN: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());
static POP_FN: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());
static MARK_FN: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());

extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut std::ffi::c_void;
    fn dlsym(handle: *mut std::ffi::c_void, symbol: *const c_char) -> *mut std::ffi::c_void;
}
const RTLD_LAZY: c_int = 1;
const RTLD_GLOBAL: c_int = 0x100;

fn init() {
    INIT.call_once(|| {
        if std::env::var("SP1_GPU_PROFILE").ok().as_deref() != Some("1") {
            return;
        }
        unsafe {
            let candidates = [
                b"libroctx64.so.4\0".as_ptr() as *const c_char,
                b"libroctx64.so\0".as_ptr() as *const c_char,
            ];
            let mut handle = std::ptr::null_mut();
            for c in candidates {
                let h = dlopen(c, RTLD_LAZY | RTLD_GLOBAL);
                if !h.is_null() {
                    handle = h;
                    break;
                }
            }
            if handle.is_null() {
                eprintln!("[profile] SP1_GPU_PROFILE=1 but libroctx64 not loadable; markers disabled");
                return;
            }
            let push = dlsym(handle, b"roctxRangePushA\0".as_ptr() as *const c_char);
            let pop = dlsym(handle, b"roctxRangePop\0".as_ptr() as *const c_char);
            let mark = dlsym(handle, b"roctxMarkA\0".as_ptr() as *const c_char);
            if push.is_null() || pop.is_null() {
                eprintln!("[profile] libroctx64 loaded but roctxRangePushA/Pop missing; markers disabled");
                return;
            }
            PUSH_FN.store(push as *mut (), Ordering::Release);
            POP_FN.store(pop as *mut (), Ordering::Release);
            MARK_FN.store(mark as *mut (), Ordering::Release);
            ENABLED.store(true, Ordering::Release);
            eprintln!("[profile] SP1_GPU_PROFILE=1 — rocTX markers active");
        }
    });
}

#[inline]
pub fn enabled() -> bool {
    init();
    ENABLED.load(Ordering::Acquire)
}

#[inline]
pub fn range_push(name: &str) {
    init();
    if !ENABLED.load(Ordering::Acquire) {
        return;
    }
    let p = PUSH_FN.load(Ordering::Acquire);
    if p.is_null() {
        return;
    }
    if let Ok(c) = CString::new(name) {
        unsafe {
            let f: RoctxRangePushFn = std::mem::transmute(p);
            f(c.as_ptr());
        }
    }
}

#[inline]
pub fn range_pop() {
    if !ENABLED.load(Ordering::Acquire) {
        return;
    }
    let p = POP_FN.load(Ordering::Acquire);
    if p.is_null() {
        return;
    }
    unsafe {
        let f: RoctxRangePopFn = std::mem::transmute(p);
        f();
    }
}

#[inline]
pub fn mark(name: &str) {
    init();
    if !ENABLED.load(Ordering::Acquire) {
        return;
    }
    let p = MARK_FN.load(Ordering::Acquire);
    if p.is_null() {
        return;
    }
    if let Ok(c) = CString::new(name) {
        unsafe {
            let f: RoctxMarkFn = std::mem::transmute(p);
            f(c.as_ptr());
        }
    }
}

/// RAII scope: `let _g = profile::scope("MyPhase");`
pub struct Scope {
    active: bool,
}

impl Scope {
    pub fn new(name: &str) -> Self {
        let active = enabled();
        if active {
            range_push(name);
        }
        Self { active }
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        if self.active {
            range_pop();
        }
    }
}
