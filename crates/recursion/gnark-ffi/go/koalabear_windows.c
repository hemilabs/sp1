//go:build windows

// On Windows this library is a DLL (see build.rs), and a DLL cannot leave these to be resolved
// against the Rust binary that loads it, as a static archive's undefined symbols are. The Rust
// side passes its implementations in before its first call, through SP1GnarkSetKoalaBear.

#include <stdint.h>
#include "koalabear.h"

typedef uint32_t (*koalabear_inv_fn)(uint32_t);
typedef uint32_t (*koalabear_extinv_fn)(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);

static koalabear_inv_fn inv_fn;
static koalabear_extinv_fn extinv_fn;

void sp1_gnark_set_koalabear(void* inv, void* extinv) {
    inv_fn = (koalabear_inv_fn)inv;
    extinv_fn = (koalabear_extinv_fn)extinv;
}

uint32_t koalabearinv(uint32_t a) {
    return inv_fn(a);
}

uint32_t koalabearextinv(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t i) {
    return extinv_fn(a, b, c, d, i);
}
