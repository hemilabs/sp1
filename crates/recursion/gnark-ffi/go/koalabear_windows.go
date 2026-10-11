//go:build windows

package main

/*
void sp1_gnark_set_koalabear(void* inv, void* extinv);
*/
import "C"

import "unsafe"

// SP1GnarkSetKoalaBear hands the DLL the Rust implementations of koalabearinv and
// koalabearextinv (see koalabear_windows.c). It must be called before anything else.
//
//export SP1GnarkSetKoalaBear
func SP1GnarkSetKoalaBear(inv unsafe.Pointer, extinv unsafe.Pointer) {
	C.sp1_gnark_set_koalabear(inv, extinv)
}
