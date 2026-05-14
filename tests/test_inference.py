#!/usr/bin/env python3
"""
OAAX runtime inference smoke test.

All fields of the input tensors_struct are allocated on the C heap via
malloc so the runtime can safely call deep_free_tensors_struct on it after
send_input — matching the expected ownership contract.

Usage:
    python3 test_inference.py \\
        --lib   /opt/oaax/bin/libRuntimeLibrary.so \\
        --model model.trt \\
        --input-name  input \\
        --input-shape 1,4
"""
import argparse
import ctypes
import ctypes.util
import sys

# ── libc (for malloc / strdup) ─────────────────────────────────────────────────
_libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
_libc.malloc.restype  = ctypes.c_void_p
_libc.malloc.argtypes = [ctypes.c_size_t]
_libc.strdup.restype  = ctypes.c_void_p
_libc.strdup.argtypes = [ctypes.c_char_p]

def _malloc(n: int) -> int:
    p = _libc.malloc(n)
    if not p:
        raise MemoryError(f"malloc({n}) failed")
    return p  # Python int = raw C heap address

# ── tensors_struct layout ──────────────────────────────────────────────────────
DATA_TYPE_FLOAT = 1

class TensorsStruct(ctypes.Structure):
    _fields_ = [
        ("num_tensors", ctypes.c_size_t),
        ("names",       ctypes.c_void_p),  # char**
        ("data_types",  ctypes.c_void_p),  # int*
        ("ranks",       ctypes.c_void_p),  # size_t*
        ("shapes",      ctypes.c_void_p),  # size_t**
        ("data",        ctypes.c_void_p),  # void**
    ]

def _build_input(name: str, shape: list) -> int:
    """
    Build a fully C-heap-allocated tensors_struct for one float32 tensor.
    Returns the raw pointer (int). After send_input the runtime owns it.
    """
    n_elems  = 1
    for d in shape:
        n_elems *= d
    ptr_size = ctypes.sizeof(ctypes.c_void_p)
    sz_size  = ctypes.sizeof(ctypes.c_size_t)
    int_size = ctypes.sizeof(ctypes.c_int)

    ts_ptr = _malloc(ctypes.sizeof(TensorsStruct))
    ts = ctypes.cast(ts_ptr, ctypes.POINTER(TensorsStruct)).contents
    ts.num_tensors = 1

    # names: char**  →  [strdup(name)]
    names_arr = _malloc(ptr_size)
    ctypes.cast(names_arr, ctypes.POINTER(ctypes.c_void_p))[0] = _libc.strdup(name.encode())
    ts.names = names_arr

    # data_types: int*  →  [DATA_TYPE_FLOAT]
    dtypes_arr = _malloc(int_size)
    ctypes.cast(dtypes_arr, ctypes.POINTER(ctypes.c_int))[0] = DATA_TYPE_FLOAT
    ts.data_types = dtypes_arr

    # ranks: size_t*  →  [rank]
    ranks_arr = _malloc(sz_size)
    ctypes.cast(ranks_arr, ctypes.POINTER(ctypes.c_size_t))[0] = len(shape)
    ts.ranks = ranks_arr

    # shapes: size_t**  →  [size_t* → [d0, d1, ...]]
    shape_arr = _malloc(sz_size * len(shape))
    for i, d in enumerate(shape):
        ctypes.cast(shape_arr, ctypes.POINTER(ctypes.c_size_t))[i] = d
    shapes_arr = _malloc(ptr_size)
    ctypes.cast(shapes_arr, ctypes.POINTER(ctypes.c_void_p))[0] = shape_arr
    ts.shapes = shapes_arr

    # data: void**  →  [float* filled with 0.5]
    buf = _malloc(n_elems * ctypes.sizeof(ctypes.c_float))
    floats = ctypes.cast(buf, ctypes.POINTER(ctypes.c_float))
    for i in range(n_elems):
        floats[i] = 0.5
    data_arr = _malloc(ptr_size)
    ctypes.cast(data_arr, ctypes.POINTER(ctypes.c_void_p))[0] = buf
    ts.data = data_arr

    return ts_ptr


def _fail(msg: str, lib=None):
    print(f"FAIL: {msg}")
    if lib:
        try:
            lib.runtime_destruction()
        except Exception:
            pass
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib",         required=True)
    parser.add_argument("--model",       required=True)
    parser.add_argument("--input-name",  default="input")
    parser.add_argument("--input-shape", default="1,4")
    args = parser.parse_args()

    shape = [int(x) for x in args.input_shape.split(",")]

    try:
        lib = ctypes.CDLL(args.lib)
    except OSError as e:
        _fail(f"cannot load runtime library: {e}")

    lib.runtime_name.restype        = ctypes.c_char_p
    lib.runtime_version.restype     = ctypes.c_char_p
    lib.runtime_initialization.restype = ctypes.c_int
    lib.runtime_model_loading.restype  = ctypes.c_int
    lib.runtime_model_loading.argtypes = [ctypes.c_char_p]
    lib.send_input.restype          = ctypes.c_int
    lib.send_input.argtypes         = [ctypes.c_void_p]
    lib.receive_output.restype      = ctypes.c_int
    lib.receive_output.argtypes     = [ctypes.POINTER(ctypes.c_void_p)]
    lib.runtime_destruction.restype = ctypes.c_int

    print(f"Runtime : {lib.runtime_name().decode()}")
    print(f"Version : {lib.runtime_version().decode()}")

    if lib.runtime_initialization() != 0:
        _fail("runtime_initialization")
    print("[OK] runtime_initialization")

    if lib.runtime_model_loading(args.model.encode()) != 0:
        _fail(f"runtime_model_loading ({args.model})", lib)
    print("[OK] runtime_model_loading")

    ts_ptr = _build_input(args.input_name, shape)
    if lib.send_input(ts_ptr) != 0:
        _fail("send_input", lib)
    print("[OK] send_input")
    # ts_ptr is now owned by the runtime — do not touch it again

    output_ptr = ctypes.c_void_p(0)
    attempts   = 0
    while lib.receive_output(ctypes.byref(output_ptr)) != 0 and attempts < 100:
        attempts += 1

    if not output_ptr.value:
        _fail("receive_output timed out", lib)

    output   = ctypes.cast(output_ptr, ctypes.POINTER(TensorsStruct)).contents
    names_p  = ctypes.cast(output.names,      ctypes.POINTER(ctypes.c_char_p))
    ranks_p  = ctypes.cast(output.ranks,      ctypes.POINTER(ctypes.c_size_t))
    shapes_p = ctypes.cast(output.shapes,     ctypes.POINTER(ctypes.c_void_p))
    dtypes_p = ctypes.cast(output.data_types, ctypes.POINTER(ctypes.c_int))
    data_p   = ctypes.cast(output.data,       ctypes.POINTER(ctypes.c_void_p))

    print(f"[OK] receive_output: {output.num_tensors} tensor(s)")
    for i in range(output.num_tensors):
        rank    = ranks_p[i]
        shape_i = ctypes.cast(shapes_p[i], ctypes.POINTER(ctypes.c_size_t))
        dims    = [shape_i[d] for d in range(rank)]
        name    = names_p[i].decode() if names_p[i] else "?"
        dtype   = dtypes_p[i]
        print(f"  [{i}] name={name}  shape={dims}  dtype={dtype}")
        if dtype == DATA_TYPE_FLOAT and data_p[i]:
            floats  = ctypes.cast(data_p[i], ctypes.POINTER(ctypes.c_float))
            n_vals  = min(5, max(1, *dims))
            preview = " ".join(f"{floats[v]:.4f}" for v in range(n_vals))
            print(f"       first values: {preview}")

    lib.runtime_destruction()
    print("[OK] runtime_destruction")
    print("=== PASS ===")


if __name__ == "__main__":
    main()
