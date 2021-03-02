import cffi

ffibuilder = cffi.FFI()
ffibuilder.cdef(
    """int kernel_train(double**, double, int, int, double**);"""
)
ffibuilder.set_source(
    "_libkernel",
    '#include "kernels.h"',
    sources=[
        "kernels.cpp",
    ],
    source_extension=".cpp",
    include_dirs=["./"],
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
