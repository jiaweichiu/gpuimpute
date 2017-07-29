# gpuimpute

# Install

## Install CUDA 8.0

We do not use the one from synaptic. If you have it, delete it. Otherwise it might cause some conflcits.

Run the two installers. Second one is the patch. Say we install to `/usr/local/cuda`.

We install CUDA to `/usr/local/cuda`.
* The libs are in `/usr/local/cuda/lib64`. Add that to `LD_LIBRARY_PATH`.
* The binaries are in `/usr/local/cuda/bin`. Add that to `PATH`.

Check `nvcc`:
```shell
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2016 NVIDIA Corporation
Built on Tue_Jan_10_13:22:03_CST_2017
Cuda compilation tools, release 8.0, V8.0.61
```

Try out the sample programs. Do a `ldd` to make sure it is using the right CUDA library.

## Install OpenBLAS

We like to use locally compiled libs.

Download OpenBLAS. Install to `/opt/OpenBLAS` by default. Make sure we have all the interfaces included:
```
 OpenBLAS build complete. (BLAS CBLAS LAPACK LAPACKE)
```

## Install MAGMA

Instead of using `cmake`, we will use our own `make.inc`.

```shell
cp make.inc-examples/make.inc.openblas ./make.inc
```

At the bottom of `make.inc`, add in
```
OPENBLASDIR = /opt/OpenBLAS
CUDADIR = /usr/local/cuda-8.0
```



make -j 10
make lib -j 10
make test -j 10
make sparse-lib -j 10
make sparse-test -j 10
```

```
-- Found OpenMP
--     OpenMP_C_FLAGS   -fopenmp
--     OpenMP_CXX_FLAGS -fopenmp
-- Found CUDA 7.5
--     CUDA_INCLUDE_DIRS:   /usr/include
--     CUDA_CUDART_LIBRARY: /usr/lib/x86_64-linux-gnu/libcudart.so
--     compile for CUDA arch 2.x (Fermi)
--     compile for CUDA arch 3.0 (Kepler)
--     compile for CUDA arch 3.5 (Kepler)
-- Define -DHAVE_CUBLAS -DMIN_CUDA_ARCH=200
-- Searching for BLAS and LAPACK. To override, set LAPACK_LIBRARIES using ccmake.
-- A library with BLAS API found.
-- A library with LAPACK API found.
--     BLAS_LIBRARIES:      /usr/lib/libblas.so
--     LAPACK_LIBRARIES:    /usr/lib/liblapack.so;/usr/lib/libblas.so
-- MKLROOT not set. To change, set MKLROOT using ccmake.
-- Flags
--     CFLAGS        -std=c99 -fopenmp -Wall -Wno-unused-function
--     CXXFLAGS      -std=c++11 -fopenmp -Wall -Wno-unused-function
--     NFLAGS        -DHAVE_CUBLAS  -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_35,code=compute_35 
--     FFLAGS        -Dmagma_devptr_t="integer(kind=8)"
--     LIBS         tester;lapacktest;magma
--     LIBS_SPARSE  tester;lapacktest;magma;magma_sparse
```