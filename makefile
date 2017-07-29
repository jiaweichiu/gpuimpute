CUDADIR=/usr/local/cuda
OPENBLASDIR=/opt/OpenBLAS

IFLAGS=-I/usr/local/include -I${CUDADIR}/include -I${OPENBLASDIR}/include
CFLAGS=-Wall -std=c++11 -O3 -DHAVE_CUBLAS -march=native
CPUFLAGS=${CFLAGS}
GPUFLAGS=${CFLAGS} -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_35,code=compute_35

CC=g++
NVCC=nvcc

common.o: common.cc common.h
	$(CC) common.cc -c $(CPUFLAGS) $(IFLAGS) -o $@

engine.o: engine.cc engine.h
	$(CC) engine.cc -c $(CPUFLAGS) $(IFLAGS) -o $@
