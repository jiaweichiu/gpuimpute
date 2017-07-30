CUDADIR=/usr/local/cuda
OPENBLASDIR=/opt/OpenBLAS
MAGMADIR=/usr/local/magma

IFLAGS=-I/usr/local/include \
-I${CUDADIR}/include \
-I${OPENBLASDIR}/include \
-I${MAGMADIR}/include

CFLAGS=-std=c++11 -O3 -DNDEBUG \
-DHAVE_CUBLAS -DMIN_CUDA_ARCH=200

CPUFLAGS=${CFLAGS} -fPIC -march=native -fopenmp -Wall
GPUFLAGS=${CFLAGS}

LFLAGS=-L${OPENBLASDIR}/lib \
-L${CUDADIR}/lib64 \
-L${MAGMADIR}/lib \
-lopenblas \
-lmagma \
-lcublas -lcusparse -lcudart -lcurand -lcusolver \
-lglog -lrt -lpthread

TESTFLAGS=-lgtest -lgtest_main

CC=g++
NVCC=nvcc

base.o: base.cc base.h
	$(CC) base.cc -c $(CPUFLAGS) $(IFLAGS) -o $@

base_test.o: base_test.cc engine.o common.o
	$(CC) $^ $(CPUFLAGS) $(IFLAGS) $(LFLAGS) $(TESTFLAGS) -o $@

cpu_vec.o: vec.cc vec.h
	$(CC) vec.cc -c $(CPUFLAGS) $(IFLAGS) -o $@

gpu_vec.o: vec.cu vec.h
	$(NVCC) vec.cu -c $(GPUFLAGS) $(IFLAGS) -o $@

clean:
	rm -f *.o