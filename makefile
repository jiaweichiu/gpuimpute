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

base_test.o: base_test.cc base.o
	$(CC) $^ $(CPUFLAGS) $(IFLAGS) $(LFLAGS) $(TESTFLAGS) -o $@

cpu_vec.o: vec.cc vec.h
	$(CC) vec.cc -c $(CPUFLAGS) $(IFLAGS) -o $@

gpu_vec.o: vec.cu vec.h
	$(NVCC) vec.cu -c $(GPUFLAGS) $(IFLAGS) -o $@

vec_test.o: vec_test.cc gpu_vec.o cpu_vec.o base.o
	$(CC) $^ $(CPUFLAGS) $(IFLAGS) $(LFLAGS) $(TESTFLAGS) -o $@

cpu_mat.o: mat.cc mat.h
	$(CC) mat.cc -c $(CPUFLAGS) $(IFLAGS) -o $@

gpu_mat.o: mat.cu mat.h
	$(NVCC) mat.cu -c $(GPUFLAGS) $(IFLAGS) -o $@

mat_test.o: mat_test.cc gpu_mat.o cpu_mat.o gpu_vec.o cpu_vec.o base.o
	$(CC) $^ $(CPUFLAGS) $(IFLAGS) $(LFLAGS) $(TESTFLAGS) -o $@

qr.o: qr.cc qr.h
	$(CC) qr.cc -c $(CPUFLAGS) $(IFLAGS) -o $@

qr_test.o: qr_test.cc qr.o gpu_mat.o cpu_mat.o gpu_vec.o cpu_vec.o base.o
	$(CC) $^ $(CPUFLAGS) $(IFLAGS) $(LFLAGS) $(TESTFLAGS) -o $@

svd.o: svd.cc svd.h
	$(CC) svd.cc -c $(CPUFLAGS) $(IFLAGS) -o $@

svd_test.o: svd_test.cc svd.o gpu_mat.o cpu_mat.o gpu_vec.o cpu_vec.o base.o
	$(CC) $^ $(CPUFLAGS) $(IFLAGS) $(LFLAGS) $(TESTFLAGS) -o $@

clean:
	rm -f *.o