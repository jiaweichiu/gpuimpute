CUDADIR=/usr/local/cuda
OPENBLASDIR=/opt/OpenBLAS
MAGMADIR=/home/jchiu/magma-2.2.0/lib

IFLAGS=-I/usr/local/include -I${CUDADIR}/include -I${OPENBLASDIR}/include
CFLAGS=-O3 -fPIC -DNDEBUG -DADD_ -Wall -fopenmp -std=c++11 \
-DHAVE_CUBLAS -DMIN_CUDA_ARCH=200
#CPUFLAGS=${CFLAGS} -march=native -fopenmp
CPUFLAGS=${CFLAGS}
GPUFLAGS=${CFLAGS}
LFLAGS=-L${OPENBLASDIR}/lib -L${CUDADIR}/lib64 -L${MAGMADIR}/lib \
-lmagma -lopenblas \
-lopenblas -lcublas -lcusparse -lcudart -lcudadevrt \
-lcurand -lcusolver \
-lglog -lrt -lpthread
TEST_FLAGS=-lgtest -lgtest_main

CC=g++
NVCC=nvcc

common.o: common.cc common.h
	$(CC) common.cc -c $(CPUFLAGS) $(IFLAGS) -o $@

engine.o: engine.cc engine.h
	$(CC) engine.cc -c $(CPUFLAGS) $(IFLAGS) -o $@

engine_test.o: engine_test.cc engine.o common.o
	$(CC) $^ $(CPUFLAGS) $(IFLAGS) $(LFLAGS) $(TEST_FLAGS) -o $@

dummy_main.o: dummy_main.cc engine.o common.o
	$(CC) $^ $(CPUFLAGS) $(IFLAGS) $(LFLAGS) -o $@

clean:
	rm -f *.o