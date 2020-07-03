CCFLAGS = -g -std=c++17 -O3 -ffast-math -march=native -fno-strict-aliasing -fcx-fortran-rules -fopenmp -DEIGEN_NO_DEBUG #code does non comply with IEEE anymore
#CCFLAGS = -std=c++17 -g -ffast-math -march=native -fno-strict-aliasing -fcx-fortran-rules -fopenmp #code does non comply with IEEE anymore
CC =  /usr/local/openmpi-4.0.2CUDA/bin/mpicxx
GPUFLAGS = -lcusparse -lcusolver -lcublas -lcudart -fopenmp
MKLFLAGS = -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_blacs_intelmpi_ilp64 -lgomp \
           -lmkl_core -lpthread -lm -ldl 
MKLROOT = /opt/intel/mkl
eigenROOT = /usr/local/eigen
LyraROOT = ../dep/Lyra
cudaROOT =/usr/local/cuda-10.1

#IPATH = -I /usr/local/cuda-10.1/include  -I ../inc/Lyra/include -I /usr/local/eigen
IPATH = -I $(cudaROOT)/include -I$(MKLROOT)/include -I $(LyraROOT)/include -I $(eigenROOT) -I ../inc
#IPATH = -I/usr/local/cuda-10.1/include -I$(MKLROOT)/include  -I ../../Lyra/include -I ../eigen
LPATH = -L /usr/local/cuda-10.1/lib64 -L$(MKLROOT)/lib/intel64
#LPATH = -L /usr/local/cuda-10.1/lib64 

#simulationTDSE : simulationTDSE.cpp
#	$(CC) $(CCFLAGS)  -I $(eigenROOT) -I $(LyraROOT)/include simulationTDSE.cpp -o simulationTDSE  

all: simulationTDSECUDA.o CUDAaccelerated.o bsplines.o simulationACC
.PHONY: all

simulationACC: CUDAaccelerated.o simulationTDSECUDA.o bsplines.o 
	$(CC) $(LPATH) CUDAaccelerated.o simulationTDSECUDA.o bsplines.o -o simulationACC $(GPUFLAGS) $(MKLFLAGS) 

simulationTDSECUDA.o: ../src/simulationMultiGPU.cpp 
	$(CC) $(CCFLAGS) $(GPUFLAGS) $(IPATH) $(LPATH) -c ../src/simulationMultiGPU.cpp -o simulationTDSECUDA.o

CUDAaccelerated.o: ../src/CUDAacceleratedMPI.cpp
	$(CC) $(CCFLAGS) $(GPUFLAGS) $(IPATH) $(LPATH) -c ../src/CUDAacceleratedMPI.cpp -o CUDAaccelerated.o

bsplines.o: ../src/bsplines.cpp 
	$(CC) $(CCFLAGS) $(GPUFLAGS) $(IPATH) $(LPATH) -c ../src/bsplines.cpp  -o bsplines.o

clean:
	rm -f simulationACC simulationTDSECUDA.o CUDAaccelerated.o bsplines.o
