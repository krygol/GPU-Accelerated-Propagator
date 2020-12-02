CCFLAGS = -g -std=c++17 -O3 -ffast-math -march=native -fno-strict-aliasing -fcx-fortran-rules -fopenmp -DEIGEN_NO_DEBUG #code does non comply with IEEE anymore
#CCFLAGS = -std=c++17 -g -ffast-math -march=native -fno-strict-aliasing -fcx-fortran-rules -fopenmp #code does non comply with IEEE anymore
CC =  $(MPIROOT)/bin/mpicxx #/usr/local/openmpi-4.0.2CUDA/:wqbin/mpicxx
GPUFLAGS = -lcusparse -lcusolver -lcublas -lcudart -fopenmp
MKLFLAGS = -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_blacs_intelmpi_ilp64 -lgomp \
           -lmkl_core -lpthread -lm -ldl 


#set the ROOT paths of the libraries spack is recommended for installation
spackROOT = ~/git/spack/opt/spack/linux-ubuntu20.04-zen2/gcc-9.3.0
MPIROOT = $(spackROOT)/openmpi-4.0.2-bzhugdplts3bbii3e7tak63anr5adjic
MKLROOT = $(spackROOT)/intel-mkl-2020.1.217-wg3bb7lxlau2oro5b2cigd3tydt3yta5/mkl/
eigenROOT = $(spackROOT)/eigen-3.3.5-nsmqiplhqiohzsuwq7er7pc65rh52nrg/include/eigen3
LyraROOT = ../dep/Lyra
cudaROOT = $(spackROOT)/cuda-11.1.0-hsatcx2imo2iwa3htzpme3lrr5j3eo2p

#IPATH = -I /usr/local/cuda-10.1/include  -I ../inc/Lyra/include -I /usr/local/eigen
IPATH = -I $(cudaROOT)/include -I $(MKLROOT)/include -I $(LyraROOT)/include -I $(eigenROOT) -I ../inc
#IPATH = -I/usr/local/cuda-10.1/include -I$(MKLROOT)/include  -I ../../Lyra/include -I ../eigen
LPATH = -L $(cudaROOT)/lib64 -L $(MKLROOT)/lib/intel64
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
