#include <iostream>
#include <vector>
#include <stdlib.h>
#include <assert.h>
#include <complex>
#include <cusparse.h>
#include <cusolverDn.h>
#include <cuComplex.h>
#include "cuda.h"
#include "cuda_runtime_api.h"
#include <cuda_runtime.h>
#include <cusparse.h>
#include "InputParameter.h"


void setupGPUs(int local_rank,int &num_devices);

void makeCSR(int nnz, int dim, int *h_cooRows, int *h_cooCols, double *h_cooVals,
             int *&d_cooCols, int *&d_rowPtr,double *&d_cooVals_sorted,
             int *&d_cscVals, int *&d_cscColPtr, double *&d_cscRows);

void SpMVCuda(int nnz,int dim,int LEGO, int *d_cooCols, int *d_rowPtr,double
          *d_cooVals_sorted,
          double *vector_x_real, double *vector_x_imag,  
          double *vector_x_conj_real, double *vector_x_conj_imag,  
          double *vector_y_real, double  *vector_y_imag);

void SpMVCuda101(int nnz,int dim,int LEGO, int *d_cooCols, int *d_rowPtr,double
          *d_cooVals_sorted,
          double *vector_x_real, double *vector_x_imag,  
          double *vector_x_conj_real, double *vector_x_conj_imag,  
          double *vector_y_real, double  *vector_y_imag);

void lanczosCUDA(std::vector<int> nnz,int dim,int dim_kryl, int nmbrLEGO, double &normStart, 
        std::vector<int*> &d_cooCols, 
        std::vector<int*> &d_rowPtr,
        std::vector<double*> &d_cooVals_sorted, 
        std::vector<int*> &d_cscRows,
        std::vector<int*>  &d_cscColPtr,
        std::vector<double*> &d_cscVals, 
        cuDoubleComplex *dX, std::complex<double> *ft,
        double *dod,cuDoubleComplex *dQ,cuDoubleComplex *dd); 

void lanczosCUDAOld(std::vector<int> nnz,int dim,int dim_kryl, int nmbrLEGO, 
        std::vector<int*> &d_cooCols, 
        std::vector<int*>  &d_rowPtr,
        std::vector<double*> &d_cooVals_sorted, 
        std::vector<int*> &d_cscRows, 
        std::vector<int*>  &d_cscColPtr,
        std::vector<double*> &d_cscVals, 
        std::complex<double> *X,std::complex<double> *ft,
        double *od,std::complex<double> *Q, std::complex<double> *d); 

void PropagatorFullGPU(
        InputParameter &Inp,
        std::vector<int> nnz,int dim, int nmbrLEGO, 
        std::vector<int*> &d_cooCols, 
        std::vector<int*>  &d_rowPtr,
        std::vector<double*> &d_cooVals_sorted, 
        std::vector<int*> &d_cscRows, 
        std::vector<int*>  &d_cscColPtr,
        std::vector<double*> &d_cscVals, 
        std::complex<double> *X
        ); 

void ft_calc(InputParameter Inp,double t,std::complex<double> *ft);

void physPropagator(
    InputParameter &Inp,
    int dim_kryl,int dim,
    double normStart, double dt, 
    cuDoubleComplex *&dQ,
    cuDoubleComplex *&dd,
    double *od,
    cuDoubleComplex *&dy,
    std::complex<double> *y);


void allocateMemoryPropagator(InputParameter Inp, 
    int dim,
    cuDoubleComplex *&dQ, 
    cuDoubleComplex *&dd,
    double *&od,
    cuDoubleComplex *&dy,
    std::complex<double> *y);

void freeMemoryPropagator(InputParameter Inp, 
    int dim,
    cuDoubleComplex *&dQ, 
    cuDoubleComplex *&dd,
    double *&od,
    cuDoubleComplex *&dy,
    std::complex<double> *y);

 


void genEigenSolverCUDA(int m,double *A, double *B,double *W,double *V);

void CUDAFinalize(int nmbrLEGO,
        std::vector<int*> &d_rowPtr,
        std::vector<int*> &d_cooCols,
        std::vector<double*> &d_cooVals_sorted,
        std::vector<int*> &d_cscRows,
        std::vector<int*>  &d_cscColPtr,
        std::vector<double*> &d_cscVals);
