#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <mpi-ext.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <complex>
#include <cusparse.h>
#include <cusolverDn.h>
#include <cuComplex.h>
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "CUDAaccelerated.h"
#include "InputParameter.h"

#define watch(x) cout << (#x) << " is " << (x) << endl
#define pwatch(x) cout <<"rank " << MPIrank <<" "<< (#x) << " is " << (x) << endl
#define bp(x) cout << "breakpoint" << (#x) << endl
#define pbp(x) cout << "breakpoint rank: " << MPIrank << " "  << (#x) << endl

using namespace std;

//from https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//version for cuSparse calls cusparseStatus_t
//cusparseGetErrorString
#define gpuSpErrchk(ans) { gpuSpAssert((ans), __FILE__, __LINE__); }
inline void gpuSpAssert(cusparseStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != CUSPARSE_STATUS_SUCCESS) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cusparseGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//version for cuBlas calls 
#define gpuBlErrchk(ans) { gpuBlAssert((ans), __FILE__, __LINE__); }
inline void gpuBlAssert(cublasStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != CUBLAS_STATUS_SUCCESS) 
   {
      fprintf(stderr,"GPUassert: %d %s %d\n", code, file, line);
      if (abort) exit(code);
   }
}

//version for cuSolver calls 
#define gpuSolErrchk(ans) { gpuSolAssert((ans), __FILE__, __LINE__); }
inline void gpuSolAssert(cusolverStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != CUSOLVER_STATUS_SUCCESS) 
   {
      fprintf(stderr,"GPUassert: %d %s %d\n", code, file, line);
      if (abort) exit(code);
   }
}



int GPU_id; 
int MPIrank;

void setupGPUs(int local_rank,int &num_devices){
//    printf("Compile time check:\n");
//#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
//    printf("This MPI library has CUDA-aware support.\n", MPIX_CUDA_AWARE_SUPPORT);
//#elif defined(MPIX_CUDA_AWARE_SUPPORT) && !MPIX_CUDA_AWARE_SUPPORT
//    printf("This MPI library does not have CUDA-aware support.\n");
//#else
//    printf("This MPI library cannot determine if there is CUDA-aware support.\n");
//#endif /* MPIX_CUDA_AWARE_SUPPORT */
    MPIrank = local_rank;

   printf("Run time check:\n");
#if defined(MPIX_CUDA_AWARE_SUPPORT)
    if (1 == MPIX_Query_cuda_support()) {
        printf("This MPI library has CUDA-aware support.\n");
    } else {
        printf("This MPI library does not have CUDA-aware support.\n");
    }
#else /* !defined(MPIX_CUDA_AWARE_SUPPORT) */
    printf("This MPI library cannot determine if there is CUDA-aware support.\n");
#endif /* MPIX_CUDA_AWARE_SUPPORT */

    //determines how many graphic cards are in the node and maps a rank to an accelerator
    gpuErrchk(cudaGetDeviceCount(&num_devices));
    gpuErrchk(cudaSetDevice(local_rank));
    gpuErrchk(cudaGetDevice(&GPU_id));

    watch(num_devices);
    watch(local_rank);
    watch(GPU_id);

    //how much memory is availble
    size_t free, total;
    cudaMemGetInfo( &free, &total );
    cout << "GPU " << GPU_id << " memory: free= " << free << ", total=" << total  
        << " % is in use: " << (1 - free/ (double) total) *100 << endl;

}

void makeCSR(int nnz, int dim, int *h_cooRows, int *h_cooCols, double *h_cooVals,
            int *&d_cooCols, int *&d_rowPtr,double *&d_cooVals_sorted,
            int *&d_cscRows,
            int *&d_cscColPtr, 
            double *&d_cscVals){ 
     
    /*Input: The pointers to the COOMatrix in the host memory
    Output: The pointers to the CSRMatrix in the device memory 
    The elements are first sorted as row-major and then the rows 
    are compressed
     */
    cout << "started make CSR" << endl;
    cusparseHandle_t handle = NULL;
    cusparseStatus_t status = CUSPARSE_STATUS_SUCCESS;
    gpuSpErrchk(cusparseCreate(&handle));
    cudaStream_t stream = NULL;
    gpuSpErrchk(cusparseSetStream(handle, stream));
    cudaError_t cudaStat1 = cudaSuccess;
    
    int *d_cooRows = NULL;
    int *d_P       = NULL;
    size_t pBufferSizeInBytes = 0;
    void *pBuffer = NULL;
    double *d_cooVals = NULL;

    //cast the std::complex to cuda complex

    size_t free, total;
    cudaMemGetInfo( &free, &total );
//    cout << "GPU " << GPU_id << " memory: free= " << free << ", total=" << total  
//        << " % is in use: " << (1 - free/ (double) total) *100 << endl;


    
    cout << "Matrix parem on GPU " <<"dim: " << dim << "nnz: " << nnz << endl;
    cout << "d_cooCols" << d_cooCols << endl;
//    cusparseHandle_t handle = NULL;
/* step 2: allocate buffer for sorting the COO matrix*/
    gpuSpErrchk(cusparseXcoosort_bufferSizeExt(
        handle,
        dim,
        dim,
        nnz,
        d_cooRows,
        d_cooCols,
        &pBufferSizeInBytes
    ));
    //assert(CUSPARSE_STATUS_SUCCESS  == status);

    printf("pBufferSizeInBytes = %lld bytes \n", (long long)pBufferSizeInBytes);


    //Allocate the device memory 
    gpuErrchk(cudaMalloc( &d_cooRows, sizeof(int)*nnz));
    gpuErrchk(cudaMalloc( &d_cooCols, sizeof(int)*nnz));
    gpuErrchk(cudaMalloc( &d_P      , sizeof(int)*nnz));
    gpuErrchk(cudaMalloc( &d_cooVals, sizeof(double)*nnz));
    gpuErrchk(cudaMalloc( &pBuffer, sizeof(char)* pBufferSizeInBytes));

    gpuErrchk(cudaMalloc(&d_rowPtr, sizeof(int)*(dim+1)));

    printf("Row pointer is fine \n");
    //copy the data to the device
    gpuErrchk(cudaMemcpy(d_cooRows, h_cooRows, sizeof(int)*nnz   , cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_cooCols, h_cooCols, sizeof(int)*nnz   , cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_cooVals, h_cooVals, sizeof(double)*nnz, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
     cudaMemGetInfo( &free, &total );
    cout << "GPU " << GPU_id << " memory: free= " << free << ", total=" << total  
        << " % is in use: " << (1 - free/ (double) total) *100 << endl;


    /* step 3: setup permutation vector P to identity */
    gpuSpErrchk(cusparseCreateIdentityPermutation(
        handle,
        nnz,
        d_P));
    /* step 4: sort COO format by Row */
    gpuSpErrchk(cusparseXcoosortByRow(
        handle,
        dim,
        dim,
        nnz,
        d_cooRows,
        d_cooCols,
        d_P,
        pBuffer
    ));

    cudaDeviceSynchronize();
/* step 5: gather sorted cooVals */
    
    gpuErrchk(cudaFree(pBuffer));
    gpuErrchk(cudaMalloc( &d_cooVals_sorted, sizeof(double)*nnz));

    gpuSpErrchk(cusparseDgthr(
        handle,
        nnz,
        d_cooVals,
        d_cooVals_sorted,
        d_P,
        CUSPARSE_INDEX_BASE_ZERO
    )) ;
    
    cudaDeviceSynchronize();
    //convert the matrix to CSR
    
    gpuSpErrchk(cusparseXcoo2csr(
            handle,
            d_cooRows,
            nnz,
            dim,
            d_rowPtr,
            CUSPARSE_INDEX_BASE_ZERO
        ));

    gpuErrchk(cudaFree(d_cooVals));
    gpuErrchk(cudaFree(d_cooRows));
    gpuErrchk(cudaFree(d_P));

    cudaDeviceSynchronize();
     cudaMemGetInfo( &free, &total );
    cout << "GPU " << GPU_id << " memory: free= " << free << ", total=" << total  
        << " % is in use: " << (1 - free/ (double) total) *100 << endl;


    //get also the CSC matrix that is the matrix the represents the lower half 
    //convert Csr to csc after freeing the Buffer
    //with the new SPMV kernels Transpose is a factor of 11 slower than normal
    //experimental
    gpuErrchk(cudaMalloc( &d_cscRows, sizeof(int)*nnz));
    gpuErrchk(cudaMalloc( &d_cscColPtr, sizeof(int)*(dim+1)));
    gpuErrchk(cudaMalloc( &d_cscVals, sizeof(double)*nnz));
     cudaMemGetInfo( &free, &total );
    cout << "GPU " << GPU_id << " memory: free= " << free << ", total=" << total  
        << " % is in use: " << (1 - free/ (double) total) *100 << endl;

    cudaDeviceSynchronize();
    gpuSpErrchk(cusparseCsr2cscEx2_bufferSize(handle,
                                     dim,
                                     dim,
                                     nnz,
                                     d_cooVals_sorted,
                                     d_rowPtr,
                                     d_cooCols,
                                     d_cscVals,
                                     d_cscColPtr,
                                     d_cscRows,
                                     CUDA_R_64F,
                                     CUSPARSE_ACTION_NUMERIC,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     CUSPARSE_CSR2CSC_ALG2,
                                     &pBufferSizeInBytes));

    gpuErrchk(cudaMalloc( &pBuffer,  pBufferSizeInBytes));
    watch(pBufferSizeInBytes);
    cudaDeviceSynchronize();

    gpuSpErrchk(cusparseCsr2cscEx2(handle,
                                     dim,
                                     dim,
                                     nnz,
                                     d_cooVals_sorted,
                                     d_rowPtr,
                                     d_cooCols,
                                     d_cscVals,
                                     d_cscColPtr,
                                     d_cscRows,
                                     CUDA_R_64F,
                                     CUSPARSE_ACTION_NUMERIC,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     CUSPARSE_CSR2CSC_ALG2,
                                     pBuffer));

    cudaDeviceSynchronize();
    gpuErrchk(cudaFree(pBuffer));
    gpuSpErrchk(cusparseDestroy(handle)) ;
    cudaDeviceSynchronize();

    cout << "finished make CSR" << endl;
     cudaMemGetInfo( &free, &total );
    cout << "GPU " << GPU_id << " memory: free= " << free << ", total=" << total  
        << " % is in use: " << (1 - free/ (double) total) *100 << endl;


    //we now have the csr matrix consisting out of 
    //d_cooCols, d_rowPtr, d_cooVals_sorted
    //these pointers are to the device memory are then returned to the c++ code
    
}



void ft_calc(InputParameter Inp,double t,complex<double> *ft){
    double phi = 0.0;
    double temp = sin(M_PI/Inp.T_puls*t);
    double AA = Inp.E0/Inp.om * temp * temp * sin(Inp.om*t+phi);
    const double c_light=137.036;
    //Velocity gauge 
    ft[0] = 1.0;
    ft[1] = AA*1i;
    ft[2] = 0.5*AA*AA*1i/c_light;
    ft[3] = 0.0;
    ft[4] = 0.0;
    ft[5] = 0.0;
    ft[13] = 1i;
    return;
}    



void allocateMemoryPropagator(InputParameter Inp, 
    int dim,
    cuDoubleComplex *&dQ, 
    cuDoubleComplex *&dd,
    double *&od,
    cuDoubleComplex *&dy,
    std::complex<double> *y)
    {
        //Initiallize the memory once
        gpuErrchk(cudaMallocManaged(&dQ,dim*Inp.dim_kryl*sizeof(cuDoubleComplex)));
        gpuErrchk(cudaMallocManaged(&od,Inp.dim_kryl*sizeof(double)));
        gpuErrchk(cudaMallocManaged(&dd,Inp.dim_kryl*sizeof(cuDoubleComplex)));
        gpuErrchk(cudaMalloc(&dy,dim*sizeof(cuDoubleComplex)));

        gpuErrchk(cudaMemcpy(dy,y,dim*2* sizeof(double),cudaMemcpyHostToDevice));
    }


void freeMemoryPropagator(InputParameter Inp, 
    int dim,
    cuDoubleComplex *&dQ, 
    cuDoubleComplex *&dd,
    double *&od,
    cuDoubleComplex *&dy,
    std::complex<double> *y)
    {

        gpuErrchk(cudaMemcpy(y,dy,dim*2* sizeof(double),cudaMemcpyDeviceToHost));
        cudaFree(dQ);
        cudaFree(dy);
        cudaFree(dd);
        cudaFree(od);
    }
    
    /*
__global__ void buildh(int *dim_kryl,
        double *od,
        cuDoubleComplex *dd,
        cuDoubleComplex *dh)
{
    //helper function that builds the tridiagonal h from od,d
    for (int i = 1; i < *dim_kryl; i++){
        dh[i-1+i* *dim_kryl] = {od[i-1],0};
        dh[i+(i-1)* *dim_kryl] = {od[i-1],0};
    }

    for (int i = 0; i < *dim_kryl; i++){
        dh[i+i* *dim_kryl] = dd[i];
    }
}
*/
//
//__global__ void propagate(){
    //the unitary propagate defined by the QM postulates
//    }
  
void physPropagator(
    InputParameter &Inp,
    int dim_kryl,int dim,
    double normStart, double dt, 
    cuDoubleComplex *&dQ,
    cuDoubleComplex *&dd,
    double *od,
    cuDoubleComplex *&dy,
    std::complex<double> *y){


    //define the descibtors
    double alpha = 1.0;
    cuDoubleComplex alphaC = {1.0,0.0};
    double beta = 1.0;
    cuDoubleComplex betaC = {1.0,0.0};


    cublasHandle_t handle_blas = NULL;
    cudaStream_t stream = NULL;
    gpuBlErrchk(cublasCreate(&handle_blas));
    gpuBlErrchk(cublasSetStream(handle_blas,stream));
       

    //how much memory is availble
    //bp("Begin phys propagator");
    size_t free, total;
    //cudaMemGetInfo( &free, &total );
    //allocate memory for h,y,x and Q (dh,dY,dX,dQ) amd allocate memory
   
    //imaginary
    cuDoubleComplex *dh,*dX,*dY,*dY_rank,*dY_temp;
    gpuErrchk(cudaMallocManaged(&dh,Inp.dim_kryl*Inp.dim_kryl*sizeof(cuDoubleComplex)));
   
   
   
    //put this to a GPU kernel
    //buildhWrapper(&Inp.dim_kryl,od,dd,dh)
    //buildh<<<1,1>>>(&Inp.dim_kryl,od,cuDoubleComplex dd,cuDoubleComplex *dh);
//{
    
    //build dh from dd and od
    gpuErrchk(cudaMemset(dh,0,Inp.dim_kryl*Inp.dim_kryl*sizeof(cuDoubleComplex)));
    cudaDeviceSynchronize();
    //construct h from od and d 
    for (int i = 1; i < Inp.dim_kryl; i++){
        dh[i-1+i*Inp.dim_kryl] = {od[i-1],0};
        dh[i+(i-1)*Inp.dim_kryl] = {od[i-1],0};
    }

    for (int i = 0; i < Inp.dim_kryl; i++){
        dh[i+i*Inp.dim_kryl] = dd[i];
    }

    cudaDeviceSynchronize();

    //set up cuSolver h 
    //allocate h,W
    double *lambda;
    cuDoubleComplex *W;
    cuDoubleComplex *tmpMatrix;
    cuDoubleComplex *vecH; 

    gpuErrchk(cudaMallocManaged(&lambda,Inp.dim_kryl*sizeof(double)));
    gpuErrchk(cudaMallocManaged(&vecH,Inp.dim_kryl*sizeof(cuDoubleComplex)));

    gpuErrchk(cudaMallocManaged(&W,Inp.dim_kryl*Inp.dim_kryl*sizeof(cuDoubleComplex)));
    gpuErrchk(cudaMallocManaged(&tmpMatrix,Inp.dim_kryl*Inp.dim_kryl*sizeof(cuDoubleComplex)));
    
    cusolverDnHandle_t handle_solver = NULL;
    cusolverDnCreate(&handle_solver);
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    //cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
    int *devInfo = NULL;
    gpuErrchk(cudaMalloc (&devInfo, sizeof(int)));

    int sizehSolverBuffer;
    cuDoubleComplex *hSolverBuffer;

    //cuda solver on h; P are the eigenvectors
    cusolverDnZheevd_bufferSize(
        handle_solver,
        jobz,
        uplo,
        Inp.dim_kryl,
        dh,
        Inp.dim_kryl,
        lambda,
        &sizehSolverBuffer);

    //allocate buffer dense solver for h
    gpuErrchk(cudaMalloc(&hSolverBuffer, sizehSolverBuffer * sizeof(cuDoubleComplex)));

 
        
        gpuErrchk(cudaMemset(lambda,0,Inp.dim_kryl*sizeof(double)));
        //watch(sizehSolverBuffer);
        //solver for h 
        gpuSolErrchk(cusolverDnZheevd(
            handle_solver,
            jobz,
            uplo,
            Inp.dim_kryl,
            dh,
            Inp.dim_kryl,
            lambda,
            hSolverBuffer,
            sizehSolverBuffer,
            devInfo));
        
        cudaDeviceSynchronize();

        gpuErrchk(cudaMemset(W,0,Inp.dim_kryl*Inp.dim_kryl*sizeof(cuDoubleComplex)));
        for (int i = 0; i < Inp.dim_kryl; i++){
            //no complex exp function in complex.h 
            W[i+i*Inp.dim_kryl] = {cos(-1 * dt * lambda[i]) , sin(-1 * dt * lambda[i])};
        }
        cudaDeviceSynchronize();
        
        alphaC = {1.0,0.0};
        betaC = {1.0,0.0};

        gpuErrchk(cudaMemset(tmpMatrix,0,Inp.dim_kryl*Inp.dim_kryl*sizeof(cuDoubleComplex)));
        //W * h.inverse()
        gpuBlErrchk(cublasZgemm(handle_blas,
                CUBLAS_OP_N,CUBLAS_OP_T,
                Inp.dim_kryl,Inp.dim_kryl,Inp.dim_kryl,
                &alphaC,
                W,Inp.dim_kryl,
                dh,Inp.dim_kryl,
                &betaC,
                tmpMatrix,Inp.dim_kryl));

        cudaDeviceSynchronize();
        //h * tmpMatrix
        gpuErrchk(cudaMemset(W,0,Inp.dim_kryl*Inp.dim_kryl*sizeof(cuDoubleComplex)));
        
        cudaDeviceSynchronize();
                   cudaDeviceSynchronize();
        gpuBlErrchk(cublasZgemm(handle_blas,
                CUBLAS_OP_N,CUBLAS_OP_N,
                Inp.dim_kryl,Inp.dim_kryl,Inp.dim_kryl,
                &alphaC,
                dh,Inp.dim_kryl,
                tmpMatrix,Inp.dim_kryl,
                &betaC,
                W,Inp.dim_kryl));

                   cudaDeviceSynchronize();
       
        gpuErrchk(cudaMemset(dy,0,dim*sizeof(cuDoubleComplex)));
        cudaDeviceSynchronize();
        //new dX
        //y = normStart * Q.block(0,0,dim,Inp.dim_kryl) * vecH
        alphaC = {normStart,0};
        gpuBlErrchk(cublasZgemv(handle_blas,
                    CUBLAS_OP_N,
                    dim,Inp.dim_kryl,
                    &alphaC,
                    dQ,dim,
                    W,1,
                    &betaC,
                    dy,1));
                    
        cudaDeviceSynchronize();
        //copy back to dY_temp, so next iteration can start

        //move new y to host
        
        //Free Memory
        gpuErrchk(cudaFree(dh));
        gpuErrchk(cudaFree(lambda));
        gpuErrchk(cudaFree(vecH));
        gpuErrchk(cudaFree(W));
        gpuErrchk(cudaFree(tmpMatrix));
        gpuErrchk(cudaFree(devInfo));
        gpuErrchk(cudaFree(hSolverBuffer));

    gpuSolErrchk(cusolverDnDestroy(handle_solver)) ;
    gpuBlErrchk(cublasDestroy(handle_blas)) ;


    
//    bp("End phys propagator");
//    cudaMemGetInfo( &free, &total );
 //   cout << "GPU " << GPU_id << " memory: free= " << free << ", total=" << total  
 //       << " % is in use: " << (1 - free/ (double) total) *100 << endl;


    
    }


void lanczosCUDA(std::vector<int> nnz,int dim,int dim_kryl, int nmbrLEGO, double &normStart, 
        std::vector<int*> &d_cooCols, 
        std::vector<int*> &d_rowPtr,
        std::vector<double*> &d_cooVals_sorted, 
        std::vector<int*> &d_cscRows,
        std::vector<int*>  &d_cscColPtr,
        std::vector<double*> &d_cscVals, 
        cuDoubleComplex *dX, std::complex<double> *ft,
        double *dod,cuDoubleComplex *dQ,cuDoubleComplex *dd){ 

    std::setprecision (15);
   /*SpMV The sparse matrix vector product we need for simulationTDSE
      dX_c is the complex conjugate of dX*/
    
   /* Input
    Sparse CSR: matrix  int *d_cooCols, int *d_rowPtr, double *d_cooVals_sorted,
    x: initial vector
    dim_kryl: dimension of Krylov subspace L in morten code
    dim: dimesion of matrix H_rank
    Output:
    Q: orthogonal Krylov space 
    h: o,od they form h together  
    */


    cusparseHandle_t handle_sparse = NULL;
    cublasHandle_t handle_blas = NULL;
    cudaStream_t stream = NULL;
    //cusparseStatus_t status = CUSPARSE_STATUS_SUCCESS;
    gpuSpErrchk(cusparseCreate(&handle_sparse));
    gpuSpErrchk(cusparseSetStream(handle_sparse, stream));

    gpuBlErrchk(cublasCreate(&handle_blas));
    gpuBlErrchk(cublasSetStream(handle_blas,stream));
    double *dX_real,*dX_imag,*dY_real,*dY_imag;

    //move data to the device
    gpuErrchk(cudaMalloc(&dX_real, dim*sizeof(double)));
    gpuErrchk(cudaMalloc(&dX_imag, dim*sizeof(double)));
   
    gpuErrchk(cudaMalloc(&dY_real, dim*sizeof(double)));
    gpuErrchk(cudaMalloc(&dY_imag, dim*sizeof(double)));
    
   
    //allocate needed device memory 
   
    //set up Lanczos 
    //real
    double *dY_r,*dY_i;
    gpuErrchk(cudaMalloc(&dY_r,dim * sizeof(double)));
    gpuErrchk(cudaMalloc(&dY_i,dim * sizeof(double)));
   
    //imaginary
    cuDoubleComplex *dY,*dY_rank,*dY_temp;
    gpuErrchk(cudaMallocManaged(&dY,dim*sizeof(cuDoubleComplex)));
    gpuErrchk(cudaMalloc(&dY_rank, dim*sizeof(cuDoubleComplex)));
    gpuErrchk(cudaMalloc(&dY_temp, dim*sizeof(cuDoubleComplex)));
        
    //normalize the input vector and retrurn the normStart
    gpuErrchk(cudaMemcpy(dY_temp,dX,sizeof(double)*2*dim,cudaMemcpyDeviceToDevice));
    gpuBlErrchk(cublasDznrm2(handle_blas, dim, dY_temp, 1, &normStart));
                
    if (normStart == 0) {bp( "Division by 0 while rescaling y"); exit(0);}
    
    //x = y/od[i];
    cuDoubleComplex alphaC = {1/normStart, 0.0};
    gpuErrchk(cudaMemset(dX,0,dim*sizeof(cuDoubleComplex)));
    gpuBlErrchk(cublasZaxpy(handle_blas, dim, &alphaC, dY_temp ,1, dX,1));
     
    //the complex vecersion of dy it is used for the reorthogonalization 
    //copy X to the device 
    //cudaMemcpy(dX, X, sizeof(double)*dim*2   , cudaMemcpyHostToDevice);
    //cout << "moved X to dX" << endl;
    //Q.setZero();
    gpuErrchk(cudaMemset(dQ,0,dim*dim_kryl*sizeof(cuDoubleComplex)));
    gpuErrchk(cudaMemset(dd,0,dim_kryl*sizeof(cuDoubleComplex)));
    gpuErrchk(cudaMemset(dod,0,(dim_kryl-1)*sizeof(double)));
    //setting dY explicitly to 0
    gpuErrchk(cudaMemset(dY,0,dim*sizeof(cuDoubleComplex)));
    //Q.col(0) = y;
    gpuErrchk(cudaMemcpy(&dQ[0],dX, dim*sizeof(cuDoubleComplex),cudaMemcpyDeviceToDevice));
 
 
    //define the descibtors
    double alpha = 1.0;
    double beta = 1.0;

    std::vector<cusparseSpMatDescr_t> matA(nmbrLEGO);
    std::vector<cusparseSpMatDescr_t> matAT(nmbrLEGO); //the transpose of A
    cusparseDnVecDescr_t vecX_real, vecY_real, vecX_imag, vecY_imag;

    if(MPIrank ==0){
        int k=0;
        gpuSpErrchk(cusparseCreateCsr(&matA[k], dim, dim, nnz[k],
            d_rowPtr[k], d_cooCols[k], d_cooVals_sorted[k],
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) );

        gpuSpErrchk(cusparseCreateCsr(&matAT[k], dim, dim, nnz[k],
            d_cscColPtr[k],d_cscRows[k],d_cscVals[k],
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F)) ;
    }

    for(int k=1; k < nmbrLEGO; k++){
        gpuSpErrchk(cusparseCreateCsr(&matA[k], dim, dim, nnz[k],
            d_rowPtr[k], d_cooCols[k], d_cooVals_sorted[k],
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) );

        gpuSpErrchk(cusparseCreateCsr(&matAT[k], dim, dim, nnz[k],
            d_cscColPtr[k],d_cscRows[k],d_cscVals[k],
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F)) ;
    }

    gpuSpErrchk(cusparseCreateDnVec(&vecX_real, dim, dX_real, CUDA_R_64F)) ;
    gpuSpErrchk(cusparseCreateDnVec(&vecX_imag, dim, dX_imag, CUDA_R_64F)) ;


    gpuSpErrchk(cusparseCreateDnVec(&vecY_real, dim, dY_real, CUDA_R_64F)) ;
    gpuSpErrchk(cusparseCreateDnVec(&vecY_imag, dim, dY_imag, CUDA_R_64F)) ;

    //declare the buffer 
    size_t bufferSize = 0;
    size_t bufferSize2 = 0;
    void *dBuffer = NULL;
    
    gpuSpErrchk(cusparseSpMV_bufferSize(
         handle_sparse, CUSPARSE_OPERATION_TRANSPOSE,
         &alpha, matA[nmbrLEGO-1], vecX_real, &beta, vecY_real, CUDA_R_64F,
         CUSPARSE_CSRMV_ALG1, &bufferSize)) ;
    
    gpuSpErrchk(cusparseSpMV_bufferSize(
         handle_sparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
         &alpha, matA[nmbrLEGO-1], vecX_real, &beta, vecY_real, CUDA_R_64F,
         CUSPARSE_CSRMV_ALG1, &bufferSize)) ;
    
    bufferSize = std::max(bufferSize,bufferSize2);

    gpuErrchk(cudaMalloc(&dBuffer, bufferSize)) ;

            cudaDeviceSynchronize();
       
          double *h_dY_real,*h_dY_imag;
    for (int i = 0; i < dim_kryl; i++){
        MPI_Bcast(dX,dim*2,MPI_DOUBLE,0,MPI_COMM_WORLD);
            cudaDeviceSynchronize();
     
        //update dX_real and dX_imaginary 
        gpuErrchk(cudaMemcpy2D(dX_real,sizeof(double),dX, 2 * sizeof(double), sizeof(double),dim,
                 cudaMemcpyDeviceToDevice));
 
        gpuErrchk(cudaMemcpy2D(dX_imag,sizeof(double), reinterpret_cast<double*>(&dX[0].y) 
                    , 2 *  sizeof(double), sizeof(double), dim,cudaMemcpyDeviceToDevice));
                  //Memset is necesarry
        gpuErrchk(cudaMemset(dY_real,0,dim*sizeof(double)));
        gpuErrchk(cudaMemset(dY_imag,0,dim*sizeof(double)));
                 //create the matrix and vector descriptors
        for(int j = 0; j < nmbrLEGO; j++){
            beta=1.0;
            //use the coreesponding ft to each LEGO
            //real part
            //this next environment is for LEGOs with only real part
            if (j==0 && MPIrank==0){
                alpha=ft[j].real();
                gpuSpErrchk(cusparseSpMV(handle_sparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA[j], vecX_real, &beta, vecY_real, CUDA_R_64F,
                                  CUSPARSE_CSRMV_ALG1, 
                                 dBuffer)) ;
                gpuSpErrchk(cusparseSpMV(handle_sparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA[j], vecX_imag, &beta, vecY_imag, CUDA_R_64F,
                                 CUSPARSE_CSRMV_ALG1,
                                 dBuffer)) ;
            }
            
            //this environment is for LEGOs with only imaginary part.
            //This leads to X_real*i*ft.imag()=X_imag*ft and vice versa for X_imag
            if(j!=0){
                //the actual SpMV product for a complex antisymmetric matrix with a complex vector
                //printf("Before SpMV product \n");
                //the real parts of vector_x and vector_y
                alpha=-ft[j].imag();
                gpuSpErrchk(cusparseSpMV(handle_sparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA[j], vecX_imag, &beta, vecY_real, CUDA_R_64F,
                                 CUSPARSE_CSRMV_ALG1, 
                                 dBuffer)) ;
                alpha=ft[j].imag();
                gpuSpErrchk(cusparseSpMV(handle_sparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA[j], vecX_real, &beta, vecY_imag, CUDA_R_64F,
                                 CUSPARSE_CSRMV_ALG1, 
                                 dBuffer)) ;

                //the imaginary parts of vector_x and vector_y
                //std::conj() is to multiply dX_imag with -1
                //if LEGO != 0 then there should be no diagonal elements
                //LEGO 0 is the diagonal therefore we multiply only once with it 
            
                alpha=ft[j].imag();
                gpuSpErrchk(cusparseSpMV(handle_sparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matAT[j], vecX_imag, &beta, vecY_real, CUDA_R_64F,
                                 CUSPARSE_CSRMV_ALG1,
                                 dBuffer)) ;
                alpha=-ft[j].imag();
                 gpuSpErrchk(cusparseSpMV(handle_sparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matAT[j], vecX_real, &beta, vecY_imag, CUDA_R_64F,
                                 CUSPARSE_CSRMV_ALG1, 
                                 dBuffer)) ;
                 


            }
                     
        }
 
            cudaDeviceSynchronize();
       // pbp( "Matrix Vector is done");
        //build complex cuda Array from two real arrays
       gpuErrchk(cudaMemcpy2D(dY_rank, 2* sizeof(double),dY_real, sizeof(double), 
                    sizeof(double),dim, cudaMemcpyDeviceToDevice));
        gpuErrchk(cudaMemcpy2D( reinterpret_cast<double*> (&dY_rank[0].y),
                    2 * sizeof(double),dY_imag, sizeof(double), sizeof(double),dim,
                 cudaMemcpyDeviceToDevice));

            cudaDeviceSynchronize();
        MPI_Reduce(dY_rank,dY,dim*2,MPI_DOUBLE,MPI_SUM, 0,MPI_COMM_WORLD);

        if(MPIrank == 0){
            gpuBlErrchk(cublasZdotc(handle_blas,
                    dim,
                    dX,
                    1,
                    dY,
                    1,
                    &dd[i]
                ));

            //Full reorthogonalization (x2):
            //does need access to temporary vector y_temp
            for(int k = 0; k < 2;k++){  //Full reorthogonalization (x2) 
                gpuErrchk(cudaMemset(dY_rank,0,dim*sizeof(cuDoubleComplex)));
                gpuErrchk(cudaMemset(dY_temp,0,dim*sizeof(cuDoubleComplex)));

                cuDoubleComplex alphaC= {1.0,0.0};
                //double alpha=1.0;
                //no memset to 0 needed
                //use dY_rank as temporary buffer
                cuDoubleComplex betaC={1.0,0.0};
                gpuBlErrchk(cublasZgemv(handle_blas,
                        CUBLAS_OP_C,
                        dim,
                        i+1,
                        &alphaC,
                        dQ,
                        dim,
                        dY,
                        1,
                        &betaC,
                        dY_rank,
                        1
                    ));
    cudaDeviceSynchronize();
                gpuBlErrchk(cublasZgemv(handle_blas,
                        CUBLAS_OP_N,
                        dim,
                        i+1,
                        &alphaC,
                        dQ,
                        dim,
                        dY_rank,
                        1,
                        &betaC,
                        dY_temp,
                        1
                    ));
                           
    cudaDeviceSynchronize();
                //subtract from other vector 
                alphaC= {-1.0,0.0};
                gpuBlErrchk(cublasZaxpy(handle_blas, 
                        dim,
                        &alphaC,
                        dY_temp,
                        1,
                        dY,
                        1
                    ));

            }

    cudaDeviceSynchronize();


            if (i < dim_kryl-1){
                //save the norm of y in od[i]
                //od(i) = dydx.norm();
                gpuBlErrchk(cublasDznrm2(handle_blas, dim, dY, 1, &dod[i]));
                
                cuDoubleComplex alphaC = {1/dod[i], 0.0};
                gpuErrchk(cudaMemset(dX,0,dim*sizeof(cuDoubleComplex)));
         
                gpuBlErrchk(cublasZaxpy(handle_blas, dim, &alphaC, dY ,1, dX,1));
                
                gpuErrchk(cudaMemcpy(&dQ[dim*(i+1)],dX, dim* sizeof(cuDoubleComplex),
                        cudaMemcpyDeviceToDevice ));
                
            }
        }
    }

    if(MPIrank == 0){
        int k=0;
        gpuSpErrchk(cusparseDestroySpMat(matA[k])) ;
        gpuSpErrchk(cusparseDestroySpMat(matAT[k])) ;

    }

    for(int k = 1; k < nmbrLEGO;k++){
        gpuSpErrchk(cusparseDestroySpMat(matA[k])) ;
        gpuSpErrchk(cusparseDestroySpMat(matAT[k])) ;
    }

    gpuSpErrchk(cusparseDestroyDnVec(vecX_real)) ;
    gpuSpErrchk(cusparseDestroyDnVec(vecX_imag)) ;
    gpuSpErrchk(cusparseDestroyDnVec(vecY_real)) ;
    gpuSpErrchk(cusparseDestroyDnVec(vecY_imag)) ;
    gpuSpErrchk(cusparseDestroy(handle_sparse)) ;
    gpuBlErrchk(cublasDestroy(handle_blas)) ;

   
    //cout << "After copy to device" << endl;
    //gpuErrchk(cudaFree(dX));
    gpuErrchk(cudaFree(dY));
    gpuErrchk(cudaFree(dY_rank));
    gpuErrchk(cudaFree(dY_temp));

    gpuErrchk(cudaFree(dX_real));
    gpuErrchk(cudaFree(dX_imag));
    gpuErrchk(cudaFree(dY_real));
    gpuErrchk(cudaFree(dY_imag));

    gpuErrchk(cudaFree(dY_r));
    gpuErrchk(cudaFree(dY_i));

    gpuErrchk(cudaFree(dBuffer));

}

void genEigenSolverCUDA(int m,double *A, double *B,double *W,double *V){
    //dim_H,H.data(),S.data(),Energy,Eigenvectors.data()
    cusolverDnHandle_t cusolverH = NULL;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    //const int m = 3;
    const int lda = m;
    double *d_A = NULL;
    double *d_B = NULL;
    double *d_W = NULL;
    int *devInfo = NULL;
    double *d_work = NULL;
    int  lwork = 0;
    int info_gpu = 0;

// step 1: create cusolver/cublas handle
    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

// step 2: copy A and B to device
    cudaStat1 = cudaMalloc ((void**)&d_A, sizeof(double) * lda * m);
    cudaStat2 = cudaMalloc ((void**)&d_B, sizeof(double) * lda * m);
    cudaStat3 = cudaMalloc ((void**)&d_W, sizeof(double) * m);
    //int *devInfo = NULL;
    cudaStat4 = cudaMalloc ((void**)&devInfo, sizeof(int));
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);

    cudaStat1 = cudaMemcpy(d_A, A, sizeof(double) * lda * m, cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(d_B, B, sizeof(double) * lda * m, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);

// step 3: query working space of sygvd
    cusolver_status = cusolverDnCreate(&cusolverH);
    cusolverEigType_t itype = CUSOLVER_EIG_TYPE_1; // A*x = (lambda)*B*x
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
    //cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
    cusolver_status = cusolverDnDsygvd_bufferSize(
        cusolverH,
        itype,
        jobz,
        uplo,
        m,
        d_A,
        lda,
        d_B,
        lda,
        d_W,
        &lwork);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
    assert(cudaSuccess == cudaStat1);

// step 4: compute spectrum of (A,B)
    cusolver_status = cusolverDnDsygvd(
        cusolverH,
        itype,
        jobz,
        uplo,
        m,
        d_A,
        lda,
        d_B,
        lda,
        d_W,
        d_work,
        lwork,
        devInfo);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);
    cudaStat1 = cudaMemcpy(W, d_W, sizeof(double)*m, cudaMemcpyDeviceToHost);
    cudaStat2 = cudaMemcpy(V, d_A, sizeof(double)*lda*m, cudaMemcpyDeviceToHost);
    cudaStat3 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);

    printf("after sygvd: info_gpu = %d\n", info_gpu);
    assert(0 == info_gpu);

// free resources
    if (d_A    ) cudaFree(d_A);
    if (d_B    ) cudaFree(d_B);
    if (d_W    ) cudaFree(d_W);
    if (devInfo) cudaFree(devInfo);
    if (d_work ) cudaFree(d_work);

    if (cusolverH) cusolverDnDestroy(cusolverH);

}



void CUDAFinalize(int nmbrLEGO,
        std::vector<int*> &d_rowPtr,
        std::vector<int*> &d_cooCols, 
        std::vector<double*> &d_cooVals_sorted, 
        std::vector<int*> &d_cscRows,
        std::vector<int*>  &d_cscColPtr,
        std::vector<double*> &d_cscVals){ 

    //free the matrices stored on the GPU
        for(int i =0 ; i< nmbrLEGO; i++){
            //the csr matrix 
            cudaFree(d_cooCols[i]);
            cudaFree(d_rowPtr[i]);
            cudaFree(d_cooVals_sorted[i]);

            //the csc matrix
            cudaFree(d_cscRows[i]);
            cudaFree(d_cscColPtr[i]);
            cudaFree(d_cscVals[i]);
        }


    //free my arrays
    cudaDeviceReset();
    cout << "CUDa memory has been reset" << endl;
}


