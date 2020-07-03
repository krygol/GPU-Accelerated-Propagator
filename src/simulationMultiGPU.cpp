#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Eigen/StdVector>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <mkl.h>
#include <mkl_spblas.h>
#include <fstream>
#include <iterator>
#include <vector>
#include <algorithm>
#include <complex>
#include <sstream>
#include <math.h>
#include <omp.h>
#include <numeric>
#include <chrono>
#include <ctime>
#include <limits>
#include <lyra/lyra.hpp>
#include <sys/stat.h>
#include <sys/types.h>
#include <cusparse.h>
#include <cusolverDn.h>
#include <cuComplex.h>
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "CUDAaccelerated.h"
#include "bsplines.h"
#include "InputParameter.h"

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#define watch(x) cout << (#x) << " is " << (x) << endl
#define pwatch(x) cout <<"rank " << rank <<" "<< (#x) << " is " << (x) << endl
#define bp(x) cout << "breakpoint" << (#x) << endl

using namespace Eigen;
using namespace std::complex_literals;
using namespace std::chrono;
using std::cout;
using std::endl;



int dim;
int rank,p,p_omp;
int64_t nnz_rank;
int states_chan_rank;
int states_chan_rank_begin;
int states_chan_rank_end;

//This is a COO matrix that holds a tupel of values for every matrix cell
class SpMatrix{
    //idea write a matrix class that has the functionality of init memory, write element,
    //multiply
    public: 
        std::vector<int> nnz;
        int dim;
        int nmbrLEGO=3;

        //The COO matrix
        std::vector<std::vector<int> > Row;
        std::vector<std::vector<int> > Col;
        std::vector<std::vector<double> > Value;
        
        //The CSR matrix
        std::vector<std::vector<int> > RowCSR;
        std::vector<std::vector<int> > ColCSR;
        std::vector<std::vector<double> > ValueCSR;

        //GPU accelerated matrix vector
        //pointers to the device memory 
        std::vector<int*> LEGO_d_cooCols;
        std::vector<int*> LEGO_d_rowPtr;
        std::vector<double* > LEGO_d_cooVals_sorted;
        std::vector<int*> LEGO_d_cscRows;
        std::vector<int*>  LEGO_d_cscColPtr;
        std::vector<double*> LEGO_d_cscVals; 

    void initMatrix(int nnz_input){
        Row.resize(nmbrLEGO);
        Col.resize(nmbrLEGO);
        Value.resize(nmbrLEGO);

        RowCSR.resize(nmbrLEGO);
        ColCSR.resize(nmbrLEGO);
        ValueCSR.resize(nmbrLEGO);

        LEGO_d_cooVals_sorted.resize(nmbrLEGO);
        LEGO_d_cooCols.resize(nmbrLEGO);
        LEGO_d_rowPtr.resize(nmbrLEGO);
        LEGO_d_cscVals.resize(nmbrLEGO);
        LEGO_d_cscColPtr.resize(nmbrLEGO);
        LEGO_d_cscRows.resize(nmbrLEGO);

        for(int i = 0; i < nmbrLEGO; i++){
            Row[i].resize(nnz_input);
            Col[i].resize(nnz_input);
            Value[i].resize(nnz_input);
        }
        nnz.resize(nmbrLEGO);

    }


    void writeElement(int ind, int row, int col, int LEGOind,double value){
        //writes an element to the matrix
        Row[LEGOind][ind] = row;
        Col[LEGOind][ind] = col;
        Value[LEGOind][ind] = value;
    }

    void CSRMKL(){
        //allocate the memeory for the CSR matrices 
        watch(nmbrLEGO);
        for(int i = 0; i < nmbrLEGO; i++){
            watch(i);
            watch(nnz[i]);
            RowCSR[i].resize(dim+2);
            ColCSR[i].resize(nnz[i]);
            ValueCSR[i].resize(nnz[i]);
        }
        bp("Allocate memory");

        //build the CSR matrix
        for (int i = 0; i< nmbrLEGO;i++){
            int job[5] = {2,0,0,nnz[i],0};
            int info = 0;
            mkl_dcsrcoo(job,
                    &dim,
                    ValueCSR[i].data(),
                    ColCSR[i].data(),
                    RowCSR[i].data(),
                    &nnz[i],
                    Value[i].data(),
                    Row[i].data(),
                    Col[i].data(),
                    &info
                );
        }
        
    }

    void CSRonDevice(){
        //converts the matrices stored in COO on the host to CSR on the device by 
        //calling the accoarding CUDA function
        //iterate over the LEGO bricks
        if(rank==0){
             int i=0;
             makeCSR(nnz[i],dim,Row[i].data(),Col[i].data(),Value[i].data(),LEGO_d_cooCols[i],
                    LEGO_d_rowPtr[i], LEGO_d_cooVals_sorted[i],
                    LEGO_d_cscRows[i],
                    LEGO_d_cscColPtr[i], 
                    LEGO_d_cscVals[i]);
        }
        for(int i=1; i < nmbrLEGO; i++){
            makeCSR(nnz[i],dim,Row[i].data(),Col[i].data(),Value[i].data(),LEGO_d_cooCols[i],
                    LEGO_d_rowPtr[i], LEGO_d_cooVals_sorted[i],
                    LEGO_d_cscRows[i],
                    LEGO_d_cscColPtr[i], 
                    LEGO_d_cscVals[i]);

       }
    }

    void matrixVectorUpper(int LEGO,VectorXcd &x,VectorXcd &b,VectorXcd &ft){ 
       //This is a COO matrix vector product M * x = b
#pragma omp parallel
        {
        VectorXcd b_private= VectorXcd::Zero(dim);
#pragma omp for 
        for(int  i = 0; i < nnz[LEGO]; i++)
        {
                b_private(Row[LEGO][i]) += Value[LEGO][i] * x(Col[LEGO][i]) * ft(LEGO);
            if (Row[LEGO][i] != Col[LEGO][i]){
                b_private(Col[LEGO][i]) += Value[LEGO][i] * x(Row[LEGO][i]) * std::conj(ft[LEGO]);
            }
        }
#pragma omp critical 
        for(int i=0; i < dim; i++){
            b(i) += b_private(i);
        }
        }
    }

    //wrapper for MKL CSR SpMV
    void matrixVectorMKL(VectorXcd &x,VectorXcd &b,VectorXcd &ft){
        //as the matrix is real we seperate the vectors in real and iamg parts 
        //this allows a real double multiplications
        VectorXd dydx_rank_real = b.real();
        VectorXd dydx_rank_imag = b.imag();

       char transa = 'N';
        char matdescra[] = {
            'G', // type of matrix
            ' ', // triangular indicator (ignored in multiplication)
            ' ', // diagonal indicator (ignored in multiplication)
            'C'  // type of indexing
        };

        double beta = 1.0;
        for(int j = 0; j < nmbrLEGO; j++){
            VectorXcd y_local = x * ft(j);
            VectorXd y_local_real = y_local.real();
            VectorXd y_local_imag = y_local.imag();

            //SpMVCuda
            double alpha=1.0;
            //non transpose real x, real y, alpha = 1 
            mkl_dcsrmv(&transa,
                    &dim,
                    &dim,
                    &alpha, 
                    matdescra, 
                    ValueCSR[j].data(),
                    ColCSR[j].data(),
                    RowCSR[j].data(),
                    RowCSR[j].data()+1,
                    y_local_real.data(),
                    &beta,
                    dydx_rank_real.data());

            //non transpose imag x, imag y, alpha = 1 
             mkl_dcsrmv(&transa,
                    &dim,
                    &dim,
                    &alpha, 
                    matdescra, 
                    ValueCSR[j].data(),
                    ColCSR[j].data(),
                    RowCSR[j].data(),
                    RowCSR[j].data()+1,
                    y_local_imag.data(),
                    &beta,
                    dydx_rank_imag.data());
   
            if (j != 0){
                alpha=-1.0;
                char transa = 'T';
                //transpose real x, real y , alpha = -1 
                mkl_dcsrmv(&transa,
                    &dim,
                    &dim,
                    &alpha, 
                    matdescra, 
                    ValueCSR[j].data(),
                    ColCSR[j].data(),
                    RowCSR[j].data(),
                    RowCSR[j].data()+1,
                    y_local_real.data(),
                    &beta,
                    dydx_rank_real.data());
   
                //transpose imag x, imag y , alpha = -1 
                mkl_dcsrmv(&transa,
                    &dim,
                    &dim,
                    &alpha, 
                    matdescra, 
                    ValueCSR[j].data(),
                    ColCSR[j].data(),
                    RowCSR[j].data(),
                    RowCSR[j].data()+1,
                    y_local_imag.data(),
                    &beta,
                    dydx_rank_imag.data());
   

            }
        }
        //join into b 
        b.real()=dydx_rank_real;
        b.imag()=dydx_rank_imag;

    }
};

void Propergator(InputParameter &Inp,SpMatrix &H_rank,std::string PATH);

void BuildMatrix(InputParameter &Inp,SpMatrix &H_rank);

void getEigenStates(InputParameter Inp, int l,int states_chan,double dX,double &tmp_d,ArrayXd &weights,
        ArrayXd &x,ArrayXXd &B, ArrayXXd &d_B,ArrayXXd &diff_B,ArrayXXd &psi,ArrayXXd &d_psi
        ,ArrayXXd &diff_psi , MatrixXd &H_base, MatrixXd &S, ArrayXd &Energy);
 

void helperEigenStates(InputParameter Inp,int states_chan,double dX,double &tmp_d,
        ArrayXd &weights, ArrayXd &x,ArrayXXd &B,ArrayXXd &diff_B,MatrixXd &H_base, 
        MatrixXd &S);

void polarization(SpMatrix &M,int states_chan, int &ind, int index,int l,int lm_index_L,int 
        lm_index_R, double LM_faktor,double dX,ArrayXd &weights, ArrayXd &x, ArrayXXd &psi_1,
        ArrayXXd  &psi_2,ArrayXXd &d_psi_2);
void lanczos(InputParameter Inp,SpMatrix &A, int m,VectorXcd &y , MatrixXcd &Q, MatrixXcd &h,  
        VectorXcd &FT);


auto stoportho = high_resolution_clock::now();
auto startortho = high_resolution_clock::now();
auto durationortho = duration_cast<milliseconds>(startortho-startortho);


int main(int argc, const char** argv){
    //Read in the sparse matrix representing the Hamiltonian
    //The matrix is in COO format(also called matrix market format)
    //The data is stored in 4 different files each of them 
    //obtains a vector. 2 files hold the indices for our sparse matrix and 
    //2 hold the values that belong to the to indices. One is the value 
    //belonging to the Hamiltion the other one is an integer telling us which 
    //ft to use
    MPI_Init(NULL,NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    cout << "rank: " << rank << " of " << p << " is here" << endl;
    InputParameter Inp;

    //map the ranks to the devices(grpahic cards)
    //local rank within a node for multi node version
    //this has to be chagned 
    int local_rank = rank; 

    p_omp = omp_get_num_threads();
    watch(p_omp);
    
    //int id = omp_get_thread_num();
    std::setprecision (15);
   
    //Inp.ReadInput(argc,argv);
    auto cli   
        =   lyra::opt(Inp.nnz, "nnz")
            ["-nnz"]["--nnz"]
            ("Number of max nnz")
        | lyra::opt(Inp.om, "om")
            ["-om"]["--om"]
            ("Photon Energy")
        | lyra::opt(Inp.E0, "E0")
            ["-E0"]["--E0"]
            ("Electric field strength")
        |lyra::opt(Inp.T_cycle, "T_cycle")
            ["-T"]["--T_cycle"]
            ("length of pulse in cycles")
        |lyra::opt(Inp.timesteps, "timesteps")
            ["-t"]["--timesteps"]
            ("Number of timesteps")
        |lyra::opt(Inp.b,"b")
            ["-b"]["--b"]
            ("Boxsize")
        |lyra::opt(Inp.N_b,"N_b")
            ["-N_b"]["--N_b"]
            ("Number of gridpoints")
        |lyra::opt(Inp.Lmax,"Lmax")
            ["-Lmax"]["--Lmax"]
            ("Maximum angular momentum")
        |lyra::opt(Inp.Mmax,"Mmax")
            ["-Mmax"]["--Mmax"]
            ("Maximum magnetic quantum number")
        |lyra::opt(Inp.Emax,"Emax")
            ["-Emax"]["--Emax"]
            ("Maximum Energy in simulation")
        |lyra::opt(Inp.I0,"I0")
            ["-I0"]["--I0"]
            ("Laser Intensity in 1e12")
        |lyra::opt(Inp.init_state,"Initial state")
            ["-init"]["--init"]
            ("Initial state of system")
        |lyra::opt(Inp.Accelerator, "Accelerator")
            ["-acc"]["--acc"]
            ("Use CPU or GPU as accelerator")
        |lyra::opt(Inp.zComp, "z-Comp")
            ["-zComp"]["--zComp"]
            ("z comp")
        |lyra::opt(Inp.xComp, "x-comp")
            ["-xComp"]["--xComp"]
            ("x Comp")
        |lyra::opt(Inp.xft, "x-ft")
            ["-xft"]["--xft"]
            ("x ft")
        |lyra::opt(Inp.zft, "z-ft")
            ["-zft"]["--zft"]
            ("z ft")
        |lyra::opt(Inp.l_quantum,"l Quantum number")
            ["-lq"]["--l_q"]
            ("l Quantum number of initial state")
        |lyra::opt(Inp.m_quantum,"n Quantum number")
            ["-mq"]["--m_q"]
            ("m Quantum number of initial state")
        |lyra::opt(Inp.n_quantum,"n Quantum number")
            ["-nq"]["--n_q"]
            ("n Quantum number of initial state");

        auto result = cli.parse({ argc, argv });
        if ( !result )
        {
                std::cerr << "Error in command line: " << result.errorMessage() << std::endl;
                    exit(1);
        }

    //number of accelerators in the system
    int num_devices = 0;
    if (Inp.Accelerator == "GPU" ||  Inp.Accelerator == "GPUFull"){
        setupGPUs(local_rank,num_devices);
        watch(num_devices);
    }

    SpMatrix H_rank;
    H_rank.nmbrLEGO = 1 + Inp.zComp +  Inp.xComp;
    BuildMatrix(Inp,H_rank);
    std::string PATH;
    Propergator(Inp,H_rank,PATH);

    if (Inp.Accelerator == "GPU" ||  Inp.Accelerator == "GPUFull"){
        CUDAFinalize(H_rank.nmbrLEGO,
                H_rank.LEGO_d_rowPtr,H_rank.LEGO_d_cooCols,H_rank.LEGO_d_cooVals_sorted,
                H_rank.LEGO_d_cscRows,H_rank.LEGO_d_cscColPtr,H_rank.LEGO_d_cscVals);
    }
    watch(rank);
    MPI_Finalize();
    cout << "MPI has been finalized" << endl;
    return 0;
        
}

void BuildMatrix(InputParameter &Inp,SpMatrix &H_rank){
    //recalculate variables that depend on command line input
        //spacing between breakpoints 
    VectorXd x_break(Inp.N_b);
    double dX = (Inp.b - Inp.a)/ (Inp.N_b-1);
    x_break = ArrayXd::LinSpaced(Inp.N_b,Inp.a,Inp.b);

    int states_chan = (int) round(Inp.b/M_PI*sqrt(2*Inp.Emax));

    if (rank==0){
        //print the Input parameters
        watch(Inp.nnz);
        watch(Inp.a);
        watch(Inp.b);
        watch(Inp.Lmax);
        watch(Inp.Mmax);
        watch(Inp.Emax);
        watch(Inp.N_b);
        watch(states_chan);
        watch(Inp.zComp);
        watch(Inp.xComp);
        watch(Inp.zft);
        watch(Inp.xft);

        watch(Inp.Accelerator);
        watch(Inp.l_quantum);
        watch(Inp.m_quantum);
        watch(Inp.n_quantum);
        watch(Inp.init_state);
        watch(Inp.om);
        watch(Inp.I0);
        watch(Inp.E0);
        watch(Inp.T_puls);
        watch(Inp.T_cycle);
        watch(Inp.Tint);
        watch(Inp.timesteps);

    }

    auto start = system_clock::now();
      nnz_rank = Inp.nnz/p;
    states_chan_rank=states_chan/p;
    //distribute the remaining channels if states_cahn%p != 0
    if(rank < states_chan%p){
        states_chan_rank++;
    }
    //offset describes how many ranks have an extra channel 
    int offset = std::min(states_chan%p,rank);
    states_chan_rank_begin=states_chan/p*rank+offset;
    states_chan_rank_end=states_chan/p*rank+states_chan_rank+offset;

    pwatch(states_chan_rank);
    pwatch(offset);
    pwatch(states_chan_rank_begin);
    pwatch(states_chan_rank_end);

    pwatch("matrix obj is created");
    bp("before memory is allocated");
    H_rank.initMatrix(Inp.nnz/p);
    H_rank.nnz[0] = nnz_rank;
    bp("after memory is allocated");
    
    //this is the ind on every rank his is a vector that holds the ind for all 
    //different matrices linked to the different LEGO bricks
    std::vector< int>ind(H_rank.nmbrLEGO,0);

    //the matrix will be generated on different MPI ranks
    //generating the B-spliens does not take much time therefore all ranks generate 
    //their own set. They also for now all hold their own set of psi 

    //Gaus Legendre integration set up
    ArrayXd x((Inp.N_b-1)*Inp.n);
    ArrayXd weights((Inp.N_b-1)*Inp.n);
    bp("before gaus set up");
    gausLegendreSetup(Inp.n,Inp.N_b,x_break,x,weights);
    //gauss legendre returns same weights and x as python script
    //allocating a vector of Eigenvector matrices 
    
    bp("after gaus set up");

    ArrayXXd B = ArrayXXd::Zero(x.size(),Inp.N_b-1+2*Inp.n);
    ArrayXXd d_B = ArrayXXd::Zero(x.size(),Inp.N_b-1+2*Inp.n);
    ArrayXXd diff_B = ArrayXXd::Zero(x.size(),Inp.N_b-1+2*Inp.n);

    int dim_H = Inp.N_b+Inp.n-3;  
    generateBsplines(Inp,dX,x,B,d_B,diff_B);
    bp("bsplines generated");
    //Memory declaration
    ArrayXXd psi_1 = ArrayXXd::Zero(x.size(),states_chan);
    ArrayXXd d_psi_1 = ArrayXXd::Zero(x.size(),states_chan);
    ArrayXXd diff_psi_1 = ArrayXXd::Zero(x.size(),states_chan);
    ArrayXd E1 = ArrayXd::Zero(dim_H);
    
    ArrayXXd psi_2 = ArrayXXd::Zero(x.size(),states_chan);
    ArrayXXd d_psi_2 = ArrayXXd::Zero(x.size(),states_chan);
    ArrayXXd diff_psi_2 = ArrayXXd::Zero(x.size(),states_chan);
    ArrayXd E2 = ArrayXd::Zero(dim_H);
    
    ArrayXXd psi_3 = ArrayXXd::Zero(x.size(),states_chan);
    ArrayXXd d_psi_3 = ArrayXXd::Zero(x.size(),states_chan);
    ArrayXXd diff_psi_3 = ArrayXXd::Zero(x.size(),states_chan);
    ArrayXd E3 = ArrayXd::Zero(dim_H);

    MatrixXd H_base = MatrixXd::Zero(dim_H,dim_H);
    MatrixXd S = MatrixXd::Zero(dim_H,dim_H);
    double tmp_d;
    helperEigenStates(Inp,states_chan,dX,tmp_d,weights, x, B,diff_B,H_base,S);
     getEigenStates(Inp,Inp.Lmin,states_chan,dX,tmp_d,weights,x,B,d_B,diff_B,psi_1,d_psi_1,diff_psi_1,H_base,S,E1);
     bp("First set of psi");

    getEigenStates(Inp,Inp.Lmin+1,states_chan,dX,tmp_d,weights,x,B,d_B,diff_B,psi_2,d_psi_2,diff_psi_2,H_base,S,E2);

     bp("second set of psi");

     ArrayXd Input_Energies = ArrayXd::Zero(5000000);
    int ind2 =0;
   int lm_index_L =-1; 
    for (int l=Inp.Lmin; l < Inp.Lmax+1; l++){
        pwatch(l);
       getEigenStates(Inp,l+2,states_chan,dX,tmp_d,weights,x,B,d_B,diff_B,psi_3,d_psi_3,diff_psi_3,H_base,S,E3);

        for(int m=-std::min(Inp.Mmax,l); m < std::min(Inp.Mmax,l)+1; m++){
            lm_index_L++;
            int lm_index_R=-1;
            for(int l2 = Inp.Lmin; l2 < Inp.Lmax+1; l2++){
                for(int m2 = -std::min(Inp.Mmax,l2);m2 < std::min(Inp.Mmax,l2)+1;m2++){
                    lm_index_R++;
                    if (l2==l+1 && m2 ==m){
                        //z-polarization
                        //< l m | x | l+1 m>:
                        if(Inp.zComp == 1){
                            double LM_faktor=sqrt(((l+1)*(l+1)-m*m)/(double) (4*((l+1)*(l+1))-1));
                            polarization(H_rank,states_chan, ind[Inp.zft],Inp.zft,l,lm_index_L,lm_index_R, 
                                LM_faktor,dX,weights,x,psi_1,psi_2,d_psi_2);
                        }
                    }
                    if(l2== l+1 && m2 == m+1){
                        //x-polarization
                        //%< l m | p_x | l+1 m+1>:
                        if(Inp.xComp == 1){
                            double LM_faktor=-sqrt((l+m+1)*(l+m+2)/(double) (4*(2*l+1)*(2*l+3)));
                            polarization(H_rank,states_chan, ind[Inp.xft],Inp.xft,l,lm_index_L,lm_index_R, 
                                LM_faktor,dX,weights,x,psi_1,psi_2,d_psi_2);
                        }    
                    }
                    if(l2== l+1 && m2 == m-1){
                        //x-polarization
                        //< l m | p_x | l+1 m-1>: 
                        if(Inp.xComp == 1){
                            double LM_faktor=sqrt((l-m+1)*(l-m+2)/(double) (4*(2*l+1)*(2*l+3)));
                            polarization(H_rank,states_chan, ind[Inp.xft],Inp.xft,l,lm_index_L,lm_index_R, 
                                    LM_faktor,dX,weights,x,psi_1,psi_2,d_psi_2);
                        } 
                    }
                }
            }

            //Lego brick 4 will unperturbated states will be only calculated on rank 0 as they are
            //not many. We also need the Input_Energies later as a txt file 
            if(rank==0){
               //Lego brick 4 
                for(int j= 0; j < states_chan;j++){
                    H_rank.writeElement(ind[0],lm_index_L*states_chan+j,lm_index_L*states_chan+j,0,E1(j));
                    Input_Energies(ind2)=std::real(E1(j));
                    ind[0]++;
                    ind2++;

                }
            }
        
        }
        //so we do not have to calculate the psi1,psi2 again
        //psi3 -> psi2, psi2 -> psi1
        E1=E2;
        psi_1=psi_2;
        d_psi_1=d_psi_2;
        diff_psi_1=diff_psi_2;

        E2=E3;
        psi_2=psi_3;
        d_psi_2=d_psi_3;
        diff_psi_2=diff_psi_3;
        if(rank==0){
            int nnzRank=0;
            for(int k=0; k < H_rank.nmbrLEGO; k++){
                watch(ind[k]);
                nnzRank+=ind[k];
            }
            watch(nnzRank);
            cout << "Total number of elements in 1e+6: " << (long long) p * nnzRank / 1.0e6 << endl; 
        }
    }
    //the number of elements every matrix on every rank holds
    nnz_rank=ind[0];
    for(int i = 0; i < H_rank.nmbrLEGO; i++){
        H_rank.nnz[i]=ind[i];
    }
    //write to file 
    if(rank ==0){
        dim=ind2;
        watch(ind2);
        std::ofstream f6 ("Energies_Input_shago.txt");
        cout << "Begin to write file" << endl;
                
              if (f6.is_open()){
               for (int i = 0; i < ind2; i++){
                    f6 << Input_Energies(i) << "\n";
               }
           }

         else cout << "Unable to open file";

        //End of timing 
        auto stop = system_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        cout << "Time taken for generating the matrix: " << (double) duration.count()/1000 << "seconds" << endl;
    }

    //Broadcast the dimension of the vector;
    MPI_Bcast(&dim,1,MPI_INT,0,MPI_COMM_WORLD);

    cout << "rank" << " " << rank << " holds elemets: " << nnz_rank << endl;
    H_rank.dim=dim;
    pwatch(dim);
    pwatch(nnz_rank);
 
    //transform the matrix to CSR after the read in is done
    //we use less memory and obtain a better performance
    auto startSort = system_clock::now();
    if(Inp.Accelerator == "GPU" || Inp.Accelerator == "GPUFull"){
        bp("Before create CSR-GPU");
        H_rank.CSRonDevice();
        bp("After create CSR-GPU");
    }
    else if(Inp.Accelerator == "MKL"){
        bp("Before create CSR-MKL");
        H_rank.CSRMKL();
        bp("After create CSR-MKL");
    }
    else if(Inp.Accelerator != "CPU"){
        cout << "Inp.Accelerator is not a vaild accelerator [CPU,GPU,MKL]" << endl;
        exit(EXIT_FAILURE);

    }
    auto stopSort = system_clock::now();
    auto durationSort = duration_cast<milliseconds>(stopSort - startSort);
    cout << "Time taken for sorting the matrix: " << (double) durationSort.count()/1000 << "seconds" << endl;
    
}//End of matrix build up 

void Propergator(InputParameter &Inp,SpMatrix &H_rank,std::string PATH){
    Inp.E0 = 5.338*sqrt(Inp.I0*1e12)*1e-9;
    Inp.T_puls = Inp.T_cycle * 2 * M_PI / Inp.om;
    
    int states_chan = (int) round(Inp.b/M_PI*sqrt(2*Inp.Emax));
    //the initial state mind that C++ index starts at 0
    //this depends on Lmax,Mmax & qunatum number n,l,m
    int l_states= std::min(Inp.l_quantum,Inp.Mmax);
    int m_states= std::min(Inp.m_quantum,Inp.Mmax);
    Inp.init_state = states_chan * (Inp.l_quantum * (l_states+1) + m_states)
         +  Inp.n_quantum-Inp.l_quantum-1;

   if(rank ==0){
        //Input param propagator
        watch(Inp.Accelerator);
        watch(Inp.l_quantum);
        watch(Inp.m_quantum);
        watch(Inp.n_quantum);
        watch(Inp.init_state);
        watch(Inp.om);
        watch(Inp.I0);
        watch(Inp.E0);
        watch(Inp.T_puls);
        watch(Inp.T_cycle);
        watch(Inp.Tint);
        watch(Inp.timesteps);
    }

    //initalize the state vector y 
    VectorXcd y = VectorXcd::Zero(dim);
    //For the TDSE we start with the electron in the lowest possible state s1
    y(Inp.init_state) = 1.0;
    VectorXcd y_init(dim);
    VectorXcd p_init(Inp.timesteps);
    VectorXcd p_excited_1(Inp.timesteps);
    VectorXcd p_excited_2(Inp.timesteps);
    y_init << y;
    //Time propagator
    double dt = (Inp.T_puls-Inp.Tint)/Inp.timesteps;

    //Only needed on rank 0 but they are small
    MatrixXcd Q = MatrixXcd::Zero(dim,Inp.dim_kryl);  //Eigenvectors
    MatrixXcd h = MatrixXcd::Zero(Inp.dim_kryl,Inp.dim_kryl); //Eigenvalues
    VectorXcd d = VectorXcd::Zero(Inp.dim_kryl);
    VectorXd od = VectorXd::Zero(Inp.dim_kryl-1);

    if(rank==0)
        cout << "decleration of variables finished" << endl;

    //Start timing the propagation
    auto start2 = system_clock::now();

    if(rank==0)
        cout << "start timepropagation" << endl; 

    MPI_Barrier(MPI_COMM_WORLD);
    if (Inp.Accelerator == "GPUFull"){
        cuDoubleComplex *dQ;
        cuDoubleComplex *dd;
        cuDoubleComplex *dy;
        double *od;
         cusparseHandle_t handle_sparse;
         cublasHandle_t handle_blas;
        cusolverDnHandle_t handle_solver;
        allocateMemoryPropagator(Inp,H_rank.dim,dQ,dd,od,dy,y.data());
        for(int i = 1; i < Inp.timesteps+1;i++){
            //normalize y 
            
            double normStart;
       
            double t = dt *i;
            //get ft 
            VectorXcd ft =  VectorXcd(116);
            ft_calc(Inp,t, ft.data());
            
        
            lanczosCUDA(
                        H_rank.nnz,H_rank.dim,Inp.dim_kryl, H_rank.nmbrLEGO,
                        normStart,
                            H_rank.LEGO_d_cooCols,
                            H_rank.LEGO_d_rowPtr,
                            H_rank.LEGO_d_cooVals_sorted,
                            H_rank.LEGO_d_cscRows,
                            H_rank.LEGO_d_cscColPtr,
                            H_rank.LEGO_d_cscVals,
                            dy,ft.data(),
                            od,dQ,dd);

            //physical propergator 
            if (rank ==0){
                physPropagator(Inp,Inp.dim_kryl,dim,normStart,dt,dQ,dd,od,dy,y.data());

                if(i%20 == 0){
                    cout << "Timesteps: " << i << endl;
                    double norm = y.norm();
                watch(y(Inp.init_state));
                watch(y(Inp.init_state+1));
                watch(y(Inp.init_state+2));
                cout << "norm: " << norm << endl;
                cout << "Pinit: " <<  abs(y(Inp.init_state))*abs(y(Inp.init_state)) << endl;

                }

            }
        }   

        freeMemoryPropagator(Inp,H_rank.dim,dQ,dd,od,dy,y.data());
                watch(y(Inp.init_state));
                watch(y(Inp.init_state+1));
                watch(y(Inp.init_state+2));
                cout << "Pinit: " <<  abs(y(Inp.init_state))*abs(y(Inp.init_state)) << endl;


    }
    else{
        for(int i = 1; i < Inp.timesteps+1; i++){ 
            double norm;
            if (rank==0){ 
                norm = y.norm();
                y = y/norm;
            }
            
            //get the ft values 
            VectorXcd ft =  VectorXcd(116);
            double t = dt * i; 
            ft_calc(Inp,t, ft.data());

            //Lanzcos with GPU for SpMV
            if(Inp.Accelerator == "CPU" || Inp.Accelerator == "MKL"){
                lanczos(Inp, H_rank,  dim, y , Q, h, ft);
            }
            cout << "h" << endl;
            cout << h << endl;
            //To get the matrix exponential first diaganolize the matrix the exponate its
            //elements.
            if(rank == 0){
                 startortho = high_resolution_clock::now();

                SelfAdjointEigenSolver<MatrixXcd> ces;
                ces.compute(h);
                MatrixXcd D = (-1i * dt * ces.eigenvalues()).array().exp().matrix().asDiagonal();
                MatrixXcd P = ces.eigenvectors();
                y = norm * Q * (P * D * P.inverse()).col(0);

                stoportho = high_resolution_clock::now();
                durationortho += duration_cast<milliseconds>(stoportho - startortho);

                //write data 
                p_init(i-1)=abs(y(Inp.init_state))*abs(y(Inp.init_state));
                p_excited_1(i-1)=abs(y(Inp.init_state+1))*abs(y(Inp.init_state+1));
                p_excited_2(i-1)=abs(y(Inp.init_state+2))*abs(y(Inp.init_state+2));
               if(i%20 == 0){
                    cout << "Timesteps: " << i << endl;
                    norm = y.norm();
                    watch(y(Inp.init_state));
                    watch(y(Inp.init_state+1));
                    watch(y(Inp.init_state+2));
                    cout << "norm: " << norm << endl;
                    cout << "Pinit: " <<  abs(y(Inp.init_state))*abs(y(Inp.init_state)) << endl;

                }
            }
        }        
    }
    watch(rank);
    //End of timing 
    auto stop2 = system_clock::now();
    auto duration2 = duration_cast<milliseconds>(stop2 - start2);
    if (rank==0){
        cout << "Time taken by propagating the matrix: " << (double) duration2.count()/1000 << "seconds" << endl;
     
        cout << "Time taken by the actual propergator " << (double) durationortho.count()/1000
            <<"seconds" << endl;

       //write the output
        std::ofstream f1 (PATH + "fort.88");
        f1.precision(15);
        if (f1.is_open()){
            for (int i = 0; i < dim; i++){
                f1 << abs(y(i)) * abs(y(i)) << "\n";
            }
        }
        else cout << "Unable to open file";
        cout << "fort.88 is written" << endl; 
        std::ofstream f2 (PATH +"fort.89");
        if (f2.is_open()){
            for (int i = 0; i < dim; i++){
                f2 << i << " " <<  y(i).real() << " " << y(i).imag() << "\n";
            }
        }
        else cout << "Unable to open file";
        cout << "fort.89 is written" << endl; 
        //write the output
        std::ofstream f3 (PATH + "pinit.dat");
        f3.precision(15);
        if (f3.is_open()){
            for (int i = 0; i < Inp.timesteps; i++){
                double time = (i+1) * Inp.T_puls/Inp.timesteps;
                f3 << time <<" " << p_init(i).real() << "\n";
            }
        }
        else cout << "Unable to open file";

        std::ofstream f4 (PATH +"pexcited_1.dat");
        f4.precision(15);
        if (f4.is_open()){
            for (int i = 0; i < Inp.timesteps; i++){
                double time = (i+1) * Inp.T_puls/Inp.timesteps;
                f4 << time <<" " << p_excited_1(i).real() << "\n";
            }
        }
        else cout << "Unable to open file";

        std::ofstream f5 (PATH +"pexcited_2.dat");
        f5.precision(15);
        if (f5.is_open()){
            for (int i = 0; i < Inp.timesteps; i++){
                double time = (i+1) * Inp.T_puls/Inp.timesteps;
                f5 << time <<" " << p_excited_2(i).real() << "\n";
            }
        }
        else cout << "Unable to open file";
        cout << "pinit,pexcited are written" << endl;
        
    }
}

void lanczos(InputParameter Inp,SpMatrix &H_rank, int m,VectorXcd &y ,MatrixXcd &Q, MatrixXcd &h,  
        VectorXcd &FT){
    /* Input
    H_rank: mxm matrix 
    b: initial vector
    Inp.drim_kryl: dimension of Krylov subspace L in morten code
    m: dimesion of matrix H_rank
    Output:
    Q: orthogonal Krylov space 
    h: tridiagonal matrix 
    */
    VectorXcd d = VectorXcd::Zero(Inp.dim_kryl);
    VectorXcd od = VectorXcd::Zero(Inp.dim_kryl-1); //L-1 from morten 
    VectorXcd dydx_rank = VectorXcd::Zero(dim);
    //will only be used on rank0
    VectorXcd dydx = VectorXcd::Zero(dim);
   
    Q.setZero();
    Q.col(0) = y;
    h.setZero();

    for (int i = 0; i < Inp.dim_kryl; i++){
        dydx_rank.setZero(dim);
        //broadcast changeing vector y to all ranks 
        MPI_Bcast(y.data(),dim*2,MPI_DOUBLE,0,MPI_COMM_WORLD);

        //Matrix vector product this is the only loop that get executed in parallel 
        //iterate over all LEGO bricks

        //the cpu COO version 
        if(Inp.Accelerator == "CPU"){
            for(int j = 0; j < H_rank.nmbrLEGO; j++){
                H_rank.matrixVectorUpper(j,y,dydx_rank,FT);
            }
        }

        if(Inp.Accelerator == "MKL"){
        // the cpu MKL CSR version
            bp("Before Matrix Vector");
            H_rank.matrixVectorMKL(y,dydx_rank,FT);
            bp("After matrix Vector");
        }
      //sum the local dydx_rank into dydx on rank 0
        MPI_Reduce(dydx_rank.data(),dydx.data(),dim*2,MPI_DOUBLE,MPI_SUM, 0,MPI_COMM_WORLD);
        
        if(rank==0){
            d(i) = y.adjoint() * dydx; 
               //Full reorthogonalization (x2):
            dydx = dydx - Q.block(0,0,dim,i+1) * ( Q.block(0,0,dim,i+1).adjoint() * dydx);
            dydx = dydx - Q.block(0,0,dim,i+1) * ( Q.block(0,0,dim,i+1).adjoint() * dydx);
            if (i < Inp.dim_kryl-1){
                od(i) = dydx.norm();
                if (abs(od(i)) == 0){
                    cout << "devision by 0 during renormalizing" <<endl;
                    break;
                }
                y = dydx/od(i);
                Q.col(i+1) = y;
                //adaptivly control the dimension of the Krylov space. At most we do dim_kryl
                //iterations if we converge earlier we terminate 
                if((Q.col(i+1)-Q.col(i)).norm()  < Inp.eps){
                    for(int j = i+1; j < Inp.dim_kryl-1 ; j++){
                        //fill the remaining vectors 
                        Q.col(j+1) = y;
                        d(j) = d(i);
                        od(j) = od(i);

                    }
                    d(Inp.dim_kryl-1)=d(i);
                    cout << "Krylov space converged after: " << i << " iterations\n"; 
                    break;
                    
                }
            }
        }
           
    }
    //set the h Matrix mortens TT matrix
    for (int i = 1; i < Inp.dim_kryl; i++){
        h(i,i-1) = od(i-1);
        h(i-1,i) = od(i-1);
    }
    for (int i = 0; i < Inp.dim_kryl; i++){
        h(i,i) = d(i);
    }
   return;
}

void polarization(SpMatrix &M,int states_chan, int &ind, int index,int l,int lm_index_L,int lm_index_R,
        double LM_faktor,double dX,ArrayXd &weights, ArrayXd &x, ArrayXXd &psi_1,ArrayXXd
        &psi_2,ArrayXXd &d_psi_2){
    //Lego brick 1 + 2 
#pragma omp parallel for collapse(2)  
    for(int j =  states_chan_rank_begin; j < states_chan_rank_end; j++){
        for(int i = 0; i < states_chan; i++){
                M.writeElement(ind+j*states_chan+i,
                        lm_index_L*states_chan+j,
                        lm_index_R*states_chan+i,
                        index,
                        dX/2*(weights*psi_1.col(j)*psi_2.col(i)/x).sum()*LM_faktor*(l+1)
                        +dX/2*(weights*psi_1.col(j)*d_psi_2.col(i)).sum()*LM_faktor);
          }

    }
   
    //number of elements written on one rank over all threads
    ind = ind + (states_chan_rank_end-states_chan_rank_begin) * states_chan;
}

void helperEigenStates(InputParameter Inp, int states_chan,double dX,double &tmp_d,
        ArrayXd &weights, ArrayXd &x,ArrayXXd &B,ArrayXXd &diff_B,MatrixXd &H_base, 
        MatrixXd &S){
    //This function should speed up the calculation
    //S and large aprts of H are always the same, by storing them and passing them to
    //getEigenStates redundant computation can be avoided 
    //S and the kinetic Energy are the same for all l, 

    int dim_H = Inp.N_b+Inp.n-3;  
    for (int i = 0; i < dim_H ; i++){ 
        for (int j = 0; j < dim_H; j++){
            H_base(i,j) = - 0.5 * dX * 0.5 * (weights * B.col(i+1)*diff_B.col(j+1)).sum(); 
            ArrayXd tmp = dX * 0.5 * (weights * B.col(i+1)*B.col(j+1));
            H_base(i,j) += (tmp * -1/x).sum(); 
            S(i,j) = (tmp).sum();
        }
    }

}


void getEigenStates(InputParameter Inp, int l,int states_chan,double dX,double &tmp_d,ArrayXd &weights,
        ArrayXd &x,ArrayXXd &B, ArrayXXd &d_B,ArrayXXd &diff_B,ArrayXXd &psi,ArrayXXd &d_psi
        ,ArrayXXd &diff_psi , MatrixXd &H_base, MatrixXd &S, ArrayXd &Energy ){
    //The H and S matrix 
    int dim_H = Inp.N_b+Inp.n-3;  //this equals to s_H
    MatrixXd H = MatrixXd::Zero(dim_H,dim_H);
    ArrayXd tmp_l = (l*(1+l)/(2*x*x)); 

//only works on rect loops
#pragma omp parallel for collapse(2) 
    for (int i = 0; i < dim_H ; i++){ 
        for (int j = 0; j < dim_H; j++){
             //Kinetic Engergy 
            H(i,j) = H_base(i,j); 
            //potential Energy V
            ArrayXd tmp =  dX * 0.5 * weights * B.col(i+1)*B.col(j+1);
            H(i,j) += (tmp * tmp_l).sum();
       }
    }
    
    //Start timing the solver
    auto start2 = system_clock::now();
    //The Gen Eigenvalue solver on CPU
    MatrixXd Eigenvectors =  MatrixXd::Zero(dim_H,dim_H);
    if(Inp.Accelerator == "CPU" || Inp.Accelerator== "MKL")
    {
        GeneralizedSelfAdjointEigenSolver<MatrixXd> ges(H,S);
        Eigenvectors = ges.eigenvectors();
        Energy = ges.eigenvalues();
    }
   //Gen Eigenvalue solver on GPU
    //H,S are in coloum major, 0 index format
    //Cusolver assumes col-major no conversion needed
    //lda,ldb,m are all the same as dim W,V = Eigenvalues,Eigenvectors
    if(Inp.Accelerator == "GPU" || Inp.Accelerator =="GPUFull")
    {
        //VectorXd Eigenvalues =  VectorXd::Zero(dim_H);
        genEigenSolverCUDA(dim_H,H.data(),S.data(),Energy.data(),Eigenvectors.data());
    }
   
    auto stop2 = system_clock::now();
    auto duration2 = duration_cast<milliseconds>(stop2 - start2);
    if (rank==0){
        cout << "Time taken by the solver in l: "<< l << " : "  << (double) duration2.count()/1000
            << "seconds" << endl;}
    
    //set the psi to 0 
    psi.setZero();
    d_psi.setZero();
    diff_psi.setZero();
    
    //watch(Energy);
    //watch(Eigenvectors);
#pragma omp parallel for collapse(2)
    for(int i = 0; i < states_chan; i++){
        for(int j=0; j < dim_H; j++){
            psi.col(i) = psi.col(i) + Eigenvectors(j,i) * B.col(j+1);
            d_psi.col(i) = d_psi.col(i) + Eigenvectors(j,i) * d_B.col(j+1);
            diff_psi.col(i) = diff_psi.col(i) + Eigenvectors(j,i) * diff_B.col(j+1);
       }
 } 
    }

