//Bsplines file
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Eigen/StdVector>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
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
#include "bsplines.h"


#define bp(x) cout << "breakpoint" << (#x) << endl

using namespace Eigen;
using namespace std::complex_literals;
using namespace std::chrono;
using std::cout;
using std::endl;

 
void generateBsplines(InputParameter Inp, double dX,ArrayXd x,ArrayXXd &BB,ArrayXXd &d_BB, ArrayXXd &diff_BB){
    std::vector<Array<double,Dynamic,Dynamic>
        ,aligned_allocator<Array<double,Dynamic,Dynamic> > > B(Inp.n+1);
    for(int i = 0 ; i < Inp.n+1; i++){
        B[i].setZero(x.size(),Inp.N_b-1+2*Inp.n);
    }
     
    ArrayXd t = ArrayXd::Zero(Inp.N_b+2*Inp.n);
    
    for(int i = Inp.n+1; i< Inp.N_b +Inp.n;i++){
        B[0].col(i-1).segment((i-Inp.n-1) * Inp.n,Inp.n) = 1;
        t(i-1) = (i-Inp.n-1) * dX;

    }
    t.segment(Inp.N_b + Inp.n - 1,Inp.n+1) = Inp.b;
    for( int k = 1; k < Inp.n+1;k++){
        for(int i = 1; i < Inp.N_b-1+2*Inp.n-k+1; i++){
            double tmp1 = t(i+k-1) - t(i-1);
            double tmp2 = t(i+k+0) - t(i-0);

            if(tmp1 == 0)
                tmp1 =1;
            if(tmp2 == 0)
                tmp2 =1;       
        
            B[k].col(i-1) = ((x - t(i-1))/tmp1) * B[k-1].col(i-1) 
                           + (t(i+k) - x)/tmp2 * B[k-1].col(i); 
       }
    }
    bp("first order derivaitve of bsplines");
    //d_B is the first derivative 
    std::vector<Array<double,Dynamic,Dynamic>
        ,aligned_allocator<Array<double,Dynamic,Dynamic> > > d_B(Inp.n+1);
    for(int i = 0 ; i < Inp.n+1; i++){
        d_B[i].setZero(x.size(),Inp.N_b-1+2*Inp.n);
    }

    for( int k = 2; k < Inp.n+1; k++){
        for(int i = 1; i < Inp.N_b-1+2*Inp.n-k+1; i++){
            double tmp1 = t(i+k-1) - t(i-1);
            double tmp2 = t(i+k+0) - t(i+0);

            if(tmp1 == 0)
                tmp1 =1;
            if(tmp2 == 0)
                tmp2 =1;       
        
            d_B[k].col(i-1) = k * (B[k-1].col(i-1)/tmp1 - B[k-1].col(i)/tmp2);

        }
    }
    bp("scond order derivative of bsplines ");
    //diff of B is the second derivative 
    std::vector<Array<double,Dynamic,Dynamic>
        ,aligned_allocator<Array<double,Dynamic,Dynamic> > > diff_B(Inp.n+1);
    for(int i = 0 ; i < Inp.n+1; i++){
        diff_B[i].setZero(x.size(),Inp.N_b-1+2*Inp.n);
    }

    for( int k = 2; k < Inp.n+1; k++){
        for(int i = 1; i < Inp.N_b-1+2*Inp.n-k+1; i++){
            double tmp1 = t(i+k-1) - t(i-1);
            double tmp2 = t(i+k+0) - t(i+0);
            double tmp3 = t(i+k-2) - t(i-1);
            double tmp4 = t(i+k-1) - t(i+0);
            double tmp5 = t(i+k+0) - t(i+1);

            if(tmp1 == 0)
                tmp1 =1;
            if(tmp2 == 0)
                tmp2 =1;       
            if(tmp3 == 0)
                tmp3 =1;       
            if(tmp4 == 0)
                tmp4 =1;       
            if(tmp5 == 0)
                tmp5 =1;       
            
        diff_B[k].col(i-1) = k * (k-1)/tmp1 * (B[k-2].col(i-1)/tmp3 - B[k-2].col(i)/tmp4)
                           - k * (k-1)/tmp2 * (B[k-2].col(i)/tmp4 - B[k-2].col(i+1)/tmp5);
        }
    }
    BB=B[Inp.n];
    d_BB=d_B[Inp.n];
    diff_BB=diff_B[Inp.n];

    cout <<"Bsplines generated"<<endl;
   
}

void gausLegendreSetup(int k, int nmbr_breakpoints,
        ArrayXd x_break, ArrayXd &x, ArrayXd &weights){
    //Return evenly spaced values within a given interval.
    std::vector<int> I(k);
    std::iota(I.begin(),I.end(),1);
    
    MatrixXd J = MatrixXd::Zero(k,k);
    for(int i = 0; i < k-1; i++){
        J(i,i+1) = 0.5 / sqrt(1-pow(2*I[i],-2));
        J(i+1,i) = 0.5 / sqrt(1-pow(2*I[i],-2));
    }
    //get the eigenvalues and eigenvectors 

    SelfAdjointEigenSolver<MatrixXd> es;
    es.compute(J);
    VectorXd eigval = es.eigenvalues();
    MatrixXd eigvect = es.eigenvectors();
    for(int i = 2; i < nmbr_breakpoints+1;i++){
        x.segment((i-2)*k,k) = (x_break(i-1)-x_break(i-2))*0.5*  eigval.array()
            +(x_break[i-1]+x_break[i-2])*0.5;
        weights.segment((i-2)*k,k) = 2*eigvect.row(0).array().square();
        
    }

}


