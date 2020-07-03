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
#include "InputParameter.h"

using namespace Eigen;
using namespace std::complex_literals;
using namespace std::chrono;
using std::cout;
using std::endl;

void gausLegendreSetup(int k, int nmbr_breakpoints,
        ArrayXd x_break, ArrayXd &x, ArrayXd &weights);

void generateBsplines(InputParameter Inp,double dX,ArrayXd
        x,ArrayXXd &BB,ArrayXXd &d_BB, ArrayXXd &diff_BB);


