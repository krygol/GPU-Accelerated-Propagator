#include <iostream>
#include <math.h>

#pragma once
class InputParameter{
    //This class handles the Input parameters to get reduce the number of global variables
    //later the input parameter should be read in from a config file
    //https://www.walletfox.com/course/parseconfigfile.php
    public:
        //max size of matrix
        int64_t nnz=4000000000; //wild guess

        //Input parameters 
        double om = 0.057;
        double E0 = 0;
        double I0 = 1000;
        int T_cycle = 10;
        double T_puls = T_cycle * 2 * M_PI / om;//15 cycles 
        double Tint = 0.0;
        int timesteps = 3000;

        //Interval [a,b] (box size)
        double a = 0;
        double b = 750;//100;


        //Input param propergator
        //Number of breakpoints/B-splines 
        int N_b = 1200;//900;
        //angular momentum 
        int Lmin=0;
        int Lmax=25;
        int Mmax=25;
        double Emax=10;
        //B-spline order 
        int n=5;//5;

        int init_state=0;
        bool zComp = 1;
        bool xComp = 1;

        int zft = 1; 
        int xft = 2;

        int l_quantum=0;
        int m_quantum=0;
        int n_quantum=1;

        std::string Accelerator = "GPU";
        //Number of iterations is the dimension of the Krylov subspace 
        int dim_kryl = 10;
        double eps = std::numeric_limits<double>::epsilon(); //the machine epsilon 

};

