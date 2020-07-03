# GPU-Accelerated-Propagator

The GPU Accelerated Propagator(GAP) was developed as part of my masters study at the University 
of Bergen (UiB). 
GAP is used to construct a given Hamiltonian in an eigenstate basis and then propagate the
Hamiltonian in time. In the research group of atomic physics it is used to simulate laser Hydrogen
interactions. 

## Implementation 

GAP as a program is divided in four blocks. First a set of b-splines is generated. Then an
eigenstate basis is formed. The Hamiltonian is then built in this eigenstate basis. The time
evolution of the system is then done with the Lanczos propagator. 

GAP is implemented in C++ making extensive use of the Eigen library. 

## Parallelization 

The Hamiltonian is distributed with MPI. For the CPU accelerated versions openMP and the MKL are
used for inner node parallelism. 
For the GPU versions every MPI rank is matched to a GPU. 

## Dependencies 

The code has been tested with the following version of the libraries:
- Eigen 3.3.5
- Lyra 
- CUDA 10.1
- MKL 2020.1.217 
- openMPI 4.0.2 (for the GPU version it has to be CUDA aware MPI)
Different versions of the libraries might work but were not tested. 


## Input 

      name       data type               description                affect performance
  ------------- ----------- -------------------------------------- --------------------
       nnz       int64_t    maximum number of nnz of Hamiltonian           yes
       om         double                photon energy                      no
       I0         double              intensity of laser                   no
    T_cycle        int            number of optical cycles                 no
        t          int              number of time steps                   yes
        b         double                   box size                        yes
      N_b          int             number of breakpoints                   yes
     L_min         int            minimum l-quantum number                 yes
     L_max         int            maximum l-quantum number                 yes
     M_min         int            minimum m-quantum number                 yes
     M_max         int            maximum m-quantum number                 yes
     E_max       double        highest energy state included               yes
        n          int               order of b-splines                    no
      zComp        bool           if true then include zComp               no
      xComp        bool           if true then include xComp               no
       zft         int      if 1 polarization, if 2 propagation            yes
       xft         int      if 1 polarization, if 2 propagation            yes
   l_quantum       int       l quantum number of initial state             no
   m_quantum       int       m quantum number of initial state             no
   n_quantum       int       n quantum number of initial state             no
    dim_kryl       int         dimension of the Krylov space               yes
   Accelerator    string         accelerator [CPU,MKL,GPU,GPUFull]         yes

The parameters given in the table above are run time parameters and can be set as command line
parameters using Lyra. The default parameters are in the InputParamter.h.
The Hamiltonian and the initial wave function have to be hard coded into GAP.  

## Output
- fort.88 contains the the element wise square of the wave function
- fort.89 contains the wave function at the end of the the laser pulse

Further information on the functioning of GAP can be found in the thesis in 'doc'.
The thesis includes a performance study on GAP and a study of laser hydrogen interaction in the 800
nm regime.

## Licensing

- The code in the directory 'dep/Lyra' has it is under the Boost Software License. 
- The code written for the project by the contributer of GAP is under the MIT License.


