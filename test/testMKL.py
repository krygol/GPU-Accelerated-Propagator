import os as os
import numpy as np

# recomplile the version of simulation TDSE
os.system("cd ../build && make all")

#run the version of simulation TDSE
os.system("../build/simulationACC --om=50 --T_cycle=15 -b=40 --N_b=530 --Emax=210 \
        --Mmax=8 --Lmax=8 --I0=5615170000 --timesteps=1000 --nnz=500000000 \
        --xft=2 --zft=1 --zComp=1 --xComp=1 --acc=MKL> output.txt")

# compare the results to the file stored in archive
a = np.loadtxt("fort.88", float)
b = np.loadtxt("archive/fort.88", float)

tolerance = 1e-10
idx = np.where(np.abs(a-b) > tolerance)[0]
if(idx != 0):
    firstidx = idx[0]
    print(firstidx, a[firstidx], b[firstidx])
else:
    print("test finsihed all the files are identical up to",tolerance)

