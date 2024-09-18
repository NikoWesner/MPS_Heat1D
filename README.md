# MPS_Heat1D
Simulation of the 1-D heat equation, with matix product states (MPS)

Tensorhelp.py:
-Functionality behind the MPS solver
-contains the classes of MPS and MPO
-contains truncated SVD
    void MPS.truncated_r(T,e)
    -Transforms N-D tensor T into right canonical MPS, with truncation threshold e
    void MPS.truncated_r(T,e)
    -Transforms N-D tensor T into left canonical MPS, with truncation threshold e
    double[] MPS.reTensor()
    -returns array, after contracting all cores
    void MPS.MPOMPS(MPO)
    -contracts cores with MPO.cores
    void MPS.returncate(e)
    -applies TT-rounding, to recompress MPS, with truncation threshold e
    void MPS.returncateCHI(X)
    -applies TT-rounding, to recompress MPS, with constant bond dims X
    void retruncateWish(ranks)
    -applies TT-rounding, to recompress MPS, with bond dims as specified in array ranks
    ____________________________________________________________________________________
    void MPO.truncate_r(T,e)
    -Transforms 2N-D tensor T into right canonical MPO, with truncation threshold e
    void MPO.truncate_l(T,e)
    -Transforms 2N-D tensor T into left canonical MPO, with truncation threshold e
    ____________________________________________________________________________________
    killSVD(U,S,V,e)
    -handles the truncated SVD, with threshold e
    killSVDChi(U,S,V,X)
    -handles the truncated SVD, with constant dim X
    -also used in MPS.retruncateWish

    

Classical.py:
-Functionality behind MPS solver

MPSSolver.py: 
-contains simulation to advance the heat equation in time, using MPS
-Tensorhelp.py and Classical.py are required

FDSolver.py:
-contains simulation to advance the heat equation in time, using finite differences
-Classical.py required
