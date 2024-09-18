# MPS_Heat1D
Simulation of the 1-D heat equation, with matix product states (MPS)

Tensorhelp.py:
-Functionality behind the MPS solver
-contains the classes of MPS and MPO
-contains truncated SVD

Classical.py:
-Functionality behind MPS solver

MPSSolver.py: 
-contains simulation to advance the heat equation in time, using MPS
-Tensorhelp.py and Classical.py are required

FDSolver.py:
-contains simulation to advance the heat equation in time, using finite differences
-Classical.py required
