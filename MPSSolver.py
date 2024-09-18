import numpy as np
from Tensorhelp import *
from Classical import *

def getdelt(expo,a):
    delx=1/(2**expo-1)
    delt=0.1*delx**2/a
    return delt


expo=10
a=98.8E-6               #Temerature conductivity
delt=getdelt(expo,a)
delx=1/(2**expo-1)
tend=1000                #Endtime
e=1E-7                   #truncation threshold
Tb=400                   #Boundary Condition

MPS_size=2*np.ones(expo,dtype=int)
MPO_size=2*np.ones(2*expo,dtype=int)
T0=giveT0(expo,Tb)
FD=giveFD(expo,a,delt,delx)
T0=T0.reshape(MPS_size)
FD=FD.reshape(MPO_size)

MPS1=MPS(expo)
MPO1=MPO(expo)

MPS1.truncated_l(T0,0)
MPO1.truncated_l(FD, 1E-13)

it=int(tend/delt)

for i in range(it):
    MPS1.MPOMPS(MPO1)
    MPS1.retruncate(e)
    Ti=MPS1.reTensor()