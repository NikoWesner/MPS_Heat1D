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


T0=giveT0(expo,Tb)
FD=giveFD(expo,a,delt,delx)

H1=Heat1D(expo)
H1.initialize(400.0,a,delt,delx)

it=int(tend/delt)

for i in range(it):
    H1.solve()
    T=H1.T