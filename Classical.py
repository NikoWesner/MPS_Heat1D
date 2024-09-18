import numpy as np
from scipy.sparse import diags

def giveFD(expo,a,delt,delx):
    FD = np.zeros((2 ** expo, 2 ** expo))
    FD[0, 0] = 1
    FD[-1, -1] = 1
    aa = a * delt / (delx ** 2)
    for i in range(1, 2 ** expo - 1):
        FD[i, i] = 1 - 2 * aa
        FD[i, i + 1] = aa
        FD[i, i - 1] = aa
    return FD
def giveFDSparse(expo,a,delt,delx):


    # Define the size of the matrix
    size = 2**expo  # Example: a 4096x4096 matrix

    # Define the values for the diagonals
    aa = a * delt / (delx ** 2)
    main_diagonal_value = 1 - 2 * aa
    upper_diagonal_value = aa
    lower_diagonal_value = aa

    # Create the diagonals as arrays
    main_diagonal = np.full(size, main_diagonal_value)  # Main diagonal
    upper_diagonal = np.full(size - 1, upper_diagonal_value)  # Upper diagonal (size-1)
    lower_diagonal = np.full(size - 1, lower_diagonal_value)  # Lower diagonal (size-1)
    main_diagonal[0]=1
    main_diagonal[-1]=1
    upper_diagonal[0]=0
    lower_diagonal[-1]=0
    # Create a sparse tridiagonal matrix using the diags function
    # The `offsets` parameter specifies the position of the diagonals:
    # 0 for the main diagonal, 1 for the upper diagonal, and -1 for the lower diagonal
    tridiagonal_matrix = diags(
        [main_diagonal, upper_diagonal, lower_diagonal],
        offsets=[0, 1, -1],
        format='csr'
    )
    return tridiagonal_matrix

def giveT0(expo,Tb):
    T0 = np.zeros((2 ** expo, 1))
    T0[0] = 400
    T0[-1] = 400
    return T0


class Heat1D:

    def __init__(self, expo):
        self.expo = expo
        self.T=np.array([0.0]*2**expo)
        self.T=self.T.reshape(-1,1)
        self.A=[0]*3
        self.p=0

    def initialize(self,T0,a,delt,delx):
        self.T[0] = T0
        self.T[-1] = T0
        fac=delt*a/(delx**2)
        self.A[0]=fac
        self.A[1]=1-2*fac
        self.A[2]=fac


    def solveP(self):
        Tn = self.T.copy()
        Tn[0] = self.T[-1] * self.A[0] + self.T[0] * self.A[1] + self.T[1] * self.A[2]
        Tn[-1] = self.T[-2] * self.A[0] + self.T[-1] * self.A[1] + self.T[0] * self.A[2]
        for i in range(1, len(self.T) - 1):
            Tn[i] = self.T[i - 1] * self.A[0] + self.T[i] * self.A[1] + self.T[i + 1] * self.A[2]
        self.T = Tn
        self.p = self.getp()
    def solve(self):
        Tn=self.T.copy()
        for i in range(1,len(self.T)-1):
          Tn[i]=self.T[i-1]*self.A[0]+self.T[i]*self.A[1]+self.T[i+1]*self.A[2]
        self.T=Tn
        self.p=self.getp()

    def getp(self):
        return np.linalg.norm(self.T,2)
    def reset(self):
        self.T = np.array([0.0] * 2 ** self.expo)
        self.T = self.T.reshape(-1, 1)