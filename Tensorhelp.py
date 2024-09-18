import numpy as np


class MPS:

    def __init__(self,expo):
        self.cores=[None]*expo
        self.expo=expo;
        self.p=1;
        self.E=0;
        self.r=[None]*(expo-1)

    def truncated_r(self,T, e):
        dim = T.shape
        N = T.size
        l = len(dim)

        T = T.reshape(dim[0], N // dim[0])
        self.p = np.linalg.norm(T, ord='fro')
        T=T/self.p
        U, S, Vt = np.linalg.svd(T, full_matrices=False)
        U, S, Vt, rs, Er = killSVD(U, S, Vt, e)
        E = Er
        G1 = U

        self.cores[0]=G1

        M = S @ Vt
        r = np.zeros(l - 1, dtype=int)
        r[0] = rs
        Gs = [None] * (l - 2)

        for i in range(1, l - 1):
            M = M.reshape(r[i - 1] * dim[i], M.size // (r[i - 1] * dim[i]))
            U, S, Vt = np.linalg.svd(M, full_matrices=False)
            U, S, Vt, rs, Er = killSVD(U, S, Vt, e)
            E += Er
            r[i] = rs
            Gi = U.reshape(r[i - 1], dim[i], r[i])
            self.cores[i]= Gi
            M = S @ Vt

        self.cores[-1] = M
        self.E=E

    def truncated_l(self,T, e):
        dim = T.shape
        N = T.size
        l = self.expo

        T = T.reshape(N//dim[-1],dim[-1])
        self.p = np.linalg.norm(T, ord='fro')
        T = T / self.p
        U, S, Vt = np.linalg.svd(T, full_matrices=False)
        U, S, Vt, rs, Er = killSVD(U, S, Vt, e)
        E = Er
        self.cores[-1] = Vt

        M = U @ S
        r = np.zeros(l - 1, dtype=int)
        r[-1] = rs
        Gs = [None] * (l - 2)

        for i in reversed(range(1,l - 1)):
            M = M.reshape(M.size // (r[i] * dim[i]), (r[i] * dim[i]))
            U, S, Vt = np.linalg.svd(M, full_matrices=False)
            U, S, Vt, rs, Er = killSVD(U, S, Vt, e)
            E+=Er
            r[i-1] = rs
            Gi = Vt.reshape(r[i - 1], dim[i], r[i])
            self.cores[i] = Gi
            M = U @ S

        self.cores[0] = M
        self.E = E
        self.r=r

    def reTensor(self):
        T=np.tensordot(self.cores[0],self.cores[1],axes=(1,0))
        for i in range(1,self.expo-1):
            T=np.tensordot(T,self.cores[i+1], axes=(i+1,0))
        return T.reshape(2**self.expo,1)*self.p

    def MPOMPS(self,MPO):
        A = np.tensordot(MPO.cores[0], self.cores[0], axes=(1, 0))
        A=A.reshape(A.shape[0],A.shape[1]*A.shape[2])
        self.cores[0]=A;
        for i in range(1,self.expo-1):
            A= np.tensordot(MPO.cores[i],self.cores[i],axes=(2,1))
            A=A.transpose(0,3,1,2,4)
            A=A.reshape(A.shape[0]*A.shape[1],A.shape[2],A.shape[3]*A.shape[4])
            self.cores[i]=A

        A=np.tensordot(MPO.cores[-1],self.cores[-1],axes=(2,1))
        A=A.transpose(0,2,1)
        A=A.reshape(A.shape[0]*A.shape[1],A.shape[2])
        self.cores[-1]=A
        self.p=self.p*MPO.p

    def retruncate(self,e):

        Q, R = np.linalg.qr(self.cores[0])
        self.cores[0]=Q
        self.cores[1]=np.tensordot(R,self.cores[1],axes=(1,0))
        for i in range(1,self.expo-1):
            A=self.cores[i]
            s=A.shape
            A=A.reshape(s[0]*s[1],s[2])
            Q, R = np.linalg.qr(A)
            self.cores[i]=Q.reshape(s[0],s[1],Q.size//(s[0]*s[1]))
            self.cores[i+1]=np.tensordot(R,self.cores[i+1],axes=(1,0))

        p=np.linalg.norm(self.cores[-1],ord='fro');
        p2=np.trace(np.matmul(np.transpose(self.cores[-1]),self.cores[-1]))
        p2=np.sqrt(p2)
        self.cores[-1]=self.cores[-1]/p2
        self.p=self.p*p2

        s = self.cores[-1].shape
        A = self.cores[-1].reshape(s[0], s[1])
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        U, S, Vt, rs, Er = killSVD(U, S, Vt, e)
        self.r[-1] = rs
        E = Er
        self.cores[-1] = Vt
        M=U@S
        a=M.shape[0]
        self.cores[-2]=self.cores[-2][:,:,0:a]
        self.cores[-2] = np.tensordot(self.cores[-2], M, axes=(-1, 0))

        for i in reversed(range(1,self.expo-1)):
            s=self.cores[i].shape
            A=self.cores[i].reshape(s[0],s[1]*s[2])
            U,S,Vt=np.linalg.svd(A,full_matrices=False)
            U, S, Vt, rs, Er = killSVD(U, S, Vt, e)
            E += Er
            self.r[i - 1] = rs
            self.cores[i] = Vt.reshape(Vt.size//(s[1]*s[2]),s[1],s[2])
            M=U@S
            a = M.shape[0]

            if i==1:
                self.cores[0]=self.cores[0][:,0:a]
                self.cores[0] = np.tensordot(self.cores[0], M, axes=(-1, 0))
                break
            self.cores[i-1] = self.cores[i-1][:, :, 0:a]
            self.cores[i-1] = np.tensordot(self.cores[i-1],M,axes=(-1,0))

        pc=np.trace(np.matmul(np.transpose(self.cores[0]),self.cores[0]))
        self.E +=E

    def retruncateData(self, e):

        Q, R = np.linalg.qr(self.cores[0])
        self.cores[0] = Q
        self.cores[1] = np.tensordot(R, self.cores[1], axes=(1, 0))
        for i in range(1, self.expo - 1):
            A = self.cores[i]
            s = A.shape
            A = A.reshape(s[0] * s[1], s[2])
            Q, R = np.linalg.qr(A)
            self.cores[i] = Q.reshape(s[0], s[1], Q.size // (s[0] * s[1]))
            self.cores[i + 1] = np.tensordot(R, self.cores[i + 1], axes=(1, 0))

        p = np.linalg.norm(self.cores[-1], ord='fro');
        p2 = np.trace(np.matmul(np.transpose(self.cores[-1]), self.cores[-1]))
        p2 = np.sqrt(p2)
        self.cores[-1] = self.cores[-1] / p2
        self.p = self.p * p2

        Sret=[None]*(self.expo-1)
        s = self.cores[-1].shape
        A = self.cores[-1].reshape(s[0], s[1])
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        Sret[-1]=S**2
        U, S, Vt, rs, Er = killSVD(U, S, Vt, e)
        self.r[-1] = rs
        E = (1 - Er) ** 2 * Er
        self.cores[-1] = Vt
        M = U @ S
        a = M.shape[0]
        self.cores[-2] = self.cores[-2][:, :, 0:a]
        self.cores[-2] = np.tensordot(self.cores[-2], M, axes=(-1, 0))


        for i in reversed(range(1, self.expo - 1)):
            s = self.cores[i].shape
            A = self.cores[i].reshape(s[0], s[1] * s[2])
            U, S, Vt = np.linalg.svd(A, full_matrices=False)
            Sret[i-1] = S**2
            U, S, Vt, rs, Er = killSVD(U, S, Vt, e)
            E += (1 - Er) ** 2 * Er
            self.r[i - 1] = rs
            self.cores[i] = Vt.reshape(Vt.size // (s[1] * s[2]), s[1], s[2])
            M = U @ S
            a = M.shape[0]

            if i == 1:
                self.cores[0] = self.cores[0][:, 0:a]
                self.cores[0] = np.tensordot(self.cores[0], M, axes=(-1, 0))
                break
            self.cores[i - 1] = self.cores[i - 1][:, :, 0:a]
            self.cores[i - 1] = np.tensordot(self.cores[i - 1], M, axes=(-1, 0))

        pc = np.trace(np.matmul(np.transpose(self.cores[0]), self.cores[0]))
        self.E += E
        return Sret
    def retruncateCHI(self, X):

        Q, R = np.linalg.qr(self.cores[0])
        self.cores[0] = Q
        self.cores[1] = np.tensordot(R, self.cores[1], axes=(1, 0))
        for i in range(1, self.expo - 1):
            A = self.cores[i]
            s = A.shape
            A = A.reshape(s[0] * s[1], s[2])
            Q, R = np.linalg.qr(A)
            self.cores[i] = Q.reshape(s[0], s[1], Q.size // (s[0] * s[1]))
            self.cores[i + 1] = np.tensordot(R, self.cores[i + 1], axes=(1, 0))

        p = np.linalg.norm(self.cores[-1], ord='fro');
        p2 = np.trace(np.matmul(np.transpose(self.cores[-1]), self.cores[-1]))
        p2 = np.sqrt(p2)
        self.cores[-1] = self.cores[-1] / p2
        self.p = self.p * p2

        s = self.cores[-1].shape
        A = self.cores[-1].reshape(s[0], s[1])
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        U, S, Vt, rs, Er = killSVDChi(U, S, Vt, X)
        self.r[-1] = rs
        E = (1 - Er) ** 2 * Er
        self.cores[-1] = Vt
        M = U @ S
        a = M.shape[0]
        self.cores[-2] = self.cores[-2][:, :, 0:a]
        self.cores[-2] = np.tensordot(self.cores[-2], M, axes=(-1, 0))

        for i in reversed(range(1, self.expo - 1)):
            s = self.cores[i].shape
            A = self.cores[i].reshape(s[0], s[1] * s[2])
            U, S, Vt = np.linalg.svd(A, full_matrices=False)
            U, S, Vt, rs, Er = killSVDChi(U, S, Vt, X)
            E += (1 - Er) ** 2 * Er
            self.r[i - 1] = rs
            self.cores[i] = Vt.reshape(Vt.size // (s[1] * s[2]), s[1], s[2])
            M = U @ S
            a = M.shape[0]

            if i == 1:
                self.cores[0] = self.cores[0][:, 0:a]
                self.cores[0] = np.tensordot(self.cores[0], M, axes=(-1, 0))
                break
            self.cores[i - 1] = self.cores[i - 1][:, :, 0:a]
            self.cores[i - 1] = np.tensordot(self.cores[i - 1], M, axes=(-1, 0))

        pc = np.trace(np.matmul(np.transpose(self.cores[0]), self.cores[0]))
        self.E += E

    def retruncateWish(self, ranks):

        Q, R = np.linalg.qr(self.cores[0])
        self.cores[0] = Q
        self.cores[1] = np.tensordot(R, self.cores[1], axes=(1, 0))
        for i in range(1, self.expo - 1):
            A = self.cores[i]
            s = A.shape
            A = A.reshape(s[0] * s[1], s[2])
            Q, R = np.linalg.qr(A)
            self.cores[i] = Q.reshape(s[0], s[1], Q.size // (s[0] * s[1]))
            self.cores[i + 1] = np.tensordot(R, self.cores[i + 1], axes=(1, 0))

        p = np.linalg.norm(self.cores[-1], ord='fro');
        p2 = np.trace(np.matmul(np.transpose(self.cores[-1]), self.cores[-1]))
        p2 = np.sqrt(p2)
        self.cores[-1] = self.cores[-1] / p2
        self.p = self.p * p2

        s = self.cores[-1].shape
        A = self.cores[-1].reshape(s[0], s[1])
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        X=ranks[-1]
        U, S, Vt, rs, Er = killSVDChi(U, S, Vt, X)
        self.r[-1] = rs
        E = (1 - Er) ** 2 * Er
        self.cores[-1] = Vt
        M = U @ S
        a = M.shape[0]
        self.cores[-2] = self.cores[-2][:, :, 0:a]
        self.cores[-2] = np.tensordot(self.cores[-2], M, axes=(-1, 0))

        for i in reversed(range(1, self.expo - 1)):
            s = self.cores[i].shape
            A = self.cores[i].reshape(s[0], s[1] * s[2])
            U, S, Vt = np.linalg.svd(A, full_matrices=False)
            X=ranks[i-1]
            U, S, Vt, rs, Er = killSVDChi(U, S, Vt, X)
            E += (1 - Er) ** 2 * Er
            self.r[i - 1] = rs
            self.cores[i] = Vt.reshape(Vt.size // (s[1] * s[2]), s[1], s[2])
            M = U @ S
            a = M.shape[0]

            if i == 1:
                self.cores[0] = self.cores[0][:, 0:a]
                self.cores[0] = np.tensordot(self.cores[0], M, axes=(-1, 0))
                break
            self.cores[i - 1] = self.cores[i - 1][:, :, 0:a]
            self.cores[i - 1] = np.tensordot(self.cores[i - 1], M, axes=(-1, 0))

        pc = np.trace(np.matmul(np.transpose(self.cores[0]), self.cores[0]))
        self.E += E

    def reset(self,T):
        self.p = 1;
        self.E = 0;
        self.truncated_l(T,0)

class MPO:

    def __init__(self,expo):
        self.cores=[None]*expo
        self.expo=expo
        self.p=1
        self.E=0

    def truncated_r(self,T, e):
        permuter=np.zeros((2*self.expo),dtype=int)
        for i in range(self.expo):
            permuter[2*i]=i
            permuter[2*i+1]=self.expo+i
        permuter=permuter.tolist()
        permuter=tuple(permuter)
        T=T.transpose(permuter)
        dim = T.shape
        N = T.size
        l = self.expo*2

        T = T.reshape(dim[0] * dim[1], N // (dim[0] * dim[1]))
        self.p=np.linalg.norm(T,ord='fro')
        T=T/self.p
        U, S, Vt = np.linalg.svd(T, full_matrices=False)
        U, S, Vt, rs, Er= killSVD(U, S, Vt, e)
        r = np.zeros(l // 2 - 1, dtype=int)
        r[0] = rs
        self.cores[0] = U.reshape(dim[0], dim[1], r[0])
        M = S @ Vt

        MPOs = [None] * (l // 2 - 2)
        for i in range(1, l // 2 - 1):
            ndim = dim[2 * i] + dim[2 * i + 1]
            M = M.reshape(r[i - 1] * ndim, M.size // (r[i - 1] * ndim))
            U, S, Vt = np.linalg.svd(M, full_matrices=False)
            U, S, Vt, rs,Er = killSVD(U, S, Vt, e)
            r[i] = rs
            self.cores[i] = U.reshape(r[i - 1], dim[2 * i], dim[2 * i + 1], r[i])
            M = S @ Vt

        self.cores[-1] = M.reshape(r[-1], dim[l - 2], dim[l - 1])

    def truncated_l(self,T, e):
        permuter = np.zeros((2 * self.expo), dtype=int)
        for i in range(self.expo):
            permuter[2 * i] = i
            permuter[2 * i + 1] = self.expo + i
        permuter = permuter.tolist()
        permuter = tuple(permuter)
        T = T.transpose(permuter)
        dim = T.shape
        N = T.size
        l = self.expo*2

        T = T.reshape(N // (dim[-1] * dim[-2]), (dim[-1] * dim[-2]))
        self.p = np.linalg.norm(T, ord='fro')
        T=T/self.p
        U, S, Vt = np.linalg.svd(T, full_matrices=False)
        U, S, Vt, rs, Er= killSVD(U, S, Vt, e)
        self.E += Er
        r = np.zeros(l // 2 - 1, dtype=int)
        r[-1] = rs
        self.cores[-1] = Vt.reshape(r[-1],dim[-2], dim[-1])
        M = U @ S


        for i in reversed(range(1, l // 2 - 1)):
            ndim = dim[2 * i]*dim[2 * i + 1]
            M = M.reshape(M.size // (r[i] * ndim), (r[i] * ndim))
            U, S, Vt = np.linalg.svd(M, full_matrices=False)
            U, S, Vt, rs,Er = killSVD(U, S, Vt, e)
            self.E += Er
            r[i-1] = rs
            self.cores[i] = Vt.reshape(r[i - 1], dim[2 * i], dim[2 * i + 1], r[i])
            M = U @ S

        self.cores[0] = M.reshape(dim[0], dim[1],r[0])

    def truncated_l(self,T, e):
        permuter = np.zeros((2 * self.expo), dtype=int)
        for i in range(self.expo):
            permuter[2 * i] = i
            permuter[2 * i + 1] = self.expo + i
        permuter = permuter.tolist()
        permuter = tuple(permuter)
        T = T.transpose(permuter)
        dim = T.shape
        N = T.size
        l = self.expo*2

        T = T.reshape(N // (dim[-1] * dim[-2]), (dim[-1] * dim[-2]))
        self.p = np.linalg.norm(T, ord='fro')
        T=T/self.p
        U, S, Vt = np.linalg.svd(T, full_matrices=False)
        U, S, Vt, rs, Er= killSVD(U, S, Vt, e)
        self.E += Er
        r = np.zeros(l // 2 - 1, dtype=int)
        r[-1] = rs
        self.cores[-1] = Vt.reshape(r[-1],dim[-2], dim[-1])
        M = U @ S


        for i in reversed(range(1, l // 2 - 1)):
            ndim = dim[2 * i]*dim[2 * i + 1]
            M = M.reshape(M.size // (r[i] * ndim), (r[i] * ndim))
            U, S, Vt = np.linalg.svd(M, full_matrices=False)
            U, S, Vt, rs,Er = killSVD(U, S, Vt, e)
            self.E += Er
            r[i-1] = rs
            self.cores[i] = Vt.reshape(r[i - 1], dim[2 * i], dim[2 * i + 1], r[i])
            M = U @ S

        self.cores[0] = M.reshape(dim[0], dim[1],r[0])

    def reTensor(self):
        T=np.tensordot(self.cores[0],self.cores[1],axes=(2,0))
        index=5;
        for i in range(1,self.expo-1):
            T=np.tensordot(T,self.cores[i+1],axes=(-1,0))
        return T.reshape(2**self.expo,2**self.expo)*self.p;

def killSVD(U, S, Vt, e):
    # Placeholder for the killSVD function.
    # Adjust this function to match your MATLAB implementation.
    # The output should be U, S, V, rs, and Er
    # Assuming rs is the rank (number of singular values greater than e)
    # and Er is some error measure.
    singular_values = S
    i=len(singular_values)
    total_sum=0
    rs=i
    while i>=0 and np.sqrt(total_sum + singular_values[i-1]**2)<e:
        rs -=1
        total_sum += singular_values[i-1]**2
        i-=1
    if rs == 0:
        rs=1
    U = U[:, :rs]
    S = np.diag(singular_values[:rs])
    Vt = Vt[:rs, :]
    Er = np.sum(singular_values[rs:]**2)
    Er=np.sqrt(Er)
    return U, S, Vt, rs, Er

def killSVDChi(U, S, Vt, X):
    singular_values = S
    i=len(singular_values)
    rs=X
    U = U[:, :rs]
    S = np.diag(singular_values[:rs])
    Vt = Vt[:rs, :]
    Er = np.sum(singular_values[rs:]**2)
    Er=np.sqrt(Er)
    return U, S, Vt, rs, Er

def exactSol(a, expo, t, acc):
    Tex = np.zeros((2 ** expo, 1))

    x = np.linspace(0, 1, 2 ** expo).reshape(-1, 1)

    # Calculate the exact solution
    for i in range(acc + 1):
        Tex -= 1600 / ((2 * i + 1) * np.pi) * np.exp(-a * np.pi ** 2 * (2 * i + 1) ** 2 * t) * np.sin(
            (2 * i + 1) * np.pi * x)

    # Add 400 to the result
    Tex += 400

    return Tex


