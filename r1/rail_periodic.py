"""
Frequency-domain FE of the periodically supported UIC 60 rail (lateral)
=======================================================================
Builds a 1D finite-element discretisation ALONG THE RAIL AXIS of the
validated Wu & Thompson 6-DOF lateral operator

        M q_tt - D q_zz - G q_z + K_R q = f(z,t)

so that (a) the Floquet band structure of one supported bay can be checked
against the independent transfer-matrix method, and (b) the driving-point
lateral mobility of a long supported rail can be computed directly, which is
the quantity the user measures.

2-node linear elements, 6 DOF/node, DOF = [v_h, psi_h, th_h, v_f, psi_f, th_f].
Element integrals (length Le):
    mass / K_R spatial factor  :  (Le/6)[[2,1],[1,2]]
    D  (from -D q_zz, by parts):  (1/Le)[[1,-1],[-1,1]]
    G  (from -G q_z)           :  -G x [[-1/2, 1/2],[-1/2, 1/2]]  (int Na dNb/dz)
"""
import numpy as np
from scipy.linalg import eig, expm
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from rail_lateral_dispersion_final import system_matrices, h_f, K_F

M6, D6, G6, KR6 = system_matrices()
NDOF = 6

# ---- element spatial factors ----
def elem_matrices(Le):
    fmass = (Le/6.0)*np.array([[2.,1.],[1.,2.]])
    fD    = (1.0/Le)*np.array([[1.,-1.],[-1.,1.]])
    fG    = np.array([[-0.5,0.5],[-0.5,0.5]])   # int Na dNb/dz
    Me = np.zeros((12,12)); Ke = np.zeros((12,12))
    for a in range(2):
        for b in range(2):
            r=slice(6*a,6*a+6); c=slice(6*b,6*b+6)
            Me[r,c] += M6*fmass[a,b]
            Ke[r,c] += D6*fD[a,b] + KR6*fmass[a,b] - G6*fG[a,b]
    return Ke, Me

# ============================================================================
# Floquet band structure of one bay (validation vs transfer matrix)
# ============================================================================
def floquet_bands(freqs, nel=1, d=None, Ks_func=None):
    """Bloch bands of a periodic cell of length d with nel elements and an
    optional support stiffness Ks (6x6, possibly f-dependent) at the left node.
    Returns list of arrays of folded real wavenumbers in [0, pi/d]."""
    if d is None: d = 1.0
    Le = d/nel
    Ke, Me = elem_matrices(Le)
    # assemble cell with (nel+1) nodes, node 0 == left, node nel == right
    nn = nel+1; N = 6*nn
    K = np.zeros((N,N), complex); Mg = np.zeros((N,N), complex)
    for e in range(nel):
        idx = np.r_[6*e:6*e+6, 6*(e+1):6*(e+1)+6]
        K[np.ix_(idx,idx)] += Ke
        Mg[np.ix_(idx,idx)] += Me
    out=[]; kbz=np.pi/d
    for f in freqs:
        w=2*np.pi*f
        Kf = K.copy()
        if Ks_func is not None:
            Kf[0:6,0:6] += Ks_func(w)
        Dyn = Kf - w**2*Mg
        # Bloch reduction: u_right = mu * u_left, mu = exp(-i k d).
        # Partition into left (L=node0), interior (I), right (R=node nel).
        L=np.arange(0,6); R=np.arange(6*nel,6*nel+6)
        I=np.arange(6,6*nel) if nel>1 else np.array([],int)
        # Solve generalized eig for mu via the standard periodic condensation.
        ks=[]
        # Build with condensed interior (if any): reduce interior DOFs by
        # dynamic condensation onto L,R.
        if len(I)>0:
            DII=Dyn[np.ix_(I,I)]; DIL=Dyn[np.ix_(I,L)]; DIR=Dyn[np.ix_(I,R)]
            DLI=Dyn[np.ix_(L,I)]; DRI=Dyn[np.ix_(R,I)]
            DIIinv=np.linalg.inv(DII)
            def cond(AB, XI, IY): return Dyn[np.ix_(AB,IY)] - XI@DIIinv@Dyn[np.ix_(I,IY)]
            DLL=Dyn[np.ix_(L,L)]-DLI@DIIinv@DIL
            DLR=Dyn[np.ix_(L,R)]-DLI@DIIinv@DIR
            DRL=Dyn[np.ix_(R,L)]-DRI@DIIinv@DIL
            DRR=Dyn[np.ix_(R,R)]-DRI@DIIinv@DIR
        else:
            DLL=Dyn[np.ix_(L,L)];DLR=Dyn[np.ix_(L,R)]
            DRL=Dyn[np.ix_(R,L)];DRR=Dyn[np.ix_(R,R)]
        # Transfer-matrix eigenproblem from condensed 2x2-block dynamic stiffness:
        #  [DLL DLR; DRL DRR][uL;uR]=[fL;fR]; periodicity uR=mu uL, fR=-mu fL?
        # Use state y=[uL; fL] with fL = -(DLL uL + DLR uR). Standard:
        #  T [uL;pL] = mu [uL;pL], pL is the left internal force.
        # Simpler robust route: solve polynomial eig via companion on
        #  (DRL) + (DLL+DRR) mu + (DLR) mu^2 = 0  (from requiring nontrivial u).
        # This arises from the periodic dynamic-stiffness of an infinite chain.
        A0=DRL; A1=DLL+DRR; A2=DLR
        # generalized companion: [0 I; -A0 -A1][x; mu x] = mu [I 0;0 A2][...]
        Z=np.zeros((6,6)); Iu=np.eye(6)
        Acomp=np.block([[Z,Iu],[-A0,-A1]])
        Bcomp=np.block([[Iu,Z],[Z,A2]])
        mu=eig(Acomp,Bcomp,right=False)
        for m in mu:
            if np.isfinite(m) and abs(abs(m)-1)<1e-6:
                k=np.real(1j*np.log(m)/d)
                if k>1e-9:
                    x=k%(2*kbz)
                    if x>kbz: x=2*kbz-x
                    ks.append(x)
        out.append(np.array(sorted(set(np.round(ks,4)))))
    return out

# ============================================================================
# reference: transfer-matrix bands (independent method, from handover)
# ============================================================================
def tm_bands(freqs, d, Ks_func=None):
    Dinv=np.linalg.inv(D6); I6=np.eye(6); kbz=np.pi/d; out=[]
    for f in freqs:
        w=2*np.pi*f
        Ac=np.block([[np.zeros((6,6)),I6],[-Dinv@(w**2*M6-KR6),-Dinv@G6]])
        Tf=expm(Ac*d)
        if Ks_func is not None:
            Ks=Ks_func(w)
            J=np.block([[I6,np.zeros((6,6))],[-Dinv@Ks,I6]])
            T=J@Tf
        else:
            T=Tf
        mu=np.linalg.eigvals(T); ks=[]
        for m in mu:
            if abs(abs(m)-1)<1e-6:
                k=np.real(1j*np.log(m)/d)
                if k>1e-9:
                    x=k%(2*kbz)
                    if x>kbz: x=2*kbz-x
                    ks.append(x)
        out.append(np.array(sorted(set(np.round(ks,4)))))
    return out

if __name__=="__main__":
    # ---- VALIDATION 1: FE-Floquet free-rail bands vs transfer matrix ----
    d=0.69
    fs=np.array([200.,400.,600.,800.,1000.])
    print("VALIDATION 1: free-rail folded wavenumbers (rad/m), FE vs TM")
    fe=floquet_bands(fs, nel=24, d=d)
    tm=tm_bands(fs, d)
    for i,f in enumerate(fs):
        print(f"  {f:6.0f} Hz  FE={np.array2string(fe[i],precision=3)}")
        print(f"           TM={np.array2string(tm[i],precision=3)}")
