"""
Driving-point lateral mobility of the Delkor-supported rail (reduced 3D SAFE)
============================================================================
Long finite rail in the SAFE cross-section modal space (reproduces SAFE
dispersion exactly). The Delkor plate resonator at each bay is CONDENSED
analytically into a frequency-dependent stiffness on the foot-underside nodes

    S(w) = kt_c * PtP  -  kt_c^2 * (b b^T) / ( (k_top+k_bot)(1+i eta_s) - m w^2 )

(kt_c = (k_top/nfoot)(1+i eta_s); PtP, b are the reduced foot-coupling terms),
so no extra DOF is added and the global matrix stays banded. A graded
hysteretic sponge at both ends emulates an infinite rail. Lateral point force
at the rail head at mid-rail; Y = i w u_head / F.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from safe_supercell import reduced_operators

def _rail_blocks(R, d, nz):
    r1, rc, r3, rm = R['r1'], R['rc'], R['r3'], R['rm']
    r = r1.shape[0]; Lz = d/nz
    mz = (Lz/6.0)*np.array([[2., 1.], [1., 2.]])
    kz = (1.0/Lz)*np.array([[1., -1.], [-1., 1.]])
    g  = np.array([[-0.5, 0.5], [-0.5, 0.5]]); rcT = rc.T
    Kel = {}; Mel = {}
    for a in range(2):
        for b in range(2):
            Kel[(a,b)] = mz[a,b]*r1 + kz[a,b]*r3 + g[a,b]*rc + g.T[a,b]*rcT
            Mel[(a,b)] = mz[a,b]*rm
    return Kel, Mel, r

def _assemble(blocks, nel, r, weight=None):
    """Assemble a block-tridiagonal reduced operator; weight[e] scales element e
    (for graded damping). Returns csr (N x N), N=(nel+1)*r."""
    N=(nel+1)*r; rows=[];cols=[];vals=[]
    idx=np.arange(r)
    for e in range(nel):
        we = 1.0 if weight is None else weight[e]
        for a in range(2):
            for b in range(2):
                rows.append(np.repeat((e+a)*r+idx, r))
                cols.append(np.tile((e+b)*r+idx, r))
                vals.append((we*blocks[(a,b)]).ravel())
    return sp.csr_matrix((np.concatenate(vals),
                          (np.concatenate(rows),np.concatenate(cols))),shape=(N,N))

def mobility(freqs, nbay=40, nz=8, d=0.69, k_top=1e8, k_bot=1e7, m_pl=9.0,
             eta_rail=0.01, eta_pad=0.08, eta_end=2.0, nsponge=8, drive='head',
             drive_offset=0, nx=10, ny=32):
    R=reduced_operators(nx,ny)
    Kel,Mel,r=_rail_blocks(R,d,nz)
    nel=nbay*nz; nplane=nel+1; N=nplane*r
    # graded rail damping: element loss factor
    eta_e=np.full(nel, eta_rail)
    for e in range(nel):
        bay=e/nz
        if bay<nsponge:
            x=(nsponge-bay)/nsponge; eta_e[e]=eta_rail+(eta_end-eta_rail)*x**3
        elif bay>nbay-nsponge:
            x=(bay-(nbay-nsponge))/nsponge; eta_e[e]=eta_rail+(eta_end-eta_rail)*x**3
    Kre=_assemble(Kel,nel,r)                      # real rail stiffness
    Kim=_assemble(Kel,nel,r,weight=eta_e)         # imag part (graded)
    Mg =_assemble(Mel,nel,r)
    # foot coupling terms (reduced)
    Phi_foot=R['Phi_foot']; nfoot=Phi_foot.shape[0]
    PtP=Phi_foot.T@Phi_foot; bvec=Phi_foot.sum(axis=0)
    kt=k_top/nfoot
    # support planes (one per bay, interior + ends)
    support_planes=[j*nz for j in range(nbay+1)]
    # blocked support (freq independent) and rank-structure for relief
    # build once: index arrays for a plane block
    idx=np.arange(r)
    # head/drive rows
    Phi=R['Phi']; nodes=R['nodes']
    head=np.argmax(nodes[:,1]); Phi_head=Phi[3*head+0,:]
    pmid=(nbay//2)*nz + drive_offset      # +nz//2 -> mid-span between supports
    F=np.zeros(N,complex); F[pmid*r:(pmid+1)*r]=Phi_head
    # precompute support-plane block placement (COO skeleton) for PtP and bb^T
    sp_rows=[];sp_cols=[]
    bb=np.outer(bvec,bvec)
    PtP_list=[];bb_list=[]
    for p in support_planes:
        rr=np.repeat(p*r+idx,r); cc=np.tile(p*r+idx,r)
        sp_rows.append(rr);sp_cols.append(cc)
    sp_rows=np.concatenate(sp_rows);sp_cols=np.concatenate(sp_cols)
    nS=len(support_planes)
    PtP_flat=np.tile(PtP.ravel(),nS)
    bb_flat=np.tile(bb.ravel(),nS)
    Y=np.zeros(len(freqs),complex)
    Mg=Mg.tocsc()
    for i,f in enumerate(freqs):
        w=2*np.pi*f
        kt_c=kt*(1+1j*eta_pad)
        denom=(k_top+k_bot)*(1+1j*eta_pad)-m_pl*w**2
        sval = kt_c*PtP_flat - (kt_c**2/denom)*bb_flat        # complex support entries
        Ssp=sp.csr_matrix((sval,(sp_rows,sp_cols)),shape=(N,N))
        A=(Kre + 1j*Kim + Ssp - w**2*Mg).tocsc()
        u=spla.spsolve(A,F)
        Y[i]=1j*w*(Phi_head@u[pmid*r:(pmid+1)*r])
    return Y

if __name__=="__main__":
    import time
    t=time.time()
    freqs=np.linspace(450,650,101)
    Y=mobility(freqs)
    mag=np.abs(Y)
    print(f'(elapsed {time.time()-t:.1f}s)')
    for f,m in zip(freqs,mag):
        bar='#'*int(50*m/mag.max())
        print(f'{f:6.1f}  {m:.3e}  {bar}')
