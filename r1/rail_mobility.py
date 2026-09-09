"""
Driving-point lateral mobility of the supported rail (frequency domain)
=======================================================================
Long finite rail, 6-DOF/node line FE (assembly from rail_periodic), with
either a continuous foundation K_F(w) (Wu & Thompson validation) or discrete
Delkor fasteners condensed to a frequency-dependent stiffness. Light
hysteretic damping + long domain suppress finite-length modes.

Y(f) = i w u_drive / F  at the driven DOF (lateral rail head, v_h).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from rail_lateral_dispersion_final import (system_matrices, h_f, K_F,
                                           Kp, Kpr, Kb, Ms)
from rail_periodic import elem_matrices, M6, D6, G6, KR6

# foot-underside lateral and roll selection (6-vectors)
S_LAT = np.zeros(6); S_LAT[3]=1.0; S_LAT[5]=-h_f/2
R_ROT = np.zeros(6); R_ROT[5]=1.0

def build_rail(L, dx):
    """Assemble global K (real, undamped) and M for a rail of length L."""
    nel=int(round(L/dx)); Le=L/nel; nn=nel+1; N=6*nn
    Ke,Me=elem_matrices(Le)
    rows=[];cols=[];kv=[];mv=[]
    for e in range(nel):
        idx=np.r_[6*e:6*e+6,6*(e+1):6*(e+1)+6]
        for a in range(12):
            for b in range(12):
                rows.append(idx[a]);cols.append(idx[b])
                kv.append(Ke[a,b]);mv.append(Me[a,b])
    K=sp.csr_matrix((kv,(rows,cols)),shape=(N,N))
    Mg=sp.csr_matrix((mv,(rows,cols)),shape=(N,N))
    return K,Mg,nel,Le,nn

def foundation_distributed(nn, Le, w, damped=True):
    """Continuous foundation K_F(w) lumped to nodes (per unit length x nodal
    tributary length). Returns sparse NxN complex."""
    KFw=K_F(w, damped=damped)
    N=6*nn; rows=[];cols=[];vals=[]
    # nodal tributary length: interior Le, ends Le/2
    for n in range(nn):
        trib = Le if 0<n<nn-1 else Le/2
        for a in range(6):
            for b in range(6):
                if KFw[a,b]!=0:
                    rows.append(6*n+a);cols.append(6*n+b);vals.append(KFw[a,b]*trib)
    return sp.csr_matrix((vals,(rows,cols)),shape=(N,N),dtype=complex)

def supports_discrete(nn, Le, d, w, k_top, k_bot, m_pl, k_rot=0.0,
                      roll_resonator=False, J_pl=None, k_rot_bot=None,
                      start=None):
    """Discrete Delkor fasteners every d metres, condensed frequency-dependent
    stiffness added at the nearest node's foot DOFs. Returns sparse complex."""
    N=6*nn; L=(nn-1)*Le
    if start is None:
        # place a support near each multiple of d, centred so a support sits mid-rail
        nbay=int(round(L/d)); zc=L/2
        zs=[zc + (i-nbay//2)*d for i in range(nbay+1)]
        zs=[z for z in zs if 0<=z<=L]
    else:
        zs=start
    Keff_lat = k_top*(k_bot-m_pl*w**2)/(k_top+k_bot-m_pl*w**2)
    if roll_resonator and J_pl is not None:
        krb = k_rot if k_rot_bot is None else k_rot_bot
        Keff_rot = k_rot*(krb - J_pl*w**2)/(k_rot+krb - J_pl*w**2)
    else:
        Keff_rot = k_rot   # static roll stiffness
    Ks = Keff_lat*np.outer(S_LAT,S_LAT) + Keff_rot*np.outer(R_ROT,R_ROT)
    rows=[];cols=[];vals=[]
    node_ids=[]
    for z in zs:
        n=int(round(z/Le)); node_ids.append(n)
        for a in range(6):
            for b in range(6):
                if Ks[a,b]!=0:
                    rows.append(6*n+a);cols.append(6*n+b);vals.append(Ks[a,b])
    return sp.csr_matrix((vals,(rows,cols)),shape=(N,N),dtype=complex), node_ids

def mobility(freqs, L, dx, support, eta=0.02, drive='head', **sk):
    """support: 'continuous' or 'discrete'. Returns (Y, drive_dof, info)."""
    K,Mg,nel,Le,nn=build_rail(L,dx)
    ndrive=nn//2
    ddof=6*ndrive + (0 if drive=='head' else 3)   # v_h or v_f
    Y=np.zeros(len(freqs),complex)
    Kc=K.astype(complex)
    info={}
    for i,f in enumerate(freqs):
        w=2*np.pi*f
        if support=='continuous':
            A=(Kc*(1+1j*eta)) + foundation_distributed(nn,Le,w) - w**2*Mg
        else:
            Ks,nid=supports_discrete(nn,Le,sk['d'],w,sk['k_top'],sk['k_bot'],
                                     sk['m_pl'],sk.get('k_rot',0.0),
                                     sk.get('roll_resonator',False),
                                     sk.get('J_pl',None),sk.get('k_rot_bot',None))
            A=(Kc*(1+1j*eta)) + Ks - w**2*Mg
            info['nsupport']=len(nid)
        F=np.zeros(6*nn,complex); F[ddof]=1.0
        u=spla.spsolve(A.tocsc(),F)
        Y[i]=1j*w*u[ddof]
    return Y, ddof, info

if __name__=="__main__":
    # VALIDATION 2: Wu & Thompson track C continuous-support point mobility.
    # Expect lateral driving-point resonances near 100,160,380 Hz and the
    # higher web-bending features; compare peak positions.
    freqs=np.linspace(20,2000,400)
    Y,dd,_=mobility(freqs, L=40.0, dx=0.02, support='continuous', eta=0.02)
    mag=np.abs(Y)
    # find local maxima
    pk=[]
    for i in range(2,len(freqs)-2):
        if mag[i]>mag[i-1] and mag[i]>mag[i+1] and mag[i]>mag[i-2] and mag[i]>mag[i+2]:
            pk.append((freqs[i],mag[i]))
    print("VALIDATION 2: Wu & Thompson track C, lateral driving-point mobility")
    print("  detected peaks (Hz, |Y| m/s/N):")
    for fp,mp in pk:
        print(f"    {fp:7.1f}   {mp:.3e}")
    print("  reference resonances ~ 100, 160, 380, (500), 900, 1700 Hz")
