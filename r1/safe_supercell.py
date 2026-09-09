"""
Discretely-supported UIC 60 rail: 3D SAFE supercell band structure (lateral)
============================================================================
The 6-DOF beam model puts the foot-lateral / "flapping" mode (the one that
couples to a foot-mounted resonator) at ~420-440 Hz, whereas the real 3D
section puts it at ~512-562 Hz (validated in safe_rail). So the measured
523.5 / 557 Hz gap can only be reproduced on the 3D section.

This module:
  1. Builds cross-section operators K1, Kc, K3, Mcs (3 DOF/node) from the
     validated UIC 60 mesh (safe_rail.mesh_section).
  2. Extrudes them into a bay of length d with linear elements along z, giving
     a full 3D FE of one periodic cell (validated against safe_rail dispersion).
  3. Adds the Delkor fastener as an explicit rigid plate (lateral DOF, mass
     9 kg) connected to the foot-underside nodes by k_top and to ground by
     k_bot, at one z-plane per bay.
  4. Solves the Floquet-Bloch eigenproblem w(k) for k in [0, pi/d] and extracts
     the lateral band gap.

Cross-section operator convention matches safe_rail: strain B = Bxy + (-ik)Bz,
Kk(k) = K1 + i k K2 + k^2 K3, K2 = Kc - Kc^T, Kc = int_A Bxy^T C Bz dA.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as sla
from safe_rail import mesh_section, E, nu, rho, G

def cross_section_operators(nx=10, ny=32):
    """Return K1, Kc, K3, Mcs (dense complex-free real arrays) + node data."""
    nodes, elems, nid, (nx_, ny_) = mesh_section(nx, ny)
    lam = E*nu/((1+nu)*(1-2*nu)); mu = G
    C = np.array([
        [lam+2*mu, lam, lam, 0, 0, 0],
        [lam, lam+2*mu, lam, 0, 0, 0],
        [lam, lam, lam+2*mu, 0, 0, 0],
        [0, 0, 0, mu, 0, 0],
        [0, 0, 0, 0, mu, 0],
        [0, 0, 0, 0, 0, mu]], float)
    nn = len(nodes); N = 3*nn
    K1 = np.zeros((N, N)); Kc = np.zeros((N, N))
    K3 = np.zeros((N, N)); Mm = np.zeros((N, N))
    gp = 1/np.sqrt(3); GP = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
    for el in elems:
        xy = nodes[list(el)]
        ke1 = np.zeros((12, 12)); kec = np.zeros((12, 12))
        ke3 = np.zeros((12, 12)); me = np.zeros((12, 12))
        for (xi, et) in GP:
            N4 = 0.25*np.array([(1-xi)*(1-et), (1+xi)*(1-et),
                                (1+xi)*(1+et), (1-xi)*(1+et)])
            dNxi = 0.25*np.array([-(1-et), (1-et), (1+et), -(1+et)])
            dNet = 0.25*np.array([-(1-xi), -(1+xi), (1+xi), (1-xi)])
            J = np.array([[dNxi@xy[:, 0], dNxi@xy[:, 1]],
                          [dNet@xy[:, 0], dNet@xy[:, 1]]])
            detJ = np.linalg.det(J); Ji = np.linalg.inv(J)
            dN = Ji@np.vstack([dNxi, dNet])
            Bxy = np.zeros((6, 12)); Bz = np.zeros((6, 12))
            for a in range(4):
                dx, dy = dN[0, a], dN[1, a]; Na = N4[a]; c = 3*a
                Bxy[0, c+0] = dx; Bxy[1, c+1] = dy
                Bxy[3, c+2] = dy; Bxy[4, c+2] = dx
                Bxy[5, c+0] = dy; Bxy[5, c+1] = dx
                Bz[2, c+2] = Na; Bz[3, c+1] = Na; Bz[4, c+0] = Na
            ke1 += (Bxy.T@C@Bxy)*detJ
            kec += (Bxy.T@C@Bz)*detJ
            ke3 += (Bz.T@C@Bz)*detJ
            Nmat = np.zeros((3, 12))
            for a in range(4):
                Nmat[0, 3*a] = N4[a]; Nmat[1, 3*a+1] = N4[a]; Nmat[2, 3*a+2] = N4[a]
            me += rho*(Nmat.T@Nmat)*detJ
        dofs = np.array([[3*n, 3*n+1, 3*n+2] for n in el]).ravel()
        ix = np.ix_(dofs, dofs)
        K1[ix] += ke1; Kc[ix] += kec; K3[ix] += ke3; Mm[ix] += me
    return K1, Kc, K3, Mm, nodes, elems, nid


# ---- SAFE cross-section modal reduction (fast, exact-in-z-free ROM) ---------
_CACHE = {}
def reduced_operators(nx=10, ny=32, nbasis=80,
                      kbasis=(0.5,1.0,1.5,2.0,2.5,3.0,3.4,3.64,3.8,3.92,4.2,4.6,5.0),
                      modes_per_k=10, tol=1e-8):
    """Reduce the cross-section operators onto a basis built from actual SAFE
    eigenvectors sampled across kbasis (real+imag parts, M-orthonormalised).
    This spans the true (k-dependent, warping) waveguide modes, so the reduced
    K1 + ik(rc-rc^T) + k^2 K3 reproduces SAFE. Returns projected operators and
    the foot coupling data. nbasis caps the final basis size."""
    key = (nx, ny, nbasis, kbasis, modes_per_k)
    if key in _CACHE: return _CACHE[key]
    import scipy.linalg as sla
    K1, Kc, K3, Mm, nodes, elems, nid = cross_section_operators(nx, ny)
    K2 = Kc - Kc.T
    cols = []
    for k in kbasis:
        Kk = K1 + 1j*k*K2 + (k**2)*K3
        Kk = 0.5*(Kk + Kk.conj().T)
        w2, V = sla.eigh(Kk, Mm, subset_by_index=[0, modes_per_k-1])
        cols.append(V.real); cols.append(V.imag)
    B = np.hstack(cols)                       # real, (P x many)
    # M-orthonormalise by SVD of the M-weighted basis
    Lm = np.linalg.cholesky(Mm + 0*np.eye(Mm.shape[0]))
    Bw = Lm.T @ B
    U, s, _ = np.linalg.svd(Bw, full_matrices=False)
    keep = s > tol*s[0]
    U = U[:, keep]
    if U.shape[1] > nbasis: U = U[:, :nbasis]
    Phi = sla.solve_triangular(Lm.T, U)       # M-orthonormal real basis (P x r)
    r1 = Phi.T @ K1 @ Phi
    rc = Phi.T @ Kc @ Phi
    r3 = Phi.T @ K3 @ Phi
    rm = Phi.T @ Mm @ Phi
    foot = foot_underside_nodes(nodes)
    # foot lateral (ux) coupling row in reduced space: sum of Phi rows at foot ux
    foot_ux = np.array([3*n+0 for n in foot])
    Phi_foot = Phi[foot_ux, :]              # (nfoot x nbasis)
    out = dict(r1=r1, rc=rc, r3=r3, rm=rm, Phi=Phi, nodes=nodes, elems=elems,
               foot=foot, foot_ux=foot_ux, Phi_foot=Phi_foot, nbasis=nbasis)
    _CACHE[key] = out
    return out


def rom_safe_dispersion(kvals, nx=10, ny=32, nbasis=80):
    """Free-rail dispersion from the reduced operators (validation vs SAFE)."""
    import scipy.linalg as sla
    R = reduced_operators(nx, ny, nbasis)
    r1, rc, r3, rm = R['r1'], R['rc'], R['r3'], R['rm']
    K2 = rc - rc.T
    out = []
    for k in kvals:
        Kk = r1 + 1j*k*K2 + (k**2)*r3
        Kk = 0.5*(Kk + Kk.conj().T)
        w2 = sla.eigh(Kk, rm, eigvals_only=True)
        w2 = w2[w2 > 1.0]
        out.append((k, np.sqrt(w2)/(2*np.pi)))
    return out


def safe_dispersion_check(K1, Kc, K3, Mm, kvals):
    """Reproduce safe_rail.dispersion_free using K2 = Kc - Kc^T (validation)."""
    import scipy.linalg as sla
    K2 = Kc - Kc.T
    out = []
    for k in kvals:
        Kk = K1 + 1j*k*K2 + (k**2)*K3
        Kk = 0.5*(Kk + Kk.conj().T)
        w2 = sla.eigh(Kk, Mm, eigvals_only=True)
        w2 = w2[w2 > 1.0]
        out.append((k, np.sqrt(w2)/(2*np.pi)))
    return out


def extrude_bay(K1, Kc, K3, Mm, d, nz):
    """Assemble full (nz+1)-plane 3D FE matrices for one bay of length d.
    Linear elements along z. Returns sparse K_full, M_full and plane DOF map.
    K_full is REAL and k-independent; Bloch k enters via the reduction R(k)."""
    P = K1.shape[0]; Lz = d/nz
    mz = (Lz/6.0)*np.array([[2., 1.], [1., 2.]])   # int Nz Nz
    kz = (1.0/Lz)*np.array([[1., -1.], [-1., 1.]])  # int Nz' Nz'
    g  = np.array([[-0.5, 0.5], [-0.5, 0.5]])       # int Nz dNz/dz
    KcT = Kc.T
    nP = nz + 1; N = nP*P
    K = sp.lil_matrix((N, N)); Mg = sp.lil_matrix((N, N))
    def blk(p): return slice(p*P, p*P+P)
    for e in range(nz):
        for a in range(2):
            for b in range(2):
                Kab = mz[a, b]*K1 + kz[a, b]*K3 + g[a, b]*Kc + g.T[a, b]*KcT
                Mab = mz[a, b]*Mm
                K[blk(e+a), blk(e+b)] += Kab
                Mg[blk(e+a), blk(e+b)] += Mab
    return K.tocsr(), Mg.tocsr(), P, nP


def bloch_reduction(P, nP, k, d, nplate=0):
    """R(k): full (nP*P + nplate) DOF from reduced (nz*P + nplate) master DOF.
    Planes 0..nz-1 are masters; plane nz == plane 0 * exp(-i k d).
    Plate DOF (if any) are extra masters appended at the end."""
    nz = nP - 1
    Nfull = nP*P + nplate; Nred = nz*P + nplate
    R = sp.lil_matrix((Nfull, Nred), dtype=complex)
    for p in range(nz):                       # masters map to themselves
        R[p*P:(p+1)*P, p*P:(p+1)*P] = sp.eye(P)
    R[nz*P:(nz+1)*P, 0:P] = np.exp(-1j*k*d)*sp.eye(P)   # last plane slaved
    for j in range(nplate):                   # plate DOF pass through
        R[nP*P+j, nz*P+j] = 1.0
    return R.tocsr()


def foot_underside_nodes(nodes, ymax=2.0e-3, xmax=0.075):
    """Nodes on the foot bottom within the rail-pad half-width (metres)."""
    sel = np.where((nodes[:, 1] <= ymax) & (np.abs(nodes[:, 0]) <= xmax))[0]
    return sel


def extrude_bay_reduced(r1, rc, r3, rm, d, nz):
    """Dense reduced-space extrusion of one bay (r DOF/plane), REAL matrices."""
    r = r1.shape[0]; Lz = d/nz
    mz = (Lz/6.0)*np.array([[2., 1.], [1., 2.]])
    kz = (1.0/Lz)*np.array([[1., -1.], [-1., 1.]])
    g  = np.array([[-0.5, 0.5], [-0.5, 0.5]])
    rcT = rc.T
    nP = nz + 1; N = nP*r
    K = np.zeros((N, N)); M = np.zeros((N, N))
    for e in range(nz):
        for a in range(2):
            for b in range(2):
                Kab = mz[a, b]*r1 + kz[a, b]*r3 + g[a, b]*rc + g.T[a, b]*rcT
                K[(e+a)*r:(e+a+1)*r, (e+b)*r:(e+b+1)*r] += Kab
                M[(e+a)*r:(e+a+1)*r, (e+b)*r:(e+b+1)*r] += mz[a, b]*rm
    return K, M, r, nP


def supercell_bands(kvals, d=0.69, nz=24, nx=10, ny=32, nbasis=160,
                    support=True, k_top=1e8, k_bot=1e7, m_pl=9.0,
                    fmax=900.0, return_vecs=False):
    """Floquet-Bloch w(k) of the (optionally) supported bay via the reduced
    SAFE 3D model. Plate = one lateral DOF (mass m_pl) at plane 0, spring
    k_top distributed to foot-underside ux nodes, spring k_bot to ground."""
    import scipy.linalg as sla
    R = reduced_operators(nx, ny, nbasis)
    r1, rc, r3, rm = R['r1'], R['rc'], R['r3'], R['rm']
    Phi_foot = R['Phi_foot']; nfoot = Phi_foot.shape[0]
    Kf, Mf, r, nP = extrude_bay_reduced(r1, rc, r3, rm, d, nz)
    nplate = 1 if support else 0
    if support:
        N = nP*r + 1
        K2 = np.zeros((N, N)); M2 = np.zeros((N, N))
        K2[:nP*r, :nP*r] = Kf; M2[:nP*r, :nP*r] = Mf
        ip = N-1; M2[ip, ip] = m_pl
        kt = k_top/nfoot
        PtP = Phi_foot.T @ Phi_foot            # (r x r) at plane 0
        b = Phi_foot.sum(axis=0)               # (r,)
        K2[:r, :r] += kt*PtP
        K2[:r, ip] += -kt*b
        K2[ip, :r] += -kt*b
        K2[ip, ip] += k_top + k_bot
        Kf, Mf = K2, M2
    out = []; vecs = []
    for k in kvals:
        # Bloch reduction in dense reduced space
        nz_ = nP-1
        Nred = nz_*r + nplate
        Rk = np.zeros((nP*r + nplate, Nred), dtype=complex)
        for p in range(nz_):
            Rk[p*r:(p+1)*r, p*r:(p+1)*r] = np.eye(r)
        Rk[nz_*r:(nz_+1)*r, 0:r] = np.exp(-1j*k*d)*np.eye(r)
        if nplate: Rk[nP*r, nz_*r] = 1.0
        Kb = Rk.conj().T @ Kf @ Rk
        Mb = Rk.conj().T @ Mf @ Rk
        Kb = 0.5*(Kb+Kb.conj().T); Mb = 0.5*(Mb+Mb.conj().T)
        if return_vecs:
            w2, V = sla.eigh(Kb, Mb)
        else:
            w2 = sla.eigh(Kb, Mb, eigvals_only=True); V=None
        m = w2 > 1.0
        fr = np.sqrt(w2[m])/(2*np.pi)
        sel = fr < fmax
        out.append((k, fr[sel]))
        if return_vecs: vecs.append((k, fr[sel], V[:, m][:, sel], Rk))
    if return_vecs: return out, vecs, R
    return out


def tm_bloch_condensed(freqs, d=0.69, nz=8, nx=10, ny=32,
                       k_top=1e8, k_bot=1e7, m_pl=9.0,
                       eta_pad=0.0, eta_rail=0.0):
    """Rigorous discrete-support Bloch dispersion via reduced z-FE transfer
    matrix with the plate CONDENSED to a frequency-dependent foot stiffness
    S(w). For each frequency returns the number of propagating Bloch waves and
    their folded wavenumbers. Frequencies with zero propagating waves that
    couple to the section = full stop band."""
    R = reduced_operators(nx, ny)
    r1, rc, r3, rm = R['r1'], R['rc'], R['r3'], R['rm']
    r = r1.shape[0]
    Phi_foot = R['Phi_foot']; nfoot = Phi_foot.shape[0]
    PtP = Phi_foot.T@Phi_foot; bvec = Phi_foot.sum(axis=0)
    bb = np.outer(bvec, bvec); kt = k_top/nfoot
    # element blocks
    Lz = d/nz
    mz = (Lz/6.0)*np.array([[2., 1.], [1., 2.]])
    kz = (1.0/Lz)*np.array([[1., -1.], [-1., 1.]])
    g  = np.array([[-0.5, 0.5], [-0.5, 0.5]]); rcT = rc.T
    def elem(a, b): return mz[a,b]*r1 + kz[a,b]*r3 + g[a,b]*rc + g.T[a,b]*rcT
    Kel = {(a,b): elem(a,b) for a in range(2) for b in range(2)}
    Mel = {(a,b): mz[a,b]*rm for a in range(2) for b in range(2)}
    kbz = np.pi/d
    nP = nz+1; N = nP*r
    out = []
    for f in freqs:
        w = 2*np.pi*f
        # assemble bay dynamic stiffness (planes 0..nz), plane r-blocks
        Dyn = np.zeros((N, N), complex)
        for e in range(nz):
            for a in range(2):
                for b in range(2):
                    Dyn[(e+a)*r:(e+a+1)*r, (e+b)*r:(e+b+1)*r] += \
                        Kel[(a,b)]*(1+1j*eta_rail) - w**2*Mel[(a,b)]
        # condensed support at plane 0
        denom = (k_top+k_bot)*(1+1j*eta_pad) - m_pl*w**2
        S = kt*(1+1j*eta_pad)*PtP - (kt*(1+1j*eta_pad))**2/denom*bb
        Dyn[0:r, 0:r] += S
        # condense interior planes onto L=plane0, R=plane nz
        L = np.arange(0, r); Rr = np.arange(nz*r, nz*r+r)
        I = np.arange(r, nz*r)
        DII = Dyn[np.ix_(I, I)]; DIIi = np.linalg.inv(DII)
        def cc(A, B): return Dyn[np.ix_(A, B)] - Dyn[np.ix_(A, I)]@DIIi@Dyn[np.ix_(I, B)]
        DLL, DLR = cc(L, L), cc(L, Rr)
        DRL, DRR = cc(Rr, L), cc(Rr, Rr)
        # polynomial eig: (DRL) + (DLL+DRR) mu + (DLR) mu^2 = 0
        Z = np.zeros((r, r)); Iu = np.eye(r)
        A = np.block([[Z, Iu], [-DRL, -(DLL+DRR)]])
        B = np.block([[Iu, Z], [Z, DLR]])
        mu = sla.eig(A, B, right=False)
        ks = []
        for m in mu:
            if np.isfinite(m) and abs(abs(m)-1) < 1e-4:
                k = np.real(1j*np.log(m)/d)
                if k > 1e-6:
                    x = k % (2*kbz)
                    if x > kbz: x = 2*kbz - x
                    ks.append(x)
        out.append((f, sorted(set(np.round(ks, 3)))))
    return out


if __name__ == "__main__":
    print("Building cross-section operators (nx=10, ny=32) ...")
    K1, Kc, K3, Mm, nodes, elems, nid = cross_section_operators(10, 32)
    print(f"  cross-section DOF = {K1.shape[0]}")
    print("\nVALIDATION 3: free-rail SAFE dispersion (lowest branches, Hz)")
    print("  k(rad/m)   lowest 6 branch frequencies")
    for k, fr in safe_dispersion_check(K1, Kc, K3, Mm, [1.0, 2.0, 3.64, 3.92, 5.0]):
        print(f"  {k:5.2f}   {np.array2string(np.sort(fr)[:6], precision=1, floatmode='fixed')}")
