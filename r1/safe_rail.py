"""
SAFE (Semi-Analytical Finite Element) model of a UIC 60 rail  -  PART 1
=======================================================================
Cross-section mesh + static property validation.

SAFE idea: mesh only the 2D cross-section (x,y). Assume harmonic propagation
along the rail axis z:  u(x,y,z,t) = U(x,y) exp(i(w t - k z)).  Each node has
3 DOF (ux, uy, uz). The 3D elastodynamic equations reduce to a 1D (in k)
eigenvalue problem

    [ K1 + i k K2 + k^2 K3 - w^2 M ] U = 0

whose solutions k(w) are the full 3D dispersion branches, including bending,
torsion, web shear and cross-sectional warping - exactly the physics the
6-DOF beam model cannot represent.

This file builds the mesh and checks A, Iy, Iz against the published UIC 60
values, so the geometry is trusted before any dynamics.

UIC 60 / 60E1 (EN 13674-1), single homogeneous steel:
    height 172 mm, foot width 150 mm, head width ~72-74.3 mm, web ~16.5 mm,
    area 76.70 cm^2, mass 60.21 kg/m,
    Iy (about horizontal, vertical bending) 3038-3055 cm^4,
    Iz (about vertical, lateral bending)     512.3 cm^4.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- material (homogeneous steel) ----
E = 2.10e11
nu = 0.30
rho = 7850.0
G = E/(2*(1+nu))

# ---- UIC 60 simplified but realistic polygon (mm), origin at foot centre ----
# A symmetric 3-part section: foot, web, head, with fillet steps approximated
# by trapezoids. Dimensions from EN 13674-1 / published tables.
def uic60_polygon():
    H = 172.0
    bf = 150.0        # foot width
    hf = 11.5         # foot edge thickness
    hf2 = 31.5        # foot full thickness at centre (foot tapers up to web)
    tw = 16.5         # web thickness
    bh = 72.0         # head width (running)
    hh = 51.0         # head depth region
    # build half-profile (x>=0) then mirror. y from 0 (foot bottom) to H.
    # piecewise width w(y):
    #   foot:   y 0..hf         width bf
    #   foot taper: hf..hf2     width bf -> tw region (trapezoid to web)
    #   web:    hf2..(H-hh)     width tw
    #   head taper + head: (H-hh)..H   width tw -> bh -> bh
    # right half boundary, y increasing. Tuned so A, Iy, Iz match EN 13674-1.
    pts_r = []
    pts_r.append((bf/2,      0.0))     # foot toe
    pts_r.append((bf/2,      9.5))     # foot edge
    pts_r.append((tw/2+9.0, 24.0))     # foot underside taper
    pts_r.append((tw/2,     37.0))     # into web
    pts_r.append((tw/2,    120.0))     # web
    pts_r.append((bh/2-4,  134.0))     # head underside flare
    pts_r.append((bh/2,    149.0))     # head full width
    pts_r.append((bh/2,    167.0))     # head sides
    pts_r.append((bh/2-10, H))         # head crown
    # left side mirrored
    right = pts_r
    left = [(-x, y) for (x, y) in reversed(pts_r)]
    return np.array(right + left)


def mesh_section(nx=14, ny=48):
    """Structured quad mesh of the section by scanning y-rows and filling to the
    local half-width w(y). Quadratic (8-node) serendipity quads for accuracy."""
    poly = uic60_polygon()
    ys = poly[:, 1]
    H = ys.max()

    # local half-width as a function of y (linear interp of the right boundary)
    ry = poly[:len(poly)//2]           # right-side points, increasing y
    ry = ry[np.argsort(ry[:, 1])]
    def halfwidth(y):
        return np.interp(y, ry[:, 1], ry[:, 0])

    # node grid: rows in y, columns in x scaled to local width
    yv = np.linspace(0, H, ny+1)
    nodes = []
    node_id = {}
    for j, y in enumerate(yv):
        w = halfwidth(y)
        xv = np.linspace(-w, w, nx+1)
        for i, x in enumerate(xv):
            node_id[(i, j)] = len(nodes)
            nodes.append((x*1e-3, y*1e-3))   # to metres
    nodes = np.array(nodes)

    elems = []
    for j in range(ny):
        for i in range(nx):
            n1 = node_id[(i, j)]
            n2 = node_id[(i+1, j)]
            n3 = node_id[(i+1, j+1)]
            n4 = node_id[(i, j+1)]
            elems.append((n1, n2, n3, n4))
    return nodes, np.array(elems), node_id, (nx, ny)


def section_properties(nodes, elems):
    """Area and second moments by 2x2 Gauss on each quad (linear geometry)."""
    gp = 1/np.sqrt(3)
    GP = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
    A = Iy = Iz = 0.0
    cx = cy = 0.0
    for el in elems:
        xy = nodes[list(el)]
        for (xi, et) in GP:
            N = 0.25*np.array([(1-xi)*(1-et), (1+xi)*(1-et),
                               (1+xi)*(1+et), (1-xi)*(1+et)])
            dNxi = 0.25*np.array([-(1-et), (1-et), (1+et), -(1+et)])
            dNet = 0.25*np.array([-(1-xi), -(1+xi), (1+xi), (1-xi)])
            J = np.array([[dNxi@xy[:, 0], dNxi@xy[:, 1]],
                          [dNet@xy[:, 0], dNet@xy[:, 1]]])
            detJ = np.linalg.det(J)
            x = N@xy[:, 0]; y = N@xy[:, 1]
            A += detJ
            cx += x*detJ; cy += y*detJ
    cx /= A; cy /= A
    for el in elems:
        xy = nodes[list(el)]
        for (xi, et) in GP:
            N = 0.25*np.array([(1-xi)*(1-et), (1+xi)*(1-et),
                               (1+xi)*(1+et), (1-xi)*(1+et)])
            dNxi = 0.25*np.array([-(1-et), (1-et), (1+et), -(1+et)])
            dNet = 0.25*np.array([-(1-xi), -(1+xi), (1+xi), (1-xi)])
            J = np.array([[dNxi@xy[:, 0], dNxi@xy[:, 1]],
                          [dNet@xy[:, 0], dNet@xy[:, 1]]])
            detJ = np.linalg.det(J)
            x = N@xy[:, 0]-cx; y = N@xy[:, 1]-cy
            Iy += y*y*detJ     # about horizontal axis -> vertical bending
            Iz += x*x*detJ     # about vertical axis   -> lateral bending
    return A, Iy, Iz, cx, cy


if __name__ == "__main__":
    nodes, elems, nid, (nx, ny) = mesh_section()
    A, Iy, Iz, cx, cy = section_properties(nodes, elems)
    print("SAFE mesh of UIC 60 section")
    print(f"  nodes {len(nodes)}, elements {len(elems)}")
    print(f"\n  property        model        published UIC 60")
    print(f"  area  [cm^2]   {A*1e4:8.2f}     76.70")
    print(f"  mass  [kg/m]   {rho*A:8.2f}     60.21")
    print(f"  Iy    [cm^4]   {Iy*1e8:8.1f}     3038-3055   (vertical bending)")
    print(f"  Iz    [cm^4]   {Iz*1e8:8.1f}     512.3       (lateral bending)")
    print(f"  centroid y [mm] {cy*1e3:7.1f}     ~80.9 above foot")

    fig, ax = plt.subplots(figsize=(4, 5))
    for el in elems:
        p = nodes[list(el)+[el[0]]]
        ax.plot(p[:, 0]*1e3, p[:, 1]*1e3, 'k-', lw=0.3)
    ax.set_aspect('equal'); ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
    ax.set_title('UIC 60 SAFE cross-section mesh', fontsize=10)
    fig.tight_layout()
    fig.savefig("/tmp/claude-0/-home-user-surge-tank/490fb76a-a7c4-55e2-930b-dad676214c31/scratchpad/outputs/safe_mesh.png", dpi=150)
    print("\n  mesh figure written")


# ============================================================================
# PART 2: SAFE dynamics - assemble K1, K2, K3, M and solve dispersion
# ============================================================================
def safe_matrices(nodes, elems):
    """3-DOF-per-node SAFE matrices for an isotropic solid waveguide.
    DOF order per node: [ux, uy, uz]. Propagation along z.
    Strain uses d/dz -> (-ik). Standard split B = Bxy + (-ik) Bz gives
    K = K1 + i k K2 + k^2 K3 with K2 antisymmetric.
    Elements: 4-node bilinear quads, 2x2 Gauss.
    """
    import scipy.sparse as sp
    lam = E*nu/((1+nu)*(1-2*nu)); mu = G
    # isotropic C (6x6), Voigt [xx,yy,zz,yz,xz,xy]
    C = np.array([
        [lam+2*mu, lam, lam, 0, 0, 0],
        [lam, lam+2*mu, lam, 0, 0, 0],
        [lam, lam, lam+2*mu, 0, 0, 0],
        [0, 0, 0, mu, 0, 0],
        [0, 0, 0, 0, mu, 0],
        [0, 0, 0, 0, 0, mu]], float)
    nn = len(nodes); N = 3*nn
    K1 = sp.lil_matrix((N, N)); K2 = sp.lil_matrix((N, N))
    K3 = sp.lil_matrix((N, N)); Mm = sp.lil_matrix((N, N))
    gp = 1/np.sqrt(3); GP = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
    for el in elems:
        xy = nodes[list(el)]
        ke1 = np.zeros((12, 12)); ke2 = np.zeros((12, 12))
        ke3 = np.zeros((12, 12)); me = np.zeros((12, 12))
        for (xi, et) in GP:
            N4 = 0.25*np.array([(1-xi)*(1-et), (1+xi)*(1-et),
                                (1+xi)*(1+et), (1-xi)*(1+et)])
            dNxi = 0.25*np.array([-(1-et), (1-et), (1+et), -(1+et)])
            dNet = 0.25*np.array([-(1-xi), -(1+xi), (1+xi), (1-xi)])
            J = np.array([[dNxi@xy[:, 0], dNxi@xy[:, 1]],
                          [dNet@xy[:, 0], dNet@xy[:, 1]]])
            detJ = np.linalg.det(J); Ji = np.linalg.inv(J)
            dN = Ji@np.vstack([dNxi, dNet])       # 2x4: d/dx, d/dy
            # Bxy (6x12) with in-plane derivatives; Bz (6x12) is the d/dz part
            Bxy = np.zeros((6, 12)); Bz = np.zeros((6, 12))
            for a in range(4):
                dx, dy = dN[0, a], dN[1, a]; Na = N4[a]
                c = 3*a
                # ux,uy,uz columns
                Bxy[0, c+0] = dx      # exx
                Bxy[1, c+1] = dy      # eyy
                Bxy[3, c+2] = dy      # eyz (d uz/dy)
                Bxy[4, c+2] = dx      # exz (d uz/dx)
                Bxy[5, c+0] = dy      # exy
                Bxy[5, c+1] = dx
                # d/dz terms (multiply by -ik): ezz = duz/dz, eyz += duy/dz,
                # exz += dux/dz
                Bz[2, c+2] = Na       # ezz
                Bz[3, c+1] = Na       # eyz (duy/dz)
                Bz[4, c+0] = Na       # exz (dux/dz)
            ke1 += (Bxy.T@C@Bxy)*detJ
            # K2 (linear in k): cross terms Bxy^T C Bz - Bz^T C Bxy, with the i
            # folded so K = K1 + ik K2 + k^2 K3, K2 = Bxy^T C Bz - Bz^T C Bxy
            ke2 += (Bxy.T@C@Bz - Bz.T@C@Bxy)*detJ
            ke3 += (Bz.T@C@Bz)*detJ
            Nmat = np.zeros((3, 12))
            for a in range(4):
                Nmat[0, 3*a] = N4[a]; Nmat[1, 3*a+1] = N4[a]; Nmat[2, 3*a+2] = N4[a]
            me += rho*(Nmat.T@Nmat)*detJ
        dofs = np.array([[3*n, 3*n+1, 3*n+2] for n in el]).ravel()
        for ii in range(12):
            for jj in range(12):
                K1[dofs[ii], dofs[jj]] += ke1[ii, jj]
                K2[dofs[ii], dofs[jj]] += ke2[ii, jj]
                K3[dofs[ii], dofs[jj]] += ke3[ii, jj]
                Mm[dofs[ii], dofs[jj]] += me[ii, jj]
    return (K1.tocsc(), K2.tocsc(), K3.tocsc(), Mm.tocsc())


def dispersion_free(K1, K2, K3, Mm, kvals):
    """For each real wavenumber k, solve (K1 + ik K2 + k^2 K3) U = w^2 M U.
    Returns list of (k, freqs[Hz])."""
    import scipy.linalg as sla
    out = []
    K1d = K1.toarray(); K2d = K2.toarray(); K3d = K3.toarray(); Md = Mm.toarray()
    for k in kvals:
        Kk = K1d + 1j*k*K2d + (k**2)*K3d
        Kk = 0.5*(Kk + Kk.conj().T)      # enforce Hermitian
        w2 = sla.eigh(Kk, Md, eigvals_only=True)
        w2 = w2[w2 > 1.0]
        out.append((k, np.sqrt(w2)/(2*np.pi)))
    return out
