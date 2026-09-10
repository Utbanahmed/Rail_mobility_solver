"""
rail_cell.py  --  shared 3D brick-mesh generator for the periodically supported
                  UIC60 / 60E1 rail (Delkor HAF fastening).

This module is imported by both tools:
    bloch_dispersion.py   (Tool A, Floquet-Bloch band diagram of one unit cell)
    transient_fullwave.py (Tool B, explicit time-domain full-wave solver)
so that BOTH tools use exactly the same cell generator, mirroring the Strand7
workflow: generate + quad the UIC60 cross-section, then extrude along the axis
into 8-node bricks.  A unit cell is one extruded period; the transient domain is
cells copied in series.

Units are SI throughout:  metres, kilograms, seconds, newtons, pascals.

-----------------------------------------------------------------------------
COORDINATE CONVENTION
    X : lateral   (horizontal, across the rail)   -> lateral bending / torsion
    Y : vertical  (up the web, head at top)        -> vertical bending
    Z : axial     (along the rail, extrusion dir)  -> wave propagation direction
The drive point (Tool B default) is a LATERAL (X) force at a rail-head node.
Support springs act in X (lateral) and Y (vertical), sourced separately.
-----------------------------------------------------------------------------
"""

import numpy as np
import scipy.sparse as sp


# ===========================================================================
# CONFIG  (shared physical constants; per-tool knobs live at the top of each tool)
# ===========================================================================
class Cfg:
    # --- steel ---
    E   = 210.0e9        # Pa   Young's modulus
    nu  = 0.30           # -    Poisson
    rho = 7850.0         # kg/m3 density

    # --- rail period (Delkor HAF support spacing) ---
    L   = 0.69           # m    support period (sleeper/baseplate spacing)

    # --- support stiffness, per direction (SI, N/m) -------------------------
    # SOURCING (see REPORT.md):
    #  Vertical k_top  = rail pad (rail foot -> baseplate top).  EN 13146 /
    #    fastening-pad practice puts static rail-pad vertical stiffness in the
    #    range 80-125 kN/mm; we take the mid value 100 kN/mm = 1.0e8 N/m.
    #  Vertical k_bot  = resilient baseplate element (baseplate -> ground).  The
    #    Delkor Egg/Alt.1 family is a *high-attenuation* direct-fixation
    #    fastener whose selling point is a LOW resilient-element stiffness;
    #    published high-attenuation baseplate vertical stiffness is typically
    #    7-20 kN/mm.  We take 10 kN/mm = 1.0e7 N/m.  (These two values reproduce
    #    the plate support resonance sqrt((k_top+k_bot)/m)/2pi ~ 556 Hz used as a
    #    cross-check.)  Firm manufacturer value not public -> documented range.
    kx_top = 1.0e8       # N/m  lateral  rail-foot -> baseplate
    ky_top = 1.0e8       # N/m  vertical rail-foot -> baseplate
    kx_bot = 1.0e7       # N/m  lateral  baseplate -> ground
    ky_bot = 1.0e7       # N/m  vertical baseplate -> ground

    # Lateral stiffness is the PRIMARY CALIBRATION PARAMETER (unknown for this
    # fastening).  By default it equals the vertical value; the tools sweep a
    # lateral scale factor s_lat over kx_top,kx_bot = s_lat * (ky_top,ky_bot).
    s_lat = 1.0

    # --- baseplate mass (V1 FULL variant) ---
    m_plate = 9.0        # kg   Delkor top plate mass (SGI), per support

    # --- cross-section mesh resolution --------------------------------------
    # The cross-section is meshed with AXIS-ALIGNED rectangular quads (element
    # edges vertical/horizontal): a graded structured x-y grid with an element
    # kept ("active", = steel) when its centre lies inside the UIC60 profile.
    # This is essential: a width-mapped (scale-x-by-halfwidth) grid produces
    # slanted columns that act as diagonal bracing and make the section ~100x
    # too stiff in lateral bending (verified).  Axis-aligned quads bend
    # correctly (cantilever within a few % of EI theory).
    #   XB_HALF : half-width column boundaries (mm), placed at the key rail
    #             widths (web 8.25, head 36, foot 75) so the staircase matches.
    #   DY      : target row height (m).
    XB_HALF = [0.0, 8.25, 18.0, 28.0, 36.0, 50.0, 63.0, 75.0]   # mm
    DY = 0.007          # m   target vertical row height (min element ~6 mm)
    REFINE = 1          # integer mesh-refinement factor (convergence studies)

    # --- fillet-resolved section (gmsh conforming quad mesh) -----------------
    # USE_GMSH=True replaces the axis-aligned staircase with a smooth,
    # boundary-conforming all-quad mesh of the true (filleted) 60E1 outline.
    # GMSH_H     : target element size (m).
    # GMSH_FILLET: corner-rounding radius applied to the outline (mm).
    # GMSH_FOOTSCALE: small foot-width compensation so the smoothed outline
    #               still reproduces Iy (foot dominates lateral bending).
    USE_GMSH = False
    GMSH_H = 0.004
    GMSH_FILLET = 2.0
    GMSH_FOOTSCALE = 1.011


# ---------------------------------------------------------------------------
# UIC60 / 60E1 half-width profile  (validated: see verify_section)
#   y   measured from foot underside (m);  hw = half width (m).
#   Fitted to published 60E1 nominals: A=76.70 cm2, Ix=3038 cm4 (vertical
#   bending), Iy=512 cm4 (lateral bending), centroid ~80.5 mm above foot.
# ---------------------------------------------------------------------------
_PROFILE_Y_MM  = np.array([0.,   8.,   20.,   35.,  75.,  115.,  128.,  140., 156., 166., 172.])
_PROFILE_HW_MM = np.array([75.,  75.,  34.66, 8.25, 8.25, 8.25,  24.71, 36.,  36.,  36.,  8.])
H_RAIL = 0.172   # m  total section height


def halfwidth(y):
    """Half width (m) of the UIC60 section at height y (m) above the foot."""
    return np.interp(y, _PROFILE_Y_MM * 1e-3, _PROFILE_HW_MM * 1e-3)


# ===========================================================================
# 1.  CROSS-SECTION QUAD MESH  (axis-aligned active-element grid)
# ===========================================================================
def build_cross_section_gmsh(h=None, fillet_mm=None, foot_scale=None):
    """
    Fillet-resolved UIC60 cross-section: a smooth, boundary-conforming ALL-QUAD
    mesh of the true (rounded) 60E1 outline, generated with gmsh (blossom
    recombination).  Unlike the staircase, the boundary is smooth and the
    elements are well shaped, so the torsion/warping behaviour and local
    cross-section modes are captured properly (no jagged corners, no spurious
    head-sway resonance from blocky junctions).  Returns (nodes2d, quads,
    meshinfo) with the same interface as build_cross_section.
    """
    import gmsh
    from scipy.ndimage import gaussian_filter1d
    if h is None: h = Cfg.GMSH_H
    if fillet_mm is None: fillet_mm = Cfg.GMSH_FILLET
    if foot_scale is None: foot_scale = Cfg.GMSH_FOOTSCALE

    # smooth (filleted) half-width profile, foot compensated for the rounding
    yf = np.linspace(0, H_RAIL, 600)
    hwf = np.interp(yf, _PROFILE_Y_MM * 1e-3, _PROFILE_HW_MM * 1e-3).copy()
    hwf[yf < 0.045] *= foot_scale
    sig = max(1.0, fillet_mm * 1e-3 / (yf[1] - yf[0]))
    hwf = np.clip(gaussian_filter1d(hwf, sig, mode="nearest"), 3e-3, None)

    npts = max(20, int(H_RAIL / h))
    yc = np.linspace(0, H_RAIL, npts); hwc = np.interp(yc, yf, hwf)
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("uic60")
        geo = gmsh.model.geo
        Rp = [geo.addPoint(hwc[i], yc[i], 0, h) for i in range(npts)]
        Lp = [geo.addPoint(-hwc[i], yc[i], 0, h) for i in range(npts - 1, -1, -1)]
        rs = geo.addSpline(Rp); top = geo.addLine(Rp[-1], Lp[0])
        ls = geo.addSpline(Lp); bot = geo.addLine(Lp[-1], Rp[0])
        geo.addPlaneSurface([geo.addCurveLoop([rs, top, ls, bot])]); geo.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMin", h)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.model.mesh.generate(2)
        nt, nc, _ = gmsh.model.mesh.getNodes(); nc = nc.reshape(-1, 3)
        tag2idx = {int(t): i for i, t in enumerate(nt)}
        nodes = nc[:, :2].copy()
        et, _, ENT = gmsh.model.mesh.getElements(2)
        quads = []
        for typ, conn in zip(et, ENT):
            if typ == 3:
                for q in conn.reshape(-1, 4):
                    quads.append([tag2idx[int(t)] for t in q])
    finally:
        gmsh.finalize()
    quads = np.array(quads, int)
    used = np.unique(quads); remap = {o: i for i, o in enumerate(used)}
    nodes = nodes[used]; quads = np.vectorize(remap.get)(quads)
    # enforce CCW (positive signed area) so hex Jacobians are positive
    for k, q in enumerate(quads):
        p = nodes[q]
        a = sum(p[i, 0] * p[(i + 1) % 4, 1] - p[(i + 1) % 4, 0] * p[i, 1] for i in range(4))
        if a < 0:
            quads[k] = q[::-1]
    foot_cs = np.where(nodes[:, 1] < 1e-6)[0]
    meshinfo = {"foot_cs": foot_cs, "nodes2d": nodes, "Ncs": len(nodes),
                "gmsh": True, "h": h}
    return nodes, quads, meshinfo


def build_cross_section(dy_target=Cfg.DY, xb_half_mm=None, refine=Cfg.REFINE):
    if Cfg.USE_GMSH:
        return build_cross_section_gmsh()
    """
    Quad-mesh the UIC60 cross-section with AXIS-ALIGNED rectangular elements.

    A graded structured x-y grid is built with column boundaries at the key
    rail half-widths (web, head, foot) and rows aligned to the profile breaks.
    A cell is kept ("active" = steel) if its centre lies inside the profile
    |x| <= halfwidth(y).  Unlike a width-mapped grid, every element edge is
    vertical or horizontal, so the mesh bends correctly (no diagonal-bracing
    over-stiffening).

    Returns
      nodes2d : (Ncs,2) (x,y) of the nodes actually used (m)
      quads   : (Nq,4)  CCW corner node ids
      meshinfo: dict with grid lines, foot-row node ids, etc.
    """
    if xb_half_mm is None:
        xb_half_mm = Cfg.XB_HALF
    # refine: subdivide each x interval and shrink dy
    xbh = np.array(xb_half_mm, float) * 1e-3
    if refine > 1:
        parts = [np.linspace(xbh[i], xbh[i + 1], refine + 1)[:-1] for i in range(len(xbh) - 1)]
        xbh = np.concatenate(parts + [xbh[-1:]])
    dy_target = dy_target / refine
    xb = np.unique(np.concatenate([-xbh[::-1], xbh]))

    # y rows aligned to profile breaks, graded to ~dy_target
    yb = _PROFILE_Y_MM * 1e-3
    ylist = [0.0]
    for a, b in zip(yb[:-1], yb[1:]):
        nsub = max(1, int(round((b - a) / dy_target)))
        ylist.extend(np.linspace(a, b, nsub + 1)[1:].tolist())
    yg = np.unique(np.round(np.array(ylist), 9))

    nxg, nyg = len(xb) - 1, len(yg) - 1
    nodemap = {}
    nodes = []

    def gn(i, j):
        key = (i, j)
        if key not in nodemap:
            nodemap[key] = len(nodes)
            nodes.append([xb[i], yg[j]])
        return nodemap[key]

    quads = []
    for j in range(nyg):
        yc = 0.5 * (yg[j] + yg[j + 1])
        hw = halfwidth(yc)
        for i in range(nxg):
            xc = 0.5 * (xb[i] + xb[i + 1])
            if abs(xc) <= hw + 1e-12:
                quads.append([gn(i, j), gn(i + 1, j), gn(i + 1, j + 1), gn(i, j + 1)])
    nodes2d = np.array(nodes)
    quads = np.array(quads, dtype=int)

    # tag foot-underside nodes (y ~ 0) for support attachment
    foot_cs = np.where(nodes2d[:, 1] < 1e-9)[0]
    meshinfo = {"xb": xb, "yg": yg, "foot_cs": foot_cs, "nodes2d": nodes2d,
                "Ncs": len(nodes2d), "dy_target": dy_target, "refine": refine}
    return nodes2d, quads, meshinfo


def verify_section(nodes2d, quads, verbose=True, tol=0.05):
    """
    Compute area, Ix (vertical bending, about horizontal centroidal axis),
    Iy (lateral bending, about vertical axis) and mass/length directly from the
    quad mesh via 2x2 Gauss integration, and compare to published 60E1 values.
    Raises AssertionError if any is off by more than `tol` (fraction).
    """
    gp = 1.0 / np.sqrt(3.0)
    gpts = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]

    def shape(xi, eta):
        N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                             (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
        dNxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
        dNet = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
        return N, dNxi, dNet

    A = Sx = Ixx0 = Iyy0 = 0.0
    for q in quads:
        xy = nodes2d[q]                                  # (4,2)
        for (xi, eta) in gpts:
            N, dNxi, dNet = shape(xi, eta)
            J = np.array([[dNxi @ xy[:, 0], dNxi @ xy[:, 1]],
                          [dNet @ xy[:, 0], dNet @ xy[:, 1]]])
            detJ = np.linalg.det(J)
            xg, yg = N @ xy[:, 0], N @ xy[:, 1]
            w = detJ  # gauss weight 1
            A += w
            Sx += yg * w
            Ixx0 += yg * yg * w
            Iyy0 += xg * xg * w
    yc = Sx / A
    Ix = Ixx0 - A * yc * yc     # about horizontal centroidal axis (vertical bending)
    Iy = Iyy0                   # about vertical axis x=0 (section symmetric) (lateral bending)

    A_t, Ix_t, Iy_t = 76.70e-4, 3038e-8, 512e-8       # m2, m4, m4
    mass_t = Cfg.rho * A_t
    res = {
        "A_cm2": A * 1e4, "Ix_cm4": Ix * 1e8, "Iy_cm4": Iy * 1e8,
        "yc_mm": yc * 1e3, "mass_kg_per_m": Cfg.rho * A,
        "eA": (A - A_t) / A_t, "eIx": (Ix - Ix_t) / Ix_t, "eIy": (Iy - Iy_t) / Iy_t,
    }
    if verbose:
        print("  [section verify]  (computed from the quad mesh that is actually used)")
        print(f"    A  = {res['A_cm2']:7.2f} cm^2   target 76.70   err {100*res['eA']:+5.2f}%")
        print(f"    Ix = {res['Ix_cm4']:7.1f} cm^4   target 3038    err {100*res['eIx']:+5.2f}%  (vertical bending)")
        print(f"    Iy = {res['Iy_cm4']:7.1f} cm^4   target 512     err {100*res['eIy']:+5.2f}%  (lateral bending)")
        print(f"    centroid y = {res['yc_mm']:.1f} mm (published ~80.5)")
        print(f"    mass = {res['mass_kg_per_m']:.2f} kg/m (published ~60.21),  target mass {mass_t:.2f}")
    ok = (abs(res["eA"]) < tol and abs(res["eIx"]) < tol and abs(res["eIy"]) < tol)
    assert ok, "SECTION VERIFY FAILED: auto-built UIC60 profile off by >%.0f%%" % (100 * tol)
    return res


# ===========================================================================
# 2.  HEX8 ELEMENT MATRICES  (trilinear brick, 2x2x2 Gauss)
# ===========================================================================
_HEX_REF = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                     [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)


def _D_matrix(E, nu):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[0, 0] = D[1, 1] = D[2, 2] = lam + 2 * mu
    D[3, 3] = D[4, 4] = D[5, 5] = mu
    return D


def _strain_disp(dphi):
    """Assemble a (6, 3*n) strain-displacement block from nodal/mode gradients
    dphi (n,3) = d(shape)/d(x,y,z).  Strain order [exx,eyy,ezz,gxy,gyz,gzx]."""
    n = dphi.shape[0]
    B = np.zeros((6, 3 * n))
    B[0, 0::3] = dphi[:, 0]
    B[1, 1::3] = dphi[:, 1]
    B[2, 2::3] = dphi[:, 2]
    B[3, 0::3] = dphi[:, 1]; B[3, 1::3] = dphi[:, 0]
    B[4, 1::3] = dphi[:, 2]; B[4, 2::3] = dphi[:, 1]
    B[5, 0::3] = dphi[:, 2]; B[5, 2::3] = dphi[:, 0]
    return B


def hex8_matrices(Xe, E=Cfg.E, nu=Cfg.nu, rho=Cfg.rho, incompatible=True):
    """
    Stiffness Ke (24x24), consistent mass Mc (24x24) and lumped mass diagonal
    mlump (24,) for one 8-node hex with nodal coords Xe (8,3).
    DOF order per node: (ux,uy,uz).  Node order matches _HEX_REF.

    `incompatible=True` -> Taylor/Wilson incompatible-modes element (the
    8-node "C3D8I"): 3 internal bending enhancement modes (9 internal dofs,
    modes 1-xi^2, 1-eta^2, 1-zeta^2 in each direction) are added and statically
    condensed out.  This removes the shear locking that cripples plain
    trilinear hexes on the rail's high-aspect-ratio, distorted mapped mesh
    (a pure-hex cantilever of rail locks to ~1% of the true deflection without
    it).  The node count and mass are unchanged; only the stiffness improves.
    """
    D = _D_matrix(E, nu)
    g = 1.0 / np.sqrt(3.0)
    gp = [-g, g]
    s = _HEX_REF

    # centroid Jacobian J0 (for the incompatible-mode metric + patch-test scaling)
    dN0 = 0.125 * s                       # d N_a / d(nat) at xi=eta=ze=0
    J0 = dN0.T @ Xe
    detJ0 = np.linalg.det(J0)
    J0inv = np.linalg.inv(J0)

    Kuu = np.zeros((24, 24))
    Kua = np.zeros((24, 9))
    Kaa = np.zeros((9, 9))
    Mc = np.zeros((24, 24))
    mrow = np.zeros(24)
    for xi in gp:
        for eta in gp:
            for ze in gp:
                N = 0.125 * (1 + s[:, 0] * xi) * (1 + s[:, 1] * eta) * (1 + s[:, 2] * ze)
                dN = np.empty((8, 3))
                dN[:, 0] = 0.125 * s[:, 0] * (1 + s[:, 1] * eta) * (1 + s[:, 2] * ze)
                dN[:, 1] = 0.125 * s[:, 1] * (1 + s[:, 0] * xi) * (1 + s[:, 2] * ze)
                dN[:, 2] = 0.125 * s[:, 2] * (1 + s[:, 0] * xi) * (1 + s[:, 1] * eta)
                J = dN.T @ Xe
                detJ = np.linalg.det(J)
                dNx = dN @ np.linalg.inv(J)
                B = _strain_disp(dNx)
                Kuu += B.T @ D @ B * detJ
                Nmat = np.zeros((3, 24))
                Nmat[0, 0::3] = N; Nmat[1, 1::3] = N; Nmat[2, 2::3] = N
                Mc += rho * (Nmat.T @ Nmat) * detJ
                for k in range(3):
                    mrow[k::3] += rho * N * detJ
                if incompatible:
                    # incompatible-mode natural gradients: modes 1-xi^2,1-eta^2,1-ze^2
                    dMnat = np.array([[-2 * xi, 0.0, 0.0],
                                      [0.0, -2 * eta, 0.0],
                                      [0.0, 0.0, -2 * ze]])
                    dMx = dMnat @ J0inv                 # map with centroid metric
                    dMx *= (detJ0 / detJ)               # Taylor patch-test scaling
                    Ba = _strain_disp(dMx)              # (6,9)
                    Kua += B.T @ D @ Ba * detJ
                    Kaa += Ba.T @ D @ Ba * detJ
    if incompatible:
        Ke = Kuu - Kua @ np.linalg.solve(Kaa, Kua.T)
        Ke = 0.5 * (Ke + Ke.T)
    else:
        Ke = Kuu
    return Ke, Mc, mrow


# ===========================================================================
# 3.  EXTRUDE  -> 3D brick mesh
# ===========================================================================
def extrude(nodes2d, quads, zs):
    """
    Extrude the cross-section along axial stations zs (Nz,) into 8-node bricks.
    Returns nodes3d (Nnode,3), hexes (Nhex,8) with node order matching _HEX_REF,
    and Ncs (cross-section node count).
    """
    Ncs = nodes2d.shape[0]
    Nz = len(zs)
    nodes3d = np.empty((Ncs * Nz, 3))
    for l, z in enumerate(zs):
        nodes3d[l * Ncs:(l + 1) * Ncs, 0:2] = nodes2d
        nodes3d[l * Ncs:(l + 1) * Ncs, 2] = z
    hexes = []
    for l in range(Nz - 1):
        base0, base1 = l * Ncs, (l + 1) * Ncs
        for q in quads:
            hexes.append([base0 + q[0], base0 + q[1], base0 + q[2], base0 + q[3],
                          base1 + q[0], base1 + q[1], base1 + q[2], base1 + q[3]])
    return nodes3d, np.array(hexes, dtype=int), Ncs


# ===========================================================================
# 4.  GLOBAL ASSEMBLY  (reuse per-quad element matrices across axial layers)
# ===========================================================================
def assemble_brick(nodes2d, quads, zs, cfg=Cfg):
    """
    Assemble the *rail* global K (sparse), consistent M (sparse), and lumped
    mass vector mlump (ndof,) for the extruded brick mesh on stations zs.
    Exploits that every axial element with the same dz reuses one element
    matrix per cross-section quad.
    Returns K, Mc, mlump, nodes3d, Ncs, ndof.
    """
    nodes3d, hexes, Ncs = extrude(nodes2d, quads, zs)
    Nnode = nodes3d.shape[0]
    ndof = 3 * Nnode

    # precompute element matrices per (quad, layer-dz).  Cache by rounded dz.
    dz_layers = np.diff(zs)
    cache = {}

    rows = []; cols = []; kdat = []; mdat = []
    mlump = np.zeros(ndof)

    for l in range(len(zs) - 1):
        dz = dz_layers[l]
        key = round(dz, 12)
        if key not in cache:
            mats = []
            z0, z1 = 0.0, dz
            for q in quads:
                xy = nodes2d[q]
                Xe = np.array([[xy[0, 0], xy[0, 1], z0], [xy[1, 0], xy[1, 1], z0],
                               [xy[2, 0], xy[2, 1], z0], [xy[3, 0], xy[3, 1], z0],
                               [xy[0, 0], xy[0, 1], z1], [xy[1, 0], xy[1, 1], z1],
                               [xy[2, 0], xy[2, 1], z1], [xy[3, 0], xy[3, 1], z1]])
                mats.append(hex8_matrices(Xe, cfg.E, cfg.nu, cfg.rho))
            cache[key] = mats
        mats = cache[key]
        base0, base1 = l * Ncs, (l + 1) * Ncs
        for qi, q in enumerate(quads):
            Ke, Mc, mrow = mats[qi]
            gnodes = np.array([base0 + q[0], base0 + q[1], base0 + q[2], base0 + q[3],
                               base1 + q[0], base1 + q[1], base1 + q[2], base1 + q[3]])
            edof = np.empty(24, dtype=int)
            edof[0::3] = 3 * gnodes
            edof[1::3] = 3 * gnodes + 1
            edof[2::3] = 3 * gnodes + 2
            R = np.repeat(edof, 24)
            C = np.tile(edof, 24)
            rows.append(R); cols.append(C)
            kdat.append(Ke.ravel()); mdat.append(Mc.ravel())
            mlump[edof] += mrow

    rows = np.concatenate(rows); cols = np.concatenate(cols)
    K = sp.coo_matrix((np.concatenate(kdat), (rows, cols)), shape=(ndof, ndof)).tocsr()
    Mc = sp.coo_matrix((np.concatenate(mdat), (rows, cols)), shape=(ndof, ndof)).tocsr()
    return K, Mc, mlump, nodes3d, Ncs, ndof


# ===========================================================================
# 5.  SUPPORT ASSEMBLY  (discrete periodic supports, two variants)
# ===========================================================================
def foot_nodes_at(Ncs, nodes2d, meshinfo, layer):
    """Global node ids of the foot-underside row (y=0) at axial `layer`."""
    return layer * Ncs + meshinfo["foot_cs"]


def add_supports(K, mlump, Mc, nodes3d, Ncs, nodes2d, meshinfo, support_layers,
                 variant="V1", cfg=Cfg, s_lat=None):
    """
    Add the discrete periodic supports to the assembled rail operators.

    variant "V1" (FULL)    : rail-foot --k_top--> [plate mass] --k_bot--> ground.
                             Each support adds ONE plate node (X,Y dofs; Z pinned)
                             carrying m_plate.  Springs act in X (lateral) and
                             Y (vertical).  k_top is split equally among the foot
                             nodes (springs in parallel -> total k_top); k_bot is
                             the single plate-to-ground spring.
    variant "V2" (REDUCED) : plate removed; rail-foot --k_series--> ground with
                             k_series = k_top*k_bot/(k_top+k_bot), per direction.

    `support_layers` : list of axial layer indices carrying a support.
    `s_lat`          : lateral stiffness scale (overrides cfg.s_lat if given).

    Returns K_new (csr), M_new (csr, consistent + plate), mlump_new (vec),
    plate_dofs (list of (dof_x, dof_y) per support; empty for V2), info dict.
    For V1 the operators are grown by 2 dofs per support (plate X,Y appended
    after the rail dofs).
    """
    if s_lat is None:
        s_lat = cfg.s_lat
    kx_top = cfg.ky_top * s_lat
    ky_top = cfg.ky_top
    kx_bot = cfg.ky_bot * s_lat
    ky_bot = cfg.ky_bot

    ndof_rail = K.shape[0]
    plate_dofs = []

    # spring stiffness contributions collected as COO triplets, then added once.
    rI, rJ, rV = [], [], []

    def spring(p, q, k):
        """add a scalar spring k between dofs p and q (q=None -> grounded)."""
        rI.append(p); rJ.append(p); rV.append(k)
        if q is not None:
            rI.append(q); rJ.append(q); rV.append(k)
            rI.append(p); rJ.append(q); rV.append(-k)
            rI.append(q); rJ.append(p); rV.append(-k)

    if variant == "V2":
        ks_x = kx_top * kx_bot / (kx_top + kx_bot)
        ks_y = ky_top * ky_bot / (ky_top + ky_bot)
        for lay in support_layers:
            fn = foot_nodes_at(Ncs, nodes2d, meshinfo, lay)
            nper = len(fn)
            for n in fn:
                spring(3 * n + 0, None, ks_x / nper)   # lateral to ground
                spring(3 * n + 1, None, ks_y / nper)   # vertical to ground
        Kadd = sp.coo_matrix((rV, (rI, rJ)), shape=(ndof_rail, ndof_rail)).tocsr()
        K = (K + Kadd).tocsr()
        info = {"variant": "V2", "ks_x": ks_x, "ks_y": ks_y,
                "kx_top": kx_top, "ky_top": ky_top, "kx_bot": kx_bot,
                "ky_bot": ky_bot, "s_lat": s_lat, "ndof": ndof_rail,
                "n_plate": 0, "support_layers": list(support_layers)}
        return K.tocsr(), Mc, mlump, plate_dofs, info

    # ---- V1 FULL: append one plate node (X,Y dofs) per support ----
    n_sup = len(support_layers)
    ndof_new = ndof_rail + 2 * n_sup
    for s, lay in enumerate(support_layers):
        px = ndof_rail + 2 * s + 0      # plate X dof
        py = ndof_rail + 2 * s + 1      # plate Y dof
        plate_dofs.append((px, py))
        fn = foot_nodes_at(Ncs, nodes2d, meshinfo, lay)
        nper = len(fn)
        for n in fn:
            spring(3 * n + 0, px, kx_top / nper)   # foot--k_top-->plate (lateral)
            spring(3 * n + 1, py, ky_top / nper)   # foot--k_top-->plate (vertical)
        spring(px, None, kx_bot)                   # plate--k_bot-->ground (lateral)
        spring(py, None, ky_bot)                   # plate--k_bot-->ground (vertical)

    Kadd = sp.coo_matrix((rV, (rI, rJ)), shape=(ndof_new, ndof_new)).tocsr()
    # embed rail K into the top-left block
    Kr = K.tocoo()
    Kbig = sp.coo_matrix((Kr.data, (Kr.row, Kr.col)), shape=(ndof_new, ndof_new))
    K = (Kbig.tocsr() + Kadd).tocsr()

    # grow mass operators: plate mass on X,Y (diagonal, lumped and consistent)
    Mr = Mc.tocoo()
    mI = list(Mr.row); mJ = list(Mr.col); mV = list(Mr.data)
    mlump_new = np.concatenate([mlump, np.zeros(2 * n_sup)])
    for s in range(n_sup):
        px = ndof_rail + 2 * s + 0
        py = ndof_rail + 2 * s + 1
        mI += [px, py]; mJ += [px, py]; mV += [cfg.m_plate, cfg.m_plate]
        mlump_new[px] += cfg.m_plate
        mlump_new[py] += cfg.m_plate
    Mc_new = sp.coo_matrix((mV, (mI, mJ)), shape=(ndof_new, ndof_new)).tocsr()

    info = {"variant": "V1", "kx_top": kx_top, "ky_top": ky_top,
            "kx_bot": kx_bot, "ky_bot": ky_bot, "m_plate": cfg.m_plate,
            "s_lat": s_lat, "ndof": ndof_new, "n_plate": n_sup,
            "support_layers": list(support_layers),
            "f_plate_lat": np.sqrt((kx_top + kx_bot) / cfg.m_plate) / (2 * np.pi),
            "f_plate_vert": np.sqrt((ky_top + ky_bot) / cfg.m_plate) / (2 * np.pi)}
    return K, Mc_new, mlump_new, plate_dofs, info


# ===========================================================================
# 6.  CONVENIENCE: tag head / web / foot nodes of a cross-section (for probes)
# ===========================================================================
def section_probe_nodes(Ncs, meshinfo, layer):
    """
    Return dict of representative node ids (at the given axial layer) for the
    lateral-motion probes used to read the modal signatures:
      'head' : centre node near the head top,
      'web'  : centre node at web mid-height,
      'foot' : centre node at foot underside.
    Found by nearest-coordinate search on the actual node set.
    """
    xy = meshinfo["nodes2d"]

    def nearest(xt, yt):
        d = (xy[:, 0] - xt) ** 2 + (xy[:, 1] - yt) ** 2
        return int(np.argmin(d))

    return {"head": layer * Ncs + nearest(0.0, H_RAIL - 0.004),
            "web":  layer * Ncs + nearest(0.0, 0.086),
            "foot": layer * Ncs + nearest(0.0, 0.0)}


# ===========================================================================
# 7.  SELF-TEST  (run this file directly)
# ===========================================================================
if __name__ == "__main__":
    print("=" * 72)
    print("rail_cell.py self-test: UIC60 / 60E1 section + assembly sanity")
    print("=" * 72)
    nodes2d, quads, mi = build_cross_section()
    print(f"cross-section: {nodes2d.shape[0]} nodes, {quads.shape[0]} quads "
          f"(axis-aligned active-element mesh)")
    verify_section(nodes2d, quads)

    # assemble one short cell and check rigid-body zero energy
    ne_z = 4
    zs = np.linspace(0.0, Cfg.L, ne_z + 1)
    K, Mc, mlump, nodes3d, Ncs, ndof = assemble_brick(nodes2d, quads, zs)
    print(f"\nbrick cell: {nodes3d.shape[0]} nodes, {ndof} dofs (ne_z={ne_z})")
    print(f"total mass (lumped) = {mlump[0::3].sum():.3f} kg  "
          f"(expect rho*A*L = {Cfg.rho*76.70e-4*Cfg.L:.3f} kg)")
    # rigid translation in X: zero strain energy
    for d, name in [(0, "X"), (1, "Y"), (2, "Z")]:
        u = np.zeros(ndof); u[d::3] = 1.0
        e = u @ (K @ u)
        print(f"  rigid-translation {name}: u^T K u = {e:.3e}  (expect ~0)")
    # symmetry
    asym = abs((K - K.T)).max()
    print(f"  K asymmetry max|K-K^T| = {asym:.2e}")

