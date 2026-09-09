"""
bloch_dispersion.py  --  TOOL A: Floquet-Bloch dispersion solver (3D bricks)
============================================================================
Assembles K, M for ONE unit cell (length L) of the extruded UIC60 brick mesh
plus one discrete support (variant V1 FULL or V2 REDUCED), applies the
Floquet-Bloch reduction u_Rf = exp(i*kappa) u_Lf, and solves the reduced
Hermitian generalized eigenproblem  K_r phi = omega^2 M_r phi  for each
kappa in [0, pi].  Branches are tracked across kappa by modal-assurance
continuity and classified by their cross-section motion (torsional/roll vs
lateral translation/bending) so the torsion<->bending avoided crossing that
(hypothetically) produces the two lateral mobility peaks can be located.

Everything SI.  Run directly to produce the band diagram; import
`run_bloch(...)` from the validation script.

COORDINATES:  X lateral, Y vertical, Z axial (see rail_cell.py).
"""

import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rail_cell as rc
from rail_cell import Cfg, build_cross_section, assemble_brick, add_supports

# ===========================================================================
# CONFIG
# ===========================================================================
VARIANT   = "V2"      # "V1" FULL (plate mass retained) or "V2" REDUCED (series spring)
S_LAT     = 1.0       # lateral stiffness scale (kx = s_lat * ky); swept in __main__
NE_Z      = 4         # axial elements per unit cell (in-band waves are long: ok)
N_KAPPA   = 31        # number of kappa samples in [0, pi]
F_MAX     = 1600.0    # Hz  band-diagram ceiling
N_BRANCH  = 24        # lowest branches to retain per kappa


# ===========================================================================
# Build the cell operators and the Bloch face partition
# ===========================================================================
def build_cell(variant=VARIANT, s_lat=S_LAT, ne_z=NE_Z):
    nodes2d, quads, mi = build_cross_section()
    zs = np.linspace(0.0, Cfg.L, ne_z + 1)
    K, Mc, mlump, nodes3d, Ncs, ndof = assemble_brick(nodes2d, quads, zs)
    # one support per cell, placed at the left-face cross-section (layer 0)
    K, M, mlump, plate_dofs, info = add_supports(
        K, mlump, Mc, nodes3d, Ncs, nodes2d, mi, support_layers=[0],
        variant=variant, s_lat=s_lat)
    ndof_full = K.shape[0]

    # DOF partition: left face (layer 0), right face (layer ne_z), interior rest
    cs = np.arange(Ncs)
    lf_nodes = 0 * Ncs + cs
    rf_nodes = ne_z * Ncs + cs
    lf_dofs = np.concatenate([[3 * n, 3 * n + 1, 3 * n + 2] for n in lf_nodes])
    rf_dofs = np.concatenate([[3 * n, 3 * n + 1, 3 * n + 2] for n in rf_nodes])
    all_dofs = np.arange(ndof_full)
    face = np.zeros(ndof_full, bool)
    face[lf_dofs] = True
    face[rf_dofs] = True
    int_dofs = all_dofs[~face]    # interior incl. plate dofs (appended > rail ndof)
    return dict(K=K.tocsr(), M=M.tocsr(), nodes3d=nodes3d, Ncs=Ncs,
                nodes2d=nodes2d, mi=mi, lf_dofs=lf_dofs, rf_dofs=rf_dofs,
                int_dofs=int_dofs, info=info, ndof=ndof_full)


# ===========================================================================
# Floquet-Bloch reduction operator T(kappa)  (ndof_full x ndof_ind)
#   independent dofs = [interior ; left face];  right face = exp(i kappa)*left
# ===========================================================================
def bloch_T(cell, kappa):
    idof, ldof, rdof = cell["int_dofs"], cell["lf_dofs"], cell["rf_dofs"]
    nfull = cell["ndof"]
    ni, nl = len(idof), len(ldof)
    nind = ni + nl
    phase = np.exp(1j * kappa)
    rows = np.concatenate([idof, ldof, rdof])
    cols = np.concatenate([np.arange(ni), ni + np.arange(nl), ni + np.arange(nl)])
    vals = np.concatenate([np.ones(ni), np.ones(nl), phase * np.ones(nl)])
    T = sp.coo_matrix((vals, (rows, cols)), shape=(nfull, nind)).tocsc()
    return T, ni, nl


# ===========================================================================
# Cross-section classification patterns (roll/torsion vs lateral translation)
# ===========================================================================
def classification_patterns(cell):
    """Build roll and lateral-translation reference vectors on the LEFT face."""
    nodes2d = cell["nodes2d"]; Ncs = cell["Ncs"]
    xy = nodes2d
    yc = 0.0797  # section centroid (m); value from verify_section
    r = np.zeros(3 * Ncs)    # roll / torsion about axial axis
    t = np.zeros(3 * Ncs)    # lateral translation (X)
    for n in range(Ncs):
        x, y = xy[n]
        r[3 * n + 0] = -(y - yc)   # ux
        r[3 * n + 1] = (x - 0.0)   # uy
        t[3 * n + 0] = 1.0         # ux
    r /= np.linalg.norm(r)
    t /= np.linalg.norm(t)
    return r, t


def torsion_fraction(u_lf_cs, r, t):
    """Fraction of (roll vs roll+lateral-translation) content of a face motion."""
    pr = np.abs(np.vdot(r, u_lf_cs))**2
    pt = np.abs(np.vdot(t, u_lf_cs))**2
    denom = pr + pt
    return pr / denom if denom > 0 else 0.0


# ===========================================================================
# Solve the band diagram
# ===========================================================================
def run_bloch(variant=VARIANT, s_lat=S_LAT, ne_z=NE_Z, n_kappa=N_KAPPA,
              f_max=F_MAX, n_branch=N_BRANCH, verbose=True):
    cell = build_cell(variant, s_lat, ne_z)
    r, t = classification_patterns(cell)
    kappas = np.linspace(1e-3, np.pi, n_kappa)
    lf_dofs = cell["lf_dofs"]

    nk = len(kappas)
    F = np.full((n_branch, nk), np.nan)     # frequencies Hz
    TF = np.full((n_branch, nk), np.nan)    # torsion fraction
    VECS = [None] * nk                      # independent-space eigvecs (for MAC)
    LFCS = [None] * nk                      # left-face cs motion per mode

    sigma = (2 * np.pi * 5.0) ** 2          # shift just above the rigid modes
    for ik, kap in enumerate(kappas):
        T, ni, nl = bloch_T(cell, kap)
        Kr = (T.conj().T @ cell["K"] @ T).tocsc()
        Mr = (T.conj().T @ cell["M"] @ T).tocsc()
        Kr = (0.5 * (Kr + Kr.conj().T)).tocsc()   # enforce Hermitian
        Mr = (0.5 * (Mr + Mr.conj().T)).tocsc()
        # lowest n_branch modes via shift-invert (sparse, complex Hermitian)
        w2, V = spla.eigsh(Kr, k=n_branch, M=Mr, sigma=sigma, which="LM")
        w2 = np.real(w2)
        order = np.argsort(w2)
        w2 = w2[order]; V = V[:, order]
        w2[w2 < 0] = 0.0
        f = np.sqrt(w2) / (2 * np.pi)
        keep = np.arange(len(f))            # keep all returned (already lowest)
        fk = f[keep]; Vk = V[:, keep]
        # reconstruct left-face cs motion for classification
        lfcs = np.zeros((len(keep), 3 * cell["Ncs"]), complex)
        tf = np.zeros(len(keep))
        # independent ordering: [interior(ni) ; left face(nl)]; left-face block:
        lf_block = Vk[ni:ni + nl, :]        # (nl, nmodes) = face dof amplitudes
        for m in range(len(keep)):
            u = lf_block[:, m]
            lfcs[m] = u
            tf[m] = torsion_fraction(u, r, t)
        F[:len(keep), ik] = fk
        TF[:len(keep), ik] = tf
        VECS[ik] = Vk
        LFCS[ik] = lfcs

    # ---- branch tracking by MAC continuity on the left-face cs motion ----
    # (left-face motion is a robust, phase-consistent signature of each mode)
    Ftr = np.full((n_branch, nk), np.nan)
    TFtr = np.full((n_branch, nk), np.nan)
    order_prev = np.arange(n_branch)
    Ftr[:, 0] = F[:, 0]; TFtr[:, 0] = TF[:, 0]
    ref = LFCS[0]
    for ik in range(1, nk):
        cur = LFCS[ik]
        nprev = ref.shape[0]; ncur = cur.shape[0]
        # MAC matrix between previous branch reps and current modes
        mac = np.zeros((nprev, ncur))
        for a in range(nprev):
            va = ref[a]
            na = np.vdot(va, va).real
            if na == 0:
                continue
            for b in range(ncur):
                vb = cur[b]
                nb = np.vdot(vb, vb).real
                if nb == 0:
                    continue
                mac[a, b] = np.abs(np.vdot(va, vb))**2 / (na * nb)
        # greedy assignment
        assigned = -np.ones(nprev, int)
        used = set()
        flat = np.dstack(np.unravel_index(np.argsort(-mac, axis=None), mac.shape))[0]
        for a, b in flat:
            if assigned[a] < 0 and b not in used:
                assigned[a] = b; used.add(b)
            if (assigned >= 0).all():
                break
        newref = [None] * nprev
        for a in range(nprev):
            b = assigned[a]
            if b >= 0:
                Ftr[a, ik] = F[b, ik]
                TFtr[a, ik] = TF[b, ik]
                newref[a] = cur[b]
            else:
                newref[a] = ref[a]
        ref = np.array(newref)

    result = dict(kappas=kappas, F=Ftr, TF=TFtr, F_raw=F, TF_raw=TF,
                  cell=cell, variant=variant, s_lat=s_lat, L=Cfg.L)

    # ---- zone-edge (kappa=pi) frequencies in the band of interest ----
    fe = Ftr[:, -1]
    tfe = TFtr[:, -1]
    sel = np.where((fe >= 400) & (fe <= 700))[0]
    result["zone_edge"] = [(float(fe[i]), float(tfe[i])) for i in np.argsort(fe)
                           if 400 <= fe[i] <= 700]

    # ---- detect avoided crossing: two branches whose frequency gap is a local
    #      minimum while their torsion fractions swap across 0.5 ----
    avoided = detect_avoided_crossing(kappas, Ftr, TFtr, fband=(400, 700))
    result["avoided"] = avoided

    if verbose:
        print(f"\n[Tool A] Bloch dispersion  variant={variant}  s_lat={s_lat}  "
              f"ne_z={ne_z}  ndof_ind~{cell['K'].shape[0]}")
        if "f_plate_lat" in cell["info"]:
            print(f"   plate lateral resonance sqrt((kx_top+kx_bot)/m)/2pi = "
                  f"{cell['info']['f_plate_lat']:.1f} Hz")
        print(f"   zone-edge (kappa=pi) branches in 400-700 Hz:")
        for f_, tf_ in result["zone_edge"]:
            kind = "torsion" if tf_ > 0.5 else "bending"
            print(f"      {f_:7.1f} Hz   torsion-fraction={tf_:.2f}  ({kind}-dominant)")
        if avoided:
            print(f"   AVOIDED CROSSING detected near kappa={avoided['kappa']:.2f} "
                  f"rad (k={avoided['kappa']/Cfg.L:.2f} rad/m):")
            print(f"      hybrid frequencies: {avoided['f_lo']:.1f} and "
                  f"{avoided['f_hi']:.1f} Hz  (gap {avoided['gap']:.1f} Hz)")
        else:
            print("   no clear avoided crossing detected in 400-700 Hz")
    return result


def detect_avoided_crossing(kappas, F, TF, fband=(400, 700)):
    """
    Find a pair of tracked branches whose frequency gap has an interior local
    minimum while their torsion fractions swap (character exchange) -> veering.
    Returns dict or None.  Reports the two branch frequencies at the gap minimum
    and at the zone edge.
    """
    nb, nk = F.shape
    best = None
    for a in range(nb):
        for b in range(a + 1, nb):
            fa, fb = F[a], F[b]
            ta, tb = TF[a], TF[b]
            gap = np.abs(fa - fb)
            valid = ~np.isnan(gap)
            if valid.sum() < 5:
                continue
            # restrict to band
            inband = valid & (fa >= fband[0]) & (fa <= fband[1]) & \
                     (fb >= fband[0]) & (fb <= fband[1])
            if inband.sum() < 4:
                continue
            idx = np.where(inband)[0]
            gseg = gap[idx]
            jmin = idx[np.argmin(gseg)]
            # interior local minimum (not at ends of the in-band segment)?
            if jmin == idx[0] or jmin == idx[-1]:
                continue
            # character swap: torsion fractions of the two branches cross
            swap = (ta[idx[0]] - tb[idx[0]]) * (ta[idx[-1]] - tb[idx[-1]]) < 0
            gmin = gap[jmin]
            # require a genuine "avoided" gap: small but nonzero, and a local min
            if gmin > 60:
                continue
            score = gmin - 100 * (1 if swap else 0)
            if best is None or score < best["score"]:
                best = dict(score=score, a=a, b=b, kappa=kappas[jmin],
                            gap=float(gmin),
                            f_lo=float(min(fa[jmin], fb[jmin])),
                            f_hi=float(max(fa[jmin], fb[jmin])),
                            f_edge=(float(F[a, -1]), float(F[b, -1])),
                            swap=bool(swap))
    return best


# ===========================================================================
# Plot
# ===========================================================================
def plot_bands(result, fname, title=None):
    kap = result["kappas"]; F = result["F"]; TF = result["TF"]
    L = result["L"]
    k_phys = kap / L
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for a in range(F.shape[0]):
        f = F[a]; tf = TF[a]
        m = ~np.isnan(f)
        sc = ax.scatter(k_phys[m], f[m], c=tf[m], cmap="coolwarm",
                        vmin=0, vmax=1, s=16, edgecolors="none")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("torsion fraction  (0 = lateral bending, 1 = torsion/roll)")
    # zone edge & pinned-pinned line
    k_edge = np.pi / L
    ax.axvline(k_edge, color="0.4", ls="--", lw=1)
    ax.text(k_edge, F_MAX * 0.02, r"  zone edge $k=\pi/L$ (pinned-pinned)",
            rotation=90, va="bottom", ha="right", fontsize=8, color="0.3")
    av = result.get("avoided")
    if av:
        ka = av["kappa"] / L
        ax.scatter([ka, ka], [av["f_lo"], av["f_hi"]], s=140, facecolors="none",
                   edgecolors="k", linewidths=1.8, zorder=5)
        ax.annotate(f"avoided crossing\n{av['f_lo']:.0f}/{av['f_hi']:.0f} Hz",
                    (ka, av["f_hi"]), textcoords="offset points", xytext=(8, 8),
                    fontsize=8)
    ax.set_xlim(0, k_edge * 1.02)
    ax.set_ylim(0, F_MAX)
    ax.set_xlabel("axial wavenumber  k  (rad/m)")
    ax.set_ylabel("frequency  (Hz)")
    ax.set_title(title or f"Bloch band diagram  ({result['variant']}, "
                 f"s_lat={result['s_lat']})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    plt.close(fig)
    print(f"   saved {fname}")


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    print("=" * 72)
    print("TOOL A  --  Floquet-Bloch dispersion of the Delkor-supported UIC60 rail")
    print("=" * 72)
    # verify the section once
    n2, q, mi = build_cross_section()
    rc.verify_section(n2, q)

    # --- baseline band diagrams for both support variants ---
    for variant in ["V2", "V1"]:
        res = run_bloch(variant=variant, s_lat=S_LAT)
        plot_bands(res, f"figures/bloch_bands_{variant}.png",
                   title=f"Bloch bands  {variant}  (s_lat={S_LAT})")

    # --- LATERAL STIFFNESS SWEEP (0.5x .. 2x vertical) on V2 ---
    print("\n" + "-" * 72)
    print("Lateral stiffness sweep (V2): zone-edge 400-700 Hz branches vs s_lat")
    print("-" * 72)
    sweep = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    plot_these = {0.5, 1.0, 2.0}
    for s in sweep:
        res = run_bloch(variant="V2", s_lat=s, n_kappa=21, verbose=False)
        ze = res["zone_edge"]
        av = res["avoided"]
        fs = ", ".join(f"{f:.0f}({'T' if tf>0.5 else 'B'})" for f, tf in ze)
        avtxt = (f"avoided {av['f_lo']:.0f}/{av['f_hi']:.0f}" if av else "none")
        print(f"  s_lat={s:4.2f}:  zone-edge 400-700Hz[{fs}]   {avtxt}")
        if s in plot_these:
            plot_bands(res, f"figures/bloch_bands_V2_slat{s:.2f}.png",
                       title=f"Bloch bands V2  s_lat={s}")
    print("\nInterpretation: the ~420 Hz pair is lateral BENDING (pinned-pinned,")
    print("node at support -> nearly s_lat-independent); the ~710-750 Hz pair is")
    print("the TORSION/flapping branch (head & foot antiphase -> s_lat-sensitive).")
    print("Done. Figures in figures/.")
