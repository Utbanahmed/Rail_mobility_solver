"""
transient_fullwave.py  --  TOOL B: explicit time-domain full-wave solver
============================================================================
A genuine numerical full-wave simulation.  The 3D brick rail (N_periods of the
same extruded UIC60 cell, support variant selectable) is marched in time with
explicit central differences:

        M u'' + C u' + K u = f(t)

M is the lumped (diagonal) mass, K the assembled brick+support stiffness,
C = alpha(x)*M the ALID absorbing-layer damping (diagonal).  A point force
time-history is applied and the response time-history is produced step by step;
every mobility below is the FFT of THAT simulated response -- NO modal
superposition, NO frequency-domain assembly, NO closed form.

Non-reflecting ends: ALID (Absorbing Layers with Increasing Damping) at BOTH
ends.  alpha(x) rises smoothly from 0 at the physical-domain interface to
alpha_max at the outer end, alpha(x)=alpha_max*(s)^p with s in [0,1], p=3.
(Why ALID and not a PML: a time-explicit displacement PML needs auxiliary split
fields / convolution memory variables per node and a specific absorbing tensor;
ALID needs only a diagonal C=alpha*M, stays trivially explicit and stable, and
-- sized for the slowest in-band group velocity -- absorbs the long bending and
near-cutoff torsion waves here without the PML bookkeeping.  See REPORT.md for
the reflection test that justifies the choice.)

Everything SI.  Run directly for the baseline case; import run_transient(...)
from the validation script.

COORDINATES: X lateral, Y vertical, Z axial.  Drive = lateral (X) at rail head.
"""

import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rail_cell as rc
from rail_cell import (Cfg, build_cross_section, assemble_brick, add_supports,
                       section_probe_nodes)

# ===========================================================================
# CONFIG
# ===========================================================================
VARIANT     = "V2"      # "V1" FULL (plate mass) or "V2" REDUCED (series spring)
S_LAT       = 1.0       # lateral support stiffness scale (kx = s_lat*ky)
L_PERIOD    = Cfg.L     # m  support spacing (validation overrides with 0.60)

N_PHYS      = 6         # physical periods (drive near centre)
N_ALID      = 4         # ALID periods at EACH end
NE_Z        = 4         # axial elements per period (>=8 per shortest wavelength)

# ALID absorbing layer
ALID_P      = 3.0       # profile exponent alpha ~ s^p
ALID_AMAX   = 6.0e4     # 1/s  peak damping rate at the outer end (tunable)
# NB: L_layer = N_ALID*L_PERIOD.  Size it for the SLOWEST in-band group velocity
# (torsion near cutoff is the worst case).  If a mode reflects, LENGTHEN the
# layer (raise N_ALID) before raising ALID_AMAX -- over-damping itself reflects.

# light interior damping (realistic rail+pad loss; keeps peaks finite-width and
# mops up any residual; mass-proportional C0 = ALPHA0*M)
ALPHA0      = 20.0      # 1/s

# forcing
PULSE       = "gauss"   # "gauss" broadband pulse, or "toneburst"
PULSE_TAU   = 1.5e-4    # s  Gaussian width (flat-ish spectrum to ~1.6 kHz)
TONE_F      = 557.0     # Hz tone-burst centre frequency (if PULSE=="toneburst")
TONE_NCYC   = 12        # tone-burst cycles

# time
T_SIM       = 0.15      # s  total simulated time (FFT df ~ 1/T ~ 6.7 Hz)
CFL_SAFE    = 0.55      # Courant safety factor on the explicit step

F_MAX_PLOT  = 1600.0    # Hz


# ===========================================================================
# Build the time-domain domain
# ===========================================================================
def build_domain(variant=VARIANT, s_lat=S_LAT, L_period=L_PERIOD,
                 n_phys=N_PHYS, n_alid=N_ALID, ne_z=NE_Z):
    nodes2d, quads, mi = build_cross_section()
    Ncs = nodes2d.shape[0]
    n_cells = n_phys + 2 * n_alid
    # axial stations: ne_z elements per period, uniform dz
    total_elem = n_cells * ne_z
    Ltot = n_cells * L_period
    zs = np.linspace(0.0, Ltot, total_elem + 1)
    dz = zs[1] - zs[0]

    K, Mc, mlump, nodes3d, Ncs, ndof = assemble_brick(nodes2d, quads, zs)

    # support layers: one support per period boundary (z = 0, L, 2L, ...)
    support_layers = [k * ne_z for k in range(n_cells + 1)]  # includes both ends
    # (both end cross-sections carry a support; this keeps the periodic pattern)
    K, M_full, mlump, plate_dofs, info = add_supports(
        K, mlump, Mc, nodes3d, Ncs, nodes2d, mi, support_layers,
        variant=variant, s_lat=s_lat)
    ndof = K.shape[0]

    # ALID damping profile on rail dofs (plate dofs get interior damping only)
    z = nodes3d[:, 2]
    L_layer = n_alid * L_period
    z_lo_iface = L_layer                 # start of physical region
    z_hi_iface = Ltot - L_layer          # end of physical region
    alpha_node = np.zeros(nodes3d.shape[0])
    # left layer: s = (z_lo_iface - z)/L_layer in [0,1]
    left = z < z_lo_iface
    alpha_node[left] = ALID_AMAX * ((z_lo_iface - z[left]) / L_layer) ** ALID_P
    right = z > z_hi_iface
    alpha_node[right] = ALID_AMAX * ((z[right] - z_hi_iface) / L_layer) ** ALID_P
    alpha_node += ALPHA0                 # uniform light interior damping
    Cdiag = np.zeros(ndof)
    rail_ndof = 3 * nodes3d.shape[0]
    Cdiag[:rail_ndof] = np.repeat(alpha_node, 3) * mlump[:rail_ndof]
    # plate dofs: light interior damping only
    if rail_ndof < ndof:
        Cdiag[rail_ndof:] = ALPHA0 * mlump[rail_ndof:]

    # drive + probe nodes (at the physical-domain centre cross-section)
    z_centre = 0.5 * Ltot
    layer_centre = int(round(z_centre / dz))
    # snap to a support layer for the "support" probe and a mid-bay layer
    layer_support = min(support_layers, key=lambda L: abs(L - layer_centre))
    layer_mid = layer_support + ne_z // 2           # midspan between supports
    probes = section_probe_nodes(Ncs, mi, layer_mid)
    probes_support = section_probe_nodes(Ncs, mi, layer_support)
    drive_node = probes["head"]                     # head, lateral, at midspan
    drive_dof = 3 * drive_node + 0                  # X

    # mid-height lateral line along z (for space-time / 2D-FFT)
    xy = mi["nodes2d"]
    cs_web = int(np.argmin((xy[:, 0]) ** 2 + (xy[:, 1] - 0.086) ** 2))
    line_nodes = np.array([l * Ncs + cs_web for l in range(len(zs))])
    line_dofs = 3 * line_nodes + 0                  # lateral X along the line
    line_z = zs.copy()

    return dict(K=K.tocsr(), mlump=mlump, Cdiag=Cdiag, ndof=ndof, dz=dz,
                nodes3d=nodes3d, Ncs=Ncs, zs=zs, L_period=L_period,
                drive_dof=drive_dof, drive_node=drive_node,
                probes=probes, probes_support=probes_support,
                line_dofs=line_dofs, line_z=line_z, info=info,
                n_phys=n_phys, n_alid=n_alid, ne_z=ne_z,
                z_phys=(z_lo_iface, z_hi_iface), plate_dofs=plate_dofs)


# ===========================================================================
# CFL-limited time step
# ===========================================================================
def critical_dt(dom, cfl_safe=CFL_SAFE):
    """
    Stability of explicit central difference with lumped mass: dt <= 2/omega_max.
    omega_max^2 is bounded by max_i (K_ii/M_ii) for the diagonal; to be safe we
    take a Courant estimate dt = cfl_safe * h_min / c_p and cross-check with the
    diagonal bound, using the smaller.
    """
    E, nu, rho = Cfg.E, Cfg.nu, Cfg.rho
    c_p = np.sqrt(E * (1 - nu) / (rho * (1 + nu) * (1 - 2 * nu)))   # dilatational
    # smallest element dimension in the mesh
    xb = build_cross_section()[2]["xb"]
    yg = build_cross_section()[2]["yg"]
    hmin = min(np.diff(xb).min(), np.diff(yg).min(), dom["dz"])
    dt_courant = cfl_safe * hmin / c_p
    # diagonal Rayleigh-quotient bound
    Kdiag = dom["K"].diagonal()
    M = dom["mlump"].copy()
    M[M <= 0] = M[M > 0].min()
    wmax2 = np.max(Kdiag / M)
    dt_diag = cfl_safe * 2.0 / np.sqrt(wmax2)
    return min(dt_courant, dt_diag), c_p, hmin


# ===========================================================================
# Forcing time history
# ===========================================================================
def force_history(t, pulse=PULSE, tau=PULSE_TAU, tone_f=TONE_F, tone_ncyc=TONE_NCYC):
    if pulse == "gauss":
        # broadband smooth pulse, flat-ish spectrum to ~1.6 kHz (has DC content)
        t0 = 5 * tau
        return np.exp(-0.5 * ((t - t0) / tau) ** 2)
    elif pulse == "ricker":
        # Gaussian doublet (1st derivative of a Gaussian): ZERO DC, bandpass,
        # spectrum peaks at f_peak = 1/(2*pi*tau).  Used for the space-time /
        # 2D-FFT dispersion so in-band propagating waves dominate (not the very
        # low-frequency rail-on-support standing mode that a flat pulse excites).
        tp = 1.0 / (2 * np.pi * tone_f)          # tune peak to tone_f
        t0 = 4 * tp
        s = (t - t0) / tp
        return -s * np.exp(-0.5 * s ** 2)
    elif pulse == "toneburst":
        Tb = tone_ncyc / tone_f
        t0 = 0.6 * Tb
        env = np.where(np.abs(t - t0) <= 0.5 * Tb,
                       0.5 * (1 + np.cos(2 * np.pi * (t - t0) / Tb)), 0.0)
        return env * np.sin(2 * np.pi * tone_f * (t - t0))
    raise ValueError(pulse)


# ===========================================================================
# Explicit central-difference time marching
# ===========================================================================
def run_transient(variant=VARIANT, s_lat=S_LAT, L_period=L_PERIOD,
                  n_phys=N_PHYS, n_alid=N_ALID, ne_z=NE_Z, T_sim=T_SIM,
                  pulse=PULSE, tau=PULSE_TAU, tone_f=TONE_F,
                  cfl_safe=CFL_SAFE, line_every=20, verbose=True):
    dom = build_domain(variant, s_lat, L_period, n_phys, n_alid, ne_z)
    K = dom["K"]
    M = dom["mlump"].copy()
    C = dom["Cdiag"].copy()
    ndof = dom["ndof"]
    dt, c_p, hmin = critical_dt(dom, cfl_safe)
    nsteps = int(np.ceil(T_sim / dt))
    t = np.arange(nsteps + 1) * dt

    # central-difference constants (all diagonal)
    inv_lhs = 1.0 / (M / dt**2 + C / (2 * dt))
    c_u = 2 * M / dt**2
    c_um = M / dt**2 - C / (2 * dt)

    f_drive = force_history(t, pulse, tau, tone_f)
    drive_dof = dom["drive_dof"]

    # state
    u_m = np.zeros(ndof)     # u_{n-1}
    u = np.zeros(ndof)       # u_n
    # recorders
    u_drive = np.zeros(nsteps + 1)
    probe_dofs = {k: 3 * v + 0 for k, v in dom["probes"].items()}         # lateral
    probe_sup = {k: 3 * v + 0 for k, v in dom["probes_support"].items()}
    rec = {k: np.zeros(nsteps + 1) for k in probe_dofs}
    rec_sup = {k: np.zeros(nsteps + 1) for k in probe_sup}
    line_dofs = dom["line_dofs"]
    nline_samp = nsteps // line_every + 1
    line_hist = np.zeros((nline_samp, len(line_dofs)))
    line_t = np.zeros(nline_samp)
    il = 0

    fvec = np.zeros(ndof)
    for n in range(nsteps):
        fvec[drive_dof] = f_drive[n]
        rhs = fvec - K.dot(u) + c_u * u - c_um * u_m
        u_p = inv_lhs * rhs
        # record
        u_drive[n + 1] = u_p[drive_dof]
        for k, d in probe_dofs.items():
            rec[k][n + 1] = u_p[d]
        for k, d in probe_sup.items():
            rec_sup[k][n + 1] = u_p[d]
        if n % line_every == 0:
            line_hist[il] = u_p[line_dofs]
            line_t[il] = t[n + 1]
            il += 1
        u_m = u
        u = u_p
    # trim line history
    line_hist = line_hist[:il]; line_t = line_t[:il]

    if not np.all(np.isfinite(u)):
        raise RuntimeError("explicit integration diverged -- reduce cfl_safe")

    # --- mobilities: Y(f) = V(f)/F(f), velocity = d/dt of displacement ---
    def mobility(u_hist):
        v = np.gradient(u_hist, dt)
        win = np.hanning(len(v))
        V = np.fft.rfft(v * win)
        F = np.fft.rfft(f_drive * win)
        freq = np.fft.rfftfreq(len(v), dt)
        Y = V / F
        return freq, Y

    freq, Y_drive = mobility(u_drive)
    Y_probes = {k: mobility(rec[k])[1] for k in rec}
    Y_sup = {k: mobility(rec_sup[k])[1] for k in rec_sup}

    result = dict(dom=dom, dt=dt, t=t, f_drive=f_drive, nsteps=nsteps,
                  freq=freq, Y_drive=Y_drive, Y_probes=Y_probes, Y_sup=Y_sup,
                  u_drive=u_drive, line_hist=line_hist, line_t=line_t,
                  line_z=dom["line_z"], variant=variant, s_lat=s_lat,
                  L_period=L_period, c_p=c_p, hmin=hmin, info=dom["info"])

    # peak finder in 0-1600 Hz
    result["peaks"] = find_peaks(freq, np.abs(Y_drive), fmax=F_MAX_PLOT)

    if verbose:
        print(f"\n[Tool B] transient  variant={variant} s_lat={s_lat} "
              f"L={L_period} n_phys={n_phys} n_alid={n_alid}")
        print(f"   ndof={ndof}  dt={dt:.2e}s (c_p={c_p:.0f} m/s, hmin={hmin*1e3:.1f} mm)"
              f"  nsteps={nsteps}  df={1/T_sim:.1f} Hz")
        if "f_plate_lat" in dom["info"]:
            print(f"   plate lateral resonance = {dom['info']['f_plate_lat']:.1f} Hz")
        pk = result["peaks"]
        print(f"   drive-point lateral mobility peaks (Hz): "
              f"{', '.join(f'{p:.1f}' for p in pk)}")
    return result


def find_peaks(freq, mag, fmax=1600.0, fmin=150.0, rel=0.08):
    """Simple local-maximum peak finder on |Y| within [fmin,fmax]."""
    band = (freq >= fmin) & (freq <= fmax)
    fb, mb = freq[band], mag[band]
    peaks = []
    thr = rel * mb.max()
    for i in range(1, len(mb) - 1):
        if mb[i] > mb[i - 1] and mb[i] >= mb[i + 1] and mb[i] > thr:
            peaks.append(fb[i])
    return peaks


# ===========================================================================
# Plots
# ===========================================================================
def plot_mobility(res, fname, title=None):
    freq, Y = res["freq"], res["Y_drive"]
    band = freq <= F_MAX_PLOT
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.semilogy(freq[band], np.abs(Y[band]), "k", lw=1.3, label="drive (head, lateral)")
    for p in res["peaks"]:
        ax.axvline(p, color="r", ls=":", lw=0.8)
        ax.text(p, ax.get_ylim()[1], f"{p:.0f}", color="r", fontsize=8,
                rotation=90, va="top")
    if "f_plate_lat" in res["info"]:
        ax.axvline(res["info"]["f_plate_lat"], color="b", ls="--", lw=0.8,
                   label=f"plate res {res['info']['f_plate_lat']:.0f} Hz")
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("|Y| point mobility (m/s/N)")
    ax.set_title(title or f"Point mobility  {res['variant']} s_lat={res['s_lat']} L={res['L_period']}")
    ax.set_xlim(0, F_MAX_PLOT); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fname, dpi=130); plt.close(fig)
    print(f"   saved {fname}")


def plot_signatures(res, fname):
    """Transfer mobilities head/web/foot + midspan-vs-support."""
    freq = res["freq"]; band = freq <= F_MAX_PLOT
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for k, c in [("head", "C0"), ("web", "C1"), ("foot", "C2")]:
        ax1.semilogy(freq[band], np.abs(res["Y_probes"][k][band]), c, lw=1.1, label=k)
    for p in res["peaks"]:
        ax1.axvline(p, color="r", ls=":", lw=0.6)
    ax1.set_title("transfer mobility at head / web / foot (lateral)")
    ax1.set_xlabel("Hz"); ax1.set_ylabel("|Y| (m/s/N)"); ax1.legend(fontsize=8)
    ax1.set_xlim(0, F_MAX_PLOT); ax1.grid(alpha=0.3, which="both")

    ax2.semilogy(freq[band], np.abs(res["Y_probes"]["head"][band]), "C0",
                 lw=1.2, label="midspan (head)")
    ax2.semilogy(freq[band], np.abs(res["Y_sup"]["head"][band]), "C3",
                 lw=1.2, label="at support (head)")
    for p in res["peaks"]:
        ax2.axvline(p, color="r", ls=":", lw=0.6)
    ax2.set_title("midspan vs support (upper peak should collapse at support)")
    ax2.set_xlabel("Hz"); ax2.set_ylabel("|Y| (m/s/N)"); ax2.legend(fontsize=8)
    ax2.set_xlim(0, F_MAX_PLOT); ax2.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(fname, dpi=130); plt.close(fig)
    print(f"   saved {fname}")


def plot_spacetime(res, fname):
    """(x,t) field of the mid-height lateral line + its 2D-FFT dispersion."""
    L = res["line_hist"]; z = res["line_z"]; tt = res["line_t"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    # normalise for display
    A = L / (np.abs(L).max() + 1e-30)
    im = ax1.pcolormesh(z, tt * 1e3, A, cmap="RdBu", vmin=-0.3, vmax=0.3, shading="auto")
    zlo, zhi = res["dom"]["z_phys"]
    for zz in (zlo, zhi):
        ax1.axvline(zz, color="k", ls="--", lw=0.8)
    ax1.set_xlabel("axial position z (m)"); ax1.set_ylabel("time (ms)")
    ax1.set_title("space-time lateral field (dashed = ALID interfaces)")
    fig.colorbar(im, ax=ax1, label="u_x (norm.)")

    # 2D FFT -> dispersion (use physical region only)
    mask = (z >= zlo) & (z <= zhi)
    Lp = L[:, mask]; zp = z[mask]
    dzp = zp[1] - zp[0]; dtp = tt[1] - tt[0]
    win_t = np.hanning(Lp.shape[0])[:, None]
    win_x = np.hanning(Lp.shape[1])[None, :]
    F2 = np.fft.fftshift(np.abs(np.fft.fft2(Lp * win_t * win_x)))
    kk = np.fft.fftshift(np.fft.fftfreq(Lp.shape[1], dzp)) * 2 * np.pi   # rad/m
    ff = np.fft.fftshift(np.fft.fftfreq(Lp.shape[0], dtp))               # Hz
    fmask = (ff >= 100) & (ff <= F_MAX_PLOT)     # skip DC / very low f in display
    kmask = kk >= 0
    Fdisp = F2[np.ix_(fmask, kmask)]
    vmax = np.percentile(Fdisp, 99.5)            # robust scaling -> branches visible
    ax2.pcolormesh(kk[kmask], ff[fmask], Fdisp, cmap="magma",
                   shading="auto", vmax=vmax)
    ax2.axvline(np.pi / res["L_period"], color="c", ls="--", lw=0.8,
                label="zone edge")
    ax2.set_xlabel("k (rad/m)"); ax2.set_ylabel("frequency (Hz)")
    ax2.set_title("2D-FFT dispersion (measured from the simulation)")
    ax2.set_xlim(0, 1.2 * np.pi / res["L_period"]); ax2.set_ylim(0, F_MAX_PLOT)
    ax2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fname, dpi=130); plt.close(fig)
    print(f"   saved {fname}")
    return kk, ff, F2


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    print("=" * 72)
    print("TOOL B  --  transient full-wave solver (explicit CD + ALID) ")
    print("=" * 72)
    n2, q, mi = build_cross_section()
    rc.verify_section(n2, q)

    # broadband run -> point/transfer mobilities (F-normalised, full 0-1.6 kHz)
    res = run_transient(variant=VARIANT, s_lat=S_LAT, pulse="gauss")
    plot_mobility(res, f"figures/mobility_{VARIANT}.png")
    plot_signatures(res, f"figures/signatures_{VARIANT}.png")
    # in-band (bandpass) run on a LONGER physical domain -> clean propagating
    # waves and finer k-resolution for the space-time + 2D-FFT dispersion
    resb = run_transient(variant=VARIANT, s_lat=S_LAT, pulse="ricker",
                         tone_f=500.0, n_phys=10, T_sim=0.08)
    plot_spacetime(resb, f"figures/spacetime_{VARIANT}.png")
    print("\nDone. Figures in figures/.")
