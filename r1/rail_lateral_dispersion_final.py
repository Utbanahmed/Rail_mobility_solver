"""
Lateral dispersion of a UIC 60 rail - Wu & Thompson (1999) multiple-beam model
=============================================================================
Reference
---------
T. X. Wu & D. J. Thompson, "Analysis of lateral vibration behavior of railway
track at high frequencies using a continuously supported multiple beam model",
J. Acoust. Soc. Am. 106(3), 1369-1376 (1999).

What this reproduces
--------------------
  Fig. 4   free rail, dispersion of waves I-IV
  Fig. 5   continuously supported rail, undamped
  Fig. 11  decay rate of each wave in the damped supported rail

Model
-----
Head and foot are infinite Timoshenko beams carrying lateral bending and
torsion about the rail axis; the web is an array of finite beams bending
laterally over the web height.

DOF (paper Eq. 5):   q = [v_h, psi_h, th_h, v_f, psi_f, th_f]^T

Free rail (paper Eq. 4):        M q_tt - D q_zz - G q_z + K_R q = 0
Supported rail (paper Eq. 8):   same with K = K_R + K_F(omega)

With q = Q exp(i w t) exp(-i k z) this becomes (paper Eq. 14)

    [ -k^2 D - i k G + (w^2 M - K) ] Q = 0

solved in state-space form (paper Eqs. 16-19), lambda = -i k.

Two interpretation points, both settled numerically below
---------------------------------------------------------
The paper describes two corrections to the raw section properties (adding the
web torsion constant to head and foot, 1/3 and 2/3; and raising I_w by 10%),
then prints a parameter list. The printed J_h, J_f and I_w are equal to the RAW
values, so the corrections must be applied ON TOP of the printed list. This is
confirmed by the dispersion curves: without the torsion redistribution wave I
reaches 44.5 rad/m at 5 kHz, which is off the top of the axis in Fig. 4, while
with it wave I reaches 35.0 rad/m, matching the figure.

Sign conventions
----------------
Vertical coordinate Z upward. A point a height delta above a centroid moves
laterally by u = v + delta*th, so du/dZ = th. The web attaches h_h/2 below the
head centroid and h_f/2 above the foot centroid:

    u_top = v_h - (h_h/2) th_h        phi_top = th_h
    u_bot = v_f + (h_f/2) th_f        phi_bot = th_f

reproducing the web DOF vector of paper Eq. (3). The foundation acts at the
underside of the foot, u = v_f - (h_f/2) th_f, reproducing paper Eq. (10).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eigh

TEAL, RED, GREY = "#0f7c85", "#c0392b", "#5a6068"
OUT = "/tmp/claude-0/-home-user-surge-tank/490fb76a-a7c4-55e2-930b-dad676214c31/scratchpad/outputs/"

# ============================================================================
# 1. Material and geometry - simplified UIC 60 section, paper Fig. 3
# ============================================================================
E, G, rho = 2.1e11, 0.77e11, 7850.0

b_h, h_h = 0.073, 0.039      # head  width x height  [m]
t_w, h_w = 0.019, 0.114      # web   thickness x height
b_f, h_f = 0.150, 0.0175     # foot  width x height
kap_h = kap_f = 0.85         # shear coefficient of the rectangle

# Foundation, paper Sec. II C ("track C" of Ref. 11)
Kp, Kpr, Kb, Ms = 83.3e6, 1.09e6, 133.3e6, 270.0
eta_p, eta_b, eta_r = 0.25, 0.6, 0.0


def rect_torsion_J(a, b):
    """St Venant torsion constant of a solid rectangle."""
    a, b = max(a, b), min(a, b)
    return a * b**3 * (1/3 - 0.21*(b/a)*(1 - b**4/(12*a**4)))


A_h, A_f, A_w = b_h*h_h, b_f*h_f, h_w*t_w
I_h = h_h * b_h**3 / 12                       # lateral bending, common axis
I_f = h_f * b_f**3 / 12
I_hp = A_h * (b_h**2 + h_h**2) / 12           # polar, about the rail axis
I_fp = A_f * (b_f**2 + h_f**2) / 12
J_h_raw, J_f_raw = rect_torsion_J(b_h, h_h), rect_torsion_J(b_f, h_f)
J_w = rect_torsion_J(h_w, t_w)
I_w_raw = t_w**3 / 12                         # per unit length of rail

APPLY_WEB_TORSION_SPLIT = True                # +J_w/3 to head, +2J_w/3 to foot
APPLY_IW_10PCT = True                         # I_w raised by 10%

J_h = J_h_raw + (J_w/3 if APPLY_WEB_TORSION_SPLIT else 0.0)
J_f = J_f_raw + (2*J_w/3 if APPLY_WEB_TORSION_SPLIT else 0.0)
I_w = I_w_raw * (1.10 if APPLY_IW_10PCT else 1.0)


# ============================================================================
# 2. System matrices
# ============================================================================
def beam_element(L, EI, mu):
    """Euler-Bernoulli element, DOF [u1, phi1, u2, phi2], phi = du/dZ,
    node 1 at Z=0, node 2 at Z=L. mu = mass per unit length."""
    K = EI/L**3 * np.array([[12,    6*L,   -12,    6*L],
                            [6*L, 4*L**2, -6*L, 2*L**2],
                            [-12,  -6*L,    12,   -6*L],
                            [6*L, 2*L**2, -6*L, 4*L**2]], float)
    M = mu*L/420 * np.array([[156,    22*L,   54,   -13*L],
                             [22*L,  4*L**2, 13*L, -3*L**2],
                             [54,     13*L,  156,   -22*L],
                             [-13*L, -3*L**2, -22*L, 4*L**2]], float)
    return K, M


def system_matrices():
    M = np.diag([rho*A_h, rho*I_h, rho*I_hp, rho*A_f, rho*I_f, rho*I_fp])
    D = np.diag([G*A_h*kap_h, E*I_h, G*J_h, G*A_f*kap_f, E*I_f, G*J_f])

    Gm = np.zeros((6, 6))
    Gm[0, 1], Gm[1, 0] = -G*A_h*kap_h, G*A_h*kap_h
    Gm[3, 4], Gm[4, 3] = -G*A_f*kap_f, G*A_f*kap_f

    K_R = np.zeros((6, 6))
    K_R[1, 1], K_R[4, 4] = G*A_h*kap_h, G*A_f*kap_f

    Ke, Me = beam_element(h_w, E*I_w, rho*t_w)
    P = np.array([[0, 0, 1, 0], [0, 0, 0, 1],
                  [1, 0, 0, 0], [0, 1, 0, 0]], float)   # reorder to [top, bot]
    Ke, Me = P @ Ke @ P.T, P @ Me @ P.T

    T = np.zeros((4, 6))
    T[0, 0], T[0, 2] = 1.0, -h_h/2
    T[1, 2] = 1.0
    T[2, 3], T[2, 5] = 1.0, +h_f/2
    T[3, 5] = 1.0

    return M + T.T @ Me @ T, D, Gm, K_R + T.T @ Ke @ T


def K_F(w, damped=False):
    """Foundation dynamic stiffness, paper Eqs. (7) and (10)."""
    kp = Kp*(1 + 1j*eta_p) if damped else Kp
    kb = Kb*(1 + 1j*eta_b) if damped else Kb
    kpr = Kpr*(1 + 1j*eta_p) if damped else Kpr
    Zt = kp*(kb - Ms*w**2)/(kp + kb - Ms*w**2)
    K = np.zeros((6, 6), complex if damped else float)
    K[3, 3] = Zt
    K[3, 5] = K[5, 3] = -Zt*h_f/2
    K[5, 5] = kpr + Zt*h_f**2/4
    return K


# ============================================================================
# 3. Solvers
# ============================================================================
def wavenumbers(M, D, Gm, K, w):
    """All 12 wavenumbers k at angular frequency w."""
    Dinv = np.linalg.inv(D)
    A = np.block([[np.zeros((6, 6)), np.eye(6)],
                  [-Dinv @ (w**2*M - K), -Dinv @ Gm]])
    return 1j*np.linalg.eigvals(A)          # lambda = -i k


def propagating(k, tol=1e-6):
    """Positive real wavenumbers, sorted descending -> waves I, II, III, IV."""
    sel = k[np.abs(k.imag) < tol*np.maximum(1.0, np.abs(k.real))]
    return np.sort(sel.real[sel.real > 1e-9])[::-1]


def cutons_free(M, K_R):
    w2, _ = eigh(K_R, M)
    return np.sqrt(np.clip(w2, 0, None))/(2*np.pi)


def cutons_supported(M, K_R, fmax=5000.0, n=200000):
    """det(K_R + K_F(w) - w^2 M) = 0. The pole of Z_t also flips the sign of
    the determinant, so that root is identified and discarded."""
    f_pole = np.sqrt((Kp + Kb)/Ms)/(2*np.pi)
    fs = np.linspace(0.5, fmax, n)
    dets = np.array([np.linalg.det(K_R + K_F(2*np.pi*f) - (2*np.pi*f)**2*M)
                     for f in fs])
    roots = []
    for i in np.where(np.sign(dets[:-1]) != np.sign(dets[1:]))[0]:
        a, b = fs[i], fs[i+1]
        sa = np.sign(dets[i])
        for _ in range(60):
            mfr = 0.5*(a + b)
            dm = np.linalg.det(K_R + K_F(2*np.pi*mfr) - (2*np.pi*mfr)**2*M)
            a, b = (mfr, b) if np.sign(dm) == sa else (a, mfr)
        r = 0.5*(a + b)
        if abs(r - f_pole) > 1.0:
            roots.append(r)
    return np.array(roots), f_pole


# ============================================================================
# 4. Validation
# ============================================================================
def validate(M, D, Gm, K_R):
    print("="*72)
    print("STEP 1  Cross-sectional properties vs. the values printed in the paper")
    print("="*72)
    paper = dict(A_h=2.847e-3, I_h=1.264e-6, I_hp=1.625e-6, J_h=0.9549e-6,
                 A_f=2.625e-3, I_f=4.921e-6, I_fp=4.988e-6, J_f=0.2471e-6,
                 A_w=2.166e-3, I_w=0.5716e-6, J_w=0.2338e-6)
    mine = dict(A_h=A_h, I_h=I_h, I_hp=I_hp, J_h=J_h_raw,
                A_f=A_f, I_f=I_f, I_fp=I_fp, J_f=J_f_raw,
                A_w=A_w, I_w=I_w_raw, J_w=J_w)
    print(f"{'':>6}  {'derived (raw)':>14}  {'paper':>13}  {'diff %':>7}")
    for k in paper:
        e = 100*(mine[k] - paper[k])/paper[k]
        print(f"{k:>6}  {mine[k]:14.5e}  {paper[k]:13.5e}  {e:+7.2f}")
    print(f"\nmass per unit length {rho*(A_h+A_f+A_w):.2f} kg/m "
          f"(UIC 60 nominal 60.21)")
    print("All raw properties reproduce the printed list to <1%, which is why "
          "the two\ncorrections described in the text must be applied on top "
          "of it.")

    print("\n" + "="*72)
    print("STEP 2  Internal consistency")
    print("="*72)
    d = h_w + h_h/2 + h_f/2
    tr = np.array([1, 0, 0, 1, 0, 0], float)
    ro = np.array([d/2, 0, 1, -d/2, 0, 1], float)
    sc = np.abs(K_R).max()
    print(f"web energy, rigid lateral translation : {tr @ K_R @ tr/sc:9.2e} (=0)")
    print(f"web energy, rigid section rotation    : {ro @ K_R @ ro/sc:9.2e} (=0)")
    print(f"M sym {np.allclose(M, M.T)} | K_R sym {np.allclose(K_R, K_R.T)} | "
          f"G antisym {np.allclose(Gm, -Gm.T)}")

    print("\n" + "="*72)
    print("STEP 3  Cut-on frequencies (Hz)")
    print("="*72)
    ff = cutons_free(M, K_R)
    rs, fp = cutons_supported(M, K_R)
    print(f"free rail      : {np.array2string(ff, precision=1, floatmode='fixed')}")
    print(f"supported rail : {np.array2string(rs, precision=1, floatmode='fixed')}")
    print(f"(Z_t pole at {fp:.1f} Hz discarded - not a cut-on)")
    print("\n  supported-rail comparison with the paper")
    print(f"  {'mechanism':<42}{'model':>9}{'paper':>9}")
    for name, val, ref in [
            ("track bouncing laterally on ballast", rs[0], 100),
            ("rail vibrating laterally on the pad", rs[1], 160),
            ("rail rotating on pad rotational stiffness", rs[2], 380),
            ("web bending, wave III", rs[3], 1400),
            ("web bending, wave IV", rs[4], 3600)]:
        print(f"  {name:<42}{val:9.1f}{ref:9.0f}")
    print(f"\n  free-rail wave III {ff[2]:7.1f} Hz  ->  supported {rs[3]:7.1f} Hz "
          f"(raised by foundation, as stated)")
    print(f"  free-rail wave IV  {ff[3]:7.1f} Hz  ->  supported {rs[4]:7.1f} Hz "
          f"(almost unchanged, as stated)")
    return ff, rs


# ============================================================================
# 5. Main
# ============================================================================
def main():
    M, D, Gm, K_R = system_matrices()
    validate(M, D, Gm, K_R)

    freqs = np.linspace(1, 5000, 1500)

    free = [propagating(wavenumbers(M, D, Gm, K_R, 2*np.pi*f)) for f in freqs]
    supp = [propagating(wavenumbers(M, D, Gm, K_R + K_F(2*np.pi*f), 2*np.pi*f))
            for f in freqs]

    print("\n" + "="*72)
    print("STEP 4  Free-rail wavenumbers (rad/m), waves I, II, III, IV")
    print("="*72)
    for f in [500, 1000, 2000, 3000, 4000, 5000]:
        i = int(np.argmin(np.abs(freqs - f)))
        print(f"  {f:5d} Hz : "
              f"{np.array2string(free[i], precision=2, floatmode='fixed')}")
    print("\n  Fig. 4 read-off at 5000 Hz: wave I reaches the top of the axis "
          "(35 rad/m).")

    # ---- decay rate, damped supported rail (paper Fig. 11) ----------------
    fd = np.logspace(np.log10(100), np.log10(5000), 400)
    decay = []
    for f in fd:
        w = 2*np.pi*f
        Dc = D*(1 + 1j*eta_r)
        k = wavenumbers(M.astype(complex), Dc, Gm.astype(complex),
                        K_R + K_F(w, damped=True), w)
        k = k[k.real > 1e-6]
        k = k[np.argsort(-k.real)]
        prop = k[np.abs(k.imag) < 0.5*np.abs(k.real)]     # exclude near-field
        decay.append(8.686*np.abs(prop.imag))

    # ---- plots ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
    for ax, data, ttl in [(axes[0], free, "Free rail  (cf. Fig. 4)"),
                          (axes[1], supp, "Continuously supported  (cf. Fig. 5)")]:
        for f, ks in zip(freqs, data):
            ax.plot(np.full_like(ks, f), ks, '.', color=TEAL, ms=1.5)
        ax.set_xlim(0, 5000)
        ax.set_ylim(0, 35)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Wavenumber (rad/m)")
        ax.set_title(ttl, fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle("Lateral vibration of UIC 60 rail — multiple-beam model",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT + "rail_lateral_dispersion.png", dpi=160,
                facecolor="white")

    fig2, ax = plt.subplots(figsize=(6.2, 4.3))
    for i, c in enumerate([TEAL, RED, GREY, "#7a5195"]):
        y = [d[i] if len(d) > i else np.nan for d in decay]
        ax.loglog(fd, y, color=c, lw=1.4, label=f"wave {'I II III IV'.split()[i]}")
    ax.set_xlim(100, 5000)
    ax.set_ylim(0.05, 100)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Decay rate (dB/m)")
    ax.set_title("Damped supported rail, lateral decay rate (cf. Fig. 11)",
                 fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(OUT + "rail_lateral_decay_rate.png", dpi=160,
                 facecolor="white")
    print("\nfigures written")


if __name__ == "__main__":
    main()
