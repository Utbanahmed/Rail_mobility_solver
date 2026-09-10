# HANDOVER — lateral mobility of the Delkor‑supported UIC60 rail (r2)

**STATUS: OPEN.** The mechanism of the measured ~525/557 Hz lateral point‑mobility
feature is **not yet conclusively explained by a validated model.** Several
hypotheses were examined; none is proven. `REPORT.md` in this folder contains
earlier conclusions that are now **superseded / under revision** — treat this
HANDOVER as the current source of truth and treat the problem as unsolved.

The two hard requirements going forward, set by the user:
1. **Whatever we conclude must physically make sense and be double‑checked** — no
   result is accepted without independent verification.
2. **We need a way to PROVE whatever we come up with.**
And the strategic instruction: **get the band structure right and proven FIRST;
everything else (mobility matching, transient, etc.) comes after.**

---

## 0. The measured target (ground truth) — "Singapore Lateral FRF"

The user provided a measured FRF figure (title "Singapore Lateral FRF").
**ACTION: attach the original image to this folder as
`figures/measured_singapore_lateral_FRF.png`** — I could not extract the pasted
image to a file programmatically, so it must be dropped in by hand. Its content,
read carefully from the plot and the user's description, is documented below so
the target is fully specified even before the file is attached.

Curves (units m/s/N vs Hz, 0–1200 Hz, log‑y):
- **Point mobility** — dashed, **no markers**: **red = midspan**, **green =
  support/baseplate location** (legend: "Midspan‑Midspan", "Midspan‑Baseplate").
- **Transfer mobility** — **with markers**, same locations but a **far station
  ~12–21.6 m away** (legend "21.6m M – Midspan"): red triangles, green x.

Features read from the red midspan **point** mobility:
| f (Hz) | feature | notes |
|---|---|---|
| ~65 | strong peak (~2e‑4) | rail lateral bounce on the supports |
| 100–350 | plateau ~1e‑5 | |
| ~360 | deep antiresonance (~6e‑7) | |
| **~400** | lateral **pinned‑pinned** | |
| **525** | **peak** (~1–2e‑4) | cross‑section: **head+foot move, WEB = node → torsion‑like**; midspan‑vs‑support: **persists at both** |
| **557** | **peak** (~1–2e‑4) | cross‑section: **head+web+foot all move**; midspan‑vs‑support: **drops a lot at support → pinned‑pinned bending‑like** |
| ~580 | deep antiresonance | in point mobility |
| ~650 | rounded hump (~1e‑6) | |
| 750–1200 | low/ragged 1e‑7…1e‑5 | |

**Transfer mobility (decisive):** the transfer **drops** (stop band) over
**~557 → >600 Hz**. The two peaks (525/557) themselves **do appear in the
transfer** (they propagate the ~12 m). So: peaks propagate, and the gap is
**above** 557, not between 525 and 557.

These are the numbers any accepted model must reproduce — **the cross‑section
shapes and the support/transfer behaviour, not just the two peak frequencies.**

---

## 1. Physical system & parameters

- Rail: UIC60 / 60E1. Steel E=210 GPa, ν=0.3, ρ=7850.
- Support period **L = 0.69 m** (Delkor HAF / Egg fastening).
- Support = **spring → plate(mass) → spring → ground, per direction** (this is
  the user's mental model and it is correct; it is the code's **V1 FULL** variant):
  - vertical: k_top = 1.0e8 N/m (rail pad), k_bot = 1.0e7 N/m (resilient baseplate).
  - lateral: unknown → **primary calibration parameter** `s_lat` (default = vertical).
  - plate mass m_plate = 9 kg.
  - **plate lateral rigid‑body resonance** √((kx_top+kx_bot)/m)/2π = **556 Hz**
    (coincides with the measured 557 — this coincidence is central and unresolved).
  - `V2 REDUCED` = plate removed, springs in series k=k_top·k_bot/(k_top+k_bot);
    a diagnostic simplification, **not** the physical support.

---

## 2. Tools in this folder (r2/) and their trust level

| file | what | trust |
|---|---|---|
| `rail_cell.py` | UIC60 section mesh, HEX8 element, extrude, supports (V1/V2), section verify | see caveats below |
| `bloch_dispersion.py` | Tool A: Floquet‑Bloch band diagram (one cell) | works |
| `transient_fullwave.py` | Tool B: explicit central‑difference + ALID, mobilities, 2D‑FFT | works |
| `validate.py` | validation suite (section, dispersion overlay, L‑shift, N‑periods, no‑reflection, V1/V2, plate) | passes offset‑independent checks |
| `rail_tools.ipynb` | executed notebook (renders on GitHub) | illustrative |
| `figures/` | all diagnostic plots (see below) | |

**What is trustworthy:**
- Section constants: A ≈ 77.6 cm² (+1.2%), Ix ≈ 3031 cm⁴ (−0.2%), Iy ≈ 505 cm⁴
  (−1.3%), mass ≈ 60.9 kg/m — all vs published 60E1. ✓
- **St‑Venant torsion constant J ≈ 191 cm⁴ (staircase) / 245 (gmsh) vs published
  ~205 cm⁴** — section torsional stiffness is essentially correct. ✓
- Axis‑aligned (staircase) mesh + incompatible‑modes HEX8 **bends correctly**
  (cantilever within 3% of EI theory, both planes; axial stretch exact).

**What is NOT trustworthy / known pitfalls (READ BEFORE CODING):**
- **The brick element only works on AXIS‑ALIGNED meshes.** A fillet‑resolved
  conforming (gmsh) quad mesh **locks catastrophically** (cantilever = 0.001×
  theory — *trapezoidal locking* of incompatible‑modes elements). `Cfg.USE_GMSH`
  exists but is **OFF for this reason — do not enable it with the current
  element.** To use true geometry you must first replace the element (see §4).
- **The staircase section's torsion resonance is mesh‑converged at ~690 Hz**
  (zone edge), NOT a discretisation artifact (refining cross‑section AND axial
  both converge there). Since J is correct, the ~690 vs measured 525 gap is
  **excess warping/distortional stiffness from the idealised (blocky) head/foot
  geometry**, not a mesh or J error.
- The head‑drive **point mobility is dominated by a spurious ~1233 Hz head‑sway
  mode** (coarse‑section artifact) that the measurement does **not** show.
- Earlier `REPORT.md` narrative ("plate local resonance", then "torsion×bending
  avoided crossing", then "two independent pin‑pin modes") is a **chronology of
  hypotheses, several since refuted** — see §3. Do not cite REPORT conclusions.

---

## 3. What the model currently produces, and the hypothesis chronology

**Mesh‑converged model numbers (staircase section):**
- lateral **bending** pinned‑pinned (zone edge k=π/L): **~425 Hz** (≈ measured ~400 ✓)
- **torsion** pinned‑pinned (zone edge): **~690 Hz** (converged; NOT 525)
- vertical bending zone edge: ~900 Hz
- plate lateral resonance: 556 Hz (correct by construction)
- **Bare rail (V2)** has an avoided crossing between the rising torsion branch and
  the back‑scattered (folded) bending branch at ~575 Hz, but the gap is only
  ~7 Hz (grows with lateral stiffness: 2/7/12 Hz at s_lat=0.5/1/2).

**Hypotheses examined (and verdicts) — do not re‑walk these blindly:**
- **H1 "plate local resonance drives the peaks"** (from the r1 SAFE iteration).
  Partly supported: with the plate (V1) the model sprouts extra branches in the
  500–640 band and the plate carries 19–35% of the modal energy there.
- **H2 "torsion×bending avoided crossing"** (the original task hypothesis).
  The bare rail *does* have such a crossing, but at ~575 Hz with a ~7 Hz gap —
  too weak and mis‑placed. A standard avoided‑crossing gap would sit *between*
  the two peaks; the measured transfer drop is *above* 557 → argues against a
  simple avoided‑crossing between 525 and 557.
- **H3 "525 and 557 are two independent rail pinned‑pinned modes (torsion +
  bending)."** **REFUTED by the user:** pinned‑pinned frequencies are fixed by L
  and the branch; the folded free‑rail band structure puts torsion pin‑pin at
  ~690 and bending pin‑pin at ~400 — **neither at 525/557.** So 525 is not a
  bare‑rail pinned‑pinned.
- **H4 "plate‑anchored locally‑resonant hybridisation" (current leading, NOT
  proven).** Because the bare rail has no mode near 525/557 (only 400 and 690),
  something must place resonances there, and the only candidate at that frequency
  is the plate (556). Idea: the plate flat resonance band opens a
  locally‑resonant stop band from 556 up (→ the transfer drop 557→>600) and
  dresses the rail branches sweeping through 556 into two resonances just below
  the gap, inheriting rail character (525 torsion‑like, 557 bending‑like). **The
  user is not convinced and it is not proven.** Open.

**The crux / open question:** why are there resonances at 525/557 when the bare
rail has none there (400, 690)? Is it (a) plate‑anchored local resonance, or
(b) the *true* section (fillets) actually placing torsion much lower (~525) so it
meets bending — which the current staircase cannot show because its torsion is
stuck at ~690? **This cannot be decided until the section/element is fixed so the
band structure is trustworthy — hence band‑structure‑first.**

---

## 4. Recommended plan for the next agent (band structure first, with proof)

**Phase A — a trustworthy element (unblock true geometry).**
- Replace the incompatible‑modes HEX8 with a **reduced‑integration + hourglass‑
  control brick (C3D8R / Flanagan–Belytschko)**, or a full EAS element that
  passes the trapezoidal‑element patch test. This is required to mesh the real
  filleted 60E1 without locking.
- **PROVE it:** constant‑strain patch test on distorted elements; rigid‑body =
  zero energy; cantilever within a few % of EI in *both* bending planes on a
  *skewed/conforming* mesh; a **torsion cantilever** matching G·J. Do not proceed
  until all pass.

**Phase B — a trustworthy section.**
- Mesh the true 60E1 outline with fillets (gmsh is installed and works; the
  `build_cross_section_gmsh` scaffold is in `rail_cell.py`).
- **PROVE it:** match published 60E1 **A, Ix, Iy, mass, St‑Venant J, and the
  warping constant Cw** (add a Cw solver — warping function on the section). The
  warping constant is the quantity that sets how far torsion rises above St‑Venant
  at short wavelength, i.e. exactly the 690‑vs‑525 discrepancy — so Cw must be
  verified, not just J.

**Phase C — the band structure, proven.**
- Compute the Bloch bands for the **real support (V1)**; classify branches
  (lateral bending / torsion / vertical / plate) by cross‑section projection.
- Compute **complex‑k** (fixed real ω → complex k, the Bloch quadratic
  eigenproblem) to locate **stop bands** (Im k > 0) directly — this is the
  rigorous version of "where does transfer drop".
- **PROVE the dispersion three independent ways** (they must agree):
  1. Tool A (Bloch) ω(k) vs Tool B (transient) 2D‑FFT dispersion — overlay.
  2. Free‑rail low‑k branches vs analytic: bending ω≈√(EI/ρA)·k², torsion
     St‑Venant+warping — must match.
  3. Reproduce the measured **signatures**: which branch has the web node,
     which nulls at the support, and where the stop band (transfer drop) sits.
- Only when the band structure reproduces the *measured signatures and the
  transfer stop‑band location* is it trusted.

**Phase D — decide the mechanism, with a falsifiable test.**
- With a trustworthy band structure, settle H1–H4 by a **parametric proof**:
  vary L, plate mass, and lateral stiffness; the predicted shifts of 525, 557,
  and the stop band must match physical expectation AND, where available, the
  measured trends (e.g. the L=0.69→0.60 shift; plate‑mass sweep; the
  fastening‑stiffness dependence the user asked about — do 525 and 557 move
  together (coupled) or independently?).
- Key discriminator already identified: **does removing the plate (V2) destroy
  the 525/557 feature?** If yes → plate‑anchored (H4). If the true‑section V2
  still shows 525/557 → rail torsion came down to meet bending (H2/H3 revised).

**Phase E — only then:** point/transfer mobility (Tool B), match the full FRF
shape (peaks + antiresonances + the transfer stop band), and the ~1233 head mode
should disappear with the better section.

---

## 5. Sanity‑check / proof checklist (apply to EVERYTHING)

- Operators: symmetry (K=Kᵀ), rigid‑body zero energy, mass = ρ·A·L.
- Element: patch test (incl. distorted), both bending planes, torsion vs GJ.
- Section: A, Ix, Iy, J, Cw, mass, centroid vs published 60E1.
- Dispersion: Bloch vs 2D‑FFT overlay; free‑rail vs analytic at low k.
- Never trust a single method — cross‑check Tool A against Tool B, and both
  against analytic limits and the measured signatures.
- Report offset‑independent checks (shapes, node structure, shift directions,
  invariances) separately from absolute frequencies.

---

## 6. Figure index (figures/)

- `dispersion_labeled_V2.png`, `dispersion_labeled_V1.png` — bands coloured by
  mode type (lateral bending / torsion / vertical / plate).
- `bloch_bands_V2.png`, `_V1.png`, `_V2_slat*.png` — Tool A bands + lateral sweep.
- `mobility_V2.png`, `signatures_V2.png`, `spacetime_V2.png` — Tool B outputs.
- `transfer_vs_point_V1.png` — point vs transfer mobility (stop‑band illustration).
- `val_*.png`, `validation_report.txt` — validation suite.
- **MISSING (attach): `measured_singapore_lateral_FRF.png`** — the ground truth.

## 7. Related prior work

- `../r1/` — earlier SAFE (semi‑analytical) iteration; its `REPORT.md`/`HANDOVER_original.md`
  reached the "plate local resonance at 556 hybridises with a flapping branch"
  conclusion. Useful context; same 556 Hz coincidence; same section‑detail caveat.

## 8. How to run

```
pip install -r requirements.txt      # numpy scipy matplotlib ; gmsh optional (needs libGLU/libXft)
python rail_cell.py                  # section verification + assembly sanity
python bloch_dispersion.py           # Tool A bands + lateral sweep
python transient_fullwave.py         # Tool B mobility / signatures / space-time
python validate.py                   # validation suite -> figures/validation_report.txt
```
All tunables are in the `Cfg`/CONFIG block at the top of each file. Everything is
SI. X = lateral, Y = vertical, Z = axial.
