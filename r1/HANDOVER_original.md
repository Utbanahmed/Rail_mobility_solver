# HANDOVER — Delkor Egg rail band structure (INCOMPLETE)

Status: **incomplete**. The band-structure spectrum does not yet match the
experiment. This document tells a new AI model exactly what the goal is, what
has been built and validated, what is broken, and — importantly — which of my
approaches were dead ends so you do not repeat them.

Read this fully before writing code. The user is an expert (rail vibration /
metamaterials); match their level, be critical, do not flatter, verify against
their data, and never use em dashes in prose.

---

## 1. THE GOAL (what the user actually wants)

Compute the **lateral** Bloch band structure of a periodically supported
UIC 60 rail carrying JIL's RCTMD / Delkor Egg fastening, and have it reproduce
the **experimentally measured spectrum**:

- The measured driving-point lateral mobility FRF (hit laterally at the rail
  head, measure lateral velocity v_y at the same point) shows **two adjacent
  sharp resonances at 523.5 Hz and 557.0 Hz** (~33 Hz apart), both ~1.5e-4
  m/s/N. See the user's FRF figure ("FRFs before and after RCTMD installation",
  the dashed red "Before Installation" curve is the experiment).
- The user's COMSOL dispersion shows these as the **edges of a NARROW band
  gap**. It is a genuine stop band, but so narrow that in the FRF it appears as
  two adjacent peaks rather than a wide attenuation trough.
- A small absolute frequency offset is acceptable. **The SPECTRUM SHAPE must
  match**: a narrow gap (~30 Hz), not a wide one.

The user validates the two mobility resonances against COMSOL dispersion; they
must be the same feature.

Downstream purpose (context, not the immediate task): this underpins an RTH-TC
research proposal on whether the RCTMD damper array suppresses rail corrugation
by opening a local-resonance / hybridisation band gap, and where corrugation
re-emerges (the gap edges). See HANDOFF_transient_solver.md for that wider
context if present.

---

## 2. THE SUPPORT MODEL (per the user, corrected at the end)

Physical Delkor Egg HAF fastening, **spring-mass-spring-ground**, parameters
from the user's Strand7 table (per fastener unless noted; "x4" means 4 springs
whose stiffnesses in the table are already the total):

    periodicity d      = 690 mm
    rail               = UIC 60, single homogeneous steel, E=210 GPa,
                         rho=7850, nu=0.30
    Iy (lateral bend)  = 514.821 cm^4   (the SMALLER one governs lateral)
    Iz (vertical bend) = 3024 cm^4
    top springs (rail->plate):    ky = 4 x 2.5e7 N/m,  kz = 4 x 5e7 N/m
    k_rot (x4)         = 4 x 0.327 MN.m/rad
    top plate          = 90 mm x 350 mm, m = 9 kg  (behaves RIGIDLY)
    rail pad area      = 90 mm x 150 mm
    bottom springs (plate->ground): ky = 4 x 2.5e6, kz = 4 x 5e6 N/m

Key geometric facts the user gave (that I mishandled — see Section 4):
- Top springs connect rail to plate at the MIDDLE of the plate.
- Bottom springs connect plate to ground at the EDGES of the plate.
- The plate behaves RIGIDLY. The user's explicit guidance: **plate dimensions
  should be irrelevant because it is rigid** — do NOT build an elaborate
  lever-arm / plate-geometry model. I made the mistake of over-focusing on
  plate edge geometry; the user stopped me. Treat the plate as a rigid mass
  with lateral + roll DOF and the springs as given; do not chase plate
  dimensions.

CONSISTENCY CHECK (done, all OK):
- Iy/Iz values are correct UIC 60 (smaller 515 cm^4 = lateral). Label wording
  in the table is ambiguous but the numbers are right.
- dx = 30 mm, d = 690 mm -> 23 elements/bay. CFL c*dt/dx = 0.69 with dt=4us. OK.
- Series lateral stiffness top||bottom = 9.1 MN/m per fastener.
- **Top-plate lateral resonance sqrt((ky_top+ky_bot)/m)/2pi = 556 Hz.** This
  coincides with the measured 557 Hz upper peak — almost certainly not a
  coincidence; the plate resonance is central to the gap.

---

## 3. WHAT IS BUILT AND VALIDATED (trust these)

All Python; the MATLAB suite mirrors it. Files in this folder.

**Beam rail model** (Wu & Thompson 1999 multiple-beam, 6-DOF lateral section):
- `rail_lateral_dispersion_final.py` : `system_matrices()` returns
  (M, D, G, KR) 6x6 per-unit-length. DOF q=[v_h,psi_h,th_h,v_f,psi_f,th_f].
  **G (3rd output) is the ANTISYMMETRIC gyroscopic matrix — use it directly,
  never C - C.'**. This caused two separate bugs; see Section 5.
  VALIDATED against Wu & Thompson JASA 106 (1999) & 108 (2000): free-rail
  cut-ons 1297/3628 Hz (waves III/IV), supported pinned-pinned 513/935/1762/
  3942 Hz, point-mobility resonances 100/160/380/500/900/1700 Hz. All <4%.
- Transfer-matrix Bloch: cell = J*expm(Ac*d), with
  Ac = [[0, I],[-Dinv(w^2 M - KR), -Dinv G]] and support jump
  J = [[I,0],[-Dinv*Ks, I]]. This is the CORRECT discrete-Bloch method (an
  earlier "S-matrix with G^T Dinv G" version was WRONG and gave spurious
  frequency-independent eigenvalues — do not use it).

**SAFE 3D model** (the promising direction — see Section 6):
- `safe_rail.py` : meshes the real UIC 60 cross-section (2D quad mesh, 3 DOF/
  node ux,uy,uz), assembles SAFE matrices K1 + i k K2 + k^2 K3 - w^2 M, solves
  dispersion w(k). Section properties validated to <1%: area 76.46 vs 76.70
  cm^2, Iz 513.1 vs 512.3 cm^4 (0.2%), centroid 80.8 vs 80.9 mm.
- Free-rail SAFE dispersion built and validated. **The torsional / head-and-
  foot lateral-flapping branch passes through 512 Hz (k=3.64) to 562 Hz
  (k=3.92)** — i.e. straight through the measured 525/557 band. This is why 3D
  matters: the beam model puts this branch at the wrong frequency (~375-440),
  the 3D section puts it right.
- NOT yet built on SAFE: the discretely-supported supercell, the gap, the
  mobility. This is the main unfinished piece.

**MATLAB suite** (`matlab/`, mirrors the beam Python, runs in Octave/MATLAB):
run_selftest, rail_cross_section, bloch_dispersion, rail_transient_bandstructure,
rail_transient_mobility, rail_point_mobility, run_all. run_selftest hardened to
check full antisymmetry of G. Transient mobility reproduces Wu & Thompson.

---

## 4. WHAT IS BROKEN (the actual open problem)

**The band-structure spectrum does not match experiment.** Specifically:

- With the physical spring-mass-spring-ground support on the BEAM rail, the
  Bloch model opens ONE WIDE stop band ~384-662 Hz (276 Hz wide), attenuation
  peaking ~535 Hz. The user confirmed this is WRONG: **experiment shows a NARROW
  gap (~30 Hz) at 523/557, not a wide one.**
- Diagnosis so far: the wide gap is driven by the ROTATIONAL support stiffness
  (krot). With krot=0 the lateral-only coupling gives no full gap (one branch
  drops out, the other propagates). With full krot it gives a 120-276 Hz gap.
  No setting I tried produced a clean NARROW (~30 Hz) full gap at 523/557.
- So the beam-model + my support coupling has the wrong structure. The two
  branches near 540 Hz either overlap (no gap) or open too wide a gap.

The narrow gap is almost certainly the **plate lateral resonance (556 Hz)
acting as a weakly-coupled LOCAL RESONATOR** that opens a narrow local-resonance
gap, hybridising with a rail branch. The wide gap came from treating the
support as a stiff series stiffness (strong coupling). The fix direction:
weakly-coupled resonator, narrow gap. But I did not get it to reproduce 30 Hz.

---

## 5. DEAD ENDS / MY MISTAKES (do not repeat)

1. **Coupling-matrix sign/count bug (fixed).** The antisymmetric G must be used
   directly. Building it as `C - C.'` from a 2-entry C doubled it; building C
   with only upper entries halved it. Both gave wrong dispersion. `run_selftest`
   now guards this.
2. **Wrong discrete-Bloch transfer matrix (fixed).** An `S = [[-Dinv C, Dinv],
   [KR - w^2 M - C^T Dinv C, C^T Dinv]]` formulation gave spurious
   frequency-independent eigenvalues (k=0.93, 4.22 constant vs frequency). The
   correct method is J*expm(Ac*d) with the companion Ac (Section 3).
3. **Overlaying continuous-support theory to "match" the transient (mistake).**
   When the transient ridge did not match discrete Bloch, I switched the overlay
   to continuous-support dispersion so it "matched". That hid the real bug
   (item 2). Do not paper over disagreements; fix the model.
4. **Single-spring model cannot place the gap at 525/557.** On the beam rail a
   single effective spring gives a gap at 375-485 Hz for any stiffness /
   attachment. Its position is set by the rail SECTION, not the support. This
   is why 3D (SAFE) is needed.
5. **Over-focusing on plate edge geometry / lever arms (mistake).** I started
   building an elaborate rigid-plate model with bottom springs at plate edges
   and lever arms. The user stopped me: the plate is RIGID and its dimensions
   should be irrelevant. Do not go there.
6. **Beam torsional softness.** The 6-DOF beam section is torsionally too soft;
   it puts the gap-forming branch ~150 Hz too low. Correcting one GJ scalar is
   not enough because the real mode is distributed 3D web/warping deformation.

---

## 6. RECOMMENDED NEXT STEPS (in order)

1. **Get the user's COMSOL dispersion in the 400-700 Hz window.** This is the
   single most useful thing. It shows whether the narrow gap is an avoided
   crossing (two branches bending apart) or a clean local-resonance gap, and
   how the branches approach it. I kept proposing shapes blind; do not. Ask for
   or use this plot to fix the model structure.
2. **Build the discretely-supported supercell on the SAFE 3D section**
   (`safe_rail.py` has the validated free-rail SAFE model). The support is the
   spring-mass-spring-ground with the 9 kg rigid plate as a 2-DOF (lateral +
   roll) resonator, coupled to the rail foot. The SAFE section already puts the
   torsional branch at 512-562 Hz, so the gap should land near 525/557 for the
   right physical reason, and — if the plate is a weakly-coupled resonator — be
   NARROW.
3. **Treat the plate as a weakly-coupled lumped resonator**, not a stiff series
   path: rail foot -- top spring --> rigid plate mass (lateral + roll) --
   bottom spring --> ground. Its lateral resonance at 556 Hz should open a
   narrow local-resonance gap. Check the gap WIDTH against the ~30 Hz measured.
4. Only after the dispersion gap matches in shape and width, compute the
   driving-point mobility (transient: 1000 N ~4 us rectangular pulse at rail
   head, record v_y, Y=FFT(v_y)/FFT(F); or frequency-domain receptance) and
   confirm the two peaks at ~523/557.
5. Then port the working model to the MATLAB suite.

The user's methodology to match (their Strand7): rectangular ~4 us, 1000 N
lateral pulse at a rail-head node; measure lateral velocity; extract mobility;
verify the two resonances against COMSOL dispersion. They use a long rail
(800 m, 1160 periods) to avoid reflections; our contribution was to replace
that with an absorbing sponge on a short domain (validated for the beam
transient; reflection ~1.35%).

---

## 7. FILE INDEX (this folder)

Trust:
- `rail_lateral_dispersion_final.py`  validated beam operator (M,D,G,KR)
- `safe_rail.py`                      validated SAFE 3D section + free dispersion
- `matlab/`                           validated beam MATLAB suite

Working / diagnostic (many iterations, not all clean):
- `fit_*.py`, `disp_fit.py`, `final_fit.py`, `delkor_*.py`  parameter-fitting
  attempts on the beam model. Historical; superseded by the SAFE direction.

Do not trust as final: any figure showing a WIDE gap (that is the bug), and any
continuous-support overlay (that was the paper-over in item 3).
