# Lateral dispersion of the Delkor-supported UIC 60 rail: results and model

This continues the handover work. Bottom line up front, then the reasoning,
the validations, and an honest account of what matches and what does not.

## 1. Bottom line

- The measured feature (two lateral driving-point peaks at 523.5 and 557 Hz,
  ~33 Hz apart) is a **local-resonance avoided crossing**: the 9 kg Delkor top
  plate, on its top+bottom lateral springs, resonates at **556.4 Hz** and
  hybridises with a rail cross-sectional **"flapping" branch** (head and foot
  translating laterally in antiphase). It is **not** a wide Bragg-type stop
  band, and it is **not** a clean full band gap either (see section 6).
- The plate lateral resonance with the rail held fixed is
  `sqrt((k_top+k_bot)/m)/2pi = sqrt((1e8+1e7)/9)/2pi = 556.4 Hz`, equal to the
  measured upper peak (557 Hz) to within 0.1%. That coincidence is the whole
  story and it is now reproduced from the section physics, not asserted.
- The old 6-DOF beam model cannot produce this, for a concrete reason
  (section 3). The 3D SAFE section is required, exactly as the handover
  suspected. A working, validated 3D SAFE Floquet-Bloch supercell is the main
  new deliverable.
- On the nominal parameters the model gives two lateral-mobility peaks at
  **535 and 573 Hz** on the coarse section mesh and **531 and 570 Hz** on the
  refined mesh (vs measured 523.5 and 557). The **width/shape matches** the
  measured ~33 Hz two-peak feature; the **absolute position is 7-13 Hz high**
  and drops as the section mesh is refined. What sets that residual offset, and
  how to close it, is in section 7.

## 2. What was re-validated before building anything

Nothing downstream is trusted unless the piece under it reproduces a known
result.

- **Beam operator** (`rail_lateral_dispersion_final.py`): reproduces Wu &
  Thompson (1999/2000) free-rail cut-ons and track-C supported resonances
  (100/160/380/1400/3600 Hz) to <4%. Confirmed by re-running.
- **SAFE cross-section** (`safe_rail.py`): A = 76.46 cm^2 (nominal 76.70),
  Iz = 509-513 cm^4 (nominal 512.3), mass 60.14 kg/m (nominal 60.21),
  centroid 80.8 mm. All <1%.
- **New reduced-order model** (section 4): reproduces the full SAFE dispersion
  to 0.1 Hz on every branch, with 57 basis vectors.
- **Operator sanity**: reduced K1, K3, M symmetric to 1e-16; uniform lateral
  translation costs ~0 strain energy (rigid-body check); mass integral 60.14
  kg/m.

## 3. Why the beam model fails (confirmed, not assumed)

At 540 Hz the two propagating lateral waves of the free beam rail are both
**pure torsion**: the section-rotation DOF dominate (|theta_h|, |theta_f| ~ 1.0)
while the net foot lateral translation `|v_f - (h_f/2) theta_f|` is 0.03-0.10.
The Delkor plate is bolted under the **foot** and couples to **foot lateral
translation**. A mode with no foot translation cannot couple to it, so the
plate resonance opens nothing near 540 Hz on the beam. The beam does have a
foot-translating mode, but its section puts it at ~420-440 Hz (the "rail
rotating on the pad" region), which is where the beam+resonator wrongly opens a
feature. Forcing a full gap there with a large static rotational support gives
the spurious ~280 Hz gap reported in the handover. This is a section-physics
error, not a coupling bug.

## 4. The 3D SAFE supercell (the new tool)

`safe_supercell.py`:

1. **Cross-section operators** K1, Kc, K3, Mcs (3 DOF/node) from the validated
   UIC 60 mesh (`cross_section_operators`).
2. **Modal reduction** (`reduced_operators`): a real basis built from SAFE
   eigenvectors sampled across k = 0.5-5.0 rad/m (real and imaginary parts,
   M-orthonormalised). 57 vectors reproduce SAFE dispersion exactly. This is
   what makes the supported problem cheap enough to converge.
3. **Bloch supercell**: the reduced operators are extruded along the rail with
   linear elements over one bay d = 0.69 m; periodicity enters through a Bloch
   phase `exp(-i k d)`. nz-converged (flapping branch 519 Hz at nz=32 vs SAFE
   517.5). Two equivalent support treatments:
   - explicit rigid plate (mass 9 kg, lateral DOF) sprung to the foot nodes by
     k_top and to ground by k_bot (`supercell_bands`);
   - the same plate **condensed analytically** into a frequency-dependent foot
     stiffness (`tm_bloch_condensed`), which keeps the driving-point mobility
     matrix banded.

The flapping mode this couples to (535 Hz, k = 3.74) has the foot translating
laterally with **full coherence** (all 11 foot-underside nodes move together,
amplitude 1.0) and the head moving in antiphase. So the coupling to the plate
is strong and physical, and the resonance lands in the right band because the
3D section, unlike the beam, places this branch at 517-567 Hz.

## 5. Driving-point lateral mobility (`safe_mobility.py`)

Long finite rail (40 bays) in the reduced space, plate condensed at each bay,
a graded hysteretic "sponge" at both ends to emulate an infinite rail
(removes finite-length ripple), lateral point force at the rail head.

Nominal result (plate 9 kg): peaks at **535 and 573 Hz** (coarse mesh) or
**531 and 570 Hz** (refined section mesh), dip near the plate resonance.
Measured: 523.5 and 557 Hz. Width 38-39 vs 33.5 Hz (good); centre +7 to +13 Hz
(offset shrinks with mesh refinement, see section 7). See `mobility.png`.

**Mechanism check (decisive).** Sweeping the plate mass moves the whole feature
with the plate resonance `sqrt((k_top+k_bot)/m)/2pi`: 6.5 kg -> 654 Hz (feature
up), 12 kg -> 482 Hz (feature down). The feature is the plate resonance, full
stop.

## 6. It is an avoided crossing, not a clean full gap (resolves the open question)

The handover asked whether the COMSOL feature is an avoided crossing or a clean
gap. The rigorous Bloch dispersion (`tm_bloch_condensed`) answers it: across
490-600 Hz there are **always at least two propagating Bloch waves**; the plate
(lateral, and also with roll added) reconfigures the branches near 556 Hz but
never drives them all evanescent. A grounded spring-mass-spring presents a
frequency-dependent *foundation stiffness*, which shifts branches but does not
open a local-resonance gap the way a hanging (mass-in-mass) resonator would;
and there is always a non-lateral branch (k ~ 0.6 rad/m, vertical/longitudinal
polarisation) that carries energy regardless. So the two measured peaks are the
two lobes of a **local-resonance hybridisation** in the laterally-coupled
flapping branch, appearing in the lateral mobility as two peaks with a dip
between, not as a wide attenuation trough. This is consistent with the FRF
showing two adjacent peaks rather than a broad notch.

## 7. What matches, what does not, and how to close it

Matches: the narrow width (~35 Hz, not ~280 Hz), the mechanism (plate lateral
resonance at 556 Hz on the flapping branch), and the band.

Does not (yet):
- **7-13 Hz upward offset.** The offset is set by the SAFE flapping-branch
  frequency, which depends on the section's torsional/warping stiffness. The
  cross-section here is the simplified polygon from `safe_rail.py` (Iz within
  0.7%, but the flapping mode is torsion-like and more sensitive to the web and
  head fillet detail than Iz is). Refining the mesh already moves the free
  branch from 517/567 (nx=10) to 513/562 (nx=14) to 511/561 (nx=16) Hz, and the
  mobility peaks from 535/573 to 531/570 Hz: monotonically toward 523.5/557. A
  truer section (your COMSOL section geometry) should close most of the rest.
- **Peak vs anti-resonance at 556 Hz.** In the driving-point mobility a grounded
  resonator makes an *anti-resonance* (tuned-damper dip) at its own frequency;
  the measurement shows a *peak* at 557 Hz. Whether 557 is a band edge of the
  hybridisation (peak) or the resonator dip depends on the exact branch shape
  near the crossing, which is precisely what the COMSOL dispersion in 400-700
  Hz would settle.

Highest-value next input, unchanged from the handover: **the COMSOL lateral
dispersion in 400-700 Hz.** With it, two knobs calibrate cleanly: the section
torsional stiffness (branch position) and the roll/vertical support coupling
(`k_rot`, `kz`; `exp_roll.py` has the roll path wired in). The framework is
ready to ingest that and does not need re-derivation.

## 8. Files

- `safe_supercell.py`   cross-section operators, SAFE modal ROM, Bloch supercell
                        (explicit plate and condensed), rigorous dispersion.
- `safe_mobility.py`    infinite-rail driving-point lateral mobility.
- `exp_roll.py`         lateral+roll rigid-pad support experiment.
- `rail_periodic.py`,`rail_mobility.py`   beam-model FE cross-checks (validation).
- `rail_lateral_dispersion_final.py`, `safe_rail.py`   inherited, re-validated.
- `dispersion_comparison.png`, `mobility.png`   figures.

Do not trust: any claim of a wide (>100 Hz) gap (that was the beam error), and
any claim of a clean full stop band (there is none; it is an avoided crossing).
