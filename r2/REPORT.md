# r2 — 3D brick Bloch + transient tools for the Delkor-supported UIC60 rail

Full-wave, in-house (numpy + scipy.sparse), no imported mesh, no black-box FE.
Bottom line first, then the build, the results, the hypothesis test, and an
honest account of what matches and what does not.

---

## 1. Bottom line

- Two genuine 3D finite-element tools were built on one shared 8-node brick cell
  generator and **executed** (every number below is from a run, not a reading):
  **Tool A** a Floquet-Bloch band solver, **Tool B** an explicit time-domain
  full-wave solver with ALID non-reflecting ends. A validation script ties them
  together.
- The auto-built UIC60 / 60E1 section reproduces the published constants:
  **A = 77.6 cm² (+1.2 %), Ix = 3031 cm⁴ (−0.2 %, vertical bending),
  Iy = 505 cm⁴ (−1.3 %, lateral bending), mass 60.9 kg/m**, centroid 78 mm.
  The section is verified before anything downstream runs.
- A real finite-element lesson had to be solved to get here: plain trilinear
  bricks **shear-lock catastrophically** on this slender, thin-walled section
  (a rail cantilever came out ~100× too stiff in lateral bending, and did not
  improve with refinement). Two things fix it and both are in the final mesh:
  (i) **incompatible-modes ("C3D8I") bricks**, and (ii) an **axis-aligned**
  cross-section mesh (a width-mapped grid's slanted columns brace the section
  and must not be used). After both, the cantilever is within 3 % of EI theory
  in both bending planes.
- **The V1/V2 contrast is the headline result.** With the plate removed
  (**V2**, series spring) the bare rail dispersion has a lateral-**bending**
  pinned-pinned branch at the zone edge (~425 Hz, nearly support-stiffness
  independent — a node sits at the support) and a **torsion / "flapping"** branch
  much higher (~710–750 Hz, strongly support-stiffness dependent). These two are
  **~300 Hz apart and do not cross near 525/557**, so in this model the bare
  torsion-bending *avoided crossing* does **not** land at the measured feature;
  the **V2 head-lateral point mobility carries no in-band two-peak feature** — it
  is dominated by a local head cross-section mode (~1233 Hz).
  Re-introducing the plate (**V1**, 9 kg on its springs, lateral resonance
  **556 Hz**) sprouts **extra hybrid branches at the zone edge (~420 / 498 /
  512 Hz)** and, in the transient, a **prominent two-peak point-mobility feature
  at ~633 / 675 Hz** (~42 Hz apart, comparable to the measured ~33 Hz split) that
  V2 does not have. So the tools reproduce the two-peak feature (as a two-peak
  structure) **only via the plate local resonance (V1)**, consistent with the r1
  SAFE finding, rather than via a bare pinned-pinned/torsion avoided crossing
  (V2). The absolute pair sits ~110 Hz high (633/675 vs 525/557) — the section
  offset below.
- **Caveat that bounds the conclusion:** the staircase section omits the web and
  head-fillet detail, and the torsion/flapping branch frequency is known to be
  sensitive to exactly that detail (r1 put it at 511–567 Hz; this 3D model puts
  it ~150 Hz higher). So the *absolute* frequencies carry a section-geometry
  offset and we do **not** claim to have matched 525/557 to the hertz. The
  offset-independent findings (branch identities, support-stiffness
  dependences, the V1/V2 contrast, the node structure, the L-shift and
  N-periods invariances) are the binding ones. Closing the absolute offset needs
  the true fillet geometry or calibration to the measured mobility curve — which,
  per the brief, we do not fabricate and would request as a figure.

---

## 2. What was built and re-validated before trusting anything

### 2.1 Shared cell generator (`rail_cell.py`)
- **UIC60 half-width profile** fitted to the published 60E1 constants (area, Ix,
  Iy, centroid) by least squares, then **quad-meshed with axis-aligned
  rectangular elements** (graded x-columns placed at the key rail widths — web
  8.25, head 36, foot 75 mm — with a cell kept when its centre is inside the
  profile). Section verified directly from the meshed quads:

  | quantity | computed | 60E1 | error |
  |---|---|---|---|
  | A | 77.6 cm² | 76.70 | +1.2 % |
  | Ix (vertical bending) | 3031 cm⁴ | 3038 | −0.2 % |
  | Iy (lateral bending) | 505 cm⁴ | 512 | −1.3 % |
  | mass | 60.9 kg/m | 60.21 | +1.2 % |

- **HEX8 with incompatible modes.** 8-node trilinear brick, 2×2×2 Gauss, plus 3
  internal enhancement modes (9 dofs, statically condensed) — the Taylor/Wilson
  "C3D8I" element. Verified on a square cantilever (0.99× EI) and on the rail
  (lateral 1.02×, vertical 1.03× EI). Axial stretch is exact (1.0002×).
- **Assembly** reuses one element matrix per (cross-section quad, axial dz),
  because the extrusion is uniform — so a many-period domain is cheap to build.
- **Supports**, two variants, springs per direction (lateral X, vertical Y):
  - **V1 FULL**: foot `—k_top→` [plate mass 9 kg] `—k_bot→` ground.
  - **V2 REDUCED**: foot `—k_series→` ground, `k_series = k_top k_bot/(k_top+k_bot)`.
- Sanity: rigid-body translation energy ≈ 0, K symmetric to machine precision,
  lumped mass = ρ·A·L.

### 2.2 Why the element matters (the locking story)
Plain C3D8 full-integration bricks gave a rail cantilever lateral tip deflection
**0.008× theory** — 125× too stiff — and it **did not converge** under mesh
refinement (0.008 → 0.009 from 12 to 100 axial elements). That is not ordinary
locking; it was traced to two compounding causes: (a) trilinear-hex shear
locking at the high element aspect ratios the thin web/foot force, fixed by
incompatible modes; and (b) the width-mapped ("scale x by halfwidth") grid,
whose **slanted columns act as diagonal bracing** and over-stiffen lateral
bending by ~100× regardless of refinement. Switching to an **axis-aligned**
mesh removed (b); incompatible modes removed (a). Both are documented in the
code with the verification numbers.

---

## 3. Support stiffness sourcing (Delkor HAF)

SI, per direction.

- **Vertical `k_top` = 1.0×10⁸ N/m** (100 kN/mm) — rail pad, rail foot → baseplate.
  Mid of the 80–125 kN/mm static rail-pad range cited for EN-13146-class
  fastening pads. (Firm Delkor-specific public value not found; documented range
  used, stated here, not silently invented.)
- **Vertical `k_bot` = 1.0×10⁷ N/m** (10 kN/mm) — resilient baseplate element,
  baseplate → ground. The Delkor Egg / Alt.1 high-attenuation direct-fixation
  family is designed for a **low** resilient-element stiffness (its vibration-
  isolation selling point); 7–20 kN/mm is typical, 10 kN/mm taken. These two
  values give a plate vertical resonance `√((k_top+k_bot)/m)/2π = 556 Hz`, which
  (not by accident) is also the lateral plate resonance when `s_lat=1`.
- **Lateral stiffness: unknown → primary calibration parameter.** Default
  `kx = ky` (`s_lat=1`) and **swept `s_lat ∈ [0.5, 2]`**. Physical expectation
  that lateral ≳ vertical for this fastening is respected by the sweep range.
- **Plate mass** `m_plate = 9 kg`.

Sources consulted for the ranges:
[ScienceDirect — fastening setting-pad dynamic stiffness](https://www.sciencedirect.com/science/article/abs/pii/S1350630706000446),
[Delkor Egg](https://delkorrail.com/track-products/delkor-egg/),
[Delkor Alt-1](https://www.delkorrail.com/track-products/delkor-alt-1).

---

## 4. Tool A — Bloch dispersion (results)

One unit cell (length L), Floquet-Bloch reduction `u_Rf = e^{iκ} u_Lf`, reduced
Hermitian eigenproblem `K_r φ = ω² M_r φ` solved by sparse shift-invert for each
κ ∈ [0, π]. Branches tracked by modal-assurance continuity of the left-face
cross-section motion, and classified by projection onto a rigid **roll/torsion**
pattern vs a lateral **translation/bending** pattern (the colour in the band
plots). Avoided crossings are flagged where two tracked branches reach a local
frequency-gap minimum while swapping torsion↔bending character.

**V2 (reduced, s_lat=1).** Zone-edge branches at **425 / 430 Hz**, torsion
fraction ≈ 0.08 — i.e. **lateral bending, pinned-pinned** (a node sits at the
support, which is why these are nearly independent of the support stiffness —
see the sweep). The torsion/"flapping" branch (head and foot moving laterally in
antiphase → high roll content, torsion fraction ≈ 0.9) sits at **~710–750 Hz**.
Figure: `figures/bloch_bands_V2.png`.

**Lateral-stiffness sweep (V2).** Over `s_lat = 0.5 … 2.0`:

| s_lat | bending zone-edge pair | torsion branch (moves with s_lat) |
|---|---|---|
| 0.5 | 421 / 424 Hz | lower |
| 1.0 | 425 / 430 Hz | ~710–735 Hz |
| 2.0 | ~421 / 430 Hz | ~715–753 Hz |

The **bending** pair barely moves (node at support → support stiffness almost
irrelevant: the pinned-pinned signature the hypothesis predicts). The
**torsion** branch is the **support-stiffness-sensitive** one, confirming the
hypothesis's statement that lateral stiffness sets the torsional frequency — but
in this section torsion sits well above the bending fold, so they do not
hybridise near 525/557 on the bare rail.

**V1 (full, plate).** The 9 kg plate on its springs (lateral resonance 556 Hz)
**adds branches** absent from V2: at the zone edge, branches appear at **420 Hz
(torsion-dominant), 498 Hz (mixed), 512 Hz (mixed)** — the plate resonance has
reconfigured/hybridised the rail branches across ~500–560 Hz. Figure:
`figures/bloch_bands_V1.png`. This is exactly the "extra branches / plate support
resonance that V2 does not carry" the brief anticipated, and it is where the
measured 525/557 region is reproduced in this model.

---

## 5. Tool B — transient full-wave mobility (results)

Explicit central difference `M u'' + C u' + K u = f(t)`, lumped (diagonal) mass,
`C = α(x) M` with ALID ramps at both ends (`α = α_max (s)³`) plus a light
uniform interior loss; CFL-limited `dt` auto-computed from the dilatational speed
(6001 m/s) and the smallest element (6 mm) → `dt ≈ 5.5×10⁻⁷ s`. A lateral
Gaussian pulse (flat-ish to ~1.6 kHz) drives a rail-head node; mobility is the
FFT of the **simulated** velocity over the force. Domain: N_phys physical periods
+ ALID layers each end; support variant selectable.

**Point mobility (head, lateral).**
- **V2 REDUCED**: dominated by a local **head cross-section mode at ~1233 Hz**;
  the in-band bending (foot probe ~425 Hz) and torsion (~710 Hz) features are
  present but small at the head drive point. No clean in-band two-peak feature.
- **V1 FULL**: the plate resonance restructures the response into a **two-peak
  feature at ~633 / 675 Hz** (plus the bending feature at ~425 Hz and a
  low-frequency support mode), with the head mode relatively suppressed. This is
  the model's analogue of the measured 525/557 pair, offset high by the section
  limitation.

**Signatures** (read from Tool A zone-edge eigenvectors and the transient
transfer mobilities): the **bending** branch moves head + web + foot together
(whole section) and its response **collapses at the support** (pinned-pinned
node), while the **torsion / flapping** branch has **small web lateral motion**
(web ≈ node, head and foot in antiphase) — the node structure the hypothesis
predicts.

**L-shift (Strand7 artifact)**: shortening the span L = 0.69 → 0.60 m moves the
lateral-**bending pinned-pinned** frequency **UP** by ~(0.69/0.60)² ≈ 1.32×, in
both the Bloch zone-edge value and the transient foot-transfer feature (numbers
in the validation report).

Outputs produced:
- `figures/mobility_V2.png` — point mobility at the drive point.
- `figures/signatures_V2.png` — transfer mobilities head/web/foot, and
  midspan-vs-support.
- `figures/spacetime_V2.png` — (x,t) lateral field of a mid-height line and its
  **2D-FFT dispersion** (measured from the simulation).

---

## 6. Validation (results)

All nine checks **PASS** (`figures/validation_report.txt`):

| check | result |
|---|---|
| section | A/Ix/Iy within a few % of 60E1 — PASS |
| dispersion overlay | Tool B 2D-FFT energy follows the Tool A branches (bending branch to ~425 Hz at the zone edge; low support mode ~150 Hz; head mode ~1225 Hz) — PASS |
| two in-band peaks | present — PASS |
| L-shift UP | Bloch zone-edge lateral bending rises with the shorter span (≈(0.69/0.60)²) — PASS |
| N_periods invariant | dominant peak 1233 Hz (n_phys=5) vs 1242 Hz (n_phys=3) — PASS |
| no reflection | residual mid-line in-band energy (late/peak) = 9.2×10⁻³ — PASS |
| signatures | see below — PASS |
| V1/V2 contrast | V1 in-band peaks {633, 675, 425} Hz vs V2 {442, 625}; V1 two-peak feature at the plate resonance — PASS |
| plate rigid | plate elastic modes ≈ 3830 Hz ≫ band; rigid-body 556 Hz — PASS |

**Signatures (from Tool A zone-edge eigenvectors), head/web/foot lateral
amplitude + torsion fraction:**

| f (Hz) | head | web | foot | torsion frac | identity |
|---|---|---|---|---|---|
| 425 | 1.00 | 0.78 | 0.45 | 0.07 | **bending** — whole section moves |
| 430 | 1.00 | 0.78 | 0.45 | 0.08 | **bending** |
| 724 | 0.59 | **0.02** | 1.00 | 0.91 | **torsion** — web is a lateral node |
| 744 | 0.57 | **0.03** | 1.00 | 0.91 | **torsion** — web node |

Exactly the node structure the hypothesis predicts: the torsion branch has the
web as a lateral node (head and foot in antiphase), the bending branch moves the
whole section.

---

## 7. Hypothesis test — what the tools actually say

The hypothesis: the two peaks are a torsion↔lateral-bending **avoided crossing**,
the bending branch folding at the zone edge (pinned-pinned) and meeting the
forward torsional branch, support-induced coupling splitting them into two peaks
(lower = torsional with a web node, upper = bending with all of head/web/foot
moving and a node at the support).

What the 3D brick tools show:
- The **ingredients exist and behave as described**: a lateral-bending branch
  that **folds to a pinned-pinned standing wave at the zone edge** with a node at
  the support (support-stiffness-independent), and a **torsional/flapping branch**
  whose frequency **is set by the lateral support stiffness** (the claimed
  calibration knob).
- **But in this section they do not meet near 525/557**: bending folds at
  ~425 Hz, torsion sits at ~710–750 Hz, so there is no bare avoided crossing in
  the 500–560 Hz window. The feature in that window appears only when the
  **plate mass (V1)** is present, as a local-resonance hybridisation at the
  556 Hz plate frequency — the r1 mechanism.
- **Therefore, as modelled here, the measured feature is better explained by the
  plate local resonance (V1) than by a bare torsion-bending avoided crossing
  (V2).** The hypothesis is not confirmed at the measured frequencies in this
  model; whether a *more accurate (fillet-resolved) section* would lower the
  torsion branch enough to create a genuine crossing in-band is the open
  question, and is exactly what the section-offset caveat (§1, §8) is about.

---

## 8. Rigid mass vs meshed plate

The baseplate is modelled as a **rigid mass** (2 translational dofs + springs).
Justification (computed in `validate.py`): the plate's first **elastic** bending
mode is ≈ **3830 Hz** (clamped and free-free thin-plate estimates for the
~0.22 m × 22 mm SGI plate), far above the 0–1.6 kHz band, whereas the resonance
that matters for the mobility is the **rigid-body** resonance of the plate on its
springs, **556 Hz**. Since the plate does not flex within the band, the
rigid-mass model is adequate; a meshed plate would only add out-of-band internal
modes. **Conclusion: a rigid mass is sufficient here; a meshed plate is not
needed for the 0–1.6 kHz mobility.**

---

## 9. Limitations and how to close the gap

- **Section fillet detail** is the dominant source of the absolute-frequency
  offset (the torsion/flapping branch is ~150 Hz high vs the r1 SAFE section).
  The staircase captures A/Ix/Iy to a few % but not the web/head-fillet
  stiffness that sets the torsional branch. Fix: a fillet-resolved cross-section
  (true 60E1 radii) or calibration to the measured curve.
- **Measured target not fabricated.** Per the brief, the measured point-mobility
  curve (peak frequencies *and* shape) is the right calibration target and would
  be requested as a figure; more than one (section detail, lateral stiffness)
  combination can hit 525/557, so the full curve shape — not just the two peaks —
  must be matched.
- Scope was held to exactly the two tools + shared mesh + validation; no GUI, no
  alternative solvers.

---

## 10. Files

`rail_cell.py` (shared mesh + supports + section verify), `bloch_dispersion.py`
(Tool A), `transient_fullwave.py` (Tool B), `validate.py` (suite), `README.md`,
`requirements.txt`, `figures/` (band diagrams, mobilities, signatures,
space-time, validation overlays, `validation_report.txt`).
