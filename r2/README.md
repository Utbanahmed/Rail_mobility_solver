# r2: in-house 3D finite-element tools for the periodically supported UIC60 rail

Second iteration. Where `r1/` used a semi-analytical (SAFE) cross-section, `r2/`
builds **full 3D brick** tools from scratch (numpy + scipy.sparse only): a shared
mesh module, a Floquet-Bloch dispersion solver, and an explicit time-domain
full-wave solver with non-reflecting ends. The goal is to *test* (not assume) the
hypothesis that the two measured lateral point-mobility peaks (~525 / 557 Hz)
come from an avoided crossing between a torsional/roll cross-section mode and a
lateral-bending (pinned-pinned) mode of the periodically supported rail.

Full technical account, results and honest limitations: **`REPORT.md`**.

## Files

- **`rail_cell.py`** — shared mesh module used by *both* tools. Auto-builds the
  UIC60 / 60E1 cross-section from published nominal dimensions (no imported
  mesh), quad-meshes it with **axis-aligned** elements, extrudes into 8-node
  bricks (incompatible-modes "C3D8I" to defeat bending shear-locking), assembles
  K / M, and attaches the discrete Delkor supports in two variants. Self-verifies
  area / Ix / Iy against 60E1. Run it directly for the section + assembly sanity.
- **`bloch_dispersion.py`** — **Tool A**. Floquet-Bloch band diagram of one unit
  cell; branch tracking + torsion-vs-bending classification; avoided-crossing
  detection; lateral-stiffness sweep; V1/V2 variants.
- **`transient_fullwave.py`** — **Tool B**. Explicit central-difference time
  marching `M u'' + C u' + K u = f(t)` on the 3D brick domain with ALID
  absorbing layers at each end. Point + transfer mobilities by FFT of the
  simulated response; space-time field and its 2D-FFT dispersion.
- **`validate.py`** — runs the validation suite (section, Bloch-vs-transient
  dispersion overlay, two peaks + L-shift, N_periods invariance + no reflection,
  modal signatures, V1/V2 contrast, rigid-vs-meshed plate check) and writes
  `figures/validation_report.txt`.
- `requirements.txt`, `figures/`.

## Coordinates & units

SI throughout (m, kg, s, N, Pa). **X** lateral, **Y** vertical, **Z** axial
(wave propagation). The drive point is a lateral (X) force at a rail-head node.

## Running

```
pip install -r requirements.txt
python rail_cell.py            # section verification + assembly sanity (fast)
python bloch_dispersion.py     # Tool A band diagrams + lateral-stiffness sweep (~min)
python transient_fullwave.py   # Tool B mobility / signatures / space-time (~min)
python validate.py             # full validation suite (several transient runs)
```

All tunable parameters live in a `CONFIG` / `Cfg` block at the top of each file.

## Support model & sourcing (Delkor HAF)

Per direction, SI. Vertical rail-pad `k_top = 1.0e8 N/m` (100 kN/mm, mid of the
80–125 kN/mm EN fastening-pad range); resilient baseplate element
`k_bot = 1.0e7 N/m` (10 kN/mm, typical of a high-attenuation direct-fixation
baseplate). Lateral stiffness is unknown and is the **primary calibration
parameter** (`s_lat`, swept 0.5×–2× vertical). Plate mass `m_plate = 9 kg`.
See `REPORT.md` and the comments in `rail_cell.py` for the full sourcing note.
