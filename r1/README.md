# r1: lateral dispersion of the Delkor-supported UIC 60 rail

First working iteration. Full technical account is in `REPORT.md`; this is the
map and run instructions.

## Result in one line

The measured lateral driving-point peaks at 523.5 and 557 Hz are a
local-resonance avoided crossing: the 9 kg Delkor plate resonates at 556.4 Hz
on its lateral springs and hybridises with a rail cross-sectional "flapping"
branch. Reproduced on a 3D SAFE section (the 6-DOF beam model cannot, because
its 540 Hz waves are pure torsion with no foot-lateral motion to couple to the
foot-mounted plate).

## Files

Trusted / primary:
- `safe_supercell.py`  cross-section operators, SAFE modal ROM (reproduces SAFE
  dispersion to 0.1 Hz), Bloch supercell (explicit plate and condensed),
  rigorous discrete-support dispersion `tm_bloch_condensed`.
- `safe_mobility.py`   infinite-rail driving-point lateral mobility (reduced
  model, plate condensed, absorbing sponge).
- `safe_rail.py`       validated UIC 60 SAFE cross-section (properties <1%).

Validation / cross-checks:
- `rail_lateral_dispersion_final.py`  6-DOF beam operator, validated vs
  Wu & Thompson (1999/2000).
- `rail_periodic.py`   beam FE-Floquet vs transfer-matrix dispersion check.
- `rail_mobility.py`   beam-model driving-point mobility (Wu & Thompson track C).

Docs / outputs:
- `REPORT.md`          full results, validations, and honest limitations.
- `HANDOVER_original.md`  the inherited handover this iteration started from.
- `figures/dispersion_comparison.png`, `figures/mobility.png`.

## Running

```
pip install -r requirements.txt
python safe_supercell.py     # cross-section + SAFE dispersion validation
python safe_mobility.py      # driving-point lateral mobility (450-650 Hz)
```

Note: `safe_mobility.py` and the supercell solvers write no files by default;
they print to stdout. The figures in `figures/` were produced by the analysis
driver described in `REPORT.md`. Output paths in the inherited beam scripts
point at a scratch directory and should be edited before use.

## Key numbers

- Plate lateral resonance (rail fixed): `sqrt((k_top+k_bot)/m)/2pi`
  = `sqrt((1e8+1e7)/9)/2pi` = 556.4 Hz (measured upper peak 557).
- SAFE flapping branch: 511-517 Hz (k=3.64) to 561-567 Hz (k=3.92), converging
  downward with mesh refinement toward the measured band.
- Model mobility peaks: 531 and 570 Hz (refined mesh) vs measured 523.5 / 557.
  Width matches (~35 vs 33.5 Hz); position 7-13 Hz high, mesh/section dependent.

## Next iteration (r2) candidates

- Calibrate the section torsional stiffness and the roll/vertical support
  coupling against the COMSOL lateral dispersion in 400-700 Hz.
- Use the true UIC 60 fillet geometry rather than the simplified polygon.
