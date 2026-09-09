# Rail mobility solver

Lateral dispersion and driving-point mobility of a periodically supported
UIC 60 rail carrying the Delkor Egg (RCTMD) fastening.

## Layout

Work is organised into numbered iteration folders so each round is
self-contained and reproducible:

- **`r1/`** first working iteration. 3D SAFE Floquet-Bloch supercell of the
  discretely-supported rail, driving-point lateral mobility, and the diagnosis
  of why the earlier 6-DOF beam model could not reproduce the measured feature.
  See `r1/README.md`.
- `r2/`, ... later iterations (e.g. calibrated against COMSOL dispersion,
  roll/vertical support coupling), added as the work progresses.

Each `rN/` carries its own code, figures, and a report, so results are never
overwritten across iterations.
