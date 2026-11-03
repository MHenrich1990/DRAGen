# Changelog

## 0.2.0 (branch `mst-integration`, based on `rework`)

### Added
- `pyproject.toml`: package `dragen` (with `InputGenerator`), Python >= 3.12. Lean core
  (numpy, scipy, pandas, pyyaml, gmsh, meshio); extras `ml`, `damask`, `extra`, `gui`, `test`.
- `dragen.api`: headless library interface `generate_rve(RVESpec) -> RVEResult` over the reworked
  generator and mesher. Metres, axes x = RD / y = TD / z = ND, grain IDs from 1, orientations as
  unit quaternions (w, x, y, z) crystal -> sample, optional tetrahedral mesh with periodic node
  pairs, metadata (version, git commit, seed, SHA-256 of spec and inputs, run times, warnings).
  Optional per-phase orientation file with measured Bunge angles.
- `dragen.export`: `export_damask` (grid.vti + material.yaml, reads back with
  `damask.GeomGrid.load`), `export_vtu`, `export_gmsh` (MSH 2.2, physical group = grain),
  `export_exodus` (one element block per grain, block ID = grain ID).
- `tests/api`: tests of interface and exports (about 30 s).
- README section "Use as a library".

### Changed
- Missing optional packages raise an ImportError that names the extra.
- The GAN source loads the GAN code with matplotlib on Agg and routes its prints to `logging`.
- Legacy tests that need `pyvista` are skipped with a reason when the extra `gui` is missing.

### Fixed
- numpy 2: the temporary MSH file for gmsh contained `np.float64(...)` in the bounding boxes, so
  every tetrahedral meshing failed.
- pandas 3: the legacy GAN reader (`HelperFunctions.read_input_gan`) used positional `Series[int]`
  access and chained assignment (which no longer writes through).
