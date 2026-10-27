"""Baseline of the legacy mesher on the meshing corpus (phase M0).

Runs `MeshingHelper.smoothen_mesh` from dragen/utilities/PvGridGeneration.py, the step that
AbaqusMesher and MooseMesher share, once without smoothing and once with the 200 iterations
AbaqusMesher uses, for C3D8 (smoothed voxel hexes) and C3D4 (tetgen per grain, merged). Both
meshes are judged with dragen.mesh.quality. Every job runs in its own process, so a crash of
tetgen on a special case is recorded instead of ending the benchmark.

Example:
    venv/Scripts/python benchmarks/mesh_baseline.py                      # all cases, both elements
    venv/Scripts/python benchmarks/mesh_baseline.py --case torus --element C3D8

Results: benchmarks/results/mesh/legacy/<case>_<element>/result.json, plus summary.csv over all
jobs. Lengths in the legacy mesh are in mm (box size / 1000); the reference volumes are converted
accordingly.
"""
import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from rework_benchmark import code_state  # noqa: E402
from dragen.mesh import corpus, quality  # noqa: E402
from dragen.mesh.surface import corner_grain_counts  # noqa: E402
from dragen.mesh.volume import VolumeMesh, grain_boundary_mesh  # noqa: E402

RESULT_DIR = os.path.join(REPO_ROOT, 'benchmarks', 'results', 'mesh', 'legacy')
ELEMENTS = ('C3D8', 'C3D4')
GENERATED = {'rve_box10_res2_seed1': (10, 2, 1), 'rve_box15_res2_seed1': (15, 2, 1)}
VTK_VOXEL, VTK_HEXAHEDRON, VTK_TETRA = 11, 12, 10
TOL = 1e-6  # relative to the box; pyvista stores points as float32


def load_case(name):
    """Label field, voxel edge in µm and periodicity of a corpus case or generated RVE."""
    if name in GENERATED:
        from legacy_baseline import lognormal_input
        from dragen.config import BoxConfig, PhaseConfig, RveConfig
        from dragen.pipeline import generate
        box, resolution, seed = GENERATED[name]
        result = generate(RveConfig(box=BoxConfig((box,) * 3, resolution),
                                    phases=(PhaseConfig('ferrite', lognormal_input(), 1.0),), seed=seed))
        return result.labels, 1.0 / resolution, (True, True, True)
    case = {c.name: c for c in corpus.cases()}[name]
    return case.labels, 1.0, case.periodic


def legacy_mesh(labels, spacing, element, smoothing):
    """Run the legacy smoothing step; returns the pyvista grid (lengths in mm)."""
    import pandas as pd
    from dragen.utilities.InputInfo import RveInfo
    from dragen.utilities.PvGridGeneration import MeshingHelper

    nx, ny, nz = labels.shape
    RveInfo.box_size, RveInfo.box_size_y, RveInfo.box_size_z = nx * spacing, ny * spacing, nz * spacing
    RveInfo.element_type = element
    RveInfo.smoothing_flag = smoothing
    RveInfo.gui_flag = False
    RveInfo.anim_flag = False
    os.makedirs(os.path.join(RveInfo.store_path, 'Generation_Data'), exist_ok=True)
    x, y, z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    rve = pd.DataFrame({'x': x.ravel(), 'y': y.ravel(), 'z': z.ravel(), 'GrainID': labels.ravel(),
                        'phaseID': 1, 'box_size': nx * spacing})
    grains = pd.DataFrame({'GrainID': np.arange(1, labels.max() + 1), 'phaseID': 1})
    helper = MeshingHelper(rve_shape=labels.shape, rve=rve, grains_df=grains)
    with contextlib.redirect_stdout(io.StringIO()):
        grid = helper.gen_grains(helper.gen_blocks())
        return helper.smoothen_mesh(grid, n_iter=200)


def to_volume_mesh(grid) -> VolumeMesh:
    types = np.unique(grid.celltypes)
    if len(types) != 1:
        raise ValueError(f'mixed cell types {types}')
    n = {VTK_VOXEL: 8, VTK_HEXAHEDRON: 8, VTK_TETRA: 4}[int(types[0])]
    cells = np.asarray(grid.cells).reshape(-1, n + 1)[:, 1:]
    if types[0] == VTK_VOXEL:
        cells = cells[:, [0, 1, 3, 2, 4, 5, 7, 6]]
    cell_type = 'tet4' if n == 4 else 'hex8'
    return VolumeMesh(np.asarray(grid.points, float), cells, cell_type,
                      np.asarray(grid.cell_data['GrainID']).astype(np.int64))


def anchors(labels, periodic, points, spacing_mm):
    """Indices of mesh nodes that sit on voxel corners touching 3 and >= 4 grains."""
    lattice = points / spacing_mm
    on = np.all(np.abs(lattice - np.round(lattice)) < 1e-4, axis=1)
    idx = np.round(lattice[on]).astype(int)
    counts = corner_grain_counts(labels, periodic)[tuple(idx.T)]
    nodes = np.flatnonzero(on)
    return nodes[counts == 3], nodes[counts >= 4]


def run_job(case, element, out_dir):
    from dragen.utilities.InputInfo import RveInfo
    labels, spacing, periodic = load_case(case)
    RveInfo.store_path = out_dir
    spacing_mm = spacing / 1000
    upper = np.array(labels.shape) * spacing_mm
    counts = np.bincount(labels.ravel()) * spacing_mm ** 3
    reference = {g: counts[g] for g in range(1, len(counts))}
    out = {'case': case, 'element': element, 'code': code_state(), 'shape': list(labels.shape),
           'spacing_um': spacing, 'periodic': list(periodic), 'n_grains': int(labels.max()), 'seconds': {}}

    t = time.perf_counter()
    before = to_volume_mesh(legacy_mesh(labels, spacing, element, smoothing=False))
    out['seconds']['unsmoothed'] = time.perf_counter() - t
    t = time.perf_counter()
    after = to_volume_mesh(legacy_mesh(labels, spacing, element, smoothing=True))
    out['seconds']['smoothed'] = time.perf_counter() - t

    for tag, mesh in (('unsmoothed', before), ('smoothed', after)):
        axes = [a for a in range(3) if periodic[a]]
        rep = {'volume': quality.volume_report(mesh, (0, 0, 0), upper, reference_volume=reference, tol=TOL,
                                               periodic_axes=axes)}
        boundary = grain_boundary_mesh(mesh, (0, 0, 0), upper, tol=TOL)
        rep['surface'] = quality.interface_report(boundary, reference_volume=reference, intersections=True)
        box_tris = boundary.triangles[(boundary.grains < 0).any(axis=1)]
        rep['periodic'] = quality.periodic_mismatch(boundary.points, box_tris, (0, 0, 0), upper,
                                                    axes=axes, tol=TOL)
        out[tag] = rep

    if len(before.points) == len(after.points):
        triple, quad = anchors(labels, periodic, before.points, spacing_mm)
        out['smoothed']['triple_line_nodes'] = quality.anchor_displacement(before.points, after.points, triple)
        out['smoothed']['quadruple_nodes'] = quality.anchor_displacement(before.points, after.points, quad)
        for d in (out['smoothed']['triple_line_nodes'], out['smoothed']['quadruple_nodes']):
            d['max_in_voxels'] = d['max'] / spacing_mm
    else:
        out['smoothed']['anchors'] = 'node count differs between unsmoothed and smoothed mesh'
    return out


def summary_row(result):
    row = {'case': result['case'], 'element': result['element']}
    if 'error' in result:
        row['error'] = result['error'][:120]
        return row
    for tag in ('unsmoothed', 'smoothed'):
        v, s = result[tag]['volume'], result[tag]['surface']
        row.update({
            f'{tag}_elements': v['n_elements'],
            f'{tag}_inverted': v['inverted_elements'],
            f'{tag}_open_inner_faces': v['open_inner_faces'],
            f'{tag}_split_grains': v['grains_with_several_bodies'],
            f'{tag}_vol_dev_max': v['volume_deviation'].get('max_abs'),
            f'{tag}_self_intersections': s['self_intersections'],
            f'{tag}_nonmanifold_edges': s['nonmanifold_edges'],
            f'{tag}_periodic_unmatched_nodes': sum(p['unmatched_nodes'] for p in result[tag]['periodic'].values()),
        })
    q = result['smoothed'].get('quadruple_nodes')
    if q:
        row['quad_nodes_moved'] = f"{q['moved']}/{q['n']}"
        row['quad_max_move_voxels'] = q['max_in_voxels']
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', nargs='*', default=None)
    parser.add_argument('--element', nargs='*', default=list(ELEMENTS))
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    names = args.case or [c.name for c in corpus.cases()] + list(GENERATED)

    if args.worker:
        case, element = names[0], args.element[0]
        out_dir = os.path.join(RESULT_DIR, f'{case}_{element}')
        os.makedirs(out_dir, exist_ok=True)
        result = run_job(case, element, out_dir)
        with open(os.path.join(out_dir, 'result.json'), 'w') as f:
            json.dump(result, f, indent=2)
        return

    rows = []
    for case in names:
        for element in args.element:
            out_dir = os.path.join(RESULT_DIR, f'{case}_{element}')
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, 'result.json')
            if os.path.exists(path):
                os.remove(path)
            proc = subprocess.run([sys.executable, __file__, '--worker', '--case', case, '--element', element],
                                  capture_output=True, text=True)
            if proc.returncode == 0 and os.path.exists(path):
                with open(path) as f:
                    result = json.load(f)
            else:
                tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or [f'exit code {proc.returncode}']
                result = {'case': case, 'element': element, 'error': tail[0],
                          'returncode': proc.returncode}
                with open(path, 'w') as f:
                    json.dump(result, f, indent=2)
            rows.append(summary_row(result))
            print(rows[-1], flush=True)
    if args.case is None:
        import pandas as pd
        pd.DataFrame(rows).to_csv(os.path.join(RESULT_DIR, 'summary.csv'), index=False)


if __name__ == '__main__':
    main()
