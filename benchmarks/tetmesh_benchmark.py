"""Volume meshing (M4) of the RVEs of the reworked generator.

For every benchmarks/results/rework/<tag>/labels.npz: clean-up, topology, surface model and
smoothing as in topology_benchmark.py, then the gmsh tet mesh (dragen.mesh.tetmesh), judged with
dragen.mesh.quality: inverted elements, gaps and overused faces, midside nodes, element quality
(scaled Jacobian, dihedral angles, radius ratio), element volume per grain against the voxel
volume, grain boundary faces of the tet mesh against the smoothed triangles, periodic node pairs
(corner and midside nodes) and facets on opposite box faces.

Example:
    venv/Scripts/python benchmarks/tetmesh_benchmark.py                        # all stored RVEs
    venv/Scripts/python benchmarks/tetmesh_benchmark.py --tag lognormal_box30_res2_seed1 --growth 0.3

Results: benchmarks/results/mesh/tetmesh/<variant>/<tag>/result.json and <variant>/summary.csv;
<variant> is 'default' unless --variant is given.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from rework_benchmark import code_state  # noqa: E402
from dragen.config import RveConfig  # noqa: E402
from dragen.mesh.cleanup import ManifoldCleaner  # noqa: E402
from dragen.mesh.model import build_model  # noqa: E402
from dragen.mesh.smoothing import smooth  # noqa: E402
from dragen.mesh.tetmesh import TetMeshConfig, check, tet_mesh  # noqa: E402
from dragen.mesh.topology import build_topology  # noqa: E402

SOURCE_DIR = os.path.join(REPO_ROOT, 'benchmarks', 'results', 'rework')
RESULT_DIR = os.path.join(REPO_ROOT, 'benchmarks', 'results', 'mesh', 'tetmesh')


def run(tag: str, config: TetMeshConfig) -> dict:
    src = os.path.join(SOURCE_DIR, tag)
    labels = np.load(os.path.join(src, 'labels.npz'))['labels']
    box = RveConfig.load(os.path.join(src, 'config.yaml')).box
    periodic, spacing = tuple(box.periodic), tuple(box.spacing)
    out = {'tag': tag, 'code': code_state(), 'config': vars(config), 'shape': list(labels.shape),
           'spacing': list(spacing), 'periodic': list(periodic), 'n_grains': int(labels.max()),
           'seconds': {}}
    t = time.perf_counter()
    cleaned, _ = ManifoldCleaner(periodic).run(labels)
    model = build_model(build_topology(cleaned, spacing, periodic))
    smoothed = smooth(model, spacing)
    out['seconds']['surface'] = time.perf_counter() - t

    t = time.perf_counter()
    result = tet_mesh(model, smoothed.points, spacing, config)
    out['seconds']['tetmesh'] = time.perf_counter() - t
    out['seconds'].update({f'tetmesh_{k}': v for k, v in result.stats.pop('seconds').items()})
    out['tetmesh'] = result.stats
    t = time.perf_counter()
    out['quality'] = check(model, smoothed.points, result.mesh, labels.shape, spacing, periodic,
                           smoothed.grain_volume_target)
    out['seconds']['checks'] = time.perf_counter() - t
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', nargs='*', default=None)
    parser.add_argument('--variant', default='default')
    parser.add_argument('--growth', type=float, default=TetMeshConfig.growth)
    parser.add_argument('--max-size', type=float, default=None)
    parser.add_argument('--optimize', default=TetMeshConfig.optimize)
    parser.add_argument('--algorithm', type=int, default=TetMeshConfig.algorithm)
    args = parser.parse_args()
    config = TetMeshConfig(growth=args.growth, max_size=args.max_size, optimize=args.optimize,
                           algorithm=args.algorithm)
    tags = args.tag or sorted(os.path.basename(os.path.dirname(p))
                              for p in glob.glob(os.path.join(SOURCE_DIR, '*', 'labels.npz')))
    rows = []
    for tag in tags:
        result = run(tag, config)
        out_dir = os.path.join(RESULT_DIR, args.variant, tag)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'result.json'), 'w') as f:
            json.dump(result, f, indent=2)
        q = result['quality']
        row = {'tag': tag, 'voxels': int(np.prod(result['shape'])), 'grains': result['n_grains'],
               'bodies': result['tetmesh']['bodies'], 'cavities': result['tetmesh']['cavities'],
               'elements': q['n_elements'], 'nodes': q['n_points'], 'elements_per_voxel': q['elements_per_voxel'],
               'inverted': q['inverted_elements'], 'open_inner_faces': q['open_inner_faces'],
               'overused_faces': q['overused_faces'], 'midside_mismatch': q.get('midside_mismatch', 0),
               'several_bodies': q['grains_with_several_bodies'],
               'volume_max_abs': q['volume_deviation']['max_abs'],
               'surface_nodes_moved': q['surface_nodes_moved'],
               'interface_missing': q['interface_faces_missing'], 'interface_extra': q['interface_faces_extra'],
               'periodic_unmatched_nodes': q['periodic_unmatched_nodes'],
               'periodic_unmatched_facets': q['periodic_unmatched_facets'],
               'dihedral_min': q['dihedral_min_deg']['min'], 'dihedral_min_p1': q['dihedral_min_deg']['p1'],
               'dihedral_max': q['dihedral_max_deg']['max'],
               'scaled_jacobian_min': q['scaled_jacobian']['min'], 'scaled_jacobian_p1': q['scaled_jacobian']['p1'],
               'radius_ratio_p1': q['radius_ratio']['p1'],
               'longest_edge_median': q['longest_edge_voxels']['median'],
               'longest_edge_max': q['longest_edge_voxels']['max']}
        row.update({f's_{k}': round(v, 2) for k, v in result['seconds'].items()})
        rows.append(row)
        print(json.dumps(row))
    import pandas as pd
    path = os.path.join(RESULT_DIR, args.variant, 'summary.csv')
    df = pd.DataFrame(rows)
    if os.path.exists(path):
        old = pd.read_csv(path)
        df = pd.concat([old[~old['tag'].isin(df['tag'])], df]).sort_values('tag')
    df.to_csv(path, index=False)


if __name__ == '__main__':
    main()
