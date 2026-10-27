"""Clean-up, topology (M1) and surface model (M2) on the RVEs of the reworked generator.

Reads every benchmarks/results/rework/<tag>/labels.npz (written by rework_benchmark.py), removes
edge and point self-contacts (dragen.mesh.cleanup), builds the topology (dragen.mesh.topology)
and the discrete surface model (dragen.mesh.model) and checks them: every grain surface a closed
2-manifold, every grain still one body, Euler check equal to 1, no anchor inside a patch, and the
model checks (every curve used, closed shells, consistent orientation, exact periodic links).

Example:
    venv/Scripts/python benchmarks/topology_benchmark.py                       # all stored RVEs
    venv/Scripts/python benchmarks/topology_benchmark.py --tag lognormal_box30_res2_seed1

Results: benchmarks/results/mesh/topology/<tag>/result.json and summary.csv.
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
from dragen.analysis.metrics import components_per_grain  # noqa: E402
from dragen.config import RveConfig  # noqa: E402
from dragen.mesh.cleanup import ManifoldCleaner  # noqa: E402
from dragen.mesh.model import build_model  # noqa: E402
from dragen.mesh.quality import grain_surfaces  # noqa: E402
from dragen.mesh.topology import build_topology  # noqa: E402

SOURCE_DIR = os.path.join(REPO_ROOT, 'benchmarks', 'results', 'rework')
RESULT_DIR = os.path.join(REPO_ROOT, 'benchmarks', 'results', 'mesh', 'topology')


def run(tag: str) -> dict:
    src = os.path.join(SOURCE_DIR, tag)
    labels = np.load(os.path.join(src, 'labels.npz'))['labels']
    box = RveConfig.load(os.path.join(src, 'config.yaml')).box
    periodic, spacing = tuple(box.periodic), tuple(box.spacing)
    out = {'tag': tag, 'code': code_state(), 'shape': list(labels.shape), 'spacing': list(spacing),
           'periodic': list(periodic), 'n_grains': int(labels.max()), 'seconds': {}}

    t = time.perf_counter()
    cleaned, stats = ManifoldCleaner(periodic).run(labels)
    out['seconds']['cleanup'] = time.perf_counter() - t
    counts = np.bincount(labels.ravel())
    rel = [abs(v) / counts[g] for g, v in stats.voxels_gained.items()]
    out['cleanup'] = {'critical_before': stats.critical_before, 'critical_after': stats.critical_after,
                      'changed_voxels': stats.changed_voxels,
                      'changed_fraction': stats.changed_voxels / labels.size,
                      'grains_changed': len(stats.voxels_gained),
                      'max_relative_grain_change': float(max(rel)) if rel else 0.0,
                      'passes': stats.passes}

    t = time.perf_counter()
    topo = build_topology(cleaned, spacing, periodic)
    out['seconds']['topology'] = time.perf_counter() - t
    t = time.perf_counter()
    out['topology'] = topo.summary()
    out['seconds']['checks'] = time.perf_counter() - t
    orders = np.array([ln.order for ln in topo.lines if not ln.imprinted])
    lengths = np.array([len(ln.vertices) - 1 for ln in topo.lines])
    out['topology']['line_order_counts'] = {int(k): int(v) for k, v in zip(*np.unique(orders, return_counts=True))}
    out['topology']['line_edges_median'] = float(np.median(lengths)) if len(lengths) else 0.0
    out['topology']['mesh_triangles'] = int(len(topo.mesh.triangles))

    t = time.perf_counter()
    model = build_model(topo)
    out['seconds']['model'] = time.perf_counter() - t
    t = time.perf_counter()
    out['model'] = model.check()
    out['seconds']['model_checks'] = time.perf_counter() - t

    surfaces = grain_surfaces(topo.mesh)
    out['checks'] = {
        'grains_not_closed_manifold': int(np.count_nonzero(~surfaces['closed_manifold'])),
        'grains_with_several_bodies': int(np.count_nonzero(components_per_grain(cleaned, periodic)[1:] > 1)),
        'grains_lost': int(np.count_nonzero(np.bincount(cleaned.ravel(), minlength=len(counts))[1:] == 0)),
        'euler_check': out['topology']['euler_check'],
        'isolated_anchors': out['topology']['isolated_anchors'],
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', nargs='*', default=None)
    args = parser.parse_args()
    tags = args.tag or sorted(os.path.basename(os.path.dirname(p))
                              for p in glob.glob(os.path.join(SOURCE_DIR, '*', 'labels.npz')))
    rows = []
    for tag in tags:
        result = run(tag)
        out_dir = os.path.join(RESULT_DIR, tag)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'result.json'), 'w') as f:
            json.dump(result, f, indent=2)
        row = {'tag': tag, 'voxels': int(np.prod(result['shape'])), 'grains': result['n_grains']}
        row.update({f'cleanup_{k}': v for k, v in result['cleanup'].items()})
        row.update({k: v for k, v in result['topology'].items() if not isinstance(v, dict)})
        row.update({f'model_{k}': v for k, v in result['model'].items()})
        row.update({f'check_{k}': v for k, v in result['checks'].items()})
        row.update({f'seconds_{k}': v for k, v in result['seconds'].items()})
        rows.append(row)
        print(row, flush=True)
    if args.tag is None:
        import pandas as pd
        pd.DataFrame(rows).to_csv(os.path.join(RESULT_DIR, 'summary.csv'), index=False)


if __name__ == '__main__':
    main()
