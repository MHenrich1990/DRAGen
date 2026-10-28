"""From a voxel label field to a C3D10 mesh with smoothed, adaptively meshed grain boundaries.

    clean-up (M1) -> topology (M1) -> surface model (M2) -> smoothing (M3)
    -> adaptive remeshing of the grain boundaries (M4a) -> gmsh tetrahedra (M4)

Feedback: tetrahedra with a smallest dihedral angle below `min_dihedral` are almost always forced
by the surface (a grain locally thinner than its boundary triangles, or a slightly concave edge
bridged by one flat element; process notes 7.12). The size field of the remeshing is halved around
them (down to h_min), the surface is remeshed further and the volume meshed again, up to
`feedback_rounds` times. What remains is reported.
"""
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from dragen.mesh.cleanup import ManifoldCleaner
from dragen.mesh.model import SurfaceModel, build_model
from dragen.mesh.remeshing import RemeshConfig, Remesher
from dragen.mesh.smoothing import SmoothingConfig, SmoothingResult, smooth
from dragen.mesh.tetmesh import TetMeshConfig, TetMeshResult, tet_mesh
from dragen.mesh.topology import build_topology
from dragen.mesh.volume import tet_dihedral_angles


@dataclass
class MeshConfig:
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    remeshing: Optional[RemeshConfig] = field(default_factory=RemeshConfig)
    """None: keep the smoothed triangles as they are (M4 without M4a)."""
    tetmesh: TetMeshConfig = field(default_factory=TetMeshConfig)
    min_dihedral: float = 5.0
    """Acceptance of the author: smallest dihedral angle of every tetrahedron, degrees."""
    feedback_rounds: int = 3
    feedback_iterations: int = 4


@dataclass
class MeshResult:
    labels: np.ndarray
    """The cleaned label field the mesh was built from."""
    staircase: SurfaceModel
    smoothed: SmoothingResult
    surface: SurfaceModel
    """The final grain boundary model (remeshed, or the staircase model with smoothed points)."""
    points: np.ndarray
    """Points of `surface`."""
    tets: TetMeshResult
    stats: dict = field(default_factory=dict)


def mesh_rve(labels: np.ndarray, spacing: Sequence[float], periodic: Sequence[bool],
             config: MeshConfig = MeshConfig()) -> MeshResult:
    stats = {'seconds': {}}
    seconds = stats['seconds']

    t = time.perf_counter()
    cleaned, cleanup = ManifoldCleaner(periodic).run(labels)
    seconds['cleanup'] = time.perf_counter() - t
    stats['cleanup'] = {'changed_voxels': cleanup.changed_voxels, 'critical_after': cleanup.critical_after}

    t = time.perf_counter()
    staircase = build_model(build_topology(cleaned, spacing, periodic))
    seconds['model'] = time.perf_counter() - t

    t = time.perf_counter()
    smoothed = smooth(staircase, spacing, config.smoothing)
    seconds['smoothing'] = time.perf_counter() - t

    if config.remeshing is None:
        surface, points, remesher = staircase, smoothed.points, None
    else:
        t = time.perf_counter()
        remesher = Remesher(staircase, smoothed.points, cleaned, spacing, smoothed.grain_volume_target,
                            config.remeshing)
        remeshed = remesher.run()
        seconds['remeshing'] = time.perf_counter() - t
        surface, points = remeshed.model, remeshed.points

    stats['feedback'] = []
    seconds['tetmesh'] = 0.0
    for round_ in range(config.feedback_rounds + 1):
        t = time.perf_counter()
        tets = tet_mesh(surface, points, spacing, config.tetmesh)
        seconds['tetmesh'] += time.perf_counter() - t
        dihedral = tet_dihedral_angles(tets.mesh).min(axis=1)
        bad = np.flatnonzero(dihedral < config.min_dihedral)
        stats['feedback'].append({'triangles': int(len(surface.oriented)), 'elements': int(len(dihedral)),
                                  'below_min_dihedral': int(len(bad)), 'min_dihedral': float(dihedral.min())})
        if not len(bad) or remesher is None or round_ == config.feedback_rounds:
            break
        t = time.perf_counter()
        centres = tets.mesh.points[tets.mesh.corners[bad]].mean(axis=1)
        remeshed = remesher.refine_near(centres, config.feedback_iterations)
        seconds['remeshing'] += time.perf_counter() - t
        surface, points = remeshed.model, remeshed.points
    if remesher is not None:
        stats['remeshing'] = remesher.stats
    return MeshResult(labels=cleaned, staircase=staircase, smoothed=smoothed, surface=surface, points=points,
                      tets=tets, stats=stats)
