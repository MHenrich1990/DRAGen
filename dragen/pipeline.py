"""Generation pipeline: grain sampling -> RSA -> tessellation -> repair.

Bands, inclusions and solver export follow in later phases (G2, E1).
"""
import time
from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from dragen.config.rve import RveConfig
from dragen.core.domain import VoxelDomain
from dragen.core.grains import GrainTable
from dragen.core.kernel import build_kernel
from dragen.sources import open_source
from dragen.stages.repair import merge_detached_pieces
from dragen.stages.rsa import RsaPlacer
from dragen.stages.tessellation import TessellationStats, Tessellator


@dataclass
class RveResult:
    labels: np.ndarray
    """Grain ID per voxel; grain i of `grains` has ID i + 1."""
    grains: GrainTable
    domain: VoxelDomain
    target_voxels: np.ndarray
    """Discrete target volume per grain (voxels of the unshrunk ellipsoid)."""
    seconds: Dict[str, float] = field(default_factory=dict)
    tessellation: TessellationStats = None
    rsa_attempts: int = 0
    repaired_voxels: int = 0


def sample_grains(config: RveConfig, domain: VoxelDomain, rng: np.random.Generator) -> GrainTable:
    tables = []
    for phase in config.phases:
        source = open_source(phase.source, phase.phase, ndim=domain.ndim)
        tables.append(source.sample(config.box.volume * phase.fraction, rng,
                                    min_radius=min(domain.spacing), max_radius=min(domain.size) / 2))
    grains = GrainTable.concat(tables)
    # largest first, as in the legacy RSA; grain i becomes label i + 1
    return grains.take(np.argsort(-grains.volume, kind='stable'))


def growth_epochs(grains: GrainTable, shrink: float, bin_size: float, margin: float):
    """Per-axis relative growth rate and the epoch at which every kernel reaches margin * radii.

    The legacy growth law scales every grain by the same factor per epoch and axis, so one epoch
    limit covers all kernels.
    """
    r0_max = grains.radii.max(axis=0) * shrink
    rate = bin_size / r0_max
    max_epoch = int(np.ceil(((margin / shrink) - 1) / rate.min()))
    return rate, max_epoch


def generate(config: RveConfig) -> RveResult:
    rng = np.random.default_rng(config.seed)
    domain = VoxelDomain.from_box(config.box)
    seconds = {}

    t = time.perf_counter()
    grains = sample_grains(config, domain, rng)
    seconds['sampling'] = time.perf_counter() - t

    t = time.perf_counter()
    shrink = config.rsa.shrink_factor
    rate, max_epoch = growth_epochs(grains, shrink, bin_size=min(domain.spacing),
                                     margin=config.tessellation.kernel_margin)
    kernels = [build_kernel(r, a, domain.spacing, shrink, rate, max_epoch)
               for r, a in zip(grains.radii, grains.alpha)]
    seconds['kernels'] = time.perf_counter() - t

    labels = domain.new_labels()
    t = time.perf_counter()
    placer = RsaPlacer(domain, config.rsa, rng)
    centers = placer.place(labels, kernels)
    seconds['rsa'] = time.perf_counter() - t

    t = time.perf_counter()
    tessellator = Tessellator(domain, config.tessellation, rng)
    tessellator.run(labels, kernels, centers)
    seconds['tessellation'] = time.perf_counter() - t

    t = time.perf_counter()
    repaired = merge_detached_pieces(labels, domain)
    seconds['repair'] = time.perf_counter() - t

    return RveResult(labels=labels, grains=grains.with_centers(centers), domain=domain,
                     target_voxels=np.array([k.target_voxels for k in kernels]), seconds=seconds,
                     tessellation=tessellator.stats, rsa_attempts=placer.attempts,
                     repaired_voxels=repaired)
