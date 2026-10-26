"""Generation pipeline: grain sampling -> RSA (bands first) -> tessellation -> repair -> inclusions.

Solver export follows in a later phase (E1).

Volume bookkeeping: band grains hold `filling` times the volume of their band slab; the matrix
phases share the rest of the box according to their fractions. Inclusions are cut out of the
grains after tessellation, their volume fraction refers to the whole box.
"""
import time
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from dragen.config.phases import Phase
from dragen.config.rve import RveConfig
from dragen.core.domain import VoxelDomain
from dragen.core.grains import GrainTable
from dragen.core.kernel import build_kernel
from dragen.sources import open_source
from dragen.stages.bands import Band, place_bands
from dragen.stages.inclusions import InclusionPlacer, InclusionStats
from dragen.stages.repair import merge_detached_pieces
from dragen.stages.rsa import RsaPlacer
from dragen.stages.tessellation import TessellationStats, Tessellator


@dataclass
class RveResult:
    labels: np.ndarray
    """Grain ID per voxel; grain i of `grains` has ID i + 1. Band grains come first, then the
    matrix grains, then the inclusions (phase INCLUSIONS)."""
    grains: GrainTable
    domain: VoxelDomain
    target_voxels: np.ndarray
    """Discrete target volume per grain (voxels of the unshrunk ellipsoid)."""
    bands: List[Band] = field(default_factory=list)
    seconds: Dict[str, float] = field(default_factory=dict)
    tessellation: TessellationStats = None
    inclusions: InclusionStats = None
    rsa_attempts: int = 0
    repaired_voxels: int = 0


def _by_volume(grains: GrainTable) -> GrainTable:
    """Largest first, as in the legacy RSA."""
    return grains.take(np.argsort(-grains.volume, kind='stable'))


def _source_limits(domain: VoxelDomain) -> dict:
    return dict(min_radius=min(domain.spacing), max_radius=min(domain.size) / 2)


def sample_band_grains(config: RveConfig, domain: VoxelDomain, bands: List[Band],
                       rng: np.random.Generator) -> List[GrainTable]:
    if not bands:
        return []
    source = open_source(config.bands.source, config.bands.phase, ndim=domain.ndim)
    tables = []
    for k, band in enumerate(bands):
        slab_volume = np.count_nonzero(band.mask(domain)) * domain.voxel_volume
        table = source.sample(config.bands.filling * slab_volume, rng, **_source_limits(domain),
                              max_extent=(band.axis, band.thickness))
        tables.append(_by_volume(table).with_band(k))
    return tables


def sample_grains(config: RveConfig, domain: VoxelDomain, rng: np.random.Generator,
                  band_volume: float = 0.0) -> GrainTable:
    matrix_volume = config.box.volume - band_volume
    tables = []
    for phase in config.phases:
        source = open_source(phase.source, phase.phase, ndim=domain.ndim)
        tables.append(source.sample(matrix_volume * phase.fraction, rng, **_source_limits(domain)))
    return _by_volume(GrainTable.concat(tables))


def growth_rates(grains: GrainTable, shrink: float, bin_size: float, margin: float, exponent: float):
    """Relative growth rate per grain and axis, and per grain the epoch at which its kernel
    reaches margin * radii.

    rate[g, i] = bin_size / (R_i ** (1 - exponent) * r_g ** exponent)

    R_i is the largest shrunk radius along axis i over all grains, r_g the largest shrunk radius
    of grain g. exponent = 0 is the legacy law (Henrich et al. 2020): every grain grows by the
    same factor per epoch, so small grains advance their front by a fraction of a voxel per epoch.
    exponent = 1 moves the front of every grain by about one voxel per epoch along its longest
    axis, keeping its shape.
    """
    r0 = grains.radii * shrink
    axis_max = r0.max(axis=0)
    grain_max = r0.max(axis=1, keepdims=True)
    rate = bin_size / (axis_max ** (1 - exponent) * grain_max ** exponent)
    max_epoch = np.ceil(((margin / shrink) - 1) / rate.min(axis=1)).astype(int)
    return rate, max_epoch


def generate(config: RveConfig) -> RveResult:
    rng = np.random.default_rng(config.seed)
    domain = VoxelDomain.from_box(config.box)
    seconds = {}

    t = time.perf_counter()
    bands = place_bands(config.bands, domain, rng) if config.bands.count else []
    band_tables = sample_band_grains(config, domain, bands, rng)
    band_volume = config.bands.filling * sum(np.count_nonzero(b.mask(domain)) for b in bands) * domain.voxel_volume
    matrix = sample_grains(config, domain, rng, band_volume=band_volume)
    grains = GrainTable.concat(band_tables + [matrix])
    seconds['sampling'] = time.perf_counter() - t

    t = time.perf_counter()
    shrink = config.rsa.shrink_factor
    rate, max_epoch = growth_rates(grains, shrink, bin_size=min(domain.spacing),
                                   margin=config.tessellation.kernel_margin,
                                   exponent=config.tessellation.growth_exponent)
    kernels = [build_kernel(r, a, domain.spacing, shrink, g, k)
               for r, a, g, k in zip(grains.radii, grains.alpha, rate, max_epoch)]
    seconds['kernels'] = time.perf_counter() - t

    labels = domain.new_labels()
    t = time.perf_counter()
    placer = RsaPlacer(domain, config.rsa, rng)
    centers, start = [], 0
    for band, table in zip(bands, band_tables):
        centers.append(placer.place(labels, kernels[start:start + len(table)], first_id=start + 1,
                                    region=band.mask(domain)))
        start += len(table)
    centers.append(placer.place(labels, kernels[start:], first_id=start + 1))
    centers = np.concatenate(centers)
    seconds['rsa'] = time.perf_counter() - t

    t = time.perf_counter()
    tessellator = Tessellator(domain, config.tessellation, rng)
    tessellator.run(labels, kernels, centers)
    seconds['tessellation'] = time.perf_counter() - t

    t = time.perf_counter()
    repaired = merge_detached_pieces(labels, domain)
    seconds['repair'] = time.perf_counter() - t

    grains = grains.with_centers(centers)
    targets = [k.target_voxels for k in kernels]
    inclusion_stats = None
    if config.inclusions.fraction > 0:
        t = time.perf_counter()
        source = open_source(config.inclusions.source, Phase.INCLUSIONS, ndim=domain.ndim)
        inclusions = _by_volume(source.sample(config.inclusions.fraction * config.box.volume, rng,
                                              **_source_limits(domain)))
        inclusion_kernels = [build_kernel(r, a, domain.spacing, 1.0, np.zeros(domain.ndim), 0)
                             for r, a in zip(inclusions.radii, inclusions.alpha)]
        inclusion_placer = InclusionPlacer(domain, rng)
        placed, inclusion_centers = inclusion_placer.place(labels, inclusion_kernels, first_id=len(grains) + 1)
        grains = GrainTable.concat([grains, inclusions.take(placed).with_centers(inclusion_centers)])
        targets += [inclusion_kernels[i].target_voxels for i in placed]
        inclusion_stats = inclusion_placer.stats
        seconds['inclusions'] = time.perf_counter() - t

    return RveResult(labels=labels, grains=grains, domain=domain, target_voxels=np.array(targets),
                     bands=bands, seconds=seconds, tessellation=tessellator.stats,
                     inclusions=inclusion_stats, rsa_attempts=placer.attempts, repaired_voxels=repaired)


def expected_phase_fractions(config: RveConfig, bands: List[Band], domain: VoxelDomain) -> Dict[int, float]:
    """Volume fraction each phase should end up with, given the band and inclusion settings."""
    band_fraction = config.bands.filling * sum(np.count_nonzero(b.mask(domain)) for b in bands) / domain.n_voxels
    keep = 1.0 - config.inclusions.fraction
    out: Dict[int, float] = {}
    for p in config.phases:
        out[int(p.phase)] = out.get(int(p.phase), 0.0) + (1 - band_fraction) * p.fraction * keep
    if bands:
        out[int(config.bands.phase)] = out.get(int(config.bands.phase), 0.0) + band_fraction * keep
    if config.inclusions.fraction > 0:
        out[int(Phase.INCLUSIONS)] = config.inclusions.fraction
    return out
