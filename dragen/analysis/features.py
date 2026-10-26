"""Metrics for bands and inclusions."""
from typing import List

import numpy as np

from dragen.config.phases import Phase
from dragen.core.domain import VoxelDomain
from dragen.core.grains import GrainTable


def band_report(labels: np.ndarray, grains: GrainTable, bands: List, domain: VoxelDomain) -> List[dict]:
    """Per band: how much of its slab the band grains fill, and how much of them lies outside.

    filling: band-grain voxels inside the slab / slab voxels (the band filling of the paper).
    outside_fraction: band-grain voxels outside the slab / all voxels of this band's grains.
    volume_ratio: all voxels of this band's grains / their discrete target (filling * slab).
    """
    flat = labels.ravel()
    band_of_label = np.concatenate([[-1], grains.band])
    owner = np.where(flat > 0, band_of_label[np.clip(flat, 0, len(grains))], -1)
    out = []
    for k, band in enumerate(bands):
        slab = band.mask(domain)
        mine = owner == k
        inside = np.count_nonzero(mine & slab)
        total = np.count_nonzero(mine)
        out.append({'band': k, 'center': band.center, 'thickness': band.thickness,
                    'slab_voxels': int(slab.sum()), 'filling': inside / slab.sum(),
                    'outside_fraction': (total - inside) / total if total else 0.0,
                    'grains': int(np.count_nonzero(grains.band == k))})
    return out


def inclusion_report(labels: np.ndarray, grains: GrainTable, domain: VoxelDomain, requested: int = None) -> dict:
    """Placed inclusions, their volume fraction, and whether any touches a grain boundary.

    requested: number of sampled inclusions (placed + not placed); defaults to the placed ones.
    expected_placed_fraction: continuous ellipsoid volume of the placed inclusions / box volume.

    touching_boundary counts inclusions whose face neighbours belong to more than one other label
    (they would sit on a grain boundary) or to another inclusion.
    """
    flat = labels.ravel()
    ids = np.flatnonzero(grains.phase == Phase.INCLUSIONS) + 1
    if ids.size == 0:
        return {'inclusions': int(requested or 0), 'placed': 0, 'volume_fraction': 0.0,
                'expected_placed_fraction': 0.0, 'touching_boundary': 0}
    counts = np.bincount(flat[flat > 0], minlength=ids.max() + 1)[ids]
    placed = ids[counts > 0]
    touching = 0
    for i in placed:
        vox = np.flatnonzero(flat == i)
        nb = domain.face_neighbors(vox).ravel()
        nb = flat[nb[nb >= 0]]
        others = np.unique(nb[nb != i])
        if len(others) != 1 or np.isin(others, ids).any():
            touching += 1
    expected = grains.volume[placed - 1].sum() / (flat.size * domain.voxel_volume)
    return {'inclusions': int(requested if requested is not None else ids.size), 'placed': int(placed.size),
            'volume_fraction': float(counts.sum() / flat.size), 'expected_placed_fraction': float(expected),
            'touching_boundary': touching}
