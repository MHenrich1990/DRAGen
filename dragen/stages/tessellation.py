"""Discrete tessellation: seeds grow epoch by epoch until the box is full.

Follows the legacy algorithm (Henrich et al. 2020, Algorithm 2): in every epoch the active grains
are shuffled and each one grows its ellipsoid by one step and takes the free voxels inside it.
A grain stops when it reaches its target volume or cannot grow any more. When no grain is active
but voxels are still free, all grains that can still grow are reactivated without volume limit.

Differences to the legacy code:
  * The ellipsoid is not re-evaluated on the whole grid. Each grain walks through its kernel,
    which is sorted by entry epoch, so one epoch costs only the voxels that newly enter.
  * With connected_growth (default) a grain only takes free voxels that are face-adjacent to its
    own body, directly or through other newly taken voxels; voxels behind a neighbour stay pending
    until the grain reaches them. The legacy code took every free voxel in the ellipsoid, which
    split about a third of all grains into several bodies.
  * Voxels no kernel reaches are filled at the end from adjacent grains.
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np

from dragen.config.rve import TessellationConfig
from dragen.core.domain import VoxelDomain
from dragen.core.kernel import EllipsoidKernel


@dataclass
class TessellationStats:
    epochs: int = 0
    reset_epoch: int = -1
    filled_after_rsa: int = 0
    filled_regular: int = 0
    filled_reset: int = 0
    filled_by_dilation: int = 0
    filled_per_epoch: List[int] = field(default_factory=list)


class Tessellator:

    def __init__(self, domain: VoxelDomain, config: TessellationConfig, rng: np.random.Generator):
        self.domain = domain
        self.config = config
        self.rng = rng
        self.stats = TessellationStats()

    def run(self, labels: np.ndarray, kernels: List[EllipsoidKernel], centers: np.ndarray,
            first_id: int = 1) -> np.ndarray:
        flat_labels = labels.reshape(-1)
        n = len(kernels)
        ids = first_id + np.arange(n)

        # Kernel voxels of every grain as flat indices, in entry-epoch order. A kernel wider than
        # half a periodic box wraps onto itself; each voxel is kept once, at its earliest epoch.
        flat, epoch = [], []
        for kernel, center in zip(kernels, centers):
            f, inside = self.domain.wrap(center + kernel.offsets)
            first = np.sort(np.unique(f, return_index=True)[1])
            flat.append(f[first])
            epoch.append(kernel.entry_epoch[inside][first])
        target = np.array([k.target_voxels for k in kernels])
        count = np.bincount(flat_labels[flat_labels > 0], minlength=ids[-1] + 1)[ids]
        # Seed voxels are candidates too (already taken ones drop out), so a grain grows even if
        # its seed was only partly written.
        pointer = np.zeros(n, dtype=np.int64)
        kernel_size = np.array([len(e) for e in epoch])
        pending = [np.empty(0, dtype=np.int64) for _ in range(n)]
        stamp = np.zeros(flat_labels.size, dtype=np.int64)

        free = int(np.count_nonzero(flat_labels == 0))
        self.stats.filled_after_rsa = flat_labels.size - free
        active = np.arange(n)
        reset = False
        # Every grain has its own epoch counter: a grain that stopped and is reactivated in the
        # fill-up phase continues from where it stopped, one step per round like all others.
        grain_epoch = np.zeros(n, dtype=np.int64)
        max_rounds = 2 * max(int(e[-1]) for e in epoch) + 2
        k = 0
        while free > 0 and k < max_rounds:
            k += 1
            grew = np.zeros(n, dtype=bool)
            for g in self.rng.permutation(active):
                grain_epoch[g] += 1
                end = np.searchsorted(epoch[g], grain_epoch[g], side='right')
                candidates = np.concatenate([pending[g], flat[g][pointer[g]:end]])
                pointer[g] = end
                candidates = candidates[flat_labels[candidates] == 0]
                if self.config.connected_growth:
                    claimed = self._claim_connected(flat_labels, candidates, ids[g], stamp, token=k * n + g + 1)
                    pending[g] = candidates[flat_labels[candidates] == 0]
                else:
                    flat_labels[candidates] = ids[g]
                    claimed = len(candidates)
                count[g] += claimed
                grew[g] = claimed > 0
                free -= claimed
                if reset:
                    self.stats.filled_reset += claimed
                else:
                    self.stats.filled_regular += claimed
                if free == 0:
                    break
            self.stats.filled_per_epoch.append(flat_labels.size - free)

            # A grain stops when it took nothing in this epoch (as in the legacy code), when its
            # kernel is used up (its last step already took every pending voxel reachable from its
            # body, and nothing new enters any more), or - outside the fill-up phase - when it
            # reached its target volume. When all grains have stopped but voxels are still free,
            # every grain with kernel left is reactivated without volume limit.
            exhausted = pointer >= kernel_size
            done = exhausted | ~grew
            if not reset:
                done |= count >= target
            active = active[~done[active]]
            if active.size == 0 and free > 0:
                if not reset:
                    reset = True
                    self.stats.reset_epoch = k
                active = np.flatnonzero(~exhausted)
            if active.size == 0:
                break

        self.stats.epochs = k
        free = int(np.count_nonzero(flat_labels == 0))
        if free > 0:
            self.stats.filled_by_dilation = self._fill_from_neighbours(flat_labels)
        return labels

    def _claim_connected(self, flat_labels, candidates, grain_id, stamp, token) -> int:
        """Take candidates reachable from the grain's body through other candidates."""
        if candidates.size == 0:
            return 0
        stamp[candidates] = token
        nb = self.domain.face_neighbors(candidates)
        touching = np.any((nb >= 0) & (flat_labels[np.maximum(nb, 0)] == grain_id), axis=1)
        layer = candidates[touching]
        claimed = 0
        while layer.size:
            flat_labels[layer] = grain_id
            claimed += layer.size
            nb = self.domain.face_neighbors(layer).ravel()
            nb = nb[nb >= 0]
            layer = np.unique(nb[(stamp[nb] == token) & (flat_labels[nb] == 0)])
        return claimed

    def _fill_from_neighbours(self, flat_labels: np.ndarray) -> int:
        """Give every remaining free voxel to a random adjacent grain, front by front."""
        filled = 0
        while True:
            free = np.flatnonzero(flat_labels == 0)
            if free.size == 0:
                return filled
            nb = self.domain.face_neighbors(free)
            nb_labels = np.where(nb >= 0, flat_labels[np.maximum(nb, 0)], 0)
            priority = np.where(nb_labels > 0, self.rng.random(nb.shape), -1.0)
            pick = priority.argmax(axis=1)
            chosen = nb_labels[np.arange(len(free)), pick]
            reachable = chosen > 0
            if not reachable.any():
                raise RuntimeError('free voxels without any grain neighbour remain')
            flat_labels[free[reachable]] = chosen[reachable]
            filled += int(reachable.sum())
