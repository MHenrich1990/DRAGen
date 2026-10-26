"""Grain table: one row per grain, stored as columns of numpy arrays."""
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np
import pandas as pd

EULER = ['phi1', 'PHI', 'phi2']


@dataclass(frozen=True)
class GrainTable:
    """Grains of one RVE.

    radii: (G, ndim) unshrunk semi-axes a, b[, c] in micrometres.
    alpha: (G,) rotation about the last axis in degrees.
    euler: (G, 3) Bunge Euler angles in degrees.
    phase: (G,) phase ID.
    centers: (G, ndim) centre voxel per grain, set by the RSA; None before.
    """

    radii: np.ndarray
    alpha: np.ndarray
    euler: np.ndarray
    phase: np.ndarray
    centers: Optional[np.ndarray] = field(default=None)

    def __post_init__(self):
        n = len(self.radii)
        if not (len(self.alpha) == len(self.euler) == len(self.phase) == n):
            raise ValueError('all grain columns need the same length')
        if self.radii.ndim != 2 or self.radii.shape[1] not in (2, 3):
            raise ValueError(f'radii must be (G, 2) or (G, 3), got {self.radii.shape}')

    def __len__(self) -> int:
        return len(self.radii)

    @property
    def ndim(self) -> int:
        return self.radii.shape[1]

    @property
    def volume(self) -> np.ndarray:
        """Continuous volume (area in 2D) of the unshrunk ellipsoids."""
        if self.ndim == 3:
            return 4 / 3 * np.pi * np.prod(self.radii, axis=1)
        return np.pi * np.prod(self.radii, axis=1)

    def take(self, index) -> 'GrainTable':
        return GrainTable(radii=self.radii[index], alpha=self.alpha[index], euler=self.euler[index],
                          phase=self.phase[index],
                          centers=None if self.centers is None else self.centers[index])

    def with_centers(self, centers: np.ndarray) -> 'GrainTable':
        return replace(self, centers=np.asarray(centers))

    @staticmethod
    def concat(tables: Sequence['GrainTable']) -> 'GrainTable':
        if any(t.centers is not None for t in tables):
            raise ValueError('concatenate grain tables before the RSA assigns centres')
        return GrainTable(radii=np.concatenate([t.radii for t in tables]),
                          alpha=np.concatenate([t.alpha for t in tables]),
                          euler=np.concatenate([t.euler for t in tables]),
                          phase=np.concatenate([t.phase for t in tables]))

    def to_frame(self) -> pd.DataFrame:
        """Grain i of the table is grain ID i + 1 in the label field."""
        axes = ['a', 'b', 'c'][:self.ndim]
        df = pd.DataFrame(self.radii, columns=axes)
        df['alpha'] = self.alpha
        df[EULER] = self.euler
        df['phaseID'] = self.phase
        df['volume'] = self.volume
        if self.centers is not None:
            df[[f'center_{a}' for a in 'xyz'[:self.ndim]]] = self.centers
        df.insert(0, 'GrainID', np.arange(1, len(self) + 1))
        return df
