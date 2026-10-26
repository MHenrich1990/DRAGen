"""The reworked generator against the acceptance criteria, on the same input as the legacy test."""
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'benchmarks'))

from dragen.analysis import metrics  # noqa: E402
from dragen.analysis.acceptance import violations  # noqa: E402
from dragen.config import BoxConfig, PhaseConfig, RveConfig  # noqa: E402
from dragen.pipeline import generate  # noqa: E402


def lognormal_config(box, seed, periodic=None):
    import legacy_baseline
    return RveConfig(box=BoxConfig((box,) * 3, 2, periodic=periodic),
                     phases=(PhaseConfig('ferrite', legacy_baseline.lognormal_input(), 1.0),), seed=seed)


def report_of(result, periodic):
    target = np.concatenate([[0.0], result.grains.volume])
    phase = np.concatenate([[0], result.grains.phase])
    return metrics.report(result.labels, target, phase, result.domain.voxel_volume, periodic)


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_rework_meets_acceptance_criteria(seed):
    result = generate(lognormal_config(15, seed))
    rep = report_of(result, (True,) * 3)
    assert violations(rep, {1: 1.0}) == []
    assert result.repaired_voxels == 0


def test_rework_with_free_surface():
    periodic = (True, True, False)
    result = generate(lognormal_config(15, 1, periodic))
    rep = report_of(result, periodic)
    assert rep.fill_ratio == 1.0
    assert rep.fragmented_grains == 0
