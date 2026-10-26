"""Legacy generator measured against the acceptance criteria of the rework.

Documents where the legacy code stands: it reproduces grain volumes and phase fractions within
the limits, but splits grains into several bodies. The new generator gets the same test without
the xfail.
"""
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'benchmarks'))

from dragen.analysis.acceptance import violations  # noqa: E402


@pytest.fixture(scope='module')
def legacy_run(tmp_path_factory):
    import legacy_baseline
    return legacy_baseline.run('lognormal', box=15, resolution=2, seed=1,
                               out_dir=str(tmp_path_factory.mktemp('legacy')))


def test_legacy_reproduces_volumes_and_phases(legacy_run):
    found = violations(legacy_run['report_obj'], legacy_run['target_phase_ratio'])
    assert [v for v in found if 'split' not in v] == []


@pytest.mark.xfail(strict=True, reason='legacy tessellation leaves grains split into several bodies')
def test_legacy_grains_are_single_bodies(legacy_run):
    assert legacy_run['report_obj'].fragmented_grains == 0
