"""The GAN source must turn WGAN output into the same ellipsoids as the legacy reader."""
import os

import numpy as np
import pytest

from dragen.sources import GanSource, ellipses_to_ellipsoids

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
GAN_FILES = ['ExampleInput/Ferrite/TrainedData_Ferrite.pkl', 'ExampleInput/Martensite/TrainedData_Martensite.pkl']


@pytest.mark.parametrize('relpath', GAN_FILES)
def test_conversion_matches_legacy_reader(relpath, tmp_path):
    torch = pytest.importorskip('torch')
    from dragen.utilities.Helpers import HelperFunctions
    from dragen.utilities.InputInfo import RveInfo
    path = os.path.join(REPO_ROOT, relpath)
    RveInfo.store_path = str(tmp_path)

    torch.manual_seed(5)
    legacy = HelperFunctions().read_input_gan(path, 3, size=400).reset_index(drop=True)

    source = GanSource(path, 'ferrite')
    torch.manual_seed(5)
    new = ellipses_to_ellipsoids(source._load().sample_batch(label=0, size=400).dropna(axis=0))

    for col in ('a', 'b', 'c', 'alpha'):
        assert np.allclose(new[col].to_numpy(), legacy[col].to_numpy()), col
    if 'phi1' in new:
        assert np.allclose(new[['phi1', 'PHI', 'phi2']].to_numpy(), legacy[['phi1', 'PHI', 'phi2']].to_numpy())


def test_gan_source_is_reproducible():
    pytest.importorskip('torch')
    source = GanSource(os.path.join(REPO_ROOT, GAN_FILES[1]), 'martensite', batch_size=2000)
    a = source.sample(500.0, np.random.default_rng(1))
    b = source.sample(500.0, np.random.default_rng(1))
    assert np.array_equal(a.radii, b.radii)
    assert np.all(a.phase == 2)
