
import sys
sys.path.append('.')
sys.path.append('./src')

import warnings
warnings.filterwarnings("ignore")

import pytest

from src import experimental
from src import backend


@pytest.mark.run(order=1)
def test_load_sample(kargs):
    sample = backend.load_sample(kargs['inputs'])
    assert sample.shape is not None


@pytest.mark.run(order=2)
def test_load_metadata(kargs):
    gen = backend.load_metadata(model_path=kargs['model'])
    assert hasattr(gen, 'psf_type')


@pytest.mark.run(order=3)
def test_reloadmodel_if_needed(kargs):
    model, modelpsfgen = experimental.reloadmodel_if_needed(modelpath=kargs['model'], preloaded=None)
    model.summary()
