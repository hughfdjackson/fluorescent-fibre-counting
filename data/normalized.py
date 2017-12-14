import data.generate as gen
from random import seed

import numpy as np

training_set_size = 100000
test_set_size = 10000

training_set_seed = 1000
test_set_seed = 1001

identity = lambda x: x

class Normalizer:
    def __init__(self, normalize, denormalize):
        self.normalize = normalize
        self.denormalize = denormalize

ImageNormalizer = Normalizer(
    ## TODO: float32 everywhere - unnecessary complication
    lambda x: x.astype('float64') / 255.,
    lambda x: (x * 255.).astype('uint8')
)

DensityMapNormalizer = Normalizer(
    lambda x: x * 100.,
    lambda x: x / 100.,
)

def training_set(config = gen.Config(), size = training_set_size):
    seed(training_set_seed)
    return _training_set_normalized(size, config)

def test_set(config = gen.Config(), size = test_set_size):
    seed(test_set_seed)
    return _training_set_normalized(size, config)

def test_set_batches(config = gen.Config(), size = test_set_size, batch_size = 10):
    seed(test_set_seed)
    batch_nums = size // batch_size
    for _ in range(batch_nums):
        yield _training_set_normalized(batch_size, config)

    remainder = size % batch_size
    if remainder != 0:
        yield _training_set_normalized(remainder, config)

def training_set_mini(config = gen.Config()):
    """
        Provides a smaller sized training set for faster iteration.
        In real applications, only `training_set` from this
        module should be used
    """
    return training_set(config, size = 10)

def test_set_mini(config = gen.Config()):
    return test_set(config, size = 10)


def _training_set_normalized(size, config):
    images, density_maps, masks, counts = gen.training_set(size, config)

    return (
        np.array([ImageNormalizer.normalize(i) for i in images]),
        np.array([DensityMapNormalizer.normalize(d) for d in density_maps]),
        np.array(masks),
        np.array(counts)
    )

