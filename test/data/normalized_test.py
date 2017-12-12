from data.normalized import DensityMapNormalizer, ImageNormalizer
from data.generate import generate_training_example, Config

from numpy import array_equal
from numpy.testing import assert_almost_equal

import math


def test_normalize_image():
    image = generate_training_example(Config())[0]
    normalized = ImageNormalizer.normalize(image)

    assert array_equal(
        ImageNormalizer.denormalize(normalized),
        image
    )
    assert math.isclose(normalized.max(), image.max() / 255.0)


def test_normalize_density_map():
    density_map = generate_training_example(Config())[1]
    normalized = DensityMapNormalizer.normalize(density_map)

    assert_almost_equal(
        DensityMapNormalizer.denormalize(normalized),
        density_map
    )

