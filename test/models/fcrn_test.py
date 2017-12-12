from models.fcrn import _mask_peaks, _count_peak_mask
from data.normalized import DensityMapNormalizer
import numpy as np
import pytest

import tensorflow as tf

a = np.array([
    [0, 1, 0, 0, 0],
    [2, 0, 0, 0, 0],
    [0, 0, 3, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0]
], dtype = 'float32').reshape(1, 5, 5, 1)

masked_a = np.array([
    [1, 1, 0, 0, 0],
    [1, 1, 1, 1, 0],
    [1, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
], dtype = 'float32').reshape(1, 5, 5, 1)

def test_mask_peaks():
    sess = tf.Session()
    result = _mask_peaks(tf.constant(a), threshold = 1, mask_size = 1)
    np.testing.assert_equal(masked_a, sess.run(result))

def test_count_peak_mask():
    np.testing.assert_equal(
        _count_peak_mask(a, threshold = 1, mask_size = 1),
        np.array([DensityMapNormalizer.denormalize(3.5)], dtype = 'float32')
    )
