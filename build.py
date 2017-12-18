from models.fcrn import (
    fcrn_a_peak_mask,
    fcrn_b_peak_mask
)

from models.model import train, test, save_all, clone, load, save_build

from data.normalized import training_set, test_set, test_set_batches, ImageNormalizer
from data.generate import Config

import os.path as path
import re

import pandas as pd
from glob import glob

import numpy as np
from PIL import Image

def train_and_test(model):
    trained, training_history = train(model, training_set(size = 50000))
    #test_results = run_realistic_tests(trained)

    save_build(trained,training_history, None)


def run_realistic_tests(model):
    ## TODO: make everything h, w format to reduce confusion
    image_dims = (3555, 1556)

    batches = test_set_batches(
        Config(image_dims = image_dims, max_fibres = 300, min_fibres = 50),
        size = 500,
        batch_size = 40)

    m = model.resize(image_dims)

    return pd.concat([
        test(m, b)[1] for b in batches
    ])

def test_with_increasing_curve(model):
    test_set_size = 100
    image_dims = (512, 512)

    model = model.resize(image_dims)

    sigma_min = 0.
    sigma_max = 0.045
    sigma_values = np.around(np.linspace(sigma_min, sigma_max, 10), 3)


    test_set_configs = [
        Config(image_dims = image_dims, curve_change_sigma = sigma, max_fibres = 10) for sigma in sigma_values
    ]

    test_results = pd.concat([
        test(model, test_set(config, size = test_set_size))[1].assign(curve_change_sigma = sigma) for sigma, config in zip(sigma_values, test_set_configs)
    ])

    test_results.to_csv('curve_test_results.csv')

if __name__ == '__main__':
    test_with_increasing_curve(load('builds/FCRN_Peak_Mask_B-fcrn_b/1/FCRN_Peak_Mask_B-fcrn_b.hdf5'))
