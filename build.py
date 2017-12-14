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
    trained, training_history = train(model, training_set(size = 20000))
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


if __name__ == '__main__':
    ### TODO: set up for training fcrn_b_peak_mask + fcrn_a_peak_mask
    #train_and_test(fcrn_a_peak_mask())
    train_and_test(fcrn_b_peak_mask())
