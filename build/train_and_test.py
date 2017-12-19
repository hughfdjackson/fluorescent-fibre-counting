from models.model import train, test, save_build

from data.normalized import training_set, test_set_batches
from data.generate import Config

import pandas as pd
import numpy as np


def train_and_test(model):
    trained, training_history = train(model, training_set(size = 50000))
    test_results = run_realistic_tests(trained)

    save_build(trained,training_history, None)

def run_realistic_tests(model):
    image_dims = (3555, 1556)

    batches = test_set_batches(
        Config(image_dims = image_dims, max_fibres = 300, min_fibres = 50),
        size = 500,
        batch_size = 40)

    m = model.resize(image_dims)

    return pd.concat([
        test(m, b)[1] for b in batches
    ])
