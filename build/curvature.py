import numpy as np
import pandas as pd

from data.normalized import test_set
from data.generate import Config
from models.model import test


def test_with_increasing_curve(model, number_of_sigmas = 11):
    test_set_size = 100
    image_dims = (3555, 1556)

    model = model.resize(image_dims)

    sigma_min = 0.
    sigma_max = 0.025
    sigma_values = np.linspace(sigma_min, sigma_max, number_of_sigmas)


    test_set_configs = [
        Config(image_dims = image_dims,
               min_curvature_sigma = sigma, max_curvature_sigma = sigma,
               min_curvature_limit = .125, max_curvature_limit = .125
               min_fibres = 100,
               max_fibres = 200) for sigma in sigma_values
    ]

    test_results = pd.concat([
        test(model, test_set(config, size = test_set_size))[1].assign(curve_change_sigma = sigma) for sigma, config in zip(sigma_values, test_set_configs)
    ])

    test_results.to_csv('curve_test_results.csv')