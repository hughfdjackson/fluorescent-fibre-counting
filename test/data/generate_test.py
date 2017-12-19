from PIL import Image
from random import seed
from data.generate import (
    generate_training_example,
    training_set,
    Config,
    Fibre,
    gen_components
)

import numpy as np
import math



def test_generation_is_deterministic():
    test_seed = 120

    seed(test_seed)
    data1, label1, count1 = training_set(10, Config())

    seed(test_seed)
    data2, label2, count2 = training_set(10, Config())

    np.array_equal(data1, data2)
    np.array_equal(label1, label2)
    np.array_equal(count1, count2)

