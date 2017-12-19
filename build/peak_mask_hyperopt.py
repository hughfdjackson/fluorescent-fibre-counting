from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np

from functools import partial

from data.normalized import test_set
from data.generate import Config



threshold_space = hp.uniform('threshold', 0.5, 3)
radius_space = hp.quniform('mask_size', 3, 12, 1)

def test_single_configuration(model, counts, density_maps, opts):
    threshold, mask_size = opts

    print('testing: threshold {}, mask_size {}'.format(threshold, mask_size))

    model.threshold = threshold
    model.mask_size = mask_size
    estimates = model.post_process(density_maps)

    loss = np.mean((counts - estimates) ** 2)

    return {
        'loss': loss,
        'status': STATUS_OK,
        'predicted_count': estimates,
        'actual_count': counts
    }

def run_trial(model, test_set):
    images, _, counts = test_set
    trials = Trials()
    density_maps = model.predict(images)

    best = fmin(
        fn = partial(test_single_configuration, model, counts, density_maps),
        space = [threshold_space, radius_space],
        algo = tpe.suggest,
        max_evals = 50,
        trials = trials
    )

    return best, trials

def optimize_hyper_params(model):
    image_dims = (256, 256)

    config = Config(image_dims = image_dims, max_fibres = 5, min_fibres = 0)

    best, trials = run_trial(
        model.resize(image_dims),
        test_set(config, size = 50)
    )

    print("---- Final Results -----")
    print("Threshold: {threshold}, Mask Size: {mask_size}".format(**best))