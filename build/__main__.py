from models.fcrn import (
    fcrn_a_peak_mask,
    fcrn_b_peak_mask
)

from build.peak_mask_hyperopt import optimize_hyper_params
from models.model import load

if __name__ == '__main__':
    optimize_hyper_params(load('builds/FCRN_Peak_Mask_B-fcrn_b/1/FCRN_Peak_Mask_B-fcrn_b.hdf5'))