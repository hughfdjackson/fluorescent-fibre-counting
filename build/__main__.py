from models.fcrn import (
    fcrn_b_peak_mask
)

from build.curvature import test_with_increasing_curve
from build.train_and_test import train_only

from models.model import load

if __name__ == '__main__':
    model = load('builds/FCRN_Peak_Mask_B-fcrn_b/3/FCRN_Peak_Mask_B-fcrn_b.hdf5')
    test_with_increasing_curve(model, number_of_sigmas = 6)