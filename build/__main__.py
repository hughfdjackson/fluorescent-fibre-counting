from models.fcrn import (
    fcrn_b_peak_mask
)

from build.curvature import test_with_increasing_curve
from models.model import load

if __name__ == '__main__':
    test_with_increasing_curve(load('builds/FCRN_Peak_Mask_B-fcrn_b/1/FCRN_Peak_Mask_B-fcrn_b.hdf5'))