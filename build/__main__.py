from models.fcrn import (
    fcrn_a_peak_mask,
    fcrn_b_peak_mask
)

from build.train_and_test import train_and_test

if __name__ == '__main__':
    train_and_test(fcrn_a_peak_mask())
    train_and_test(fcrn_b_peak_mask())
