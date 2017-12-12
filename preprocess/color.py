from matplotlib.colors import rgb_to_hsv
from scipy.stats import norm
import numpy as np
from scipy.ndimage import gaussian_filter

def normalise(xs):
    return xs / (xs.max() + 1e-10)

def slope(xs, mean, std_div):
    lower_bound = .2
    return normalise(norm.pdf(xs, loc = mean, scale = std_div))  * (1 - lower_bound) + lower_bound

def region_with_slopes(xs, range, std_div):
    lower_slope = slope(xs, range[0], std_div)
    upper_slope = slope(xs, range[1], std_div)

    within_range = np.where(
        (xs >= range[0]) & (xs <= range[1]), 1., 0.
    )

    max_slope = np.maximum(lower_slope, upper_slope)
    max_boost = np.maximum(max_slope, within_range)

    return max_boost

def correct_color(image):

    hsv = rgb_to_hsv(image)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    hue_slope = region_with_slopes(
        hue,
        range = (0.40, 0.57),
        std_div = 0.025)

    return value * gaussian_filter(hue_slope, sigma = 1)
