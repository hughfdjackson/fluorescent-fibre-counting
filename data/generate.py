from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

from random import random
from math import floor, pi, cos, sin, tanh, sqrt

import numpy as np
import itertools

from scipy.ndimage import gaussian_filter, filters, interpolation
from multiprocessing import Pool

class Component():
    """
        A component is some part of the generated image that knows
        how to draw itself.
    """

    def __init__(self, state):
        self.state = state

    @staticmethod
    def generate(config):
        """
            All randomness should be constrained to the this method; allowing
            the drawing of the images to be done in parallel.
        """
        pass

    def draw(self, image):
        return image

    def update_density_map(self, array):
        return array


def _generate_point(config, point, angle, rate_of_angle_change):
    np.random.seed(_pick_natural(maximum = 1000))

    angle += rate_of_angle_change
    rate_of_angle_change += np.random.normal(loc = 0, scale = config.curve_change_sigma)
    rate_of_angle_change = np.clip(rate_of_angle_change, -pi * 0.125, pi * 0.125)

    vector = _vector(angle, 1)
    new_point = (point[0] + vector[0], point[1] + vector[1])

    return new_point, angle, rate_of_angle_change


def _generate_path(config, length, bounds):
    start_ = (_pick_float(config.image_dims[0]), _pick_float(config.image_dims[1]))

    point = clip_within_border(start_, config)
    angle = (_pick_float(0, 2 * pi))
    rate_of_angle_change = 0
    path = [point]

    for length_so_far in range(length):
        point, angle, rate_of_angle_change = _generate_point(config, point, angle, rate_of_angle_change)

        path.append(point)

    return path, length

def _clip_color_change_rate(rate_of_color_change):
    return np.clip(rate_of_color_change, -50, 50)

def _generate_segment_color(color_range, color, rate_of_color_change):
    np.random.seed(_pick_natural(maximum = 1000))

    color += rate_of_color_change
    color = int(np.clip(color, *color_range))
    rate_of_color_change += np.random.normal(loc = 0, scale = 30)
    rate_of_color_change = _clip_color_change_rate(rate_of_color_change)

    return color, rate_of_color_change

def _apply_end_colour_penality(i, length, color, color_range, amount):
    start = tanh(i / amount) * 255
    end = tanh(length - i / amount) * 255

    alpha = int(min(start, end))

    return color[:3] + (alpha,)

def _generate_colors(length, color_range, alpha_range):
    np.random.seed(_pick_natural(maximum = 1000))

    alpha = _pick_natural(*alpha_range)

    color_bound_1 = _pick_natural(*color_range)
    color_bound_2 = np.clip(
        int(np.random.normal(loc = color_bound_1, scale = 200)),
        *color_range
    )

    color_bounds = (
        min([color_bound_1, color_bound_2]),
        max([color_bound_1, color_bound_2]),
    )

    penalty_amount = _pick_float(1, 2)

    color = _pick_natural(*color_bounds)

    color_with_penalty = _apply_end_colour_penality(0, length, _color(color, alpha), color_range, penalty_amount)
    colors = [color_with_penalty]
    rate_of_color_change = _clip_color_change_rate(_pick_natural(-50, 50))

    for i in range(length):
        color, rate_of_color_change = _generate_segment_color(color_bounds, color, rate_of_color_change)

        color_with_penalty = _apply_end_colour_penality(i + 1, length, _color(color, alpha), color_range, penalty_amount)
        colors.append(color_with_penalty)

    return colors

def _color(i, alpha = 255):
    return (i, i, i, alpha)

def _draw_fibre(state, image):
    image = state['bubble'].draw(image)
    path = state['path']

    draw = ImageDraw.Draw(image, 'RGBA')

    segments = zip(path[:-1], path[1:])

    for segment, color in zip(segments, state['color']):
        draw.line(
            segment,
            fill = color,
            width = state['width']
        )

    return image

def _vector(angle, length):
    return (cos(angle) * length, sin(angle) * length)

class Fibre(Component):

    @staticmethod
    def generate(config):
        width = _pick_natural(config.min_fibre_width, config.max_fibre_width)
        max_length = _pick_natural(config.min_fibre_length, config.max_fibre_length)
        path, length = _generate_path(config, max_length, config.image_dims)

        return Fibre({
            'path': path,
            'color': _generate_colors(length, (125, 200), (150, 255)),
            'width': width,
            'bubble': FibreBubble.generate(path, width),
            'length': length,
            'image_dims': config.image_dims
        })

    def draw(self, image):
        return _draw_fibre(self.state, image)

    def update_density_map(self, array):
        def point_to_int(point):
            return (int(point[0]), int(point[1]))

        path = self.state['path']
        start = point_to_int(path[0])
        end = point_to_int(path[-1])

        if _point_is_within_bounds(start, array.shape):
            array[start[1], start[0]] += 1

        if _point_is_within_bounds(end, array.shape):
            array[end[1], end[0]] += 1

        return array

def _point_is_within_bounds(point, bounds):
    return (point[1] < bounds[0] and
            point[1] >= 0 and
            point[0] < bounds[1] and
            point[0] >= 0)

class NonFluorescentFibre(Component):
    """
        Fibres that are much darker than the fluorescent fibres in the image.
    """
    @staticmethod
    def generate(config):
        width = _pick_natural(config.min_fibre_width, config.max_fibre_width)
        max_length = _pick_natural(config.min_fibre_length, config.max_fibre_length)
        path, length = _generate_path(config, max_length, config.image_dims)

        return NonFluorescentFibre({
            'path': path,
            'color': _generate_colors(length, (0, 50), (0, 25)),
            'width': width,
            'bubble': FibreBubble.generate(path, width)
        })

    def draw(self, image):
        return _draw_fibre(self.state, image)

class FibreBubble(Component):

    @staticmethod
    def generate(path, width):
        return FibreBubble({
            'path': path,
            'width': floor(width * _pick_float(2.0, 7.0)),
            'alpha': _pick_natural(0, 3)
        })

    def draw(self, image):
        points = self.state['path']

        bubble_color = _color(255, self.state['alpha'])

        draw = ImageDraw.Draw(image, 'RGBA')
        draw.line(points, width = self.state['width'], fill = bubble_color)

        return image

_noise_shape = (2 ** 12), (2 ** 12)
_noise = np.random.normal(np.empty(_noise_shape)).repeat(3).reshape(_noise_shape + (3,))


class Background(Component):

    @staticmethod
    def generate(config):
        np.random.seed(_pick_natural(maximum = 324230432))
        return Background({
            'color': _color(_pick_natural(0, 50)),
            'bounding_box': [(0, 0), config.image_dims],
            'noise_degree': _pick_float(0, 3),
            'noise_shift': (_pick_natural(0, 100), _pick_natural(0, 100)),
            'image_dims': config.image_dims
        })

    def draw(self, image):
        draw = ImageDraw.Draw(image, 'RGBA')
        draw.rectangle(self.state['bounding_box'], fill = self.state['color'])

        w, h = self.state['image_dims']
        noise = np.roll(_noise[:h, :w, :], self.state['noise_shift'], axis = (0, 1))
        noise *= self.state['noise_degree']

        array = np.asarray(image).astype('float32')
        array[:, :, :3] += noise
        array = np.clip(array, 0, array.max())

        return Image.fromarray(array.astype('uint8'))

class TapeLine(Component):

    @staticmethod
    def generate(config):
        w, h = config.image_dims

        angle = _pick_float(-pi, pi)

        # If we go this length in each direction from any starting point,
        # we're guaranteed to be outside of the image
        length_from_point = int(sqrt(w ** 2 + h ** 2))

        start_vec = _vector(angle, length_from_point)
        end_vec = _vector(angle, -length_from_point)

        xy = (_pick_natural(0, w), _pick_natural(0, h))

        start = _tuple_addition(xy, start_vec)
        end = _tuple_addition(xy, end_vec)

        num_segments = length_from_point * 2

        path = list(zip(
            np.linspace(start[0], end[0], num_segments),
            np.linspace(start[1], end[1], num_segments),
        ))

        colors = _generate_colors(num_segments, (50, 150), (50, 50))

        return TapeLine({
            'path': path,
            'colors': colors
        })

    def draw(self, image):
        draw = ImageDraw.Draw(image, 'RGBA')
        path = self.state['path']

        segments = zip(path[:-1], path[1:])

        for segment, color in zip(segments, self.state['colors']):
            draw.line(segment, fill = color)

        return image

def _tuple_addition(xs, ys):
    return tuple(x + y for x, y in zip(xs, ys))

class WhitePaperBand(Component):

    @staticmethod
    def generate(config):
        return WhitePaperBand({
            'color': _pick_natural(200, 255),
            'noise': _gen_noise(config.image_dims),
            'poly_points': [
                _pick_point_within(config.image_dims),
                _pick_point_within(config.image_dims),
                _pick_point_within(config.image_dims)
            ]
        })

    def draw(self, image):
        draw = ImageDraw.Draw(image, 'RGBA')
        draw.polygon(self.state['poly_points'], fill = _color(self.state['color']))

        noise = self.state['noise']
        noise = noise.repeat(3).reshape(noise.shape + (3,))

        array = np.asarray(image).astype('float32')
        array += noise
        array = np.clip(array, 0, array.max())

        return Image.fromarray(array.astype('uint8'))


class Blur(Component):

    @staticmethod
    def generate(config):
        return Blur({
            'radius': _pick_float(.5, 1.)
        })

    def draw(self, image):
         return image.filter(ImageFilter.GaussianBlur(radius = self.state['radius']))


class DensityMapBlur(Component):

    sigma = 1

    @staticmethod
    def generate(config):
        return DensityMapBlur({})

    def update_density_map(self, array):
        return gaussian_filter(array, sigma = self.sigma, mode = 'constant', cval = 0.0)


class Brightness(Component):

    @staticmethod
    def generate(config):
        return Brightness({
            'scalar': _pick_float(0.25, 1.5)
        })

    def draw(self, image):
        enhance = ImageEnhance.Brightness(image)
        return enhance.enhance(self.state['scalar'])

def identity(x): return x

class Config:

    def __init__(self,
           image_dims = (64, 64),
           max_fibres = 3, min_fibres = 1,
           max_fibre_width = 3, min_fibre_width = 1,
           max_fibre_length = 125, min_fibre_length = 20,
           max_background_fibres = 2,
           curve_change_sigma = 0.075
           ):

        self.image_dims = image_dims
        self.max_fibres = max_fibres
        self.min_fibres = min_fibres
        self.max_fibre_width = max_fibre_width
        self.min_fibre_width = min_fibre_width
        self.max_fibre_length = max_fibre_length
        self.min_fibre_length = min_fibre_length
        self.max_background_fibres = max_background_fibres
        self.curve_change_sigma = curve_change_sigma

def _pick_natural(minimum = 0, maximum = 1):
    return floor(random() * (maximum - minimum)) + minimum

def _pick_float(minimum = 0, maximum = 1.0):
    return (random() * (maximum - minimum)) + minimum

def _pick_point_within(bounding_box):
    return (
        _pick_natural(maximum = bounding_box[0]),
        _pick_natural(maximum = bounding_box[1])
    )

def get_mid_point(a, b):
    return a + (b - a) * 0.5

def clip_within_border(point, config):
    x, y = point
    w_, h_ = config.image_dims

    ## Accounting for border of 5 standard deviations of the image edge
    ## when the training dots are made.  This makes sure that none of the
    ## density map's blurred dots overlap the edge of the image.
    w, h = w_ - 5, h_ - 5
    return np.clip(x, 5, w), np.clip(y, 5, h)

def pick_fibre_number(config):
    return _pick_natural(config.min_fibres, config.max_fibres + 1)


def gen_components(config):
    num_fibres = pick_fibre_number(config)
    num_background_fibres = _pick_natural(maximum = config.max_background_fibres)

    background = Background.generate(config)
    #white_paper = WhitePaperBand.generate(config)

    fluorescent_fibres = [Fibre.generate(config) for i in range(num_fibres)]
    background_fibres = [NonFluorescentFibre.generate(config) for i in range(num_background_fibres)]
    fibres = (fluorescent_fibres + background_fibres)
    fibres.sort(key = lambda x: random())
    tape_line = TapeLine.generate(config)

    blur = Blur.generate(config)
    # brightness = Brightness.generate(config)
    density_map_blur = DensityMapBlur.generate(config)

    return [background] + fibres + [tape_line, blur, density_map_blur]

def create_fibre_image(components, config):
    image = Image.new('RGB', config.image_dims)

    for component in components:
        image = component.draw(image)

    return image.convert('L')

def create_density_map(components, config):
    w, h = config.image_dims
    array = np.zeros((h, w)) # for whatever reason, numpy prefers it in h -> w format

    for component in components:
        array = component.update_density_map(array)

    return array

def render_components(components, config):
    w, h = config.image_dims
    image = np.asarray(create_fibre_image(components, config)).reshape(h, w, 1)
    density_map = np.asarray(create_density_map(components, config)).reshape(h, w, 1)
    count = np.sum(density_map) / 2.

    return (image, density_map, count)

def render_components_set(components_set, config):
    with Pool() as p:
        return p.starmap(render_components, zip(components_set, itertools.repeat(config)))

def generate_training_example(config):
    return render_components(gen_components(config), config)

## Training set
def training_set(size, config):
    components_set = [gen_components(config) for i in range(size)]
    values = zip(*render_components_set(components_set, config))
    return tuple(np.array(v) for v in values)