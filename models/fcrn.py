from data.normalized import DensityMapNormalizer

from keras.models import Model, Input
from keras.layers import Convolution2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D
from keras.optimizers import SGD

import tensorflow as tf

from models.wrapper import Wrapper
from models.model import sequence_layers, inner_layer, clone
from data.normalized import DensityMapNormalizer

def fcrn_a_peak_mask():
    return FCRN_Peak_Mask_A(_fcrn_a_base())

def fcrn_b_peak_mask():
    return FCRN_Peak_Mask_B(_fcrn_b_base())

def _conv_bn_relu(filters, kernel_size):
    return inner_layer([
        Convolution2D(filters, kernel_size, kernel_initializer = 'orthogonal', padding = 'same', use_bias = False),
        BatchNormalization(),
        Activation('relu'),
    ])



def _fcrn_a_base():
    inputs = Input(batch_shape = (None, 64, 64, 1))
    outputs = sequence_layers([
        inputs,
        _conv_bn_relu(32, (3, 3)),
        MaxPooling2D((2, 2)),

        # Conv
        _conv_bn_relu(64, (3, 3)),
        MaxPooling2D((2, 2)),

        # Conv
        _conv_bn_relu(128, (3, 3)),
        MaxPooling2D((2, 2)),

        # FC
        ## TODO:  this is (3, 3) in the paper - switch to that?
        _conv_bn_relu(512, (5, 5)),

        # UnConv
        ## TODO: Missing ReLUs
        UpSampling2D((2, 2)),
        _conv_bn_relu(128, (3, 3)),

        UpSampling2D((2, 2)),
        _conv_bn_relu(64, (3, 3)),

        UpSampling2D((2, 2)),
        _conv_bn_relu(32, (3, 3)),

        Convolution2D(1, (1, 1), kernel_initializer = 'orthogonal', padding = 'same', use_bias = False)
    ])

    return Model(name = 'fcrn_a', inputs = inputs, outputs = outputs)


def _fcrn_b_base():
    inputs = Input(batch_shape = (None, 64, 64, 1))
    outputs = sequence_layers([
        inputs,
        _conv_bn_relu(32, (3, 3)),
        MaxPooling2D((2, 2)),

        # Conv
        _conv_bn_relu(64, (3, 3)),

        # Conv
        _conv_bn_relu(128, (3, 3)),
        MaxPooling2D((2, 2)),

        # Conv
        _conv_bn_relu(256, (5, 5)),

        # Conv
        _conv_bn_relu(128, (3, 3)),

        # UnConv
        UpSampling2D((2, 2)),
        _conv_bn_relu(256, (5, 5)),


        UpSampling2D((2, 2)),
        Convolution2D(1, (1, 1), kernel_initializer = 'orthogonal', padding = 'same', use_bias = False)
    ])

    return Model(name = 'fcrn_b', inputs = inputs, outputs = outputs)



## Wrappers - for pre/post processing and resizing ##

class FCRN_Peak_Mask(Wrapper):
    def __init__(self, model, threshold, mask_size):
        self.threshold = threshold
        self.mask_size = mask_size
        super().__init__(model)

    def post_process(self, predictions):
        return _count_peak_mask(predictions, self.threshold, self.mask_size)

    def resize(self, target_size):
        return FCRN_Peak_Mask(_resize_fcrn(self.model, target_size),
                              threshold = self.threshold,
                              mask_size = self.mask_size)

    def compile(self):
        optimizer = SGD(momentum = 0.9, nesterov = True)
        self.model.compile(optimizer, 'mean_squared_error')

    def fit(self, images, labels, **kwargs):
        density_map = labels[0]
        return self.model.fit(images, density_map, **kwargs)



class FCRN_Peak_Mask_A(FCRN_Peak_Mask):

    def __init__(self, model):
        super().__init__(model, threshold = 2.30, mask_size = 4)

class FCRN_Peak_Mask_B(FCRN_Peak_Mask):

    def __init__(self, model):
        super().__init__(model, threshold = 1.91, mask_size = 3)



def _count_peak_mask(density_maps, threshold, mask_size):
   with tf.Graph().as_default():
        density_map_tensor = tf.constant(density_maps)

        masks = _mask_peaks(density_map_tensor, threshold, mask_size)
        mask_float32 = tf.cast(masks, tf.float32)
        masked = mask_float32 * density_map_tensor


        fibre_count = tf.reduce_sum(masked, axis = [1, 2, 3]) / tf.constant(2.0)
        with tf.Session() as sess:
            return DensityMapNormalizer.denormalize(sess.run(fibre_count))

def _mask_peaks(a, threshold, mask_size):
    kernel_size = mask_size * 2 + 1

    # Find peaks - i.e. pixels that are greater than their 9 immediate neighbours
    windows = _sliding_window(a, 3)
    is_peak = tf.equal(tf.argmax(windows, axis = 3), 4)
    peaks = tf.reshape(tf.cast(is_peak, tf.int8), tf.shape(a))

    # Filter out the low peaks - many of these will belong to random low-level
    # noise.
    high_peaks = tf.where(tf.greater(a, threshold), peaks, tf.zeros_like(peaks))

    # Create a square mask around the high peaks, so we capture the whole (meaningful)
    # of the estimated gaussian.
    masks = tf.nn.max_pool(
        high_peaks,
        ksize = [1, kernel_size, kernel_size, 1],
        strides = [1, 1, 1, 1],
        padding = 'SAME')

    return masks

def _sliding_window(a, kernel_size):
    return tf.extract_image_patches(
        a,
        ksizes = [1, kernel_size, kernel_size, 1],
        strides = [1, 1, 1, 1],
        rates = [1, 1, 1, 1],
        padding = 'SAME')



def _resize_fcrn(model, target_size):
    m = clone(model)
    target_x, target_y = target_size

    config = m.get_config()
    batch, x, y, z = config['layers'][0]['config']['batch_input_shape']
    config['layers'][0]['config']['batch_input_shape'] = batch, target_y, target_x, z

    new_model = Model.from_config(config)
    new_model.set_weights(model.get_weights())
    new_model.name = model.name

    return new_model

