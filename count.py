import argparse
from textwrap import dedent

import cv2

import os.path as path
from glob import glob
from data.normalized import ImageNormalizer, DensityMapNormalizer
import numpy as np
import warnings
import itertools


def _load_image(path):
    image = ImageNormalizer.normalize(np.asarray(cv2.imread(path, cv2.IMREAD_GRAYSCALE)))
    return image.reshape(image.shape + (1,))

def predict_from_real_images(model, image_dims, images, image_labels):
    from models.model import predict

    model = model.resize(image_dims)
    return predict(model, images, image_labels = image_labels)

def count(args):
    from models.model import load
    import models.fcrn

    model_path = path.join(path.dirname(path.abspath(__file__)), 'builds', args.model, args.build_number, args.model + '.hdf5')
    model = load(model_path)

    image_paths = glob(path.join(args.input_dir, '*'))
    file_names = [path.basename(p) for p in image_paths]
    images = (_load_image(p) for p in image_paths)

    results = predict_from_real_images(model, (args.width, args.height), images, file_names)
    results.to_csv(args.output_file)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = dedent("""
            Predict fibre counts from a set of images.  If the images haven't yet been pre-processed, use
            `python -m preprocess` first.
        """),
        formatter_class = argparse.RawTextHelpFormatter
    )

    parser.add_argument('input_dir', help = 'directory containing pre-processed images')
    parser.add_argument('output_file', help = 'file to save test results in (results in csv format)')

    model_group = parser.add_argument_group('model', dedent("""
        By default, the most effective model in our tests (FCRN_Peak_Mask_B-fcrn_b) is used to count fibres in the input images.
        If you want to change this, or use a new build of your own, use the following parameters to specify your preference.
    """))

    model_group.add_argument('--model', help = 'name of the model to use - see model name formats in `builds` directory', default = 'FCRN_Peak_Mask_B-fcrn_b')
    model_group.add_argument('--build_number', help = 'build number of the model to use', default = '1')

    image_dims_group = parser.add_argument_group('image dimensions', dedent("""
        By default, the preprocessing step creates images that are 3555 x 1556.  If your pre-processed images are
        of a different size, use the following parameters to adjust this.
    """))
    image_dims_group.add_argument('--width', default = 3555, type = int)
    image_dims_group.add_argument('--height', default = 1556, type = int)

    args = parser.parse_args()

    with warnings.catch_warnings():
        warnings.simplefilter('default')
        count(args)
