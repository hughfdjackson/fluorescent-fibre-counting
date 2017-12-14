import argparse
from textwrap import dedent

from models.model import load, predict
import models.fcrn
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
    model = model.resize(image_dims)
    return predict(model, images, image_labels = image_labels)

def count(args):
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

        """),
        formatter_class = argparse.RawTextHelpFormatter
    )

    parser.add_argument('input_dir', help = 'directory containing pre-processed images')
    parser.add_argument('output_file', help = 'file to save test results in')

    parser.add_argument('--model', help = 'directory containing pre-processed images', default = 'FCRN_Peak_Mask_B-fcrn_b')
    parser.add_argument('--build_number', help = 'directory containing pre-processed images', default = '1')

    parser.add_argument('--width', default = 3555, type = int)
    parser.add_argument('--height', default = 1556, type = int)

    args = parser.parse_args()

    with warnings.catch_warnings():
        warnings.simplefilter('default')
        count(args)
