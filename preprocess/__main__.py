from preprocess.color import correct_color
from preprocess.snip import snip
from preprocess.orientation import correct_orientation

from glob import glob
import os.path as path
import numpy as np
import cv2
from PIL import Image

from functools import reduce

import argparse

def compose(*fns):
    return reduce(
        lambda fn, fns:
            lambda x: fn(fns(x)),
        fns)


as_uint8 = lambda x: x.astype('uint8')

default_pipeline = compose(correct_color, snip, correct_orientation)

def base_pipeline(args):
    if not any([args.snip, args.color, args.orientate]):
        return default_pipeline
    else:
        pipeline_candidates = [
            args.color and correct_color,
            args.snip and snip,
            args.orientate and correct_orientation,
        ]
        return compose(*[x for x in pipeline_candidates if not x is None])


def create_pipeline(args):
    return compose(as_uint8, base_pipeline(args), cast_to_rgb)

def cast_to_rgb(bgr):
    return np.dstack([
        bgr[:,:,2],
        bgr[:,:,1],
        bgr[:,:,0],
    ])

def _preprocess_all(input_dir, output_dir, pipeline):
    for input_path in glob(path.join(input_dir, '*')):
        _preprocess_file(input_path, output_dir, pipeline)

def _preprocess_file(input_path, output_dir, pipeline):
    try:
        file_name = path.basename(input_path)
        print("-- pre-processing {} --".format(file_name))

        image = cv2.imread(input_path)
        Image.fromarray(pipeline(image)).save(path.join(output_dir, file_name))

    except Exception as e:
        print("couldn't pre-process {}".format(input_path))
        raise e


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = 'automatically correct the orientation of slide images'
    )

    parser.add_argument('--input_dir', help = 'directory including images.')
    parser.add_argument('--output_dir', help = 'directory to put the resulting images in.')

    parser.add_argument('--snip', action = 'store_true', default = None)
    parser.add_argument('--orientate', action = 'store_true', default = None)
    parser.add_argument('--color', action = 'store_true', default = None)

    args = parser.parse_args()

    pipeline = create_pipeline(args)
    _preprocess_all(args.input_dir, args.output_dir, pipeline)
