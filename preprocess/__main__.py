from preprocess.color import correct_color
from preprocess.snip import snip
from preprocess.orientation import correct_orientation
from textwrap import dedent

from glob import glob
import os.path as path
import numpy as np
import cv2
from PIL import Image

from functools import reduce, partial

import argparse

def compose(*fns):
    return reduce(
        lambda fn, fns:
            lambda x: fn(fns(x)),
        fns)


as_uint8 = lambda x: x.astype('uint8')

def base_pipeline(args):
    color_stage = partial(correct_color, args)
    snip_stage = snip
    orientate_stage = correct_orientation

    if not any([args.snip, args.color, args.orientate]):
        return compose(color_stage, snip_stage, orientate_stage)
    else:
        pipeline_candidates = [
            args.color and color_stage,
            args.snip and snip_stage,
            args.orientate and orientate_stage,
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

def _add_optional_boolean_flag(parser, arg, help = None):
    return parser.add_argument(arg, action = 'store_true', default = None, help = help)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = dedent("""
            Pre-processes images of fibre slides, taken as per the 'taking-photos-of-slides' guide. After pre-processing, the resulting images are usable with the model.
        """),
        formatter_class = argparse.RawTextHelpFormatter
    )

    parser.add_argument('input_dir', help = 'directory including images')
    parser.add_argument('output_dir', help = 'directory to put the resulting images in')

    pipeline_stage_groups = parser.add_argument_group('select pipeline stages', dedent("""
        By default, all stages of the preprocessing pipeline are run in this order:

        1. orientate stage: finds the aruco orientation markers in the image, and corrects perspective warp
        2. snip stage: reduces the image to the black background area in the template
        3. color stage: selects a range of hues to highlight.  By default, calibrated for finding green fluorescent fibres.

        If you want to pick-and-choose which stages to run, use any combination of the flags below
    """))

    _add_optional_boolean_flag(pipeline_stage_groups, '--snip',      help = 'explicitly enable the snip stage')
    _add_optional_boolean_flag(pipeline_stage_groups, '--orientate', help = 'explicitly enable the orientate stage')
    _add_optional_boolean_flag(pipeline_stage_groups, '--color',     help = 'explicitly enable the color stage')


    color_stage_group = parser.add_argument_group('color stage settings', dedent("""
        By default, the color stage's hue min and max are calibrated for highlighting green fluorescent fibres.
        If your fibres are of a different color, shade, or variability, you may need to select different values.
    """))
    color_stage_group.add_argument('--min-hue', help = 'the maximum hue to highlight in the color stage', type = float)
    color_stage_group.add_argument('--max-hue', help = 'the minimum hue to highlight in the color stage', type = float)

    args = parser.parse_args()
    pipeline = create_pipeline(args)
    _preprocess_all(args.input_dir, args.output_dir, pipeline)
