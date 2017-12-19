# fluorescent-fibre-counting

This project aims to estimate the number of fluorescent fibres present in a UV-illuminated slide image.

## Installation

After installing [conda](https://docs.anaconda.com/anaconda/install/):

```bash
$ git clone https://github.com/hughfdjackson/fluorescent-fibre-counting.git
$ cd fluorescent-fibre-counting
$ conda env create --file environment
$ source activate fluorescent-fibre-counting
```

## Usage

Counting fluorescent fibres involves:

1. Taking an image of your slide under UV light
2. Preprocessing the images
3. Feeding the preprocessed images to the model

### Taking images under UV light

Refer to the ["taking photos of slides" guide](https://github.com/hughfdjackson/fluorescent-fibre-counting/wiki/Taking-photos-of-Slides) for more details.

### Preprocessing images

Once you've gotten your images (preferably in an uncompressed `.tiff` format):

```bash
$ source activate fluorescent-fibre-counting
$ python -m preprocess
```

The `--min-hue` and `--max-hue` values should be adjusted to best highlight the colour of fibre you're attempting to isolate.

For more details, see the ["image preprocessing" guide](guides/image-preprocessing.ipynb)

### Counting fibres

To obtain an estimate of the count from pre-processed images:

```bash
$ source activate fluorescent-fibre-counting
$ python -m count.py <directory of preprocessed images> results.csv
```


## Running on the GPU

Since this project uses tensorflow.  To install with GPU support:

1. find the installation instructions for your platform from https://www.tensorflow.org/install/
2. follow instructions to prepare your platform for use with tensorflow-gpu
3. change `tensorflow` to `tensorflow-gpu` in `environment.yaml`
4. initialise the environment with `conda env create --file environment.yaml`