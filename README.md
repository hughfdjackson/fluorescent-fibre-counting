# fluorescent-fibre-counting

This project aims to estimate the number of fluorescent fibres present in a UV-illuminated slide image.

## Installation

After installing [conda](https://docs.anaconda.com/anaconda/install/)

```bash
$ git clone https://github.com/hughfdjackson/fluorescent-fibre-counting.git
$ cd fluorescent-fibre-counting
$ conda env create --file environment
$ source activate fluorescent-fibre-counting
```

## Usage

Obtaining a count of fluorescent fibres involves two steps - obtaining preprocessed images of the fibres, and then counting them with the model.

### Taking and pre-processing images

TODO

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