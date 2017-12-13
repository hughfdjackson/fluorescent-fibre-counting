# Fluorescent-Fibre-Counting


## Running on the GPU

Automatically counting fibres on an appropriate GPU can speed up execution by a large degree.  To install with GPU support:

1. find the installation instructions for your platform from https://www.tensorflow.org/install/
2. follow instructions to prepare your platform for use with tensorflow-gpu
3. change `tensorflow` to `tensorflow-gpu` in `environment.yaml`
4. initialise the environment with `conda env create --file environment.yaml`