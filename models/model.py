from models.wrapper import Wrapper, _clone

from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.optimizers import SGD
from keras.models import load_model

import numpy as np
import pandas as pd

import tempfile
import os.path as path
from os import listdir, makedirs
import re

from functools import reduce


clone = _clone

def sequence_layers(layers):
    return reduce(lambda l1, l2: l2(l1), layers)

def inner_layer(layers):
    return reduce(lambda l1, l2: lambda x: l2(l1(x)), layers)


def load(file_path):
    file_base_name = path.splitext(path.basename(file_path))[0]
    model = load_model(file_path)
    model.name = file_base_name.split('-')[-1]
    return Wrapper.resolve(file_base_name)(model)

def save_model(model, output_directory):
    model_path = path.join(output_directory,  '{}.hdf5'.format(model.name))
    model.save(model_path)


## Save results from training + testing ##
_number_re = re.compile('^[0-9]+$')

def _is_build_dir(item_path):
    return path.isdir(item_path) and _number_re.match(path.basename(item_path))

def _list_build_directories(model_build_path):
    try:
        items = listdir(model_build_path)
        return [i for i in items if _is_build_dir(path.join(model_build_path, i))]
    except (NotADirectoryError, FileNotFoundError):
        return []

def save_build(model, training_history = None, test_results = None, predictions = None):
    base_path = path.dirname(path.abspath(__file__))

    # increment build number
    model_build_path = path.abspath(path.join(base_path, '..', 'builds', model.name))
    builds = _list_build_directories(model_build_path)
    build_number = len(builds) + 1

    # create new build dir
    build_dir = path.join(model_build_path, str(build_number))
    makedirs(build_dir, exist_ok = True)

    # save results
    save_all(
        model,
        test_results,
        training_history,
        output_directory = build_dir,
        predictions = predictions
    )


def save_predictions(model, predictions, test_labels, output_directory):
    predictions_path = path.join(output_directory, 'predictions.npz')
    arrays = { l: np.round(p[0], decimals = 2) for l, p in zip(test_labels, predictions) }

    np.savez_compressed(predictions_path, **arrays)

def save_training_history(model, history, output_directory):
    history_path = path.join(
        output_directory,
        'training_history.csv'
    )

    pd.DataFrame({
        'epoch': history.epoch,
        'loss': history.history['loss']
    }).to_csv(history_path)

def save_test_results(model, test_results, output_directory):
    test_path = path.join(
        output_directory,
        'test_results.csv'
    )
    test_results.to_csv(test_path)

def save_all(model, test_results, training_history, output_directory, predictions):
    _print_title_card(
        title = model.name,
        message = 'saving build to {}'.format(output_directory)
    )

    save_model(model, output_directory)

    if not test_results is None:
        save_test_results(model, test_results, output_directory)

    if not training_history is None:
        save_training_history(model, training_history, output_directory)

    if not predictions is None:
        save_predictions(model, predictions, test_results.label, output_directory)

## Testing ##

def test(model, test_set, test_labels = None):
    _print_title_card(
        title = model.name,
        message = 'testing model.'
    )
    images, density_maps, masks, counts = test_set

    def predict(i, image):
        print('predicting image {}'.format(i))
        return model.predict_one(image)

    # Each image has to be predicted separately
    # due to the potentially large size exhausting GPU
    # resources.
    predictions = [
        predict(i, image) for i, image in enumerate(images)
    ]

    predicted_counts = [
        model.post_process_one(p) for p in predictions
    ]

    _, w, h, _ = model.input_shape

    return predictions, pd.DataFrame({
        'model': [model.name] * len(images),
        'index': range(len(images)),
        'label': range(len(images)) if test_labels is None else test_labels,
        'predicted_count': predicted_counts,
        'actual_count':  counts,
        'image_dims': str(w) + 'x' + str(h),
        'density': counts / (w * h)
    })

## Predictions ##
def predict(model, images, image_labels):
    _print_title_card(
        title = model.name,
        message = 'predicting fluorescent fibre counts using {}.'.format(model.name)
    )

    def count_one(image, label):
        print('predicting image {}'.format(label))
        return model.post_process_one(model.predict_one(image))

    predicted_counts = [
        count_one(image, label) for label, image in zip(image_labels, images)
    ]

    return pd.DataFrame({
        'model': [model.name] * len(image_labels),
        'label': image_labels,
        'predicted_count': predicted_counts,
    })



## Training ##

def _model_checkpoint_tmp_path(model):
    return path.join(
        tempfile.gettempdir(),
        model.name + '.hdf5'
    )

def _print_title_card(title, message):
    top_line = ('-' * 20) + ' {} '.format(title) + ('-' * 20)
    bottom_line = '-' * len(top_line)

    print(top_line)
    print(message)
    print(bottom_line)


## Keras callbacks + training ##
def _early_stopping():
    return EarlyStopping(
        monitor = 'val_loss',
        min_delta = 0.0001,
        patience = 1,
        mode = 'auto'
    )

def _model_checkpoint(file_path):
    return ModelCheckpoint(
        file_path,
        monitor = 'val_loss',
        save_best_only = True
    )

def _learning_rate_scheduler(step = 20, initial_learning_rate = 1e-3, minimum_learning_rate = 1e-5):
    def anneal(epoch):
        learning_rate = initial_learning_rate / (10 ** (epoch // step))

        print('learning rate for epoch {} is {}'.format(epoch + 1, learning_rate))
        return np.maximum(learning_rate, minimum_learning_rate)
    return LearningRateScheduler(anneal)


def train(model, training_set):
    data, *labels = training_set
    checkpoint_path = _model_checkpoint_tmp_path(model)
    model.compile()

    _print_title_card(
        title = model.name,
        message = 'checkpointed at {}'.format(checkpoint_path)
    )

    training_history = model.fit(
              data,
              labels,
              epochs = 50,
              validation_split = 0.2,
              callbacks = [
                _model_checkpoint(checkpoint_path),
                _early_stopping(),
                _learning_rate_scheduler(),
            ])

    # return the very best model
    return (load(checkpoint_path), training_history)
