from models.wrapper import Wrapper

import keras.models as models
from keras.layers import Dense
import numpy as np
import pytest

class MyWrapper(Wrapper):
    pass

def init_model():
    return models.Sequential([
        Dense(1, input_shape = (1,))
    ], name = 'model_name')

def test_wrapper_init():
    model = init_model()
    wrapped_model = MyWrapper(model)

    assert wrapped_model.model != model
    assert isinstance(wrapped_model.model, models.Sequential)
    assert wrapped_model.model.name == model.name

    assert wrapped_model.name == 'MyWrapper-model_name'
    assert model.name == 'model_name'

    np.testing.assert_equal(
        wrapped_model.model.get_weights(),
        model.get_weights(),
    )

def test_wrapper_throws_if_it_recieves_already_wrapped_model():
    model = init_model()

    with pytest.raises(Exception):
        MyWrapper(MyWrapper(model))


def test_wrapper_resolve():
    model = init_model()
    wrapped_model = MyWrapper(model)

    assert Wrapper.resolve(wrapped_model.name) == MyWrapper

    class SubWrapper(MyWrapper):
        pass

    wrapped_model_b = SubWrapper(model)
    assert Wrapper.resolve(wrapped_model_b.name) == SubWrapper


def test_wrapper_resolve_throws_if_model_is_untagged():
    model = init_model()

    with pytest.raises(Exception):
        Wrapper.resolve(model)

def test_resolve_uses_latest_wrapper():
    """
    This is mostly in place to support interactive use within Jupyter.
    We want to resolve the version of the wrapper that the user can
    actually _see_ - the latest version they wrote
    """
    global MyWrapper
    TestWrapperGlobal = MyWrapper

    class MyWrapper(Wrapper):
        pass

    model = models.Sequential([], name = 'model_name')
    wrapped_model = TestWrapperGlobal(model)

    assert Wrapper.resolve(wrapped_model.name) == MyWrapper
    assert Wrapper.resolve(wrapped_model.name) != TestWrapperGlobal


def test_wrapper_delegates_to_underlying_model():
    model = init_model()
    wrapped_model = Wrapper(model)

    assert wrapped_model.summary == wrapped_model.model.summary
    assert wrapped_model.fit     == wrapped_model.model.fit
    assert wrapped_model.predict == wrapped_model.model.predict

def test_post_process_defaults_to_identity_function():
    model = init_model()
    wrapped_model = Wrapper(model)

    prediction = np.arange(64 * 64 * 1).reshape(1, 64, 64, 1)
    np.testing.assert_equal(
        wrapped_model.post_process(prediction),
        prediction
    )

def test_resize_defaults_to_throwing_not_implemented():
    wrapped_model = Wrapper(init_model())

    with pytest.raises(NotImplementedError):
        wrapped_model.resize((128, 128))

def test_compile_defaults_to_throwing_not_implemented():
    wrapped_model = Wrapper(init_model())

    with pytest.raises(NotImplementedError):
        wrapped_model.compile()

