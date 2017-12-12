from keras.models import clone_model
import numpy as np

def _clone(model):
    m2 = clone_model(model)
    if model.weights:
        m2.set_weights(model.get_weights())

    m2.trainable = model.trainable
    return m2

def _has_tag(name):
    return '-' in name

def _get_tag(name):
    return name.split('-')[0]

def _all_subclasses(cls):
    subclasses = cls.__subclasses__()
    subsubclasses = sum((_all_subclasses(s) for s in subclasses), [])
    return subclasses + subsubclasses

def _find_subclass(cls, name):
    subclasses = _all_subclasses(cls)
    matching = [c for c in subclasses if c.__name__ == name]

    if len(matching) == 0: return None
    else: return matching[-1]

class Wrapper:
    """
    Provides a place for:
    - post-processing hooks (deriving a count from density maps)
    - resizing
    """

    _pass_through = [
        'summary', 'predict',
        'fit', 'compile',
        'input_shape', 'output_shape',
        'save'
    ]

    def __init__(self, model):
        assert not isinstance(model, Wrapper)

        self.name = self.__class__.__name__ + '-' + model.name
        self.model = _clone(model)

    def __getattr__(self, name):
        if name in self.__class__._pass_through:
            return getattr(self.model, name)

    def post_process(self, predictions):
        return predictions

    def resize(self, target_size):
        raise NotImplementedError()

    def compile(self):
        raise NotImplementedError()

    def predict_one(self, x):
        return self.predict(np.array([x]))[0]

    def post_process_one(self, prediction):
        return self.post_process(np.array([prediction]))[0]

    @staticmethod
    def resolve(tagged_name):
        """
        Loads the correct wrapper for a model.
        """
        assert _has_tag(tagged_name)
        return _find_subclass(Wrapper, _get_tag(tagged_name))