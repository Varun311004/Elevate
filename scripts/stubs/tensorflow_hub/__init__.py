"""
tensorflow_hub — minimal stub for tensorflowjs converter compatibility.

The installed tensorflow_hub version uses ``tf.compat.v1.estimator`` in
``estimator.py``, which was removed in TensorFlow 2.16.x, causing:

    AttributeError: module 'tensorflow.compat.v1' has no attribute 'estimator'

tensorflowjs 4.x unconditionally imports tensorflow_hub in
``tf_saved_model_conversion_v2.py`` even when converting plain Keras models
that contain no TF Hub layers at all.

This stub satisfies the import, providing no-op versions of the symbols
tensorflowjs actually calls.  For our MobileNetV3 emotion model:
  - hub.resolve()          → returns the path unchanged (no Hub URL to resolve)
  - isinstance(x, KerasLayer) → always False (we have no Hub layers)
  - load()                 → never called for non-Hub models

Do NOT add real TF Hub functionality here — this file must stay dep-free.
"""


class KerasLayer:
    """
    Stub for ``hub.KerasLayer``.

    tensorflowjs uses ``isinstance(layer, hub.KerasLayer)`` to detect Hub
    layers in the graph.  Since our model has no Hub layers, this check always
    returns False and the Hub-specific code path is never entered.
    """


class Module:
    """Stub for the deprecated ``hub.Module`` class."""


def resolve(handle, *args, **kwargs):
    """
    Return the handle unchanged.

    The real implementation resolves Hub URLs to local paths.  Our model
    path is already a local SavedModel directory — no resolution needed.
    """
    return str(handle)


def load(handle, *args, **kwargs):
    """
    Raise a clear error if someone accidentally calls hub.load() via stub.

    This is never called during plain Keras model conversion.
    """
    raise NotImplementedError(
        "tensorflow_hub.load() is not available in stub mode.  "
        "Plain Keras model conversion does not require Hub support.  "
        "Install tensorflow-hub manually if you need to convert Hub models."
    )


# Provide a minimal 'estimator' sub-namespace so that any accidental
# ``hub.estimator.*`` reference doesn't raise AttributeError.
class _EstimatorStub:
    pass


estimator = _EstimatorStub()
