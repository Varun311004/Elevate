"""
tensorflow_decision_forests — minimal stub.

tensorflowjs 4.x unconditionally imports this module at the top of
``tf_saved_model_conversion_v2.py`` even when converting plain Keras / MobileNet
models that have nothing to do with TF-DF.

The real ``tensorflow-decision-forests`` package requires tensorflow~=2.15 and
therefore cannot be installed alongside tensorflow 2.16.x.  This stub satisfies
the bare import so the tensorflowjs converter can load, while the actual TF-DF
code path is never reached during Keras model conversion.

Do NOT add real TF-DF functionality here — this file must stay dependency-free.
"""
