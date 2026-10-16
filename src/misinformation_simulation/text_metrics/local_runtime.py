"""Bound memory use by serializing local model loading and inference."""

from functools import wraps
from threading import RLock

_model_lock = RLock()


def serialized_inference(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with _model_lock:
            return function(*args, **kwargs)

    for name in ("cache_clear", "cache_info", "cache_parameters"):
        if hasattr(function, name):
            setattr(wrapped, name, getattr(function, name))
    return wrapped
