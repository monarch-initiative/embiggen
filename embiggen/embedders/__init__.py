"""Submodule providing TensorFlow and Ensmallen-based embedders."""
from typing import List
from .ensmallen_embedders import *
try:
    from tensorflow_embedders import *
except ModuleNotFoundError as e:
    pass


def get_available_libraries() -> List[str]:
    """Return names of the available embedding libraries."""
    available_libraries = ["Ensmallen"]
    try:
        import tensorflow
        available_libraries.append("TensorFlow")
    except ModuleNotFoundError as e:
        pass

    return available_libraries
