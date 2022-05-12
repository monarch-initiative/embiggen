"""Submodule providing models for edge-label prediction."""
from .node_label_prediction_sklearn import *
from .node_label_prediction_tensorflow import *

# Export all non-internals.
__all__ = [
    variable_name
    for variable_name in locals().keys()
    if not variable_name.startswith("_")
]
