"""Submodule providing edge prediction models based on TensorFlow Models."""
from ..edge_prediction_model import AbstractEdgePredictionModel
from ...utils import build_init

build_init(
    module_library_names="tensorflow",
    formatted_library_name="TensorFlow",
    expected_parent_class=AbstractEdgePredictionModel
)