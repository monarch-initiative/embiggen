"""Submodule providing node-label prediction models based on TensorFlow."""
from ..node_label_prediction_model import AbstractNodeLabelPredictionModel
from ...utils import build_init

build_init(
    module_library_names="tensorflow",
    formatted_library_name="TensorFlow",
    expected_parent_class=AbstractNodeLabelPredictionModel
)
