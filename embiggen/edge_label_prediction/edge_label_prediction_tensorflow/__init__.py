"""Submodule providing edge-label prediction models based on Sklearn Models."""
from ..edge_label_prediction_model import AbstractEdgeLabelPredictionModel
from ...utils import build_init

build_init(
    module_library_names="tensorflow",
    formatted_library_name="TensorFlow",
    expected_parent_class=AbstractEdgeLabelPredictionModel
)