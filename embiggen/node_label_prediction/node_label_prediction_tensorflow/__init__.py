"""Submodule providing node-label prediction models based on TensorFlow."""
from embiggen.node_label_prediction.node_label_prediction_model import AbstractNodeLabelPredictionModel
from embiggen.utils.abstract_models import build_init

build_init(
    module_library_names="tensorflow",
    formatted_library_name="TensorFlow",
    expected_parent_class=AbstractNodeLabelPredictionModel
)
