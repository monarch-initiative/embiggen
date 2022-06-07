"""Submodule providing edge-label prediction models based on Sklearn Models."""
from embiggen.edge_label_prediction.edge_label_prediction_model import AbstractEdgeLabelPredictionModel
from embiggen.utils.abstract_models import build_init

build_init(
    module_library_names="tensorflow",
    formatted_library_name="TensorFlow",
    expected_parent_class=AbstractEdgeLabelPredictionModel
)