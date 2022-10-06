"""Submodule providing node embedding models implemented in Ensmallen in Rust."""
from embiggen.node_label_prediction.node_label_prediction_model import AbstractNodeLabelPredictionModel
from embiggen.utils.abstract_models import build_init

build_init(
    module_library_names="ensmallen",
    formatted_library_name="Ensmallen",
    expected_parent_class=AbstractNodeLabelPredictionModel
)
