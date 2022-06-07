"""Submodule providing node embedding models implemented in Ensmallen in Rust."""
from embiggen.edge_prediction.edge_prediction_model import AbstractEdgePredictionModel
from embiggen.utils.abstract_models import build_init

build_init(
    module_library_names="ensmallen",
    formatted_library_name="Ensmallen",
    expected_parent_class=AbstractEdgePredictionModel
)
