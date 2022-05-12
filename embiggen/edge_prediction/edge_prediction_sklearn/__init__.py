"""Submodule providing edge prediction models based on Sklearn Models."""
from .sklearn_edge_prediction_adapter import SklearnEdgePredictionAdapter
from ...utils import build_init

build_init(
    module_library_name="sklearn",
    formatted_library_name="scikit-learn",
    expected_parent_class=SklearnEdgePredictionAdapter
)