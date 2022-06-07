"""Submodule providing edge-label prediction models based on Sklearn Models."""
from embiggen.edge_label_prediction.edge_label_prediction_sklearn.sklearn_edge_label_prediction_adapter import SklearnEdgeLabelPredictionAdapter
from embiggen.utils.abstract_models import build_init

build_init(
    module_library_names="sklearn",
    formatted_library_name="scikit-learn",
    expected_parent_class=SklearnEdgeLabelPredictionAdapter
)