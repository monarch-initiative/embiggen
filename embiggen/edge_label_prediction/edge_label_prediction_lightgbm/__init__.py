"""Submodule providing edge-label prediction models based on LightGBM Models."""
from embiggen.edge_label_prediction.sklearn_like_edge_label_prediction_adapter import (
    SklearnLikeEdgeLabelPredictionAdapter,
)
from embiggen.utils.abstract_models import build_init

build_init(
    module_library_names="lightgbm",
    formatted_library_name="LightGBM",
    task_name="Edge Label Prediction",
    expected_parent_class=SklearnLikeEdgeLabelPredictionAdapter,
)
