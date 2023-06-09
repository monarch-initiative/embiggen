"""Submodule providing edge-label prediction models based on XGBoost Models."""
from embiggen.edge_label_prediction.sklearn_like_edge_label_prediction_adapter import (
    SklearnLikeEdgeLabelPredictionAdapter,
)
from embiggen.utils.abstract_models import build_init

build_init(
    module_library_names="xgboost",
    formatted_library_name="XGBoost",
    task_name="Edge-label Prediction",
    expected_parent_class=SklearnLikeEdgeLabelPredictionAdapter,
)
