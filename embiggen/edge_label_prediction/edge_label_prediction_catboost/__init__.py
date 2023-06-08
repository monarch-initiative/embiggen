"""Submodule providing Edge-label prediction models based on CatBoost Models."""
from embiggen.edge_label_prediction.sklearn_like_edge_label_prediction_adapter import (
    SklearnLikeEdgeLabelPredictionAdapter,
)
from embiggen.utils.abstract_models import build_init

build_init(
    module_library_names="catboost",
    formatted_library_name="CatBoost",
    task_name="Edge-label Prediction",
    expected_parent_class=SklearnLikeEdgeLabelPredictionAdapter,
)
