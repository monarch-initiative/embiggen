"""Submodule providing edge prediction models based on CatBoost Models."""
from embiggen.edge_prediction.sklearn_like_edge_prediction_adapter import (
    SklearnLikeEdgePredictionAdapter,
)
from embiggen.utils.abstract_models import build_init

build_init(
    module_library_names="catboost",
    formatted_library_name="CatBoost",
    task_name="Edge Prediction",
    expected_parent_class=SklearnLikeEdgePredictionAdapter,
)
