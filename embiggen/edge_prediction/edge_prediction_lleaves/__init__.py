"""Submodule providing edge prediction models based on LightGBM Models."""
from embiggen.edge_prediction.sklearn_like_edge_prediction_adapter import (
    SklearnLikeEdgePredictionAdapter,
)
from embiggen.utils.abstract_models import build_init

build_init(
    module_library_names=["lleaves", "lightgbm"],
    formatted_library_name="lleaves",
    task_name="Edge Prediction",
    expected_parent_class=SklearnLikeEdgePredictionAdapter,
)
