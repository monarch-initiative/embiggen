"""Submodule providing node-label prediction models based on LightGBM Models."""
from embiggen.node_label_prediction.sklearn_like_node_label_prediction_adapter import (
    SklearnLikeNodeLabelPredictionAdapter,
)
from embiggen.utils.abstract_models import build_init

build_init(
    module_library_names="lightgbm",
    formatted_library_name="LightGBM",
    task_name="Node Label Prediction",
    expected_parent_class=SklearnLikeNodeLabelPredictionAdapter,
)
