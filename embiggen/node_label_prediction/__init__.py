"""Submodule providing models for edge-label prediction."""
from embiggen.node_label_prediction.sklearn_like_node_label_prediction_adapter import *
from embiggen.node_label_prediction.node_label_prediction_sklearn import *
from embiggen.node_label_prediction.node_label_prediction_tensorflow import *
from embiggen.node_label_prediction.node_label_prediction_lightgbm import *
from embiggen.node_label_prediction.node_label_prediction_catboost import *
from embiggen.node_label_prediction.node_label_prediction_xgboost import *
from embiggen.node_label_prediction.node_label_prediction_evaluation import node_label_prediction_evaluation

# Export all non-internals.
__all__ = [
    variable_name
    for variable_name in locals().keys()
    if not variable_name.startswith("_")
]
