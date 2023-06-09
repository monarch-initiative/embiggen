"""Submodule providing models for edge-label prediction."""
from embiggen.edge_label_prediction.edge_label_prediction_sklearn import *
from embiggen.edge_label_prediction.edge_label_prediction_tensorflow import *
from embiggen.edge_label_prediction.sklearn_like_edge_label_prediction_adapter import *
from embiggen.edge_label_prediction.edge_label_prediction_catboost import *
from embiggen.edge_label_prediction.edge_label_prediction_xgboost import *
from embiggen.edge_label_prediction.edge_label_prediction_evaluation import edge_label_prediction_evaluation

# Export all non-internals.
__all__ = [
    variable_name
    for variable_name in locals().keys()
    if not variable_name.startswith("_")
]
