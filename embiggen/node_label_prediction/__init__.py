"""Submodule providing models for edge-label prediction."""
from .node_label_prediction_sklearn import SklearnModelNodeLabelPredictionAdapter

__all__ = [
    "SklearnModelNodeLabelPredictionAdapter"
]
