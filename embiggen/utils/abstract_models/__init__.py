"""Submodule providing abstract models."""
from .abstract_classifier_model import AbstractClassifierModel
from .abstract_embedding_model import AbstractEmbeddingModel
from .abstract_model import abstract_class, AbstractModel
from .get_models_dataframe import (
    get_models_dataframe,
    get_available_models_for_edge_label_prediction,
    get_available_models_for_edge_prediction,
    get_available_models_for_node_label_prediction,
    get_available_models_for_node_embedding
)
from .auto_init import build_init

__all__ = [
    "AbstractClassifierModel",
    "AbstractEmbeddingModel",
    "get_standardized_model_map",
    "abstract_class",
    "AbstractModel",
    "get_models_dataframe",
    "get_available_models_for_edge_label_prediction",
    "get_available_models_for_edge_prediction",
    "get_available_models_for_node_label_prediction",
    "get_available_models_for_node_embedding",
    "build_init"
]
