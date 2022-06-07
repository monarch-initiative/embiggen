"""Submodule providing abstract models."""
from embiggen.utils.abstract_models.abstract_classifier_model import AbstractClassifierModel
from embiggen.utils.abstract_models.abstract_embedding_model import AbstractEmbeddingModel, EmbeddingResult
from embiggen.utils.abstract_models.abstract_model import abstract_class, AbstractModel
from embiggen.utils.abstract_models.get_models_dataframe import (
    get_models_dataframe,
    get_available_models_for_edge_label_prediction,
    get_available_models_for_edge_prediction,
    get_available_models_for_node_label_prediction,
    get_available_models_for_node_embedding
)
from embiggen.utils.abstract_models.auto_init import build_init
from embiggen.utils.abstract_models.list_formatting import format_list

__all__ = [
    "AbstractClassifierModel",
    "AbstractEmbeddingModel",
    "EmbeddingResult",
    "get_standardized_model_map",
    "abstract_class",
    "AbstractModel",
    "get_models_dataframe",
    "get_available_models_for_edge_label_prediction",
    "get_available_models_for_edge_prediction",
    "get_available_models_for_node_label_prediction",
    "get_available_models_for_node_embedding",
    "build_init",
    "format_list"
]
