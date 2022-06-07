"""Sub-module with utilities to make experiments easier to write."""
from embiggen.utils.abstract_models import (
    AbstractClassifierModel,
    AbstractEmbeddingModel,
    EmbeddingResult,
    AbstractModel,
    get_models_dataframe,
    get_available_models_for_node_label_prediction,
    get_available_models_for_edge_prediction,
    get_available_models_for_edge_label_prediction,
    get_available_models_for_node_embedding,
    abstract_class,
    format_list
)
from embiggen.utils.pipeline import classification_evaluation_pipeline
from embiggen.utils.number_to_ordinal import number_to_ordinal

__all__ = [
    "AbstractClassifierModel",
    "AbstractEmbeddingModel",
    "EmbeddingResult",
    "AbstractModel",
    "classification_evaluation_pipeline",
    "build_init",
    "format_list",
    "get_models_dataframe",
    "get_available_models_for_node_label_prediction",
    "get_available_models_for_edge_prediction",
    "get_available_models_for_edge_label_prediction",
    "get_available_models_for_node_embedding",
    "abstract_class",
    "number_to_ordinal",
    "format_list"
]
