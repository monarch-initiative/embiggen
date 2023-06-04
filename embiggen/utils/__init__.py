"""Sub-module with utilities to make experiments easier to write."""
from embiggen.utils.abstract_models import (
    AbstractClassifierModel,
    AbstractEmbeddingModel,
    EmbeddingResult,
    AbstractModel,
    AbstractFeaturePreprocessor,
    get_models_dataframe,
    get_available_models_for_node_label_prediction,
    get_available_models_for_edge_prediction,
    get_available_models_for_edge_embedding,
    get_available_models_for_edge_label_prediction,
    get_available_models_for_node_embedding,
    abstract_class,
    format_list
)
from embiggen.utils.pipeline import classification_evaluation_pipeline
from embiggen.utils.number_to_ordinal import number_to_ordinal
from embiggen.utils.normalize_kwargs import normalize_kwargs
from embiggen.utils.abstract_edge_feature import AbstractEdgeFeature
from embiggen.utils.abstract_feature import AbstractFeature

__all__ = [
    "AbstractClassifierModel",
    "AbstractEmbeddingModel",
    "AbstractFeaturePreprocessor",
    "EmbeddingResult",
    "AbstractModel",
    "classification_evaluation_pipeline",
    "format_list",
    "get_models_dataframe",
    "get_available_models_for_node_label_prediction",
    "get_available_models_for_edge_prediction",
    "get_available_models_for_edge_embedding",
    "get_available_models_for_edge_label_prediction",
    "get_available_models_for_node_embedding",
    "abstract_class",
    "number_to_ordinal",
    "normalize_kwargs",
    "AbstractEdgeFeature",
    "AbstractFeature"
]
