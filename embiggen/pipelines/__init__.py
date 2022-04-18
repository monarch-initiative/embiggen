"""Submodule providing experiments pipelines."""
import warnings
from .compute_node_embedding import (
    compute_node_embedding,
    get_available_node_embedding_methods
)
try:
    # TODO!: add non-tensorflow models to the edge embedding pipeline
    # such as the models from Sklearn.
    from .edge_prediction import evaluate_embedding_for_edge_prediction
    __all__ = [
        "compute_node_embedding",
        "get_available_node_embedding_methods",
        "evaluate_embedding_for_edge_prediction"
    ]
except:
    __all__ = [
        "compute_node_embedding",
        "get_available_node_embedding_methods",
    ]
