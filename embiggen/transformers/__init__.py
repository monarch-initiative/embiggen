"""Module with method to embed graphs using a precomputed embedding."""
from .edge_transformer import EdgeTransformer
from .node_transformer import NodeTransformer
from .graph_transformer import GraphTransformer
from .link_prediction_transformer import LinkPredictionTransformer

import warnings
try:
    from .corpus_transformer import CorpusTransformer
except ModuleNotFoundError:
    warnings.warn(
        "You do not have TensorFlow installed, all tensorflow models "
        "will not work."
    )

__all__ = [
    "EdgeTransformer",
    "NodeTransformer",
    "GraphTransformer",
    "CorpusTransformer",
    "LinkPredictionTransformer"
]
