"""Module with edge embedding layers."""
from .concatenate_edge_embedding import ConcatenateEdgeEmbedding
from .l1_edge_embedding import L1EdgeEmbedding
from .l2_edge_embedding import L2EdgeEmbedding
from .average_edge_embedding import AverageEdgeEmbedding
from .sum_edge_embedding import SumEdgeEmbedding
from .min_edge_embedding import MinEdgeEmbedding
from .max_edge_embedding import MaxEdgeEmbedding
from .hadamard_edge_embedding import HadamardEdgeEmbedding

edge_embedding_layer = {
    "Concatenate": ConcatenateEdgeEmbedding,
    "L1": L1EdgeEmbedding,
    "L2": L2EdgeEmbedding,
    "Average": AverageEdgeEmbedding,
    "Hadamard": HadamardEdgeEmbedding,
    "Min": MinEdgeEmbedding,
    "Max": MaxEdgeEmbedding,
    "Sum": SumEdgeEmbedding,
}

__all__ = [
    "ConcatenateEdgeEmbedding",
    "L1EdgeEmbedding",
    "L2EdgeEmbedding",
    "HadamardEdgeEmbedding",
    "AverageEdgeEmbedding",
    "MinEdgeEmbedding",
    "MaxEdgeEmbedding",
    "SumEdgeEmbedding"
]
