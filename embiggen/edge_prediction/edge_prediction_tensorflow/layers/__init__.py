"""Module with edge embedding layers."""
from embiggen.edge_prediction.edge_prediction_tensorflow.layers.concatenate_edge_embedding import ConcatenateEdgeEmbedding
from embiggen.edge_prediction.edge_prediction_tensorflow.layers.l1_edge_embedding import L1EdgeEmbedding
from embiggen.edge_prediction.edge_prediction_tensorflow.layers.l2_edge_embedding import L2EdgeEmbedding
from embiggen.edge_prediction.edge_prediction_tensorflow.layers.average_edge_embedding import AverageEdgeEmbedding
from embiggen.edge_prediction.edge_prediction_tensorflow.layers.sum_edge_embedding import SumEdgeEmbedding
from embiggen.edge_prediction.edge_prediction_tensorflow.layers.min_edge_embedding import MinEdgeEmbedding
from embiggen.edge_prediction.edge_prediction_tensorflow.layers.max_edge_embedding import MaxEdgeEmbedding
from embiggen.edge_prediction.edge_prediction_tensorflow.layers.hadamard_edge_embedding import HadamardEdgeEmbedding

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
