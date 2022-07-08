"""Module with method to embed graphs using a precomputed embedding."""
from embiggen.embedding_transformers.edge_transformer import EdgeTransformer
from embiggen.embedding_transformers.node_transformer import NodeTransformer
from embiggen.embedding_transformers.graph_transformer import GraphTransformer
from embiggen.embedding_transformers.edge_prediction_transformer import EdgePredictionTransformer
from embiggen.embedding_transformers.edge_label_prediction_transformer import EdgeLabelPredictionTransformer
from embiggen.embedding_transformers.node_label_prediction_transformer import NodeLabelPredictionTransformer

__all__ = [
    "EdgeTransformer",
    "NodeTransformer",
    "GraphTransformer",
    "EdgePredictionTransformer",
    "EdgeLabelPredictionTransformer",
    "NodeLabelPredictionTransformer"
]
