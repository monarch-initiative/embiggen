"""Module with Keras Sequences."""
from .node2vec_sequence import Node2VecSequence
from .edge_prediction_sequence import EdgePredictionSequence
from .edge_label_prediction_sequence import EdgeLabelPredictionSequence
from .node_label_prediction_sequence import NodeLabelPredictionSequence
from .gnn_edge_prediction_sequence import GNNEdgePredictionSequence
from .gnn_bipartite_edge_prediction_sequence import GNNBipartiteEdgePredictionSequence
from .siamese_sequence import SiameseSequence
from .kgsiamese_sequence import KGSiameseSequence

__all__ = [
    "Node2VecSequence",
    "EdgePredictionSequence",
    "EdgeLabelPredictionSequence",
    "NodeLabelPredictionSequence",
    "GNNEdgePredictionSequence",
    "GNNBipartiteEdgePredictionSequence",
    "SiameseSequence",
    "KGSiameseSequence"
]
