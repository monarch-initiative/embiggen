"""Module with Keras Sequences."""
from .node2vec_sequence import Node2VecSequence
from .edge_prediction_training_sequence import EdgePredictionTrainingSequence
from .gcn_edge_prediction_training_sequence import GCNEdgePredictionTrainingSequence
from .gcn_edge_prediction_sequence import GCNEdgePredictionSequence
from .gcn_edge_label_prediction_training_sequence import GCNEdgeLabelPredictionTrainingSequence
from .edge_prediction_sequence import EdgePredictionSequence
from .siamese_sequence import SiameseSequence
from .kgsiamese_sequence import KGSiameseSequence

__all__ = [
    "Node2VecSequence",
    "EdgePredictionTrainingSequence",
    "EdgePredictionSequence",
    "GCNEdgePredictionTrainingSequence",
    "GCNEdgePredictionSequence",
    "GCNEdgeLabelPredictionTrainingSequence",
    "SiameseSequence",
    "KGSiameseSequence"
]
