"""Module with Keras Sequences."""
from embiggen.sequences.tensorflow_sequences.node2vec_sequence import Node2VecSequence
from embiggen.sequences.tensorflow_sequences.edge_prediction_training_sequence import EdgePredictionTrainingSequence
from embiggen.sequences.tensorflow_sequences.gcn_edge_prediction_training_sequence import GCNEdgePredictionTrainingSequence
from embiggen.sequences.tensorflow_sequences.gcn_edge_prediction_sequence import GCNEdgePredictionSequence
from embiggen.sequences.tensorflow_sequences.gcn_edge_label_prediction_training_sequence import GCNEdgeLabelPredictionTrainingSequence
from embiggen.sequences.tensorflow_sequences.edge_prediction_sequence import EdgePredictionSequence
from embiggen.sequences.tensorflow_sequences.siamese_sequence import SiameseSequence
from embiggen.sequences.tensorflow_sequences.kgsiamese_sequence import KGSiameseSequence

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
