"""Module with models for graph and text embedding and their Keras Sequences."""
import warnings
try:
    from .embedders import (
        CBOW, GloVe, GraphCBOW, GraphGloVe, GraphSkipGram,
        SkipGram, TransE, TransH, TransR, SimplE, Siamese
    )
    from .node_prediction import GraphConvolutionalNeuralNetwork
    from .sequences import Word2VecSequence
except ModuleNotFoundError:
    warnings.warn(
        "You do not have TensorFlow installed, all tensorflow models "
        "will not work."
    )
from .transformers import (CorpusTransformer, EdgeTransformer,
                           GraphTransformer, LinkPredictionTransformer,
                           NodeTransformer)
from .visualizations import GraphVisualization

__all__ = [
    "CBOW",
    "SkipGram",
    "GloVe",
    "GraphCBOW",
    "GraphSkipGram",
    "GraphGloVe",
    "Word2VecSequence",
    "NodeTransformer",
    "EdgeTransformer",
    "GraphTransformer",
    "CorpusTransformer",
    "LinkPredictionTransformer",
    "GraphVisualization",
    "TransE",
    "TransH",
    "TransR",
    "SimplE",
    "Siamese",
    "GraphConvolutionalNeuralNetwork"
]
