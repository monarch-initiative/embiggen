"""Module with models for graph and text embedding and their Keras Sequences."""
import warnings
try:
    from .embedders.tensorflow_embedders import (
        CBOW, GloVe, GraphCBOW, GraphGloVe, GraphSkipGram,
        SkipGram, TransE, TransH, TransR, SimplE, Siamese
    )

    all_exports = [
        "CBOW", "SkipGram", "GloVe", "GraphGloVe", "Word2VecSequence",
        "TransE", "TransH", "TransR", "SimplE", "Siamese"
    ]
except ModuleNotFoundError as e:
    from .embedders.ensmallen_embedders import *
    all_exports = []
    warnings.warn(
        "You do not have TensorFlow installed, all tensorflow models "
        "will not work. We have replaced some default models, namely GraphCBOW "
        "and GraphSkipGram with the Ensmallen Rust versions."
    )

from .embedders.ensmallen_embedders import SPINE
from .transformers import (EdgeTransformer,
                           GraphTransformer, EdgePredictionTransformer,
                           NodeTransformer)
from .visualizations import GraphVisualizer

__all__ = [
    "GraphCBOW",
    "GraphSkipGram",
    "NodeTransformer",
    "EdgeTransformer",
    "GraphTransformer",
    "EdgePredictionTransformer",
    "GraphVisualizer",
    "SPINE",
    *all_exports
]
