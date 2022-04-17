"""Submodule providing TensorFlow and Ensmallen-based embedders."""
try:
    from . import tensorflow_embedders
    from . import ensmallen_embedders

    SUPPORTED_NODE_EMBEDDING_METHODS = {
        "CBOW": {
            "gpu": tensorflow_embedders.GraphCBOW,
            "cpu": ensmallen_embedders.GraphCBOW,
        },
        "GloVe": tensorflow_embedders.GraphGloVe,
        "SkipGram": {
            "gpu": tensorflow_embedders.GraphSkipGram,
            "cpu": ensmallen_embedders.GraphSkipGram,
        },
        "Siamese": tensorflow_embedders.Siamese,
        "TransE": tensorflow_embedders.TransE,
        "SimplE": tensorflow_embedders.SimplE,
        "TransH": tensorflow_embedders.TransH,
        "TransR": tensorflow_embedders.TransR,
        "SPINE": ensmallen_embedders.SPINE
    }

    __all__ = [
        "tensorflow_embedders",
        "ensmallen_embedders",
        "SUPPORTED_NODE_EMBEDDING_METHODS"
    ]
except ModuleNotFoundError as e:
    print("Cactus", e)
    from .ensmallen_embedders import *

    SUPPORTED_NODE_EMBEDDING_METHODS = {
        "CBOW": GraphCBOW,
        "SkipGram": GraphSkipGram,
        "SPINE": SPINE
    }

    __all__ = [
        "ensmallen_embedders",
        "SUPPORTED_NODE_EMBEDDING_METHODS"
    ]
