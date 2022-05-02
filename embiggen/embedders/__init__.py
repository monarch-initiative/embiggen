"""Submodule providing TensorFlow and Ensmallen-based embedders."""
try:
    from . import tensorflow_embedders
    from . import ensmallen_embedders

    SUPPORTED_NODE_EMBEDDING_METHODS = {
        "cbow": {
            "gpu": tensorflow_embedders.GraphCBOW,
            "cpu": ensmallen_embedders.GraphCBOW,
        },
        "glove": tensorflow_embedders.GraphGloVe,
        "skipgram": {
            "gpu": tensorflow_embedders.GraphSkipGram,
            "cpu": ensmallen_embedders.GraphSkipGram,
        },
        "siamese": tensorflow_embedders.Siamese,
        "transe": tensorflow_embedders.TransE,
        "simple": tensorflow_embedders.SimplE,
        "transh": tensorflow_embedders.TransH,
        "transr": tensorflow_embedders.TransR,
        "spine": ensmallen_embedders.SPINE,
        "weightedspine": ensmallen_embedders.WeightedSPINE,
    }

    __all__ = [
        "tensorflow_embedders",
        "ensmallen_embedders",
        "SUPPORTED_NODE_EMBEDDING_METHODS"
    ]
except ModuleNotFoundError as e:
    from .ensmallen_embedders import *

    SUPPORTED_NODE_EMBEDDING_METHODS = {
        "cbow": GraphCBOW,
        "skipgram": GraphSkipGram,
        "spine": SPINE,
        "weightedspine": WeightedSPINE
    }

    __all__ = [
        "ensmallen_embedders",
        "SUPPORTED_NODE_EMBEDDING_METHODS",
    ]
