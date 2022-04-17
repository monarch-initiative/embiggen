"""Submodule providing node embedding models implemented in Ensmallen in Rust.

These models are NOT dependant on TensorFlow and execute in CPU, not GPU.
"""

from .ensmallen_embedder import EnsmallenEmbedder
from .graph_cbow import GraphCBOW
from .graph_skipgram import GraphSkipGram
from .spine import SPINE

__all__ = [
    "GraphCBOW",
    "GraphSkipGram",
    "SPINE",
    "EnsmallenEmbedder"
]