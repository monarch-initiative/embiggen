"""Layer for executing Concatenation edge embedding."""
from typing import Dict, List

from tensorflow.keras.layers import Concatenate, Layer
from .edge_embedding import EdgeEmbedding


class ConcatenateEdgeEmbedding(EdgeEmbedding):

    def __init__(self, *args: List, **kwargs: Dict):
        """Create new ConcatenateEdgeEmbedding object."""
        super().__init__(*args, **kwargs)

    def _call(self, left_embedding: Layer, right_embedding: Layer) -> Layer:
        """Compute the edge embedding layer.

        Parameters
        --------------------------
        left_embedding: Layer,
            The left embedding layer.
        right_embedding: Layer,
            The right embedding layer.

        Returns
        --------------------------
        New output layer.
        """
        return Concatenate()([
            left_embedding,
            right_embedding
        ])
