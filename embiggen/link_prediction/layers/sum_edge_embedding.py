"""Layer for executing Sum edge embedding."""
from typing import Dict, List

from tensorflow.keras.layers import Lambda, Layer
from tensorflow.keras import backend as K
from .edge_embedding import EdgeEmbedding


class SumEdgeEmbedding(EdgeEmbedding):

    def _call(self, source_node_embedding: Layer, destination_node_embedding: Layer) -> Layer:
        """Compute the edge embedding layer.

        Parameters
        --------------------------
        source_node_embedding: Layer,
            The source node embedding vector
        destination_node_embedding: Layer,
            The destination node embedding vector

        Returns
        --------------------------
        New output layer.
        """
        return Lambda(lambda x: K.sum(x, axis=0))(source_node_embedding, destination_node_embedding)
