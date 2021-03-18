"""Layer for executing L1 edge embedding."""
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda, Layer, Subtract

from .edge_embedding import EdgeEmbedding


class L1EdgeEmbedding(EdgeEmbedding):

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
        return Lambda(K.abs)(Subtract()([
            source_node_embedding,
            destination_node_embedding
        ]))
