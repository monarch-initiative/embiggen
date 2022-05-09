"""Layer for executing Concatenation edge embedding."""
from tensorflow.keras.layers import Concatenate, Layer  # pylint: disable=import-error,no-name-in-module
from .edge_embedding import EdgeEmbedding


class ConcatenateEdgeEmbedding(EdgeEmbedding):

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
        return Concatenate()([
            source_node_embedding,
            destination_node_embedding
        ])
