"""Layer for executing Hadamard edge embedding."""
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda, Layer

from .edge_embedding import EdgeEmbedding


class HadamardEdgeEmbedding(EdgeEmbedding):

    def _call(self, source_node_embedding: Layer, destination_node_embedding: Layer) -> Layer:
        """Compute the edge embedding layer.

        Parameters
        --------------------------
        source_node_embedding: Layer,
            The source node embedding layer.
        destination_node_embedding: Layer,
            The destination node embedding layer.

        Returns
        --------------------------
        New output layer.
        """
        return Lambda(lambda x: K.prod(x, axis=0))([
            source_node_embedding,
            destination_node_embedding
        ])
