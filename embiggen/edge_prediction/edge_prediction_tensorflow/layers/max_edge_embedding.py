"""Layer for executing Maximum edge embedding."""
from tensorflow.keras.layers import Maximum, Layer  # pylint: disable=import-error,no-name-in-module
from embiggen.edge_prediction.edge_prediction_tensorflow.layers.edge_embedding import EdgeEmbedding

class MaxEdgeEmbedding(EdgeEmbedding):

    def _call(self, source_node_embedding: Layer, destination_node_embedding: Layer) -> Layer:
        """Compute the edge embedding layer using a maximum reduce method.

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
        return Maximum()([
            source_node_embedding,
            destination_node_embedding
        ])
