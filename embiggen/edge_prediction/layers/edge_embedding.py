"""Layer for executing Concatenation edge embedding."""
from typing import Dict, Optional

import numpy as np
from tensorflow.keras.layers import Embedding  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.layers import Flatten, Input, Layer, Dropout  # pylint: disable=import-error,no-name-in-module

from ...embedders import Embedder


class EdgeEmbedding(Layer):

    def __init__(
        self,
        nodes_number: Optional[int] = None,
        embedding_size: Optional[int] = None,
        embedding: Optional[np.ndarray] = None,
        embedding_layer: Optional[Embedding] = None,
        use_dropout: Optional[bool] = True,
        dropout_rate: Optional[float] = 0.5,
        **kwargs: Dict
    ):
        """Create new EdgeEmbedding object.

        Parameters
        --------------------------
        nodes_number: Optional[int] = None
            Number of nodes in the graph to embed.
        embedding_size: Optional[int] = None
            Size of the embedding.
        embedding: np.ndarray = None
            The embedding vector.
            Either this or the embedding layer MUST be provided.
        embedding_layer: Embedding = None
            The embedding layer.
            Either this or the embedding layer MUST be provided.
        use_dropout: bool = True
            Wether to enable or not the Dropout layer after the embedding.
        dropout_rate: float = 0.5
            The rate for the dropout.
        **kwargs: Dict
            Kwargs to pass to super call.

        Raises
        --------------------------
        ValueError,
            If neither the embedding and the embedding layer are provided.
        ValueError,
            If the given dropout rate is not a strictly positive real number.
        """
        if not isinstance(dropout_rate, float) or dropout_rate <= 0:
            raise ValueError(
                "The dropout rate must be a strictly positive real number."
            )
        if embedding is None and (nodes_number is None or embedding_size is None):
            raise ValueError(
                "If the embedding was not provided, the number of nodes or the embedding "
                "size must be provided!"
            )
        self._embedding_layer = embedding_layer
        self._embedding = embedding
        self._nodes_number = nodes_number
        self._embedding_size = embedding_size
        self._use_dropout = use_dropout
        self._dropout_rate = dropout_rate
        self._source_node_input = Input((1,), name="source_node_input")
        self._destination_node_input = Input(
            (1,), name="destination_node_input")
        super(EdgeEmbedding, self).__init__(**kwargs)

    @property
    def inputs(self):
        return [self._source_node_input, self._destination_node_input]

    def build(self, *args):
        """Build the Edge embedding layer."""
        if self._embedding is not None:
            self._embedding_layer = Embedding(
                *self._embedding.shape,
                input_length=1,
                name=Embedder.TERMS_EMBEDDING_LAYER_NAME,
                weights=[self._embedding]
            )
        else:
            self._embedding_layer = Embedding(
                self._nodes_number,
                self._embedding_size,
                input_length=1,
                name=Embedder.TERMS_EMBEDDING_LAYER_NAME
            )

        self._embedding_layer.trainable = self._trainable
        super().build((None, 2, ))

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
        raise NotImplementedError(
            "The _call method must be implemented in child classes."
        )

    def call(self, *args) -> Layer:
        """Create call graph for current layer.

        Returns
        ---------------------------
        New output layer.
        """
        # Updated edge embedding layer
        edge_embedding = Flatten()(self._call(
            self._embedding_layer(self._source_node_input),
            self._embedding_layer(self._destination_node_input)
        ))
        if self._use_dropout:
            return Dropout(self._dropout_rate)(
                edge_embedding
            )
        return edge_embedding
