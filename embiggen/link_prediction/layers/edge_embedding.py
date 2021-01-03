"""Layer for executing Concatenation edge embedding."""
from typing import Dict, Tuple

import numpy as np
from tensorflow.keras.layers import Layer, Embedding, Input   # pylint: disable=import-error


class EdgeEmbedding(Layer):

    def __init__(
        self,
        embedding: np.ndarray = None,
        embedding_layer: Embedding = None,
        trainable: bool = True,
        **kwargs: Dict
    ):
        """Create new EdgeEmbedding object.

        Parameters
        --------------------------
        embedding: np.ndarray = None,
            The embedding vector.
            Either this or the embedding layer MUST be provided.
        embedding_layer: Embedding = None,
            The embedding layer.
            Either this or the embedding layer MUST be provided.
        trainable: bool = True,
            Wether to make this layer trainable.
        **kwargs: Dict,
            Kwargs to pass to super call.

        Raises
        --------------------------
        ValueError,
            If neither the embedding and the embedding layer are provided.
        """
        if embedding is None and embedding_layer is None:
            raise ValueError(
                "You must provide either the embedding or the embedding layer."
            )
        self._embedding_layer = embedding_layer
        self._embedding = embedding
        self._trainable = trainable
        self._left_input = Input((1,), name="left_input")
        self._right_input = Input((1,), name="right_input")
        super(EdgeEmbedding, self).__init__(
            trainable=trainable,
            **kwargs
        )

    @property
    def inputs(self):
        return [self._left_input, self._right_input]

    def build(self, *args):
        """Build the Edge embedding layer."""
        if self._embedding is not None:
            self._embedding_layer = Embedding(
                *self._embedding.shape,
                input_length=1,
            )
            self._embedding_layer.build((None,))
            self._embedding_layer.set_weights([self._embedding])

        self._embedding_layer.trainable = self._trainable
        super().build((None, 2, ))

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
        return self._call(
            self._embedding_layer(self._left_input),
            self._embedding_layer(self._right_input)
        )
