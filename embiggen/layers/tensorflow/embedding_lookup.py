"""Submodule providing embedding lookup layer."""
from typing import Tuple, Dict
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Layer  # pylint: disable=import-error,no-name-in-module


class EmbeddingLookup(Layer):
    """Layer implementing simple embedding lookup layer."""

    def __init__(
        self,
        **kwargs: Dict
    ):
        """Create new Embedding Lookup layer.

        Parameters
        ----------------------
        **kwargs: Dict,
            Kwargs to pass to the parent Layer class.
        """
        super().__init__(**kwargs)
        self._flatten_layer = None

    def build(self, input_shape) -> None:
        """Build the embedding lookup layer.

        Parameters
        ------------------------------
        input_shape
            Shape of the output of the previous layer.
        """
        self._flatten_layer = Flatten()
        super().build(input_shape)

    def call(
        self,
        inputs: Tuple[tf.Tensor],
    ) -> tf.Tensor:
        """Returns called embeddingg lookup.

        Parameters
        ---------------------------
        inputs: Tuple[tf.Tensor],
        """
        node_ids, node_features = inputs
        return self._flatten_layer(tf.nn.embedding_lookup(
            node_features,
            ids=node_ids
        ))