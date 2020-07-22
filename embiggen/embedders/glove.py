"""GloVe model for graph and words embedding."""
from typing import Union

import tensorflow as tf
from tensorflow.keras import backend as K  # pylint: disable=import-error
from tensorflow.keras.layers import Add, Dot, Embedding, Flatten, Input  # pylint: disable=import-error
from tensorflow.keras.models import Model   # pylint: disable=import-error
from tensorflow.keras.optimizers import Optimizer   # pylint: disable=import-error

from .embedder import Embedder


class GloVe(Embedder):
    """GloVe model for graph and words embedding.

    The GloVe model for graoh embedding receives two words and is asked to
    predict its cooccurrence probability.
    """

    def __init__(
        self,
        vocabulary_size: int,
        embedding_size: int,
        optimizer: Union[str, Optimizer] = "nadam",
        alpha: float = 0.75
    ):
        """Create new GloVe-based Embedder object.

        Parameters
        ----------------------------
        vocabulary_size: int,
            Number of terms to embed.
            In a graph this is the number of nodes, while in a text is the
            number of the unique words.
        embedding_size: int,
            Dimension of the embedding.
        optimizer: Union[str, Optimizer] = "nadam",
            The optimizer to be used during the training of the model.
        alpha: float = 0.75,
            Alpha to use for the function
        """
        self._alpha = alpha
        super().__init__(
            vocabulary_size=vocabulary_size,
            embedding_size=embedding_size,
            optimizer=optimizer
        )

    def _glove_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
        """Compute the glove loss function.

        Parameters
        ---------------------------
        y_true: tf.Tensor,
            The true values Tensor for this batch.
        y_pred: tf.Tensor,
            The predicted values Tensor for this batch.

        Returns
        ---------------------------
        Loss function score related to this batch.
        """
        return K.sum(
            K.pow(K.clip(y_true, 0.0, 1.0), self._alpha) *
            K.square(y_pred - K.log(y_true)),
            axis=-1
        )

    def _build_model(self):
        """Create new Glove model."""
        # Creating the input layers
        input_layers = [
            Input((1,), name=Embedder.EMBEDDING_LAYER_NAME),
            Input((1,))
        ]

        # Creating the dot product of the embedding layers
        dot_product_layer = Dot(axes=2)([
            Embedding(
                self._vocabulary_size,
                self._embedding_size,
                input_length=1
            )(input_layer)
            for input_layer in input_layers
        ])

        # Creating the biases laye
        biases = [
            Embedding(
                self._vocabulary_size,
                1,
                input_length=1
            )(input_layer)
            for input_layer in input_layers
        ]

        # Concatenating with an add the three layers
        prediction = Flatten()(Add()([dot_product_layer, *biases]))

        # Creating the model
        glove = Model(
            inputs=input_layers,
            outputs=prediction,
            name="GloVe"
        )
        glove.compile(
            loss=self._glove_loss,
            optimizer=self._optimizer
        )

        return glove
