from .embedder import Embedder
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dot, Dense, Input, Flatten, Add
from tensorflow.keras.models import Model


class GloVe(Embedder):

    def __init__(self, alpha: float = 0.75, **kwargs):
        """Create new GloVe-based Embedder object.

        Parameters
        ----------------------------
        alpha: float = 0.75,
            Alpha to use for the function
        """
        self._alpha = alpha
        super().__init__(**kwargs)

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
            name="Glove"
        )
        glove.compile(
            loss=self._glove_loss,
            optimizer=self._optimizer
        )

        return glove
