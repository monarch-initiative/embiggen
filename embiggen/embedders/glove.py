from .embedder import Embedder
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Reshape, Dot, Dense, Input, Flatten
from tensorflow.keras.models import Model


class Glove(Embedder):

    def __init__(self, alpha: float = 0.75, **kwargs):
        """Create new Glove-based Embedder object.

        Parameters
        ----------------------------
        alpha: float = 0.75,
            Alpha to use for the function
        """
        self._alpha = alpha

    def glove_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
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
        return K.sum(K.pow(K.clip(y_true, 0.0, 1.0), self._alpha) * K.square(y_pred - K.log(y_true)), axis=-1)

    def _build_model(self):
        """
        A Keras implementation of the GloVe architecture
        :param vocab_size: The number of distinct words
        :param vector_dim: The vector dimension of each word
        :return:
        """
        input_layers = [
            Input((1,), name=Embedder.EMBEDDING_LAYER_NAME),
            Input((1,), name='contexts')
        ]

        central_embedding = Embedding(
            self._vocabulary_size,
            self._embedding_size,
            input_length=1
        )

        central_bias = Embedding(
            vocab_size, 1, input_length=1)

        context_embedding = Embedding(
            vocab_size, vector_dim, input_length=1)
        context_bias = Embedding(
            vocab_size, 1, input_length=1)

        bias_target = central_bias(input_target)
        bias_context = context_bias(input_context)

        dot_product = Dot(axes=-1)([vector_target, vector_context])
        dot_product = Reshape((1, ))(dot_product)
        bias_target = Reshape((1,))(bias_target)
        bias_context = Reshape((1,))(bias_context)

        prediction = Add()([dot_product, bias_target, bias_context])

        model = Model(inputs=[input_target, input_context], outputs=prediction)
        model.compile(
            loss=self._glove_loss,
            optimizer=Nadam(learning_rate=0.1),
        )

        return model
