import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, Embedding, Input, Add, Dot, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import skipgrams
from tensorflow.keras.models import Model


class Glove:
    """
    An implementation of the Global Vectors method
    :param embedding_size: dimension of embedded vectors
    :param learning_rate: update rate for stochastic gradient descent
    :param alpha: factor for the exponent of the weighting function of GloVe
    :param max_count: maximum word/concept count for the weighting function of GloVe
    """

    def __init__(self, embedding_size=100, learning_rate=0.05,
                 alpha=0.75, max_count=100):
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.max_count = max_count
        self.model = self._create_glove_model()

    def _create_glove_model(self):
        """
        w_i = tf.keras.Input(shape=(1,), name="w_i") leads to a tensor with shape (None,1)
        emb_i = Flatten()(Embedding(v_size, 96, input_length=1)(w_i)) leads to a tensor with shape (None, v_size)
        both have dtype=float32
        :param v_size:
        :return:
        """
        v_size = 2000  # vocabulary size
        embed_dim = self.embedding_size
        w_i = tf.keras.Input(shape=(1,), name="w_i")
        w_j = tf.keras.Input(shape=(1,), name="w_j")

        emb_i = Flatten()(Embedding(v_size, embed_dim, input_length=1)(w_i))
        emb_j = Flatten()(Embedding(v_size, embed_dim, input_length=1)(w_j))

        ij_dot = Dot(axes=-1)([emb_i, emb_j])

        b_i = Flatten()(Embedding(v_size, 1, input_length=1)(w_i))
        b_j = Flatten()(Embedding(v_size, 1, input_length=1)(w_j))

        pred = Add()([ij_dot, b_i, b_j])

        def glove_loss(y_true, y_pred):
            import tensorflow.keras.backend.pow as k_pow
            import tensorflow.keras.backend.sum as k_sum
            import tensorflow.keras.backend.square as k_square
            import tensorflow.keras.backend.log as l_log
            return k_sum(
                k_pow((y_true - 1) / 100.0, 0.75) * k_square(y_pred - l_log(y_true))
            )

        model = Model(inputs=[w_i, w_j], outputs=pred)
        model.compile(loss=glove_loss, optimizer=Adam(lr=0.0001))
        return model
