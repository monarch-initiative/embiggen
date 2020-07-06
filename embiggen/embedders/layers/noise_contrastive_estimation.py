from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Layer
import tensorflow as tf
import tensorflow.keras.backend as K
from typing import Dict, Tuple


class NoiseContrastiveEstimation(Layer):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_size: int,
        negative_samples: int,
        **kwargs: Dict
    ):
        """Create new NoiseContrastiveEstimation layer.

        This layer behaves as the NCE loss function.
        No loss function is required when using this layer.

        Parameters
        -------------------------
        vocabulary_size: int,
            Number of vectors in the embedding.
            In a graph this values are the number of nodes.
            In a text, this is the number of unique words.
        embedding_size: int,
            Dimension of the embedding.
        negative_samples: int,
            The number of negative classes to randomly sample per batch.
            This single sample of negative classes is evaluated for each element in the batch.

        """
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.negative_samples = negative_samples
        super().__init__(**kwargs)

    def build(self, input_shape: Tuple[int, int]):
        """Build the NCE layer.

        Parameters
        ------------------------------
        input_shape: Tuple[int, int],
            Shape of the output of the previous layer.
        """
        self._weights = self.add_weight(
            name="approx_softmax_weights",
            shape=(self.vocabulary_size, self.embedding_size),
            initializer="glorot_normal",
        )

        self._biases = self.add_weight(
            name="approx_softmax_biases",
            shape=(self.vocabulary_size,),
            initializer="zeros"
        )

        super().build(input_shape)

    # keras Layer interface
    def call(self, inputs: Tuple[Layer]):
        """Create call graph for current layer.

        Parameters
        ---------------------------
        inputs: Tuple[Layer],
            Tuple with vector of labels and inputs.
        """

        predictions, labels = inputs

        # Computing NCE loss.
        self.add_loss(tf.nn.nce_loss(
            self._weights,
            self._biases,
            labels=labels,
            inputs=predictions,
            num_sampled=self.negative_samples,
            num_classes=self.vocabulary_size
        ))

        # Computing logits for closing TF graph
        return K.dot(predictions, K.transpose(self._weights))

    def compute_output_shape(self, *args):
        """return output shape of the layer."""
        return (self.vocabulary_size, )
