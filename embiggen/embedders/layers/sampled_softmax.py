"""Layer for executing Sampled Softmax in Keras models."""
from typing import Dict, Tuple

import tensorflow as tf
import tensorflow.keras.backend as K   # pylint: disable=import-error
from tensorflow.keras.layers import Layer   # pylint: disable=import-error


class SampledSoftmax(Layer):
    """Layer for executing Sampled Softmax in Keras models."""

    def __init__(
        self,
        vocabulary_size: int,
        embedding_size: int,
        negative_samples: int,
        remove_accidental_hits: bool = False,
        **kwargs: Dict
    ):
        """Create new SampledSoftmax layer.

        This layer behaves as the Sampled Softmax function.
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
        remove_accidental_hits: bool = False,
            Whether to remove accidental hits.
        """
        self._vocabulary_size = vocabulary_size
        self._embedding_size = embedding_size
        self._negative_samples = negative_samples
        self._remove_accidental_hits = remove_accidental_hits
        self._weights = None
        self._biases = None
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
            shape=(self._vocabulary_size, self._embedding_size),
            initializer="glorot_normal",
        )

        self._biases = self.add_weight(
            name="approx_softmax_biases",
            shape=(self._vocabulary_size,),
            initializer="zeros"
        )

        super().build(input_shape)

    def call(self, inputs: Tuple[Layer], **kwargs) -> Layer:
        """Create call graph for current layer.

        Parameters
        ---------------------------
        inputs: Tuple[Layer],
            Tuple with vector of labels and inputs.
        """

        predictions, labels = inputs

        # Computing Sampled Softmax.
        loss = K.mean(tf.nn.sampled_softmax_loss(
            self._weights,
            self._biases,
            labels=labels,
            inputs=predictions,
            num_sampled=self._negative_samples,
            num_classes=self._vocabulary_size,
            remove_accidental_hits=self._remove_accidental_hits
        ), axis=0)

        self.add_loss(loss)

        # Computing logits for closing TF graph
        return loss
