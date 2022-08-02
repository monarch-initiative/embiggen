"""CBOW model for sequence embedding."""
from tensorflow.keras.layers import (   # pylint: disable=import-error,no-name-in-module
    GlobalAveragePooling1D, Input, Embedding
)
from ensmallen import Graph
import tensorflow as tf
from tensorflow.keras.models import Model  # pylint: disable=import-error,no-name-in-module

from embiggen.embedders.tensorflow_embedders.node2vec import Node2Vec
from embiggen.layers.tensorflow import SampledSoftmax


class CBOWTensorFlow(Node2Vec):
    """CBOW model for sequence embedding.

    The CBOW model for graph embedding receives a list of contexts and tries
    to predict the central word. The model makes use of an NCE loss layer
    during the training process to generate the negatives.
    """

    @classmethod
    def model_name(cls) -> str:
        """Returns name of the model."""
        return "Node2Vec CBOW"

    def _build_model(self, graph: Graph) -> Model:
        """Return CBOW model."""
        # Creating the inputs layers

        # Create first the input with the central terms
        central_terms = Input((1, ), dtype=tf.int32)

        # Then we create the input of the contextual terms
        contextual_terms = Input((self._window_size*2, ), dtype=tf.int32)

        # Getting the average context embedding
        average_context_embedding = GlobalAveragePooling1D()(Embedding(
            input_dim=graph.get_number_of_nodes(),
            output_dim=self._embedding_size,
            input_length=self._window_size*2,
            name="node_embedding",
        )(contextual_terms))

        # Adding layer that also executes the loss function
        sampled_softmax = SampledSoftmax(
            vocabulary_size=graph.get_number_of_nodes(),
            embedding_size=self._embedding_size,
            number_of_negative_samples=self._number_of_negative_samples,
        )((average_context_embedding, central_terms))

        # Creating the actual model
        model = Model(
            inputs=[contextual_terms, central_terms],
            outputs=sampled_softmax,
            name=self.model_name().replace(" ", "")
        )

        model.compile(optimizer=self._optimizer)

        return model
