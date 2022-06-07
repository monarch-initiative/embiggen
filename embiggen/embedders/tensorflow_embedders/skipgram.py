"""SkipGram model for sequence embedding."""
from typing import Dict, Union
from tensorflow.keras.layers import (  # pylint: disable=import-error,no-name-in-module
    Input, Embedding, Flatten
)
from ensmallen import Graph
import tensorflow as tf  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.models import Model  # pylint: disable=import-error,no-name-in-module

from embiggen.embedders.tensorflow_embedders.node2vec import Node2Vec
from embiggen.layers.tensorflow import NoiseContrastiveEstimation


class SkipGramTensorFlow(Node2Vec):
    """SkipGram model for sequence embedding.

    The SkipGram model for graph embedding receives a central word and tries
    to predict its contexts. The model makes use of an NCE loss layer
    during the training process to generate the negatives.
    """

    NODE_EMBEDDING = "node_embedding"

    @staticmethod
    def model_name() -> str:
        """Returns name of the model."""
        return "SkipGram"

    def _build_model(self, graph: Graph) -> Model:
        """Return SkipGram model."""
        # Create first the input with the central terms
        central_terms = Input((1, ), dtype=tf.int32)

        # Then we create the input of the contextual terms
        contextual_terms = Input((self._window_size*2, ), dtype=tf.int32)

        # Creating the embedding layer for the contexts
        central_term_embedding = Flatten()(Embedding(
            input_dim=graph.get_nodes_number(),
            output_dim=self._embedding_size,
            input_length=1,
            name=self.NODE_EMBEDDING,
        )(central_terms))

        # Adding layer that also executes the loss function
        output = NoiseContrastiveEstimation(
            vocabulary_size=graph.get_nodes_number(),
            embedding_size=self._embedding_size,
            number_of_negative_samples=self._number_of_negative_samples,
            positive_samples=self._window_size*2,
        )((central_term_embedding, contextual_terms))

        # Creating the actual model
        model = Model(
            inputs=[contextual_terms, central_terms],
            outputs=output,
            name=self.model_name()
        )

        model.compile(optimizer=self._optimizer)

        return model