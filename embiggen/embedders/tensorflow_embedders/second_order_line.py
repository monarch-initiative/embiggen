"""Second order LINE TensorFlo model."""
from typing import Union, List

import tensorflow as tf
from ensmallen import Graph
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (  # pylint: disable=import-error,no-name-in-module
    Dot, Embedding, Activation, Flatten
)
from embiggen.utils.abstract_models import EmbeddingResult

from embiggen.embedders.tensorflow_embedders.edge_prediction_based_tensorflow_embedders import EdgePredictionBasedTensorFlowEmbedders


class SecondOrderLINETensorFlow(EdgePredictionBasedTensorFlowEmbedders):
    """Second order LINE TensorFlo model."""

    def _build_edge_prediction_based_model(
        self,
        graph: Graph,
        sources: tf.Tensor,
        destinations: tf.Tensor
    ) -> Union[List[tf.Tensor], tf.Tensor]:
        """Return the model implementation.

        Parameters
        -------------------
        sources: tf.Tensor
            The source nodes to be used in the model.
        destinations: tf.Tensor
            The destinations nodes to be used in the model.
        """
        node_embedding = Embedding(
            input_dim=graph.get_nodes_number(),
            output_dim=self._embedding_size,
            input_length=1,
            name="node_embeddings"
        )
        context_embedding = Embedding(
            input_dim=graph.get_nodes_number(),
            output_dim=self._embedding_size,
            input_length=1,
            name="context_embeddings"
        )
        return Activation("sigmoid")(Dot(axes=-1)([
            Flatten()(node_embedding(sources)),
            Flatten()(context_embedding(destinations))
        ]))

    @staticmethod
    def model_name() -> str:
        """Returns name of the current model."""
        return "Second Order LINE"

    def _extract_embeddings(
        self,
        graph: Graph,
        model: Model,
        return_dataframe: bool
    ) -> EmbeddingResult:
        """Returns embedding from the model.

        Parameters
        ------------------
        graph: Graph
            The graph that was embedded.
        model: Model
            The Keras model used to embed the graph.
        return_dataframe: bool
            Whether to return a dataframe of a numpy array.
        """
        node_embeddings = self.get_layer_weights(
            "node_embeddings",
            model,
        )
        context_embeddings = self.get_layer_weights(
            "context_embeddings",
            model,
        )
        if return_dataframe:
            node_names = graph.get_node_names()
            node_embeddings = pd.DataFrame(
                node_embeddings,
                index=node_names
            )
            context_embeddings = pd.DataFrame(
                context_embeddings,
                index=node_names
            )

        return EmbeddingResult(
            embedding_method_name=self.model_name(),
            node_embeddings=[node_embeddings, context_embeddings]
        )