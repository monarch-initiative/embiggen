"""Unstructured model."""
from ensmallen import Graph
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from embiggen.embedders.tensorflow_embedders.siamese import Siamese
from embiggen.utils.abstract_models import EmbeddingResult


class UnstructuredTensorFlow(Siamese):
    """Unstructured model."""

    def _build_output(
        self,
        srcs_embedding: tf.Tensor,
        dsts_embedding: tf.Tensor,
        not_srcs_embedding: tf.Tensor,
        not_dsts_embedding: tf.Tensor,
        graph: Graph
    ):
        """Returns the five input tensors, unchanged."""
        return (
            None,
            0.0,
            srcs_embedding,
            dsts_embedding,
            not_srcs_embedding,
            not_dsts_embedding
        )

    @classmethod
    def model_name(cls) -> str:
        """Returns name of the current model."""
        return "Unstructured"

    @classmethod
    def can_use_edge_types(cls) -> bool:
        return False

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
        node_embedding = self.get_layer_weights(
            "NodeEmbedding",
            model,
            drop_first_row=False
        )

        if return_dataframe:
            node_embedding = pd.DataFrame(
                node_embedding,
                index=graph.get_node_names()
            )

        return EmbeddingResult(
            embedding_method_name=self.model_name(),
            node_embeddings=node_embedding,
        )

    @classmethod
    def can_use_node_types(cls) -> bool:
        """Returns whether the model can optionally use node types."""
        return False
