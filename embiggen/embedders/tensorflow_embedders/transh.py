"""TransH model."""
import tensorflow as tf
from ensmallen import Graph
import pandas as pd
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dot
from embiggen.embedders.tensorflow_embedders.siamese import Siamese
from embiggen.utils.abstract_models import EmbeddingResult
from embiggen.layers.tensorflow import FlatEmbedding


class TransHTensorFlow(Siamese):
    """TransH model."""

    def _build_output(
        self,
        srcs_embedding: tf.Tensor,
        dsts_embedding: tf.Tensor,
        not_srcs_embedding: tf.Tensor,
        not_dsts_embedding: tf.Tensor,
        graph: Graph
    ):
        """Returns the five input tensors, unchanged."""
        edge_types = Input((1,), dtype=tf.int32, name="Edge Types")
        bias_edge_type_embedding = FlatEmbedding(
            vocabulary_size=graph.get_number_of_edge_types(),
            dimension=self._embedding_size,
            input_length=1,
            mask_zero=graph.has_unknown_edge_types(),
            name="BiasEdgeTypeEmbedding",
        )(edge_types)
        multiplicative_edge_type_embedding = FlatEmbedding(
            vocabulary_size=graph.get_number_of_edge_types(),
            dimension=self._embedding_size,
            input_length=1,
            mask_zero=graph.has_unknown_edge_types(),
            name="MultiplicativeEdgeTypeEmbedding",
        )(edge_types)

        dot = Dot(axes=-1)([
            bias_edge_type_embedding,
            multiplicative_edge_type_embedding
        ])

        return (
            edge_types,
            dot * dot / Dot(axes=-1)([
                bias_edge_type_embedding,
                bias_edge_type_embedding
            ]),
            srcs_embedding - multiplicative_edge_type_embedding*Dot(axes=-1)([
                multiplicative_edge_type_embedding,
                srcs_embedding
            ]) + bias_edge_type_embedding,
            dsts_embedding - multiplicative_edge_type_embedding*Dot(axes=-1)([
                multiplicative_edge_type_embedding,
                dsts_embedding
            ]),
            not_srcs_embedding - multiplicative_edge_type_embedding*Dot(axes=-1)([
                multiplicative_edge_type_embedding,
                not_srcs_embedding
            ]) + bias_edge_type_embedding,
            not_dsts_embedding - multiplicative_edge_type_embedding*Dot(axes=-1)([
                multiplicative_edge_type_embedding,
                not_dsts_embedding
            ])
        )

    @classmethod
    def model_name(cls) -> str:
        """Returns name of the current model."""
        return "TransH"

    @classmethod
    def requires_edge_types(cls) -> bool:
        return True

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
        bias_edge_type_embedding = self.get_layer_weights(
            "BiasEdgeTypeEmbedding",
            model,
            drop_first_row=graph.has_unknown_edge_types()
        )
        multiplicative_edge_type_embedding = self.get_layer_weights(
            "MultiplicativeEdgeTypeEmbedding",
            model,
            drop_first_row=graph.has_unknown_edge_types()
        )

        if return_dataframe:
            node_embedding = pd.DataFrame(
                node_embedding,
                index=graph.get_node_names()
            )

            edge_type_names = graph.get_unique_edge_type_names()

            bias_edge_type_embedding = pd.DataFrame(
                bias_edge_type_embedding,
                index=edge_type_names
            )

            multiplicative_edge_type_embedding = pd.DataFrame(
                multiplicative_edge_type_embedding,
                index=edge_type_names
            )

        return EmbeddingResult(
            embedding_method_name=self.model_name(),
            node_embeddings=node_embedding,
            edge_type_embeddings=[
                bias_edge_type_embedding,
                multiplicative_edge_type_embedding
            ]
        )

    @classmethod
    def can_use_node_types(cls) -> bool:
        """Returns whether the model can optionally use node types."""
        return False

    @classmethod
    def requires_edge_types(cls) -> bool:
        """Returns whether the model requires edge types."""
        return True