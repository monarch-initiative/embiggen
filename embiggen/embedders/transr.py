"""Siamese network for node-embedding including optionally node types and edge types."""
from typing import Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from embiggen.embedders.transe import TransE
from ensmallen_graph import EnsmallenGraph
from tensorflow.keras import backend as K  # pylint: disable=import-error
from tensorflow.keras.layers import Embedding, Reshape
from tensorflow.keras.optimizers import \
    Optimizer  # pylint: disable=import-error


class TransR(TransE):
    """Siamese network for node-embedding including optionally node types and edge types."""

    def __init__(
        self,
        graph: EnsmallenGraph,
        embedding_size: int = 100,
        distance_metric: str = "COSINE",
        embedding: Union[np.ndarray, pd.DataFrame] = None,
        extra_features: Union[np.ndarray, pd.DataFrame] = None,
        model_name: str = "TransR",
        optimizer: Union[str, Optimizer] = None
    ):
        """Create new sequence Embedder model.

        Parameters
        -------------------------------------------
        vocabulary_size: int = None,
            Number of terms to embed.
            In a graph this is the number of nodes, while in a text is the
            number of the unique words.
            If None, the seed embedding must be provided.
            It is not possible to provide both at once.
        embedding_size: int = 100,
            Dimension of the embedding.
            If None, the seed embedding must be provided.
            It is not possible to provide both at once.
        distance_metric: str = "COSINE",
            The distance to use for the loss function.
            Supported methods are L1, L2 and COSINE.
        embedding: Union[np.ndarray, pd.DataFrame] = None,
            The seed embedding to be used.
            Note that it is not possible to provide at once both
            the embedding and either the vocabulary size or the embedding size.
        extra_features: Union[np.ndarray, pd.DataFrame] = None,
            Optional extra features to be used during the computation
            of the embedding. The features must be available for all the
            elements considered for the embedding.
        model_name: str = "Word2Vec",
            Name of the model.
        optimizer: Union[str, Optimizer] = "nadam",
            The optimizer to be used during the training of the model.
        window_size: int = 4,
            Window size for the local context.
            On the borders the window size is trimmed.
        negative_samples: int = 10,
            The number of negative classes to randomly sample per batch.
            This single sample of negative classes is evaluated for each element in the batch.
        """
        super().__init__(
            graph=graph,
            embedding_size=embedding_size,
            distance_metric=distance_metric,
            embedding=embedding,
            extra_features=extra_features,
            model_name=model_name,
            optimizer=optimizer
        )

    def _build_output(
        self,
        src_node_embedding: tf.Tensor,
        dst_node_embedding: tf.Tensor,
        edge_type_embedding: Optional[tf.Tensor] = None,
        edge_types_input: Optional[tf.Tensor] = None,
    ):
        """Return output of the model."""
        normal_edge_type_embedding = Embedding(
            input_dim=self._edge_types_number,
            output_dim=self._edge_type_embedding_size*self._vocabulary_size,
            input_length=1,
            name="normal_edge_type_embedding_layer",
        )(edge_types_input)

        normal_edge_type_embedding_matrix = Reshape((
            self._edge_type_embedding_size,
            self._vocabulary_size
        ))(normal_edge_type_embedding)

        src_node_embedding = K.l2_normalize(
            normal_edge_type_embedding_matrix * src_node_embedding,
            axis=-1
        )
        dst_node_embedding = K.l2_normalize(
            normal_edge_type_embedding_matrix * dst_node_embedding,
            axis=-1
        )

        return super()._build_output(
            edge_type_embedding,
            dst_node_embedding,
            edge_type_embedding,
            edge_types_input
        )
