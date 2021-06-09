"""Siamese network for node-embedding including optionally node types and edge types."""
from typing import Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from embiggen.embedders.transe import TransE
from ensmallen_graph import EnsmallenGraph
from tensorflow.keras import backend as K  # pylint: disable=import-error
from tensorflow.keras.constraints import UnitNorm
from tensorflow.keras.layers import Embedding
from tensorflow.keras.optimizers import \
    Optimizer  # pylint: disable=import-error

from .transe import TransE


class TransH(TransE):
    """Siamese network for node-embedding including optionally node types and edge types."""

    def __init__(
        self,
        graph: EnsmallenGraph,
        embedding_size: int = 100,
        distance_metric: str = "COSINE",
        embedding: Union[np.ndarray, pd.DataFrame] = None,
        extra_features: Union[np.ndarray, pd.DataFrame] = None,
        model_name: str = "TransH",
        optimizer: Union[str, Optimizer] = None,
        support_mirror_strategy: bool = False,
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
        model_name: str = "TransH",
            Name of the model.
        optimizer: Union[str, Optimizer] = "nadam",
            The optimizer to be used during the training of the model.
        support_mirror_strategy: bool = False,
            Wethever to patch support for mirror strategy.
            At the time of writing, TensorFlow's MirrorStrategy does not support
            input values different from floats, therefore to support it we need
            to convert the unsigned int 32 values that represent the indices of
            the embedding layers we receive from Ensmallen to floats.
            This will generally slow down performance, but in the context of
            exploiting multiple GPUs it may be unnoticeable.
        """
        super().__init__(
            graph=graph,
            embedding_size=embedding_size,
            distance_metric=distance_metric,
            embedding=embedding,
            extra_features=extra_features,
            model_name=model_name,
            optimizer=optimizer,
            support_mirror_strategy=support_mirror_strategy
        )

    def _build_output(
        self,
        source_node_embedding: tf.Tensor,
        destination_node_embedding: tf.Tensor,
        edge_type_embedding: Optional[tf.Tensor] = None,
        edge_types_input: Optional[tf.Tensor] = None,
    ):
        """Return output of the model."""
        normal_edge_type_embedding = Embedding(
            input_dim=self._edge_types_number,
            output_dim=self._edge_type_embedding_size,
            input_length=1,
            name="normal_edge_type_embedding_layer",
        )(edge_types_input)
        normal_edge_type_embedding = UnitNorm(axis=-1)(normal_edge_type_embedding)
        source_node_embedding -= K.transpose(normal_edge_type_embedding) * \
            source_node_embedding * normal_edge_type_embedding
        destination_node_embedding -= K.transpose(normal_edge_type_embedding) * \
            destination_node_embedding * normal_edge_type_embedding

        return super()._build_output(
            source_node_embedding + edge_type_embedding,
            destination_node_embedding,
            edge_type_embedding,
            edge_types_input
        )
