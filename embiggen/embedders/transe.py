"""Siamese network for node-embedding including optionally node types and edge types."""
from typing import Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from ensmallen import Graph
from tensorflow.keras.optimizers import Optimizer  # pylint: disable=import-error,no-name-in-module

from .siamese import Siamese


class TransE(Siamese):
    """Siamese network for node-embedding including optionally node types and edge types."""

    def __init__(
        self,
        graph: Graph,
        embedding_size: int = 100,
        distance_metric: str = "COSINE",
        use_node_types: Union[bool, str] = "auto",
        node_types_combination: str = "Add",
        embedding: Union[np.ndarray, pd.DataFrame] = None,
        extra_features: Union[np.ndarray, pd.DataFrame] = None,
        model_name: str = "TransE",
        optimizer: Union[str, Optimizer] = None,
        support_mirrored_strategy: bool = False,
        use_gradient_centralization: str = "auto"
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
        node_types_combination: str = "Add",
            Method to combine the node embedding with the node type ambedding.
            The supported methods are "Add" and "Concatenate".
        embedding: Union[np.ndarray, pd.DataFrame] = None,
            The seed embedding to be used.
            Note that it is not possible to provide at once both
            the embedding and either the vocabulary size or the embedding size.
        extra_features: Union[np.ndarray, pd.DataFrame] = None,
            Optional extra features to be used during the computation
            of the embedding. The features must be available for all the
            elements considered for the embedding.
        model_name: str = "TransE",
            Name of the model.
        optimizer: Union[str, Optimizer] = "nadam",
            The optimizer to be used during the training of the model.
        support_mirrored_strategy: bool = False,
            Wethever to patch support for mirror strategy.
            At the time of writing, TensorFlow's MirrorStrategy does not support
            input values different from floats, therefore to support it we need
            to convert the unsigned int 32 values that represent the indices of
            the embedding layers we receive from Ensmallen to floats.
            This will generally slow down performance, but in the context of
            exploiting multiple GPUs it may be unnoticeable.
        use_gradient_centralization: bool = True,
            Whether to wrap the provided optimizer into a normalized
            one that centralizes the gradient.
            It is automatically enabled if the current version of
            TensorFlow supports gradient transformers.
            More detail here: https://arxiv.org/pdf/2004.01461.pdf
        """
        super().__init__(
            graph=graph,
            use_node_types=use_node_types,
            node_types_combination=node_types_combination,
            use_edge_types=True,
            embedding_size=embedding_size,
            node_type_embedding_size=embedding_size,
            edge_type_embedding_size=embedding_size,
            distance_metric=distance_metric,
            embedding=embedding,
            extra_features=extra_features,
            model_name=model_name,
            optimizer=optimizer,
            support_mirrored_strategy=support_mirrored_strategy,
            use_gradient_centralization=use_gradient_centralization
        )

    def _build_output(
        self,
        source_node_embedding: tf.Tensor,
        destination_node_embedding: tf.Tensor,
        edge_type_embedding: Optional[tf.Tensor] = None,
        edge_types_input: Optional[tf.Tensor] = None,
    ):
        """Return output of the model."""
        return super()._build_output(
            source_node_embedding + edge_type_embedding,
            destination_node_embedding,
            edge_type_embedding,
            edge_types_input
        )
