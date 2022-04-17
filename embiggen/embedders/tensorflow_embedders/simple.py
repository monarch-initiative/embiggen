"""Siamese network for node-embedding including optionally node types and edge types."""
from typing import Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from ensmallen import Graph
from tensorflow.keras.layers import Embedding  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.constraints import UnitNorm  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras import backend as K  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.optimizers import \
    Optimizer  # pylint: disable=import-error,no-name-in-module

from .siamese import Siamese


class SimplE(Siamese):
    """Siamese network for node-embedding including optionally node types and edge types."""

    def __init__(
        self,
        graph: Graph,
        embedding_size: int = 100,
        embedding: Union[np.ndarray, pd.DataFrame] = None,
        model_name: str = "SimplE",
        optimizer: Union[str, Optimizer] = None,
        use_gradient_centralization: str = "auto"
    ):
        """Create new sequence TensorFlowEmbedder model.

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
        embedding: Union[np.ndarray, pd.DataFrame] = None,
            The seed embedding to be used.
            Note that it is not possible to provide at once both
            the embedding and either the vocabulary size or the embedding size.
        model_name: str = "SimplE",
            Name of the model.
        optimizer: Union[str, Optimizer] = "nadam",
            The optimizer to be used during the training of the model.
        use_gradient_centralization: bool = True,
            Whether to wrap the provided optimizer into a normalized
            one that centralizes the gradient.
            It is automatically enabled if the current version of
            TensorFlow supports gradient transformers.
            More detail here: https://arxiv.org/pdf/2004.01461.pdf
        """
        super().__init__(
            graph=graph,
            use_node_types=False,
            use_edge_types=True,
            embedding_size=embedding_size,
            edge_type_embedding_size=embedding_size,
            embedding=embedding,
            model_name=model_name,
            optimizer=optimizer,
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
        reverse_edge_type_embedding = Embedding(
            input_dim=self._edge_types_number,
            output_dim=self._edge_type_embedding_size,
            input_length=1,
            name="reverse_edge_type_embedding_layer",
        )(edge_types_input)
        reverse_edge_type_embedding = UnitNorm(
            axis=-1
        )(reverse_edge_type_embedding)
        return K.sum(
            source_node_embedding * edge_type_embedding * destination_node_embedding,
            axis=-1
        ) + K.sum(
            destination_node_embedding * reverse_edge_type_embedding * source_node_embedding,
            axis=-1
        )
