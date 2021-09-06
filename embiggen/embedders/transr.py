"""Siamese network for node-embedding including optionally node types and edge types."""
from typing import Optional, Union, Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from ensmallen import Graph
from tensorflow.keras import backend as K  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.layers import Embedding, Reshape  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.optimizers import \
    Optimizer  # pylint: disable=import-error,no-name-in-module
from .transe import TransE


class TransR(TransE):
    """Siamese network for node-embedding including optionally node types and edge types."""

    def __init__(
        self,
        graph: Graph,
        embedding_size: int = 100,
        distance_metric: str = "COSINE",
        embedding: Union[np.ndarray, pd.DataFrame] = None,
        extra_features: Union[np.ndarray, pd.DataFrame] = None,
        model_name: str = "TransR",
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
        embedding: Union[np.ndarray, pd.DataFrame] = None,
            The seed embedding to be used.
            Note that it is not possible to provide at once both
            the embedding and either the vocabulary size or the embedding size.
        extra_features: Union[np.ndarray, pd.DataFrame] = None,
            Optional extra features to be used during the computation
            of the embedding. The features must be available for all the
            elements considered for the embedding.
        model_name: str = "TransR",
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
            embedding_size=embedding_size,
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
        normal_edge_type_embedding = Embedding(
            input_dim=self._edge_types_number,
            output_dim=self._edge_type_embedding_size*self._embedding_size,
            input_length=1,
            name="normal_edge_type_embedding_layer",
        )(edge_types_input)

        normal_edge_type_embedding_matrix = Reshape((
            self._edge_type_embedding_size,
            self._embedding_size
        ))(normal_edge_type_embedding)

        source_node_embedding = K.l2_normalize(
            normal_edge_type_embedding_matrix * source_node_embedding,
            axis=-1
        )
        destination_node_embedding = K.l2_normalize(
            normal_edge_type_embedding_matrix * destination_node_embedding,
            axis=-1
        )

        return super()._build_output(
            source_node_embedding,
            destination_node_embedding,
            edge_type_embedding,
            edge_types_input
        )

    def fit(
        self,
        batch_size: int = 2**15,
        negative_samples_rate: float = 0.5,
        avoid_false_negatives: bool = False,
        graph_to_avoid: Graph = None,
        batches_per_epoch: Union[int, str] = "auto",
        elapsed_epochs: int = 0,
        epochs: int = 1000,
        early_stopping_monitor: str = "loss",
        early_stopping_min_delta: float = 0.01,
        early_stopping_patience: int = 5,
        early_stopping_mode: str = "min",
        reduce_lr_monitor: str = "loss",
        reduce_lr_min_delta: float = 0.01,
        reduce_lr_patience: int = 2,
        reduce_lr_mode: str = "min",
        reduce_lr_factor: float = 0.9,
        verbose: int = 2,
        **kwargs: Dict
    ) -> pd.DataFrame:
        """Return pandas dataframe with training history.

        Parameters
        -----------------------
        graph: Graph,
            Graph to embed.
        epochs: int = 100,
            Epochs to train the model for.
        early_stopping_monitor: str = "loss",
            Metric to monitor for early stopping.
        early_stopping_min_delta: float = 0.1,
            Minimum delta of metric to stop the training.
        early_stopping_patience: int = 5,
            Number of epochs to wait for when the given minimum delta is not
            achieved after which trigger early stopping.
        early_stopping_mode: str = "min",
            Direction of the variation of the monitored metric for early stopping.
        reduce_lr_monitor: str = "loss",
            Metric to monitor for reducing learning rate.
        reduce_lr_min_delta: float = 1,
            Minimum delta of metric to reduce learning rate.
        reduce_lr_patience: int = 3,
            Number of epochs to wait for when the given minimum delta is not
            achieved after which reducing learning rate.
        reduce_lr_mode: str = "min",
            Direction of the variation of the monitored metric for learning rate.
        reduce_lr_factor: float = 0.9,
            Factor for reduction of learning rate.
        verbose: int = 2,
            Wethever to show the loading bar.
            Specifically, the options are:
            * 0 or False: No loading bar.
            * 1 or True: Showing only the loading bar for the epochs.
            * 2: Showing loading bar for both epochs and batches.
        **kwargs: Dict,
            Additional kwargs to pass to the Keras fit call.

        Returns
        -----------------------
        Dataframe with training history.
        """
        return super().fit(
            batch_size=batch_size,
            avoid_false_negatives=avoid_false_negatives,
            graph_to_avoid=graph_to_avoid,
            negative_samples_rate=negative_samples_rate,
            elapsed_epochs=elapsed_epochs,
            batches_per_epoch=batches_per_epoch,
            epochs=epochs,
            early_stopping_monitor=early_stopping_monitor,
            early_stopping_min_delta=early_stopping_min_delta,
            early_stopping_patience=early_stopping_patience,
            early_stopping_mode=early_stopping_mode,
            reduce_lr_monitor=reduce_lr_monitor,
            reduce_lr_min_delta=reduce_lr_min_delta,
            reduce_lr_patience=reduce_lr_patience,
            reduce_lr_mode=reduce_lr_mode,
            reduce_lr_factor=reduce_lr_factor,
            verbose=verbose,
            **kwargs
        )
