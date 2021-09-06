from typing import Dict, Union

import numpy as np
import pandas as pd
from ensmallen import Graph
from extra_keras_metrics import get_standard_binary_metrics
from tensorflow.keras.layers import Layer, Input, Concatenate
from tensorflow.keras.models import Model

from ..embedders import Embedder
from ..sequences import EdgePredictionSequence
from .layers import edge_embedding_layer


class EdgePredictionModel(Embedder):

    def __init__(
        self,
        nodes_number: int = None,
        embedding_size: int = None,
        embedding: Union[np.ndarray, pd.DataFrame] = None,
        edge_embedding_method: str = "Concatenate",
        optimizer: str = "nadam",
        trainable_embedding: bool = True,
        use_dropout: bool = True,
        dropout_rate: float = 0.5,
        use_edge_metrics: bool = False,
    ):
        """Create new Perceptron object.

        Parameters
        --------------------
        nodes_number: int = None,
            Number of terms to embed.
            In a graph this is the number of nodes, while in a text is the
            number of the unique words.
            If None, the seed embedding must be provided.
            It is not possible to provide both at once.
        embedding_size: int = None,
            Dimension of the embedding.
            If None, the seed embedding must be provided.
            It is not possible to provide both at once.
        embedding: Union[np.ndarray, pd.DataFrame] = None,
            The seed embedding to be used.
            Note that it is not possible to provide at once both
            the embedding and either the vocabulary size or the embedding size.
        edge_embedding_method: str = "Concatenate",
            Method to use to create the edge embedding.
        optimizer: str = "nadam",
            Optimizer to use during the training.
        trainable_embedding: bool = True,
            Whether to allow for trainable embedding.
        use_dropout: bool = True,
            Whether to use dropout.
        dropout_rate: float = 0.5,
            Dropout rate.
        use_edge_metrics: bool = False,
            Whether to return the edge metrics.
        """
        if edge_embedding_method not in edge_embedding_layer:
            raise ValueError(
                "The given edge embedding method `{}` is not supported.".format(
                    edge_embedding_method
                )
            )
        self._use_edge_metrics = use_edge_metrics
        self._edge_embedding_method = edge_embedding_method
        self._model_name = self.__class__.__name__
        self._use_dropout = use_dropout
        self._dropout_rate = dropout_rate
        super().__init__(
            vocabulary_size=nodes_number,
            embedding_size=embedding_size,
            optimizer=optimizer,
            embedding=embedding,
            trainable_embedding=trainable_embedding
        )

    def _compile_model(self) -> Model:
        """Compile model."""
        self._model.compile(
            loss="binary_crossentropy",
            optimizer=self._optimizer,
            metrics=get_standard_binary_metrics(),
        )

    def _build_model_body(self, input_layer: Layer) -> Layer:
        """Build new model body for Edge prediction."""
        raise NotImplementedError(
            "The method _build_model_body must be implemented in the child classes."
        )

    def _build_model(self) -> Model:
        """Build new model for Edge prediction."""
        embedding_layer = edge_embedding_layer[self._edge_embedding_method](
            embedding=self._embedding,
            use_dropout=self._use_dropout,
            dropout_rate=self._dropout_rate
        )

        inputs = [*embedding_layer.inputs]
        edge_embedding = embedding_layer(None)

        if self._use_edge_metrics:
            # TODO! update the shape using an ensmallen method
            edge_metrics_input = Input((4,), name="EdgeMetrics")
            inputs.append(edge_metrics_input)
            edge_embedding = Concatenate()([edge_embedding, edge_metrics_input])

        return Model(
            inputs=inputs,
            outputs=self._build_model_body(edge_embedding),
            name="{}_{}".format(
                self._model_name,
                self._edge_embedding_method
            )
        )

    def fit(
        self,
        graph: Graph,
        batch_size: int = 2**10,
        batches_per_epoch: Union[int, str] = "auto",
        negative_samples_rate: float = 0.5,
        epochs: int = 10000,
        support_mirrored_strategy: bool = False,
        early_stopping_monitor: str = "loss",
        early_stopping_min_delta: float = 0.01,
        early_stopping_patience: int = 5,
        early_stopping_mode: str = "min",
        reduce_lr_monitor: str = "loss",
        reduce_lr_min_delta: float = 0.1,
        reduce_lr_patience: int = 3,
        reduce_lr_mode: str = "min",
        reduce_lr_factor: float = 0.9,
        verbose: int = 2,
        ** kwargs: Dict
    ) -> pd.DataFrame:
        """Train model and return training history dataframe.

        Parameters
        -------------------
        graph: Graph,
            Graph object to use for training.
        batch_size: int = 2**16,
            Batch size for the training process.
        batches_per_epoch: int = 2**10,
            Number of batches to train for in each epoch.
        negative_samples_rate: float = 0.5,
            Rate of unbalancing in the batch.
        epochs: int = 10000,
            Epochs to train the model for.
        support_mirrored_strategy: bool = False,
            Wethever to patch support for mirror strategy.
            At the time of writing, TensorFlow's MirrorStrategy does not support
            input values different from floats, therefore to support it we need
            to convert the unsigned int 32 values that represent the indices of
            the embedding layers we receive from Ensmallen to floats.
            This will generally slow down performance, but in the context of
            exploiting multiple GPUs it may be unnoticeable.
        early_stopping_monitor: str = "loss",
            Metric to monitor for early stopping. 
        early_stopping_min_delta: float = 0.01,
            Minimum delta of metric to stop the training.
        early_stopping_patience: int = 5,
            Number of epochs to wait for when the given minimum delta is not
            achieved after which trigger early stopping.
        early_stopping_mode: str = "min",
            Direction of the variation of the monitored metric for early stopping.
        reduce_lr_monitor: str = "loss",
            Metric to monitor for reducing learning rate.
        reduce_lr_min_delta: float = 0.1,
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
        ** kwargs: Dict,
            Parameteres to pass to the dit call.

        Returns
        --------------------
        Dataframe with traininhg history.
        """
        sequence = EdgePredictionSequence(
            graph,
            batch_size=batch_size,
            batches_per_epoch=batches_per_epoch,
            negative_samples_rate=negative_samples_rate,
            support_mirrored_strategy=support_mirrored_strategy,
            use_edge_metrics=self._use_edge_metrics,
        )
        return super().fit(
            sequence,
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

    def predict(self, *args, **kwargs):
        """Run predict."""
        return self._model.predict(*args, **kwargs)

    def evaluate(self, *args, **kwargs) -> Dict[str, float]:
        """Run predict."""
        return dict(zip(
            self._model.metrics_names,
            self._model.evaluate(*args, **kwargs)
        ))
