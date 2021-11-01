"""Class for an abstract edge prediction model."""
from typing import Dict, Union
import warnings

import numpy as np
import pandas as pd
from ensmallen import Graph
from extra_keras_metrics import get_standard_binary_metrics, get_sparse_multiclass_metrics
from tensorflow.keras.layers import Layer, Input, Concatenate  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.models import Model  # pylint: disable=import-error,no-name-in-module
from userinput import set_validator
from userinput.utils import closest

from ..embedders import Embedder
from ..sequences import EdgePredictionSequence, EdgeLabelPredictionSequence
from .layers import edge_embedding_layer

edge_prediction_supported_tasks = [
    "EDGE_PREDICTION",
    "EDGE_LABEL_PREDICTION"
]


class EdgePredictionModel(Embedder):

    def __init__(
        self,
        graph: Graph,
        embedding_size: int = None,
        embedding: Union[np.ndarray, pd.DataFrame] = None,
        edge_embedding_method: str = "Concatenate",
        optimizer: str = "nadam",
        trainable_embedding: bool = False,
        use_dropout: bool = True,
        dropout_rate: float = 0.5,
        use_edge_metrics: bool = False,
        task_name: str = "EDGE_PREDICTION"
    ):
        """Create new abstract Edge Prediction Model object.

        Parameters
        --------------------
        graph: Graph,
            The graph object to base the model on.
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
        trainable_embedding: bool = False,
            Whether to allow for trainable embedding.
        use_dropout: bool = True,
            Whether to use dropout.
        dropout_rate: float = 0.5,
            Dropout rate.
        use_edge_metrics: bool = False,
            Whether to return the edge metrics.
        task_name: str = "EDGE_PREDICTION",
            The name of the task to build the model for.
            The currently supported task names are `EDGE_PREDICTION` and `EDGE_LABEL_PREDICTION`.
            The default task name is `EDGE_PREDICTION`.
        """
        supported_edge_embedding_methods = list(edge_embedding_layer.keys())
        if not set_validator(supported_edge_embedding_methods)(edge_embedding_method):
            raise ValueError(
                (
                    "The provided edge embedding method name `{edge_embedding_method}` is not supported. Did you mean {closest}?\n"
                    "The supported edge embedding method names are: {supported_tasks}."
                ).format(
                    edge_embedding_method=edge_embedding_method,
                    closest=closest(edge_embedding_method,
                                    supported_edge_embedding_methods),
                    supported_tasks=", ".join(
                        supported_edge_embedding_methods),
                )
            )
        if not set_validator(edge_prediction_supported_tasks)(task_name):
            raise ValueError(
                (
                    "The provided task name `{task_name}` is not supported. Did you mean {closest}?\n"
                    "The supported task names are: {supported_tasks}."
                ).format(
                    task_name=task_name,
                    closest=closest(
                        task_name, edge_prediction_supported_tasks),
                    supported_tasks=", ".join(edge_prediction_supported_tasks),
                )
            )
        if embedding is None and not trainable_embedding:
            warnings.warn(
                "The embedding was not provided, therefore a new random Normal embedding "
                "will be allocated, but also the trainable embedding flag was left set to false.\n"
                "This means that the embedding will not be trained, and the nodes will have "
                "exclusively random features associated to them."
            )
        self._graph = graph
        self._task_name = task_name
        self._use_edge_metrics = use_edge_metrics
        self._edge_embedding_method = edge_embedding_method
        self._model_name = self.__class__.__name__
        self._use_dropout = use_dropout
        self._dropout_rate = dropout_rate
        super().__init__(
            vocabulary_size=self._graph.get_nodes_number(),
            embedding_size=embedding_size,
            optimizer=optimizer,
            embedding=embedding,
            trainable_embedding=trainable_embedding
        )

    def _compile_model(self) -> Model:
        """Compile model."""
        if self._task_name == "EDGE_PREDICTION":
            self._model.compile(
                loss="binary_crossentropy",
                optimizer=self._optimizer,
                metrics=get_standard_binary_metrics(),
            )
        elif self._task_name == "EDGE_LABEL_PREDICTION":
            self._model.compile(
                loss="sparse_categorical_crossentropy",
                optimizer=self._optimizer,
                metrics=get_sparse_multiclass_metrics(),
            )
        else:
            raise ValueError("Unreacheable!")

    def _build_model_body(self, input_layer: Layer) -> Layer:
        """Build new model body for Edge prediction.

        Parameters
        --------------------
        input_layer: Layer,
            The previous layer to be used as input of the model.
        """
        raise NotImplementedError(
            "The method _build_model_body must be implemented in the child classes."
        )

    def _build_model(self) -> Model:
        """Build new model for Edge prediction."""
        embedding_layer = edge_embedding_layer[self._edge_embedding_method](
            nodes_number=self._graph.get_nodes_number(),
            embedding_size=self._embedding_size,
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
            edge_embedding = Concatenate()([
                edge_embedding,
                edge_metrics_input
            ])

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
        if self._task_name == "EDGE_PREDICTION":
            sequence = EdgePredictionSequence(
                graph,
                batch_size=batch_size,
                batches_per_epoch=batches_per_epoch,
                negative_samples_rate=negative_samples_rate,
                support_mirrored_strategy=support_mirrored_strategy,
                use_edge_metrics=self._use_edge_metrics,
            )
        elif self._task_name == "EDGE_LABEL_PREDICTION":
            sequence = EdgeLabelPredictionSequence(
                graph,
                batch_size=batch_size,
                batches_per_epoch=batches_per_epoch,
                support_mirrored_strategy=support_mirrored_strategy,
                use_edge_metrics=self._use_edge_metrics,
            )
        else:
            raise ValueError("Unreacheable!")
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
