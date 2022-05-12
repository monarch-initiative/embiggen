"""Abstract Keras Model wrapper for embedding models."""
from typing import Dict, List, Union, Optional, Tuple, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from ensmallen import Graph
from embiggen.utils.parameter_validators import validate_verbose
from tensorflow.keras.callbacks import (  # pylint: disable=import-error,no-name-in-module
    EarlyStopping, ReduceLROnPlateau)
from tensorflow.keras.models import \
    Model  # pylint: disable=import-error,no-name-in-module

from ...utils.tensorflow_utils import execute_gpu_checks, get_available_gpus_number
from ...utils import AbstractEmbeddingModel


class TensorFlowEmbedder(AbstractEmbeddingModel):
    """Abstract Keras Model wrapper for embedding models."""

    def __init__(
        self,
        embedding_size: int,
        early_stopping_min_delta: float,
        early_stopping_patience: int,
        learning_rate_plateau_min_delta: float,
        learning_rate_plateau_patience: int,
        epochs: int = 10,
        optimizer: str = "sgd",
        use_mirrored_strategy: bool = False
    ):
        """Create new TensorFlowEmbedder object.

        Parameters
        ----------------------------------
        embedding_size: int = None
            Dimension of the embedding.
        early_stopping_min_delta: float
            The minimum variation in the provided patience time
            of the loss to not stop the training.
        early_stopping_patience: int
            The amount of epochs to wait for better training
            performance.
        learning_rate_plateau_min_delta: float
            The minimum variation in the provided patience time
            of the loss to not reduce the learning rate.
        learning_rate_plateau_patience: int
            The amount of epochs to wait for better training
            performance without decreasing the learning rate.
        epochs: int = 10
            Number of epochs to train.
        optimizer: str = "sgd"
            Optimizer to use during the training.
        use_mirrored_strategy: bool = False
            Whether to use mirrored strategy.
        """
        execute_gpu_checks()
        if use_mirrored_strategy and get_available_gpus_number() <= 1:
            raise ValueError(
                "Mirrored strategy was requested, one "
                "or less GPUs where detected."
            )
        self._epochs = epochs
        self._optimizer = optimizer
        self._early_stopping_min_delta = early_stopping_min_delta
        self._early_stopping_patience = early_stopping_patience
        self._learning_rate_plateau_min_delta = learning_rate_plateau_min_delta
        self._learning_rate_plateau_patience = learning_rate_plateau_patience
        super().__init__(embedding_size=embedding_size)

    def _build_model(self, graph: Graph) -> Model:
        """Build new model for embedding.

        Parameters
        ------------------
        graph: Graph
            The graph to build the model for.
        """
        raise NotImplementedError(
            "The method _build_model must be implemented in the child classes."
        )

    def _build_input(self, graph: Graph, verbose: bool) -> Tuple[Any]:
        """Returns values to be fed as input into the model.

        Parameters
        ------------------
        graph: Graph
            The graph to build the model for.
        verbose: bool
            Whether to show loading bars while building input.
        """
        raise NotImplementedError(
            "The method _build_input must be implemented in the child classes."
        )

    def _extract_embeddings(
        self,
        graph: Graph,
        model: Model,
        return_dataframe: bool
    ) -> Union[np.ndarray, pd.DataFrame, Dict[str, np.ndarray], Dict[str, pd.DataFrame]]:
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
        raise NotImplementedError(
            "The method _build_input must be implemented in the child classes."
        )

    def summary(self):
        """Print model summary."""
        self._model.summary()

    def get_layer_weights(self, layer_name: str) -> np.ndarray:
        """Return weights from the requested layer.

        Parameters
        -----------------------
        layer_name: str
            Name of the layer to query for.
        """
        for layer in self._model.layers:
            if layer.name == layer_name:
                return layer.get_weights()[0]
        raise NotImplementedError(
            "This model does not have a layer called {}.".format(
                layer_name
            )
        )

    def _fit_transform(
        self,
        graph: Graph,
        return_dataframe: bool = True,
        verbose: bool = True
    ) -> Union[np.ndarray, pd.DataFrame, Dict[str, np.ndarray], Dict[str, pd.DataFrame]]:
        """Return node embedding"""
        try:
            from tqdm.keras import TqdmCallback
            traditional_verbose = False
        except AttributeError:
            traditional_verbose = True

        # Build the model
        model = self._build_model(graph)

        # Get the model input
        training_input = self._build_input(graph)

        # Fit the model
        model.fit(
            *training_input,
            epochs=self._epochs,
            verbose=traditional_verbose and verbose > 0,
            callbacks=[
                EarlyStopping(
                    monitor="loss",
                    min_delta=self._early_stopping_min_delta,
                    patience=self._early_stopping_patience,
                    mode="min",
                ),
                ReduceLROnPlateau(
                    monitor="loss",
                    min_delta=self._learning_rate_plateau_min_delta,
                    patience=self._learning_rate_plateau_patience,
                    factor=0.5,
                    mode="min",
                ),
                *((TqdmCallback(verbose=verbose-1),)
                  if not traditional_verbose and verbose > 0 else ()),
            ],
        )

        # Extract and return the embedding
        return self._extract_embeddings(
            graph,
            model,
            return_dataframe=return_dataframe
        )
