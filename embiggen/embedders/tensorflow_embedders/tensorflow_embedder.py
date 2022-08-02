"""Abstract Keras Model wrapper for embedding models."""
from typing import Dict, Sequence, Tuple, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from ensmallen import Graph
from tensorflow.keras.callbacks import (  # pylint: disable=import-error,no-name-in-module
    EarlyStopping, ReduceLROnPlateau)
from tensorflow.keras.models import \
    Model  # pylint: disable=import-error,no-name-in-module

from embiggen.utils.tensorflow_utils import execute_gpu_checks, get_available_gpus_number, has_gpus
from embiggen.utils.abstract_models import AbstractEmbeddingModel, abstract_class, EmbeddingResult


@abstract_class
class TensorFlowEmbedder(AbstractEmbeddingModel):
    """Abstract Keras Model wrapper for embedding models."""

    def __init__(
        self,
        embedding_size: int = 100,
        early_stopping_min_delta: float = 0.001,
        early_stopping_patience: int = 1,
        learning_rate_plateau_min_delta: float = 0.001,
        learning_rate_plateau_patience: int = 1,
        epochs: int = 10,
        batch_size: int = 2**10,
        optimizer: str = "nadam",
        verbose: bool = False,
        use_mirrored_strategy: bool = False,
        enable_cache: bool = False,
        random_state: int = 42
    ):
        """Create new TensorFlowEmbedder object.

        Parameters
        ----------------------------------
        embedding_size: int = 100
            Dimension of the embedding.
        early_stopping_min_delta: float = 0.001
            The minimum variation in the provided patience time
            of the loss to not stop the training.
        early_stopping_patience: int = 1
            The amount of epochs to wait for better training
            performance.
        learning_rate_plateau_min_delta: float = 0.001
            The minimum variation in the provided patience time
            of the loss to not reduce the learning rate.
        learning_rate_plateau_patience: int = 1
            The amount of epochs to wait for better training
            performance without decreasing the learning rate.
        epochs: int = 10
            Number of epochs to train.
        batch_size: int = 2**10
            Batch size to use during the training.
        optimizer: str = "nadam"
            Optimizer to use during the training.
        verbose: bool = False
            Whether to show loading bars.
        use_mirrored_strategy: bool = False
            Whether to use mirrored strategy.
        enable_cache: bool = False
            Whether to enable the cache, that is to
            store the computed embedding.
        random_state: Optional[int] = None
            The random state to use if the model is stocastic.
        """
        execute_gpu_checks()
        if use_mirrored_strategy and get_available_gpus_number() <= 1:
            raise ValueError(
                "Mirrored strategy was requested, one "
                "or less GPUs where detected."
            )
        self._use_mirrored_strategy = use_mirrored_strategy
        self._epochs = epochs
        self._batch_size = batch_size
        self._optimizer = optimizer
        self._verbose = verbose
        self._early_stopping_min_delta = early_stopping_min_delta
        self._early_stopping_patience = early_stopping_patience
        self._learning_rate_plateau_min_delta = learning_rate_plateau_min_delta
        self._learning_rate_plateau_patience = learning_rate_plateau_patience
        super().__init__(
            embedding_size=embedding_size,
            enable_cache=enable_cache,
            random_state=random_state,
        )

    @classmethod
    def smoke_test_parameters(cls) -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            embedding_size=5,
            epochs=1
        )

    def parameters(self) -> Dict[str, Any]:
        return dict(
            **super().parameters(),
            **dict(
                use_mirrored_strategy=self._use_mirrored_strategy,
                epochs=self._epochs,
                batch_size=self._batch_size,
                optimizer=self._optimizer,
                verbose=self._verbose,
                early_stopping_min_delta=self._early_stopping_min_delta,
                early_stopping_patience=self._early_stopping_patience,
                learning_rate_plateau_min_delta=self._learning_rate_plateau_min_delta,
                learning_rate_plateau_patience=self._learning_rate_plateau_patience,
            )
        )

    @classmethod
    def library_name(cls) -> str:
        return "TensorFlow"

    def _build_model(self, graph: Graph) -> Model:
        """Build new model for embedding.

        Parameters
        ------------------
        graph: Graph
            The graph to build the model for.
        """
        raise NotImplementedError(
            f"In the child class {self.__class__.__name__} of {super().__name__.__name__} "
            f"implementing the model {self.model_name()} we could not find the method "
            "called `_build_model`. Please do implement it."
        )

    def _build_input(self, graph: Graph) -> Tuple[Any]:
        """Returns values to be fed as input into the model.

        Parameters
        ------------------
        graph: Graph
            The graph to build the model for.
        verbose: bool
            Whether to show loading bars while building input.
        """
        raise NotImplementedError(
            f"In the child class {self.__class__.__name__} of {super().__name__.__name__} "
            f"implementing the model {self.model_name()} we could not find the method "
            "called `_build_input`. Please do implement it."
        )

    def _get_steps_per_epoch(self, graph: Graph) -> Tuple[Any]:
        """Returns number of steps per epoch.

        Parameters
        ------------------
        graph: Graph
            The graph to compute the number of steps.
        """
        return None

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
        raise NotImplementedError(
            f"In the child class {self.__class__.__name__} of {super().__class__.__name__} "
            f"implementing the model {self.model_name()} we could not find the method "
            "called `_extract_embeddings`. Please do implement it."
        )

    def get_layer_weights(
        self,
        layer_name: str,
        model: Model,
        drop_first_row: bool = False
    ) -> np.ndarray:
        """Return weights from the requested layer.

        Parameters
        -----------------------
        layer_name: str
            Name of the layer to query for.
        model: Model
            The model from where to extract the layer weights.
        drop_first_row: bool = False
            Whether to drop the first row of the weights.
            May be useful for when the embedding included
            padding, such as when dealing with unknown
            edge and node types or multilabel node types.
        """
        for layer in model.layers:
            if layer.name == layer_name:
                weights = layer.get_weights()[0]
                if drop_first_row:
                    weights = weights[1:]
                return weights
        raise NotImplementedError(
            f"The model {self.model_name()} implemented in the class {self.__class__.__name__} "
            f"does not have a layer called {layer_name}."
        )

    def _fit_transform(
        self,
        graph: Graph,
        return_dataframe: bool = True,
    ) -> EmbeddingResult:
        """Return node embedding"""
        try:
            from tqdm.keras import TqdmCallback
            traditional_verbose = False
        except AttributeError:
            traditional_verbose = True

        if has_gpus() and self._use_mirrored_strategy:
            strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        else:
            strategy = tf.distribute.get_strategy()

        # # Build the model
        with strategy.scope():
            model = self._build_model(graph)

        # Get the model input
        training_input = self._build_input(
            graph,
        )

        if not isinstance(training_input, tuple):
            raise ValueError(
                "The provided input data is not a tuple."
            )

        # Fit the model
        model.fit(
            *training_input,
            epochs=self._epochs,
            verbose=traditional_verbose and self._verbose > 0,
            batch_size=(
                self._batch_size
                if issubclass(training_input[0].__class__, Sequence)
                else None
            ),
            steps_per_epoch=self._get_steps_per_epoch(graph),
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
                *((TqdmCallback(verbose=1, leave=False),)
                  if not traditional_verbose and self._verbose > 0 else ()),
            ],
        )

        # Extract and return the embedding
        return self._extract_embeddings(
            graph,
            model,
            return_dataframe=return_dataframe
        )

    @classmethod
    def is_stocastic(cls) -> bool:
        """Returns whether the model is stocastic and has therefore a random state."""
        return True
