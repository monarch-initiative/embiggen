"""Abstract Keras Model object for embedding models."""
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from embiggen.utils.parameter_validators import validate_verbose
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import (  # pylint: disable=import-error,no-name-in-module
    EarlyStopping, ReduceLROnPlateau)
from tensorflow.keras.models import \
    Model  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.optimizers import \
    Nadam  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.optimizers import \
    Optimizer  # pylint: disable=import-error,no-name-in-module

from .optimizers import apply_centralized_gradients


class Embedder:
    """Abstract Keras Model object for embedding models."""

    TERMS_EMBEDDING_LAYER_NAME = "terms_embedding_layer"

    def __init__(
        self,
        vocabulary_size: int = None,
        embedding_size: int = None,
        embedding: Union[np.ndarray, pd.DataFrame] = None,
        extra_features: Union[np.ndarray, pd.DataFrame] = None,
        optimizer: Union[str, Optimizer] = None,
        trainable_embedding: bool = True,
        use_gradient_centralization: bool = True
    ):
        """Create new Embedder object.

        Parameters
        ----------------------------------
        vocabulary_size: int = None,
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
        extra_features: Union[np.ndarray, pd.DataFrame] = None,
            Optional extra features to be used during the computation
            of the embedding. The features must be available for all the
            elements considered for the embedding.
        optimizer: Union[str, Optimizer] = None,
            The optimizer to be used during the training of the model.
            By default, if None is provided, Nadam with learning rate
            set at 0.01 is used.
        trainable_embedding: bool = True,
            Wether to allow for trainable embedding.
            By default true.
        use_gradient_centralization: bool = True,
            Whether to wrap the provided optimizer into a normalized
            one that centralizes the gradient.
            It is automatically enabled if the current version of
            TensorFlow supports gradient transformers.
            More detail here: https://arxiv.org/pdf/2004.01461.pdf

        Raises
        -----------------------------------
        ValueError,
            When the given vocabulary size is not a strictly positive integer.
        ValueError,
            When the given embedding size is not a strictly positive integer.
        ValueError,
            When both vocabulary size or embedding size are provided with also
            the seed embedding.
        """
        if embedding is not None:
            if isinstance(embedding, pd.DataFrame):
                embedding = embedding.values
            if not isinstance(embedding, np.ndarray):
                raise ValueError(
                    "Given embedding is not a numpy array."
                )
            if vocabulary_size is not None and embedding_size is not None and vocabulary_size != embedding_size.shape[0]:
                raise ValueError((
                    "Both seed embedding and vocabulary size were provided but the two values "
                    "are not compatible. Namely, the vocabulary size is {} while the embedding "
                    "shape is {}."
                ).format(
                    vocabulary_size,
                    embedding_size.shape
                ))
            embedding_size = embedding.shape[1]
            vocabulary_size = embedding.shape[0]

        if extra_features is not None:
            if isinstance(extra_features, pd.DataFrame):
                extra_features = extra_features.values
            if not isinstance(extra_features, np.ndarray):
                raise ValueError(
                    "Given extra features is not a numpy array."
                )
            if vocabulary_size != extra_features.shape[0]:
                raise ValueError(
                    (
                        "The number of samples in the extra features should "
                        "be the same as the provided vocabulary size {} "
                        "but was {}."
                    ).format(
                        vocabulary_size,
                        extra_features.shape[0]
                    )
                )

        if not isinstance(vocabulary_size, int) or vocabulary_size < 1:
            raise ValueError((
                "The given vocabulary size ({}) "
                "is not a strictly positive integer."
            ).format(
                vocabulary_size
            ))
        if not isinstance(embedding_size, int) or embedding_size < 1:
            raise ValueError((
                "The given embedding size ({}) "
                "is not a strictly positive integer."
            ).format(
                embedding_size
            ))
        self._vocabulary_size = vocabulary_size
        self._embedding_size = embedding_size
        self._embedding = embedding
        self._extra_features = extra_features

        if optimizer is None:
            optimizer = Nadam(learning_rate=0.01)

        if isinstance(optimizer, str):
            optimizer = tf.keras.optimizers.get(optimizer)

        if use_gradient_centralization:
            apply_centralized_gradients(optimizer)

        self._optimizer = optimizer
        self._model = self._build_model()
        self.trainable = trainable_embedding

    def _build_model(self) -> Model:
        """Build new model for embedding."""
        raise NotImplementedError(
            "The method _build_model must be implemented in the child classes."
        )

    def _compile_model(self) -> Model:
        """Compile model."""
        raise NotImplementedError(
            "The method _compile_model must be implemented in the child classes."
        )

    def summary(self):
        """Print model summary."""
        self._model.summary()

    def get_layer_weights(self, layer_name: str) -> np.ndarray:
        """Return weights from the requested layer.

        Parameters
        -----------------------
        layer_name: str,
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

    @property
    def embedding(self) -> np.ndarray:
        """Return model embeddings.

        Raises
        -------------------
        NotImplementedError,
            If the current embedding model does not have an embedding layer.
        """
        return self.get_layer_weights(Embedder.TERMS_EMBEDDING_LAYER_NAME)

    @property
    def trainable(self) -> bool:
        """Return whether the embedding layer can be trained.

        Raises
        -------------------
        NotImplementedError,
            If the current embedding model does not have an embedding layer.
        """
        for layer in self._model.layers:
            if layer.name == Embedder.TERMS_EMBEDDING_LAYER_NAME:
                return layer.trainable
        raise NotImplementedError(
            "This embedding model does not have an embedding layer."
        )

    @trainable.setter
    def trainable(self, trainable: bool):
        """Set whether the embedding layer can be trained or not.

        Parameters
        -------------------
        trainable: bool,
            Whether the embedding layer can be trained or not.
        """
        for layer in self._model.layers:
            if layer.name == Embedder.TERMS_EMBEDDING_LAYER_NAME:
                layer.trainable = trainable
        self._compile_model()

    def get_embedding_dataframe(self, term_names: List[str]) -> pd.DataFrame:
        """Return terms embedding using given index names.

        Parameters
        -----------------------------
        term_names: List[str],
            List of terms to be used as index names.
        """
        return pd.DataFrame(
            self.embedding,
            index=term_names
        )

    def save_embedding(self, path: str, term_names: List[str]):
        """Save terms embedding using given index names.

        Parameters
        -----------------------------
        path: str,
            Save embedding as csv to given path.
        term_names: List[str],
            List of terms to be used as index names.
        """
        self.get_embedding_dataframe(term_names).to_csv(path, header=False)

    @property
    def name(self) -> str:
        """Return model name."""
        return self._model.name

    def save_weights(self, path: str):
        """Save model weights to given path.

        Parameters
        ---------------------------
        path: str,
            Path where to save model weights.
        """
        self._model.save_weights(path)

    def load_weights(self, path: str):
        """Load model weights from given path.

        Parameters
        ---------------------------
        path: str,
            Path from where to load model weights.
        """
        self._model.load_weights(path)

    def fit(
        self,
        *args,
        early_stopping_min_delta: float = 0.1,
        early_stopping_patience: int = 3,
        reduce_lr_min_delta: float = 1,
        reduce_lr_patience: int = 1,
        epochs: int = 10000,
        early_stopping_monitor: str = "loss",
        early_stopping_mode: str = "min",
        reduce_lr_monitor: str = "loss",
        reduce_lr_mode: str = "min",
        reduce_lr_factor: float = 0.9,
        verbose: int = 1,
        **kwargs: Dict
    ) -> pd.DataFrame:
        """Return pandas dataframe with training history.

        Parameters
        -----------------------
        early_stopping_min_delta: float,
            Minimum delta of metric to stop the training.
        early_stopping_patience: int,
            Number of epochs to wait for when the given minimum delta is not
            achieved after which trigger early stopping.
        reduce_lr_min_delta: float,
            Minimum delta of metric to reduce learning rate.
        reduce_lr_patience: int,
            Number of epochs to wait for when the given minimum delta is not
            achieved after which reducing learning rate.
        epochs: int = 10000,
            Epochs to train the model for.
        early_stopping_monitor: str = "loss",
            Metric to monitor for early stopping.
        early_stopping_mode: str = "min",
            Direction of the variation of the monitored metric for early stopping.
        reduce_lr_monitor: str = "loss",
            Metric to monitor for reducing learning rate.
        reduce_lr_mode: str = "min",
            Direction of the variation of the monitored metric for learning rate.
        reduce_lr_factor: float = 0.9,
            Factor for reduction of learning rate.
        verbose: int = 1,
            Wethever to show the loading bar.
            Specifically, the options are:
            * 0 or False: No loading bar.
            * 1 or True: Showing only the loading bar for the epochs.
            * 2: Showing loading bar for both epochs and batches.
        **kwargs: Dict,
            Additional kwargs to pass to the Keras fit call.

        Raises
        -----------------------
        ValueError,
            If given verbose value is not within the available set (-1, 0, 1).

        Returns
        -----------------------
        Dataframe with training history.
        """
        try:
            from tqdm.keras import TqdmCallback
            traditional_verbose = False
        except AttributeError:
            traditional_verbose = True
        verbose = validate_verbose(verbose)
        callbacks = kwargs.pop("callbacks", ())
        return pd.DataFrame(self._model.fit(
            *args,
            epochs=epochs,
            verbose=traditional_verbose and verbose > 0,
            callbacks=[
                EarlyStopping(
                    monitor=early_stopping_monitor,
                    min_delta=early_stopping_min_delta,
                    patience=early_stopping_patience,
                    mode=early_stopping_mode,
                ),
                ReduceLROnPlateau(
                    monitor=reduce_lr_monitor,
                    min_delta=reduce_lr_min_delta,
                    patience=reduce_lr_patience,
                    factor=reduce_lr_factor,
                    mode=reduce_lr_mode,
                ),
                *((TqdmCallback(verbose=verbose-1),)
                  if not traditional_verbose and verbose > 0 else ()),
                *callbacks
            ],
            **kwargs
        ).history)
