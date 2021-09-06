"""GloVe model for graph and words embedding."""
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.layers import Add  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.layers import Concatenate, Dot, Embedding, Flatten, Input  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.models import Model  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.optimizers import \
    Optimizer  # pylint: disable=import-error,no-name-in-module

from ..sequences import GloveSequence
from .embedder import Embedder


class GloVe(Embedder):
    """GloVe model for graph and words embedding.

    The GloVe model for graph embedding receives two words and is asked to
    predict its cooccurrence probability.
    """

    def __init__(
        self,
        vocabulary_size: int,
        embedding_size: int,
        embedding: Union[np.ndarray, pd.DataFrame] = None,
        extra_features: Union[np.ndarray, pd.DataFrame] = None,
        optimizer: Union[str, Optimizer] = None,
        alpha: float = 0.75,
        random_state: int = 42,
        directed: bool = False,
        use_gradient_centralization: bool = True,
    ):
        """Create new GloVe-based Embedder object.

        Parameters
        -------------------------------
        vocabulary_size: int,
            Number of terms to embed.
            In a graph this is the number of nodes, while in a text is the
            number of the unique words.
        embedding_size: int,
            Dimension of the embedding.
        embedding: Union[np.ndarray, pd.DataFrame] = None,
            The seed embedding to be used.
            Note that it is not possible to provide at once both
            the embedding and either the vocabulary size or the embedding size.
        extra_features: Union[np.ndarray, pd.DataFrame] = None,
            Optional extra features to be used during the computation
            of the embedding. The features must be available for all the
            elements considered for the embedding.
        optimizer: Union[str, Optimizer] = "nadam",
            The optimizer to be used during the training of the model.
            By default, if None is provided, Nadam with learning rate
            set at 0.01 is used.
        alpha: float = 0.75,
            Alpha to use for the function.
        random_state: int = 42,
            The random state to reproduce the training sequence.
        directed: bool = False,
            Whether to treat the data as directed or not.
        use_gradient_centralization: bool = True,
            Whether to wrap the provided optimizer into a normalized
            one that centralizes the gradient.
            It is automatically enabled if the current version of
            TensorFlow supports gradient transformers.
            More detail here: https://arxiv.org/pdf/2004.01461.pdf
        """
        self._alpha = alpha
        self._random_state = random_state
        self._directed = directed
        super().__init__(
            vocabulary_size=vocabulary_size,
            embedding_size=embedding_size,
            embedding=embedding,
            extra_features=extra_features,
            optimizer=optimizer,
            use_gradient_centralization=use_gradient_centralization
        )

    def _glove_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
        """Compute the glove loss function.

        Parameters
        ---------------------------
        y_true: tf.Tensor,
            The true values Tensor for this batch.
        y_pred: tf.Tensor,
            The predicted values Tensor for this batch.

        Returns
        ---------------------------
        Loss function score related to this batch.
        """
        return K.sum(
            K.pow(K.clip(y_true, 0.0, 1.0), self._alpha) *
            K.square(y_pred - K.log(y_true)),
            axis=-1
        )

    def _build_model(self):
        """Create new Glove model."""
        # Creating the input layers
        left_input_layer = Input((1,), name="left_input_layer")
        right_input_layer = Input((1,), name="right_input_layer")

        trainable_left_embedding = Embedding(
            self._vocabulary_size,
            self._embedding_size,
            input_length=1,
            weights=None if self._embedding is None else [
                self._embedding
            ],
            name=Embedder.TERMS_EMBEDDING_LAYER_NAME
        )(left_input_layer)

        trainable_right_embedding = Embedding(
            self._vocabulary_size,
            self._embedding_size,
            input_length=1,
        )(right_input_layer)

        if self._extra_features is not None:
            extra_features_matrix = Embedding(
                *self._extra_features,
                input_length=1,
                weights=self._extra_features,
                trainable=False,
                name="extra_features_matrix"
            )
            trainable_left_embedding = Concatenate()([
                extra_features_matrix(left_input_layer),
                trainable_left_embedding
            ])
            trainable_right_embedding = Concatenate()([
                extra_features_matrix(right_input_layer),
                trainable_right_embedding
            ])

        # Creating the dot product of the embedding layers
        dot_product_layer = Dot(axes=2)([
            trainable_left_embedding,
            trainable_right_embedding
        ])

        # Creating the biases layer
        biases = [
            Embedding(self._vocabulary_size, 1, input_length=1)(input_layer)
            for input_layer in (left_input_layer, right_input_layer)
        ]

        # Concatenating with an add the three layers
        prediction = Flatten()(Add()([dot_product_layer, *biases]))

        # Creating the model
        glove = Model(
            inputs=[
                left_input_layer,
                right_input_layer
            ],
            outputs=prediction,
            name="GloVe"
        )

        return glove

    def _compile_model(self) -> Model:
        """Compile model."""
        self._model.compile(
            loss=self._glove_loss,
            optimizer=self._optimizer
        )

    def fit(
        self,
        X: Tuple[np.ndarray, np.ndarray],
        frequencies: np.ndarray,
        *args: List,
        epochs: int = 1000,
        batch_size: int = 2**20,
        early_stopping_monitor: str = "loss",
        early_stopping_min_delta: float = 0.001,
        early_stopping_patience: int = 10,
        early_stopping_mode: str = "min",
        reduce_lr_monitor: str = "loss",
        reduce_lr_min_delta: float = 0.01,
        reduce_lr_patience: int = 10,
        reduce_lr_mode: str = "min",
        reduce_lr_factor: float = 0.9,
        verbose: int = 1,
        **kwargs: Dict
    ) -> pd.DataFrame:
        """Return pandas dataframe with training history.

        Parameters
        -----------------------
        X: Tuple[np.ndarray, np.ndarray],
            Tuple with source and destinations.
        frequencies: np.ndarray,
            The frequencies to predict.
        *args: List,
            Other arguments to provide to the model.
        epochs: int = 1000,
            Epochs to train the model for.
        batch_size: int = 2**20,
            The batch size.
            Tipically batch sizes for the GloVe model can be immense.
        early_stopping_monitor: str = "loss",
            Metric to monitor for early stopping.
        early_stopping_min_delta: float = 0.001,
            Minimum delta of metric to stop the training.
        early_stopping_patience: int = 10,
            Number of epochs to wait for when the given minimum delta is not
            achieved after which trigger early stopping.
        early_stopping_mode: str = "min",
            Direction of the variation of the monitored metric for early stopping.
        reduce_lr_monitor: str = "loss",
            Metric to monitor for reducing learning rate.
        reduce_lr_min_delta: float = 0.01,
            Minimum delta of metric to reduce learning rate.
        reduce_lr_patience: int = 10,
            Number of epochs to wait for when the given minimum delta is not
            achieved after which reducing learning rate.
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
        return super().fit(
            GloveSequence(
                *X, frequencies,
                batch_size=batch_size,
                directed=self._directed,
                random_state=self._random_state
            ),
            *args,
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
