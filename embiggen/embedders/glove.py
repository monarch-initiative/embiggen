"""GloVe model for graph and words embedding."""
from typing import Dict, List, Union

import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K  # pylint: disable=import-error
from tensorflow.keras.layers import (Add, Dot,  # pylint: disable=import-error
                                     Embedding, Flatten, Input)
from tensorflow.keras.models import Model  # pylint: disable=import-error
from tensorflow.keras.optimizers import \
    Optimizer  # pylint: disable=import-error

from .embedder import Embedder


class GloVe(Embedder):
    """GloVe model for graph and words embedding.

    The GloVe model for graoh embedding receives two words and is asked to
    predict its cooccurrence probability.
    """

    def __init__(
        self,
        vocabulary_size: int,
        embedding_size: int,
        optimizer: Union[str, Optimizer] = None,
        alpha: float = 0.75,
    ):
        """Create new GloVe-based Embedder object.

        Parameters
        ----------------------------
        vocabulary_size: int,
            Number of terms to embed.
            In a graph this is the number of nodes, while in a text is the
            number of the unique words.
        embedding_size: int,
            Dimension of the embedding.
        optimizer: Union[str, Optimizer] = "nadam",
            The optimizer to be used during the training of the model.
            By default, if None is provided, Nadam with learning rate
            set at 0.01 is used.
        alpha: float = 0.75,
            Alpha to use for the function.
        """
        self._alpha = alpha
        super().__init__(
            vocabulary_size=vocabulary_size,
            embedding_size=embedding_size,
            optimizer=optimizer
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
        input_layers = [
            Input((1,)),
            Input((1,))
        ]

        embedding_layers = [
            Embedding(
                self._vocabulary_size,
                self._embedding_size,
                input_length=1,
                weights=None if self._embedding is None else [
                    self._embedding
                ],
                name=Embedder.EMBEDDING_LAYER_NAME
            )(input_layers[0]),
            Embedding(
                self._vocabulary_size,
                self._embedding_size,
                input_length=1,
            )(input_layers[1])
        ]

        # Creating the dot product of the embedding layers
        dot_product_layer = Dot(axes=2)(embedding_layers)

        # Creating the biases layer
        biases = [
            Embedding(
                self._vocabulary_size,
                1,
                input_length=1
            )(input_layer)
            for input_layer in input_layers
        ]

        # Concatenating with an add the three layers
        prediction = Flatten()(Add()([dot_product_layer, *biases]))

        # Creating the model
        glove = Model(
            inputs=input_layers,
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
        *args: List,
        epochs: int = 1000,
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
        epochs: int = 1000,
            Epochs to train the model for.
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
