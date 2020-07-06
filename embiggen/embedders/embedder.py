import tensorflow as tf
from typing import Union, Tuple, Dict, List
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer


class Embedder:

    EMBEDDING_LAYER_NAME = "words_embedding"

    def __init__(
        self,
        vocabulary_size: int,
        embedding_size: int,
        optimizer: Union[str, Optimizer] = "nadam"
    ):
        """Create new Embedder object.

        Parameters
        ----------------------------------
        vocabulary_size: int,
            Number of terms to embed.
            In a graph this is the number of nodes, while in a text is the
            number of the unique words.
        embedding_size: int,
            Dimension of the embedding.
        optimizer: Union[str, Optimizer] = "nadam",
            The optimizer to be used during the training of the model.

        Raises
        -----------------------------------
        ValueError,
            When the given vocabulary size is not a strictly positive integer.
        ValueError,
            When the given embedding size is not a strictly positive integer.
        """
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
        self._optimizer = optimizer
        self._model = self._build_model()

    def _build_model(self) -> Model:
        """Build new model for embedding."""
        raise NotImplementedError(
            "The method _build_model must be implemented in the child classes."
        )

    def summary(self):
        """Print model summary."""
        self._model.summary()

    @property
    def embedding(self) -> np.ndarray:
        """Return model embeddings."""
        for layer, weights in zip(self._model.layers, self._model.weights):
            if layer.name == Embedder.EMBEDDING_LAYER_NAME:
                return weights.numpy()

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

    def fit(self, *args, **kwargs)->pd.DataFrame:
        """Return pandas dataframe with training history."""
        return pd.DataFrame(self._model.fit(*args, **kwargs).history)