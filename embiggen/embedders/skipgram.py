"""SkipGram model for sequences embedding."""
from typing import Tuple, Union

import numpy as np
import pandas as pd
from tensorflow.keras.layers import (Flatten,  # pylint: disable=import-error
                                     Layer)
from tensorflow.keras.optimizers import \
    Optimizer  # pylint: disable=import-error

from .word2vec import Word2Vec


class SkipGram(Word2Vec):
    """SkipGram model for sequences embedding.

    The SkipGram model for graoh embedding receives a central word and tries
    to predict its contexts. The model makes use of an NCE loss layer
    during the training process to generate the negatives.
    """

    def __init__(
        self,
        vocabulary_size: int,
        embedding_size: int,
        model_name: str = "SkipGram",
        embedding: Union[np.ndarray, pd.DataFrame] = None,
        extra_features: Union[np.ndarray, pd.DataFrame] = None,
        optimizer: Union[str, Optimizer] = None,
        window_size: int = 16,
        negative_samples: int = 10,
    ):
        """Create new CBOW-based Embedder object.

        Parameters
        -------------------------------------------
        vocabulary_size: int,
            Number of terms to embed.
            In a graph this is the number of nodes, while in a text is the
            number of the unique words.
        embedding_size: int,
            Dimension of the embedding.
        model_name: str = "SkipGram",
            Name of the model.
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
        window_size: int = 16,
            Window size for the local context.
            On the borders the window size is trimmed.
        negative_samples: int = 10,
            The number of negative classes to randomly sample per batch.
            This single sample of negative classes is evaluated for each element in the batch.
        """
        super().__init__(
            vocabulary_size=vocabulary_size,
            embedding_size=embedding_size,
            model_name=model_name,
            embedding=embedding,
            extra_features=extra_features,
            optimizer=optimizer,
            window_size=window_size,
            negative_samples=negative_samples,
        )

    def _get_true_input_length(self) -> int:
        """Return length of true input layer."""
        return 1

    def _get_true_output_length(self) -> int:
        """Return length of true output layer."""
        return self._window_size*2

    def _merging_layer(self, embedding_layer: Layer) -> Layer:
        """Return layer to be used to compose the layer from the input node(s).

        Parameters
        ----------------------------
        embedding_layer: Layer,
            The embedding layer.

        Returns
        ----------------------------
        Layer with composition of the embedding layers.
        """
        return Flatten()(embedding_layer)

    def _sort_input_layers(
        self,
        true_input_layer: Layer,
        true_output_layer: Layer
    ) -> Tuple[Layer]:
        """Return input layers for training with the same input sequence.

        Parameters
        ----------------------------
        true_input_layer: Layer,
            The input layer that will contain the true input.
        true_output_layer: Layer,
            The input layer that will contain the true output.

        Returns
        ----------------------------
        Return tuple with the tuple of layers.
        """
        return (
            true_output_layer,
            true_input_layer
        )
