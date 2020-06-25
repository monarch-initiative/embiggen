import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import math
import random
import tensorflow as tf  # type: ignore

from typing import Dict, List, Optional, Tuple, Union
import os

from ..embedder import Embedder


class Word2Vec(Embedder):
    """Superclass of all of the word2vec family algorithms."""

    def __init__(self) -> None:
        """Create a new instance of Word2Vec.

        Parameters
        --------------------
        !TODO Bring this up to date
        vocabulary_size: An integer storing the total number of unique words in the vocabulary.
        reverse_worddictionary: 
        display: An integer of the number of words to display.

        """
        super().__init__()
        self._embedding = None
        self._is_list_of_lists = None  # must be set in fit method
        self.context_window = None  # must be set in fit method
        self.number_negative_samples = None  # must be set in fit method

    def nce_loss(self, x_embed: tf.Tensor, y: np.ndarray) -> Union[float, int]:
        """Calculates the noise-contrastive estimation (NCE) training loss estimation for each batch.
        Args:
            x_embed: A Tensor with shape [batch_size, dim].
            y: An array containing the target classes with shape [batch_size, num_true].
        Returns:
            loss: The NCE losses.
        """
        y = tf.cast(y, tf.int64)

        return tf.reduce_mean(tf.nn.nce_loss(
            weights=self._nce_weights,
            biases=self._nce_biases,
            labels=y,
            inputs=x_embed,
            num_sampled=self.number_negative_samples,
            num_classes=self._vocabulary_size
        ))

    def fit(
        self,
        *args,
        learning_rate: float = 0.05,
        embedding_size: int = 200,
        number_negative_samples: int = 7,
        **kwargs
    ):
        """Fit the Word2Vec model.

        Parameters
        ---------------------
        data: Union[tf.Tensor, tf.RaggedTensor],

        learning_rate: float,
            A float between 0 and 1 that controls how fast the model learns to solve the problem (Default 0.05)
        batch_size: int,
            The size of each "batch" or slice of the data to sample when training the model.
        num_epochs: int,
            The number of epochs to run when training the model.
        embedding_size: int,
            Dimension of embedded vectors.
        max_vocabulary_size: int,
            Maximum number of words (i.e. total number of different words in the vocabulary).
            An integer storing the total number of unique words in the vocabulary.
        context_window: int,
            How many words to consider left and right.
        number_negative_samples: int,
            Number of negative examples to sample (default=7).

        """
        super().fit(
            *args,
            learning_rate=learning_rate,
            embedding_size=embedding_size,
            **kwargs
        )

        # ensure the following ops & var are assigned on CPU (some ops are not compatible on GPU)
        with tf.device('cpu'):
            self._embedding: tf.Variable = tf.Variable(
                tf.random.uniform(
                    [self._vocabulary_size, embedding_size], -1.0, 1.0, dtype=tf.float32)
            )
            # get weights and biases
            # construct the variables for the softmax loss
            tf_distribution = \
                tf.random.truncated_normal([self._vocabulary_size, embedding_size],
                                           mean=0.0,
                                           stddev=0.5 /
                                           math.sqrt(embedding_size),
                                           dtype=tf.float32)
            self._nce_weights: tf.Variable = tf.Variable(tf_distribution)
            self._nce_biases = tf.Variable(
                tf.random.uniform([self._vocabulary_size], 0.0, 0.01))

        self.optimizer = tf.keras.optimizers.SGD(learning_rate)

        # Note that the super method sets the vocabulary size
        if number_negative_samples > self._vocabulary_size:
            raise ValueError((
                "Given number_negative_samples {} is larger than the vocabulary size. "
                "Probably this is a toy example. Consider reducing "
                "number_negative_samples"
            ).format(number_negative_samples))
        else:
            self.number_negative_samples = number_negative_samples

         # This method should be callable by any class extending this one, but
        # it can't be called since this is an "abstract method"
        if self.__class__ == Word2Vec:
            raise NotImplementedError(
                "The fit method must be implemented in the child classes of Word2Vec."
            )

    @property
    def list_of_lists(self) -> bool:
        return self._is_list_of_lists
