import numpy as np  # type: ignore
import pandas as pd  # type: ignore
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
        vocabulary_size: An integer storing the total number of unique words in the vocabulary.
        reverse_worddictionary: 
        display: An integer of the number of words to display.

        """
        self._embedding = None
        # ensure the following ops & var are assigned on CPU (some ops are not compatible on GPU)
        with tf.device('cpu'):
            self._embedding: tf.Variable = tf.Variable(
                tf.random.uniform(
                    [self.vocabulary_size, self.embedding_size], -1.0, 1.0, dtype=tf.float32)
            )
            # get weights and biases
            # construct the variables for the softmax loss
            tf_distribution = \
                tf.random.truncated_normal([self.vocabulary_size, self.embedding_size],
                                           stddev=0.5 / tf.math.sqrt(self.embedding_size),
                                           dtype=tf.float32)
            self._softmax_weights: tf.Variable = tf.Variable(tf_distribution)
            self._softmax_biases = tf.Variable(
                tf.random.uniform([self.vocabulary_size], 0.0, 0.01))


    def fit(
            self,
            X: Union[tf.Tensor, tf.RaggedTensor],
            learning_rate: float = 0.05,
            batch_size: int = 128,
            num_epochs: int = 1,
            embedding_size: int = 128,
            context_window: int = 3,
            number_negative_samples: int = 7,
            callbacks: Tuple["Callback"] = ()

            # !TODO! Add callbacks for displaying how the learning is going.
    ):
        """Fit the Word2Vec model.
        
        Parameters
        ---------------------
        X: Union[List, tf.Tensor, tf.RaggedTensor],
            
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

        raise NotImplementedError(
            "The fit method must be implemented in the child classes of Word2Vec."
        )
