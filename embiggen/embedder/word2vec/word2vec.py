import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import random
import tensorflow as tf  # type: ignore

from typing import Dict, List, Optional, Tuple, Union
import os
from .utils import get_embedding, calculate_cosine_similarity
import logging


class Word2Vec:
    """Superclass of all of the word2vec family algorithms."""

    def __init__(self) -> None:
        """Create a new instance of Word2Vec.

        Parameters
        --------------------
        vocabulary_size: An integer storing the total number of unique words in the vocabulary.
        reverse_worddictionary: 
        display: An integer of the number of words to display.

        """
        self.eval_step = 100
        self._embedding = None
        self.context_window = context_window
        self.samples_per_window = samples_per_window
        self.number_negative_samples = number_negative_samples
        self.display = display
        self.display_examples: List = []
        self.device_type = '/CPU:0' if 'cpu' in device_type.lower() else '/GPU:0'


    def fit(
        self,
        X: Union[tf.Tensor, tf.RaggedTensor],
        learning_rate: float,
        batch_size: int,
        num_epochs: int,
        embedding_size: int,
        vocabulary_size: int = 50000,
        context_window: int = 3,
        number_negative_samples: int = 7,
        # !TODO! Add callbacks for displaying how the learning is going.
    ):
        """Fit the Word2Vec model.
        
        Parameters
        ---------------------
        X: Union[List, tf.Tensor, tf.RaggedTensor],
            
        learning_rate: float,
            A float between 0 and 1 that controls how fast the model learns to solve the problem.
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

    def add_display_words(self, count: list, num: int = 5) -> None:
        """Creates a list of display nodes/words by obtaining a random sample of 'num' nodes/words from the full
        sample.

        If the argument 'display' is not None, then we expect that the user has passed an integer amount of words
        that are to be displayed together with their nearest neighbors, as defined by the word2vec algorithm. It is
        important to note that display is a costly operation. Up to 16 nodes/words are permitted. If a user asks for
        more than 16, a random validation set of 'num' nodes/words, that includes common and uncommon nodes/words, is
        selected from a 'valid_window' of 50 nodes/words.
        Args:
            count: A list of tuples (key:word, value:int).
            num: An integer representing the number of words to sample.
        Returns:
            None.
        Raises:
            TypeError: If the user does not provide a list of display words.
        """

        if not isinstance(count, list):
            self.display = None
            raise TypeError(
                'self.display requires a list of tuples with key:word, value:int (count)')

        if num > 16:
            logging.warning(
                'maximum of 16 display words allowed (you passed {num_words})'.format(num_words=num))
            num = 16

        # pick a random validation set of 'num' words to sample
        valid_window = 50
        valid_examples = np.array(random.sample(range(2, valid_window), num))

        # sample less common words - choose 'num' points randomly from the first 'valid_window' after element 1000
        self.display_examples = np.append(valid_examples, random.sample(
            range(1000, 1000 + valid_window), num), axis=0)

        return None

    def calculate_vocabulary_size(self) -> None:
        """Calculates the vocabulary size for the input data, which is a list of words (i.e. from a text),
        or list of lists (i.e. from a collection of sentences or random walks).
        The function checks that self.vocabulary size has not been set

        Returns:
            None.
        """
        if self.word2id is None and self.vocabulary_size == 0:
            self.vocabulary_size = 0
        elif self.vocabulary_size == 0:
            self.vocabulary_size = len(self.word2id)+1
        return None

    @property
    def embedding(self) -> np.ndarray:
        """Return the embedding obtained from the model.

        Raises
        ---------------------
        ValueError,
            If the model is not yet fitted.
        """
        if self._embedding is None:
            raise ValueError("Model is not yet fitted!")
        return self._embedding.numpy()

    def save_embedding(self, path: str):
        """Save the computed embedding to the given file.

        Parameters
        -------------------
        path: str,
            Path where to save the embedding.

        Raises
        ---------------------
        ValueError,
            If the model is not yet fitted.
        """
        if self._embedding is None:
            raise ValueError("Model is not yet fitted!")
        pd.DataFrame({
            word: tf.nn.embedding_lookup(self._embedding, key).numpy()
            for key, word in enumerate(self.id2word)
        }).T.to_csv(path, header=False)

    def load_embedding(self, path: str):
        """Save the computed embedding to the given file.

        Raises
        ---------------------
        ValueError,
            If the give path does not exists.

        Parameters
        -------------------
        path: str,
            Path where to save the embedding.
        """
        if not os.path.exists(path):
            raise ValueError(
                "Embedding file at path {} does not exists.".format(path)
            )
        embedding = pd.read_csv(path, header=None, index_col=0)
        nodes_mapping = [
            self.word2id[node_name]
            for node_name in embedding.index.values.astype(str)
        ]
        self._embedding = embedding.values[nodes_mapping]
