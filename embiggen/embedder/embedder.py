import tensorflow as tf
from typing import Union, Tuple, Dict, List
import numpy as np
import pandas as pd


class Embedder:

    def __init__(self, word2id: Dict[str, int], id2word: List[str], devicetype: "cpu"):
        """Abstract class for Embedder objects.

        Parameters
        ----------------------
        word2id: Dict[str, int],
            Mapping between the word name and a numeric ID.
        id2word: List[str],
            Mapping between the numeric ID and word name

        Raises
        -----------------------
        ValueError,
            If the given word2id dictionary is empty.
        ValueError,
            If the given id2word list is empty.
        ValueError,
            If the given word mapping and and word reverse mapping have not the
            same length.
        
        """

        if not word2id:
            raise ValueError("Given word2id dictionary is empty.")
        if not id2word:
            raise ValueError("Given id2word list is empty.")
        if len(word2id) != len(id2word):
            raise ValueError(
                "Given word mapping and word reverse "
                "mapping have not the same length."
            )

        self._word2id = word2id
        self._id2word = id2word
        self._devicetype = devicetype
        # TODO! Understand why there is a +1 here.
        self._vocabulary_size = len(self._word2id)+1

    def fit(
        self,
        X: Union[tf.Tensor, tf.RaggedTensor],
        learning_rate: float = 0.05,
        batch_size: int = 128,
        epochs: int = 1,
        embedding_size: int = 128,
        context_window: int = 3,
        number_negative_samples: int = 7,
        callbacks: Tuple["Callback"] = ()
    ):
        """Fit the Embedder model.

        Parameters
        ---------------------
        X: Union[tf.Tensor, tf.RaggedTensor],

        learning_rate: float = 0.05,
            A float between 0 and 1 that controls how fast the model learns to solve the problem.
        batch_size: int = 128,
            The size of each "batch" or slice of the data to sample when training the model.
        epochs: int = 1,
            The number of epochs to run when training the model.
        embedding_size: int = 128,
            Dimension of embedded vectors.
        context_window: int = 3,
            How many words to consider left and right.
        number_negative_samples: int = 7,
            Number of negative examples to sample (default=7).
        callbacks: Tuple["Callback"] = (),
            List of callbacks to be called on epoch end and on batch end.

        Raises
        ----------------------
        ValueError,
            If given learning rate is not a strictly positive real number.
        ValueError,
            If given tensor is not 2-dimensional.
        ValueError,
            If given batch_size is not a strictly positive integer number.
        ValueError,
            If given epochs is not a strictly positive integer number.
        ValueError,
            If context_size is not a strictly positive integer

        """
        if len(X.shape) != 2:
            raise ValueError(
                "Given tensor is not 2-dimensional. "
                "If your tensor has only one dimension, "
                "you should use tensor.reshape(1, -1)."
            )

        if not isinstance(learning_rate, float) or learning_rate <= 0:
            raise ValueError((
                "Given learning_rate {} is not a "
                "strictly positive real number."
            ).format(learning_rate))

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError((
                "Given batch_size {} is not a "
                "strictly positive integer number."
            ).format(learning_rate))

        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError((
                "Given epochs {} is not a strictly positive real number."
            ).format(epochs))

        if not isinstance(context_window, int) or context_window < 1:
            raise ValueError((
                "Context window {} invalid. Must be a positive integer, "
                "usually in range (1,10)"
            ).format(context_window))

        if not isinstance(embedding_size, int) or embedding_size < 1:
            raise ValueError((
                "Given embedding_size {} is not an int or is less than 1"
            ).format(embedding_size))


        # Checking if the context window is valid for given tensor shape.
        for sequence in X:
            if context_window > len(sequence):
                raise ValueError((
                    "Given context window {} is larger than at least one of "
                    "the tensors in X {}"
                ).format(context_window, len(sequence)))

        # This method should be callable by any class extending this one, but
        # it can't be called since this is an "abstract method"
        if self.__class__ == Embedder:
            raise NotImplementedError(
                "The fit method must be implemented in the child classes of Embedder."
            )

    def transform(self):
        pass

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

    @property
    def devicetype(self) -> str:
        """Return the device type (cpu/gpu)
        return self._devicetype

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
            for key, word in enumerate(self._id2word)
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
