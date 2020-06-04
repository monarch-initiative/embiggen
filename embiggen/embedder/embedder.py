import tensorflow as tf
from typing import Union


class Embedder:

    def __init__(self):
        """Abstract class for Embedder objects."""
        pass

    def fit(
        self,
        X: Union[tf.Tensor, tf.RaggedTensor],
        learning_rate: float = 0.05,
        batch_size: int = 128,
        epochs: int = 1,
        embedding_size: int = 128,
        context_window: int = 3,
        number_negative_samples: int = 7,
        # !TODO! Add callbacks for displaying how the learning is going.
    ):
        """Fit the Embedder model.
        
        Parameters
        ---------------------
        X: Union[tf.Tensor, tf.RaggedTensor],
            
        learning_rate: float,
            A float between 0 and 1 that controls how fast the model learns to solve the problem.
        batch_size: int,
            The size of each "batch" or slice of the data to sample when training the model.
        epochs: int,
            The number of epochs to run when training the model.
        embedding_size: int,
            Dimension of embedded vectors.
        context_window: int,
            How many words to consider left and right.
        number_negative_samples: int,
            Number of negative examples to sample (default=7).

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
            if len(sequence) < context_window:
                raise ValueError((
                    "Given context window {} is larger than at least one of "
                    "the tensors in X {}"
                ).format(context_window, len(sequence))

        # This method should be callable by any class extending this one, but
        # it can't be called since this is an "abstract method"
        if self.__class__ == Embedder:
            raise NotImplementedError(
                "The fit method must be implemented in the child classes of Word2Vec."
            )

    def transform(self):
        pass
