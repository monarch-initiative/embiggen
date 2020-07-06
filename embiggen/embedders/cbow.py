from .embedder import Embedder
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dot, Dense, Input, Flatten, Lambda
from tensorflow.keras.models import Model
from .layers import NoiseContrastiveEstimation

class CBOW(Embedder):

    def __init__(
        self,
        window_size: int = 4,
        negatives_samples:int=10,
        **kwargs
    ):
        """Create new CBOW-based Embedder object.
        
        Parameters
        -------------------------------------------
        negative_samples: int,
            The number of negative classes to randomly sample per batch.
            This single sample of negative classes is evaluated for each element in the batch.
        """
        self._window_size = window_size
        self._negatives_samples = negatives_samples
        super().__init__(**kwargs)

    def _build_model(self):
        """Return CBOW model.

        Parameters
        --------------------------
        vocabulary_size:int,
            Number of different elements of the dataset.
            In a text corpus this is the cardinality of the set of different words.
            In a graph this is the number of nodes within the graph.
        embedding_size:int,
            The size of the embedding to be obtained.
        window_size:int,
            The sliding window size.

        Returns
        --------------------------
        Sequential model of CBOW.
        """
        # Creating the inputs layers
        contexts_input = Input(
            (self._window_size*2, ),
            name=Embedder.EMBEDDING_LAYER_NAME
        )
        words_input = Input((self._window_size*2, ))

        # Creating the embedding layer for the contexts
        embedding = Embedding(
            input_dim=self._vocabulary_size,
            output_dim=self._embedding_size,
            input_length=self._window_size*2,
        )(contexts_input)

        # Computing mean of the embedding of all the contexts
        mean_embedding = Lambda(
            lambda x: K.mean(x, axis=1),
            output_shape=(self._embedding_size,)
        )(embedding)

        # Adding layer that also executes the loss function
        nce_loss = NoiseContrastiveEstimation(
            vocabulary_size=self._vocabulary_size,
            embedding_size=self._embedding_size,
            negative_samples=self._negatives_samples
        )((mean_embedding, words_input))

        # Creating the actual model
        cbow = Model(
            inputs=[contexts_input, words_input],
            outputs=nce_loss,
            name="CBOW"
        )

        # No loss function is needed because it is already executed in
        # the NCE loss layer.
        cbow.compile(optimizer=self._optimizer)
        return cbow
