"""CBOW model for sequence embedding."""
from typing import Dict, Union
from tensorflow.keras.layers import (   # pylint: disable=import-error,no-name-in-module
    GlobalAveragePooling1D, Input, Embedding
)
import tensorflow as tf
from tensorflow.keras.models import Model  # pylint: disable=import-error,no-name-in-module

from .tensorflow_embedder import TensorFlowEmbedder
from .layers import SampledSoftmax
from ...utils import validate_window_size


class CBOW(TensorFlowEmbedder):
    """CBOW model for sequence embedding.

    The CBOW model for graoh embedding receives a list of contexts and tries
    to predict the central word. The model makes use of an NCE loss layer
    during the training process to generate the negatives.
    """

    def __init__(
        self,
        window_size: int = 4,
        number_of_negative_samples: int = 5,
        **kwargs: Dict
    ):
        """Create new sequence TensorFlowEmbedder model.

        Parameters
        -------------------------------------------
        window_size: int = 4
            Window size for the local context.
            On the borders the window size is trimmed.
        number_of_negative_samples: int = 5
            The number of negative classes to randomly sample per batch.
            This single sample of negative classes is evaluated for each element in the batch.
        **kwargs: Dict
            Additional kwargs to pass to parent constructor.
        """
        self._number_of_negative_samples = number_of_negative_samples
        super().__init__(
            **kwargs
        )

    def _build_model(self) -> Model:
        """Return CBOW model."""
        # Creating the inputs layers

        # Create first the input with the central terms
        central_terms_input = Input(
            (1, ),
            dtype=tf.int32,
            name="CentralTermsInput"
        )

        # Then we create the input of the contextual terms
        contextual_terms_input = Input(
            (self._window_size*2, ),
            dtype=tf.int32,
            name="ContextualTermsInput"
        )

        # Creating the embedding layer for the contexts
        contextual_terms_embedding_layer = Embedding(
            input_dim=self._vocabulary_size,
            output_dim=self._embedding_size,
            input_length=self._window_size*2,
            name=TensorFlowEmbedder.TERMS_EMBEDDING_LAYER_NAME,
        )

        # Query the embedding to get the embedding vector
        # of the contextual terms provided as input.
        contextual_terms_embedding = contextual_terms_embedding_layer(
            contextual_terms_input)

        # Compute the average of the context embedding
        contextual_embedding = GlobalAveragePooling1D()(
            contextual_terms_embedding
        )

        # Adding layer that also executes the loss function
        sampled_softmax = SampledSoftmax(
            vocabulary_size=self._vocabulary_size,
            embedding_size=self._embedding_size,
            number_of_negative_samples=self._number_of_negative_samples,
            embedding=contextual_terms_embedding_layer if self._siamese else None
        )((contextual_embedding, central_terms_input))

        # Creating the actual model
        model = Model(
            inputs=[contextual_terms_input, central_terms_input],
            outputs=sampled_softmax,
            name="CBOW"
        )
        return model

    def _compile_model(self) -> Model:
        """Compile model."""
        # No loss function is needed because it is already executed in
        # the Sampled Softmax loss layer.
        self._model.compile(
            optimizer=self._optimizer
        )
