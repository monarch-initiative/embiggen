"""CBOW model for sequence embedding."""
from typing import Dict, Union
from tensorflow.keras.layers import (   # pylint: disable=import-error,no-name-in-module
    GlobalAveragePooling1D, Input, Embedding
)
import tensorflow as tf
from tensorflow.keras.models import Model  # pylint: disable=import-error,no-name-in-module

from .embedder import Embedder
from .layers import SampledSoftmax
from ..utils import validate_window_size


class CBOW(Embedder):
    """CBOW model for sequence embedding.

    The CBOW model for graoh embedding receives a list of contexts and tries
    to predict the central word. The model makes use of an NCE loss layer
    during the training process to generate the negatives.
    """

    def __init__(
        self,
        window_size: int = 4,
        negative_samples: int = 10,
        use_gradient_centralization: bool = True,
        **kwargs: Dict
    ):
        """Create new sequence Embedder model.

        Parameters
        -------------------------------------------
        window_size: int = 4,
            Window size for the local context.
            On the borders the window size is trimmed.
        negative_samples: int = 10,
            The number of negative classes to randomly sample per batch.
            This single sample of negative classes is evaluated for each element in the batch.
        use_gradient_centralization: bool = True,
            Whether to wrap the provided optimizer into a normalized
            one that centralizes the gradient.
            It is automatically enabled if the current version of
            TensorFlow supports gradient transformers.
            More detail here: https://arxiv.org/pdf/2004.01461.pdf
        **kwargs: Dict,
            Additional kwargs to pass to parent constructor.
        """
        # TODO! Figure out a way to test for Zifian distribution in the
        # data used for the word2vec sampling! The values in the vocabulary
        # should have a decreasing node degree order!
        self._window_size = validate_window_size(window_size)
        self._negative_samples = negative_samples
        super().__init__(
            use_gradient_centralization=use_gradient_centralization,
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
        contextual_terms_embedding = Embedding(
            input_dim=self._vocabulary_size,
            output_dim=self._embedding_size,
            input_length=self._window_size*2,
            name=Embedder.TERMS_EMBEDDING_LAYER_NAME,
        )(contextual_terms_input)

        contextual_embedding = GlobalAveragePooling1D()(
            contextual_terms_embedding
        )

        # Adding layer that also executes the loss function
        sampled_softmax = SampledSoftmax(
            vocabulary_size=self._vocabulary_size,
            embedding_size=self._embedding_size,
            negative_samples=self._negative_samples,
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
