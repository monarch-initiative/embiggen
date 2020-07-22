"""Abstract class for graph embedding models."""
from typing import Union, Tuple

from tensorflow.keras import backend as K   # pylint: disable=import-error
from tensorflow.keras.layers import Embedding, Input, Lambda, Layer, Flatten   # pylint: disable=import-error
from tensorflow.keras.models import Model   # pylint: disable=import-error
from tensorflow.keras.optimizers import Optimizer   # pylint: disable=import-error

from .embedder import Embedder
from .layers import NoiseContrastiveEstimation


class Node2Vec(Embedder):
    """Abstract class for graph embedding models."""

    def __init__(
        self,
        vocabulary_size: int,
        embedding_size: int,
        model_name: str,
        optimizer: Union[str, Optimizer] = "nadam",
        window_size: int = 4,
        negatives_samples: int = 10
    ):
        """Create new Graph Embedder model.

        Parameters
        -------------------------------------------
        vocabulary_size: int,
            Number of terms to embed.
            In a graph this is the number of nodes, while in a text is the
            number of the unique words.
        embedding_size: int,
            Dimension of the embedding.
        model_name: str,
            Name of the model.
        optimizer: Union[str, Optimizer] = "nadam",
            The optimizer to be used during the training of the model.
        negative_samples: int,
            The number of negative classes to randomly sample per batch.
            This single sample of negative classes is evaluated for each element in the batch.
        """
        self._model_name = model_name
        self._window_size = window_size
        self._negatives_samples = negatives_samples
        super().__init__(
            vocabulary_size=vocabulary_size,
            embedding_size=embedding_size,
            optimizer=optimizer
        )

    def _get_true_input_length(self) -> int:
        """Return length of true input layer."""
        raise NotImplementedError((
            "The method '_get_true_input_length' "
            "must be implemented in child class."
        ))

    def _get_true_output_length(self) -> int:
        """Return length of true output layer."""
        raise NotImplementedError((
            "The method '_get_true_output_length' "
            "must be implemented in child class."
        ))

    def _sort_input_layers(
        self,
        true_input_layer: Layer,
        true_output_layer: Layer
    ) -> Tuple[Layer, Layer]:
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
        raise NotImplementedError((
            "The method '_sort_input_layers' "
            "must be implemented in child class."
        ))

    def _build_model(self):
        """Return Node2Vec model."""
        # Creating the inputs layers
        true_input_layer = Input(
            (self._get_true_input_length(), ),
            name=Embedder.EMBEDDING_LAYER_NAME
        )
        true_output_layer = Input(
            (self._get_true_output_length(), ),
        )

        # Creating the embedding layer for the contexts
        embedding = Embedding(
            input_dim=self._vocabulary_size,
            output_dim=self._embedding_size,
            input_length=self._get_true_input_length()
        )(true_input_layer)

        # If there is more than one value per single sample
        if self._get_true_input_length() > 1:
            # Computing mean of the embedding of all the contexts
            mean_embedding = Lambda(
                lambda x: K.mean(x, axis=1),
                output_shape=(self._embedding_size,)
            )(embedding)
        else:
            mean_embedding = Flatten()(embedding)

        # Adding layer that also executes the loss function
        nce_loss = NoiseContrastiveEstimation(
            vocabulary_size=self._vocabulary_size,
            embedding_size=self._embedding_size,
            negative_samples=self._negatives_samples,
            positive_samples=self._get_true_output_length()
        )((mean_embedding, true_output_layer))

        # Creating the actual model
        model = Model(
            inputs=self._sort_input_layers(
                true_input_layer,
                true_output_layer
            ),
            outputs=nce_loss,
            name=self._model_name
        )

        # No loss function is needed because it is already executed in
        # the NCE loss layer.
        model.compile(optimizer=self._optimizer)
        return model
