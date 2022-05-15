"""TransE model."""
from ensmallen import Graph
from tensorflow.keras.optimizers import Optimizer  # pylint: disable=import-error,no-name-in-module

from .siamese import Siamese


class TransETensorFlow(Siamese):
    """TransE model."""

    def _build_output(
        self,
        *args
    ):
        """Returns the five input tensors, unchanged."""
        return args
