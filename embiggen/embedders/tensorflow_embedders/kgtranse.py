"""KGTransE model."""
from ensmallen import Graph
from tensorflow.keras.optimizers import Optimizer  # pylint: disable=import-error,no-name-in-module

from .siamese import Siamese


class KGTransETensorFlow(Siamese):
    """KGTransE model."""

    def _build_output(
        self,
        *args
    ):
        """Returns the five input tensors, unchanged."""
        return args

    @staticmethod
    def requires_node_types() -> bool:
        return True

    @staticmethod
    def requires_edge_types() -> bool:
        return True