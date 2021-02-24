"""GraphCBOW model for graph and words embedding."""
from typing import Union
from tensorflow.keras.optimizers import Optimizer   # pylint: disable=import-error
from ensmallen_graph import EnsmallenGraph
from .cbow import CBOW


class GraphCBOW(CBOW):
    """GraphCBOW model for graph and words embedding.

    The GraphCBOW model for graoh embedding receives a list of contexts and tries
    to predict the central word. The model makes use of an NCE loss layer
    during the training process to generate the negatives.
    """

    def __init__(
        self,
        graph: EnsmallenGraph,
        embedding_size: int,
        optimizer: Union[str, Optimizer] = "nadam",
        window_size: int = 4,
        negative_samples: int = 10
    ):
        """Create new GraphCBOW-based Embedder object.

        Parameters
        -------------------------------------------
        graph: EnsmallenGraph,
            Graph to be embedded.
        embedding_size: int,
            Dimension of the embedding.
        optimizer: Union[str, Optimizer] = "nadam",
            The optimizer to be used during the training of the model.
        window_size: int = 4,
            Window size for the local context.
            On the borders the window size is trimmed.
        negative_samples: int,
            The number of negative classes to randomly sample per batch.
            This single sample of negative classes is evaluated for each element in the batch.
        """
        super().__init__(
            vocabulary_size=graph.get_nodes_number(),
            embedding_size=embedding_size,
            model_name="GraphCBOW",
            optimizer=optimizer,
            window_size=window_size,
            negative_samples=negative_samples
        )

    