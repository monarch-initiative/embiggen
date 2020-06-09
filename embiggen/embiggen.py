from .graph import Graph
import tensorflow as tf  # type: ignore
from embiggen import SkipGram, Cbow, GloVeModel, CooccurrenceEncoder
from .transformers import GraphPartitionTransfomer
from typing import Tuple
import numpy as np  # type: ignore


class Embiggen:

    def __init__(self, embedding_method: str = "skipgram", edge_creation="hadamard"):
        """Returns new instance of Embiggen.

        Parameters
        ----------------------
        method: str = "skipgram",
            Method to use to transform the nodes embedding to edges.

        Raises
        ----------------------
        ValueError,
            If the given embedding method is not supported.
        """
        # TODO: a very long docstring showing the possible usages of Embiggen.
        self._model = None  # TODO! move the constructor of the model here!
        self._transformer = GraphPartitionTransfomer(method=edge_creation)

    def _get_embedding_model(
        self,
        graph: Graph,
        walks: tf.Tensor,
        embedding_model: str,
        epochs: int,
        embedding_size: int,
        context_window: int,
        window_size: int
    ):
        # TODO: add notes for the various parameters relative to which parameters
        # and add exceptions relative to the invalid ranges for the specific
        # parameters.
        if embedding_model == "skipgram":
            return SkipGram(
                data=walks,
                worddictionary=graph.worddictionary,
                reverse_worddictionary=graph.reverse_worddictionary,
                num_epochs=epochs
            )
        if embedding_model == "cbow":
            return Cbow(
                walks,
                worddictionary=graph.worddictionary,
                reverse_worddictionary=graph.reverse_worddictionary,
                num_epochs=epochs
            )
        cencoder = CooccurrenceEncoder(
            walks,
            window_size=window_size,
            vocab_size=graph.nodes_number
        )
        return GloVeModel(
            co_oc_dict=cencoder.build_dataset(),
            vocab_size=graph.nodes_number,
            embedding_size=embedding_size,
            context_size=context_window,
            num_epochs=epochs
        )

    def fit(
        self,
        data: Union[tf.Tensor, tf.RaggedTensor],
        walks_number: int = 100,
        walks_length: int = 100,
        embedding_model: str = "cbow",
        epochs: int = 10,
        embedding_size: int = 200,
        context_window: int = 3,
        window_size: int = 2
    ):
        """Fit model using input data (Tensors dervied from a Graph or a text).

        #TODO: add notes for the various parameters relative to which parameters
        # are best in which range! Add links to papers if any are relevant!

        Parameters
        -----------------

        Raises
        -----------------
        ValueError,
            If given embedding model must be 'cbow', 'skipgram' or 'glove'.

        """

        if embedding_model not in ("skipgram", "cbow", "glove"):
            raise ValueError(
                "Given embedding model must be 'cbow', 'skipgram' or 'glove'")

        walks = graph.random_walk(number=walks_number, length=walks_length)

        # TODO! move this constructor to the constructor of the class.
        self._model = self._get_embedding_model(
            graph, walks,
            embedding_model=embedding_model,
            epochs=epochs,
            embedding_size=embedding_size,
            context_window=context_window,
            window_size=window_size
        )

        # TODO! this train method must receive the arguments that we don't need
        # to pass to the constructor of the model.
        self._model.train()

        self._transformer.fit(self.embedding)

    def transform(self, positives: Graph, negatives: Graph) -> Tuple[np.ndarray, np.ndarray]:
        """Return tuple of embedded positives and negatives graph partitions.

        Parameters
        ----------------------
        positive: Graph,
            The positive partition of the Graph.
        negative: Graph,
            The negative partition of the Graph.

        Returns
        ----------------------
        Tuple of X and y to be used for training.
        """
        return self._transformer.transform(positives, negatives)

    def transform_nodes(self, positives: Graph, negatives: Graph) -> Tuple[
            np.ndarray, np.ndarray, np.ndarray]:
        """Return triple of embedded positives and negatives graph partitions.

        Parameters
        ----------------------
        positive: Graph,
            The positive partition of the Graph.
        negative: Graph,
            The negative partition of the Graph.

        Returns
        ----------------------
        Triple of X for the source nodes,
        X for the destination nodes and the labels to be used for training.
        """
        return self._transformer.transform_nodes(positives, negatives)

    @property
    def embedding(self) -> np.ndarray:
        """Return computed embedding.

        Raises
        ---------------------
        ValueError,
            If the model is not yet fitted.

        Returns
        ---------------------
        Computed embedding.
        """
        return self._model.embedding

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
        self._model.save_embedding(path)

    def load_embedding(self, path: str):
        """Load the embedding at the given path.

        Parameters
        -------------------
        path: str,
            Path from where to load the embedding.

        Raises
        ---------------------
        ValueError,
            If the give path does not exists.
        """
        self._model.load_embedding(path)
