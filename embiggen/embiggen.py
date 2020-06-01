from .graph import Graph
import tensorflow as tf
from .word2vec import SkipGramWord2Vec, ContinuousBagOfWordsWord2Vec
from .glove import GloVeModel
from .coocurrence_encoder import CooccurrenceEncoder
from .graph_partition_transformer import GraphPartitionTransfomer
from typing import Dict, Tuple
import numpy as np


class Embiggen:

    def __init__(self):
        # TODO: a very long docstring showing the possible usages of Embiggen.
        self._model = None

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
            return SkipGramWord2Vec(
                data=walks,
                worddictionary=graph.worddictionary,
                reverse_worddictionary=graph.reverse_worddictionary,
                num_epochs=epochs
            )
        if embedding_model == "cbow":
            return ContinuousBagOfWordsWord2Vec(
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
        graph: Graph,
        walks_number: int = 100,
        walks_length: int = 100,
        embedding_model: str = "skipgram",
        epochs: int = 10,
        embedding_size: int = 200,
        context_window: int = 3,
        window_size: int = 2,
        edges_embedding_method: str = "hadamard"
    ):
        """Fit model using given graph.

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

        walks = graph.random_walks(number=walks_number, length=walks_length)

        self._model = self._get_embedding_model(
            graph, walks,
            embedding_model=embedding_model,
            epochs=epochs,
            embedding_size=embedding_size,
            context_window=context_window,
            window_size=window_size
        )

        self._model.train()

        self._transformer = GraphPartitionTransfomer(
            self.embedding,
            method=edges_embedding_method
        )

    def transform(self, positives: Graph, negatives: Graph) -> Tuple[np.ndarray, np.ndarray]:
        """Return transformed positives and negatives graph partitions.

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

    @property
    def embedding(self) -> Dict[str, np.ndarray]:
        """Return computed embedding.

        Raises
        ---------------------
        ValueError,
            If the model is not yet fitted.

        Returns
        ---------------------
        Computed embedding.
        """

        if self._model is None:
            raise ValueError("Model is not yet fitted!")

        return self._model.embedding

    def save_embedding(self, path: str):
        """Save the computed embedding to the given file.

        Raises
        ---------------------
        ValueError,
            If the model is not yet fitted.

        Parameters
        -------------------
        path: str,
            Path where to save the embedding.
        """

        if self._model is None:
            raise ValueError("Model is not yet fitted!")

        self._model.save(path)
