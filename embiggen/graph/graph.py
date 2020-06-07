from typing import List, Dict
import gc
import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
from dict_hash import sha256, Hashable  # type: ignore
from .numba_graph import NumbaGraph
from .random_walks import (
    random_walk, random_walk_with_traps, lazy_random_walk_with_traps)
from ..utils import logger


class Graph(Hashable):
    def __init__(self, **kwargs):
        """Create new instance of Graph."""
        # self._consistent_hash = sha256({
        #     "directed": directed,
        #     **kwargs
        # })
        self._graph = NumbaGraph(**kwargs)

    def random_walk(self, number: int, length: int) -> tf.Tensor:
        """Return a list of graph walks

        Parameters
        ----------
        number: int,
            Number of walks to execute.
        length:int,
            The length of the walks in edges traversed.

        Returns
        -------
        Tensor with walks containing the numeric IDs of nodes.
        """
        if not self._graph.preprocessed:
            raise ValueError(
                "Given graph was not properly setup for random walk."
            )
        logger.info("Starting random walks.")
        if self._graph.has_traps:
            logger.info("Using trap-aware algorithm for random walks.")
            walks = random_walk_with_traps(self._graph, number, length)
            logger.info("Building RaggedTensor from walks.")
            return tf.ragged.constant(walks)
        logger.info("Using trap-unaware algorithm fo random walks.")
        walks = random_walk(self._graph, number, length)
        logger.info("Building Tensor from walks.")
        return tf.constant(walks)

    def lazy_random_walk(self, number: int, length: int) -> tf.Tensor:
        """Return a list of graph walks with lazy evaluation.

        Parameters
        ----------
        number: int,
            Number of walks to execute.
        length:int,
            The length of the walks in edges traversed.

        Returns
        -------
        Tensor with walks containing the numeric IDs of nodes.
        """
        logger.info("Starting lazy random walks.")
        walks = lazy_random_walk_with_traps(self._graph, number, length)
        logger.info("Building Tensor from walks.")
        return tf.ragged.constant(walks)

    def build_graph_alias(self):
        """Create objects necessary for quick random search over graph.

        Note that these objects can get VERY big, for example in a graph with
        15 million edges they get up to around 80GBs.

        Consider using the lazy random walk that renders the probabilities as
        the the walk proceeds as an alternative solution when the graph gets
        too big for this quick walk.

        After the walk is executed, to keep the graph object but destroy these
        big objects, call the method graph.destroy_graph_alias().
        """
        self._graph.build_graph_alias()

    def destroy_graph_alias(self):
        """Destroys object related to graph alias."""
        self._graph.destroy_graph_alias()
        gc.collect()

    @property
    def sources(self) -> np.ndarray:
        """Return sources indices.

        Returns
        ---------------------
        Numpy array of sources indices.
        """
        return self._graph.sources

    @property
    def destinations(self) -> np.ndarray:
        """Return destinations indices.

        Returns
        ---------------------
        Numpy array of destinations indices.
        """
        return self._graph.destinations

    # def consistent_hash(self) -> str:
    #     """Return hash for the current instance of the graph."""
    #     return self._consistent_hash
