from typing import List, Dict
import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
from dict_hash import sha256, Hashable  # type: ignore
from .undirected_graph import DirectedGraph, UndirectedGraph
from .random_walks import random_walk, random_walk_with_traps
from ..utils import logger


class Graph(Hashable):
    def __init__(self, directed: bool = True, **kwargs):
        """Create new instance of Graph."""
        # self._consistent_hash = sha256({
        #     "directed": directed,
        #     **kwargs
        # })
        if directed:
            logger.info("Building directed graph")
            self._graph = DirectedGraph(**kwargs)
        else:
            logger.info("Building undirected graph graph")
            self._graph = UndirectedGraph(**kwargs)

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

        if self._graph.has_traps:
            return tf.ragged.constant(
                random_walk_with_traps(self._graph.core, number, length)
            )
        return tf.constant(
            random_walk(self._graph.core, number, length)
        )

    @property
    def preprocessed(self) -> bool:
        """Return boolean representing if the graph was preprocessed."""
        return self._graph.preprocessed

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
