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
    def worddictionary(self) -> Dict[str, int]:
        """Return mapping for nodes to numeric node ID."""
        # TODO: this method is used too much internally within the w2v.
        # We should use the numeric indices internally.
        return self._graph._nodes_mapping

    @property
    def reverse_worddictionary(self) -> List[str]:
        """Return mapping for numeric nodes ID to nodes."""
        return self.nodes_indices

    # !TODO: Integrate this with graph!
    def DegreeProduct(self, node_1, node_2) -> float:
        ''' Function takes a CSF graph object and list of edges calculates the Degree Product or Preferential Attachment for these
        nodes given the structure of the graph.
        :param graph: Graph  object
        :param node_1: one node of graph
        :param node_2: one node of a graph
        :return: Degree Product score of the nodes
        '''

        return self._graph.degree(node_1) * self._graph.degree(node_2)

    # !TODO: Integrate this with graph!
    # !TODO: what exactly is a common neighbors score???
    def CommonNeighbors(self, node_1, node_2):
        ''' Function takes a CSF graph object and list of edges calculates the Common Neighbors for these nodes given the
        structure of the graph.
        :param node_1: one node of graph
        :param node_2: one node of a graph
        :return: Common Neighbors score of the nodes
        '''

        node_1_neighbors = set(self._graph.neighbors(node_1))
        node_2_neighbors = set(self._graph.neighbors(node_2))

        if len(node_1_neighbors) == 0 or len(node_2_neighbors) == 0:
            score = 0.0

        elif len(node_1_neighbors.intersection(node_2_neighbors)) == 0:
            score = 0.0

        else:
            n_intersection = node_1_neighbors.intersection(node_2_neighbors)
            score = float(len(n_intersection))

        return score

    # !TODO: Integrate this with graph!
    def Jaccard(self, node_1, node_2):
        ''' Function takes a CFS graph object and list of edges calculates the Jaccard for these nodes given the
        structure of the graph.
        :param graph: CFS graph object
        :param node_1: one node of graph
        :param node_2: one node of a graph
        :return: The Jaccard score of two nodes
        '''

        node_1_neighbors = set(self._graph.neighbors(node_1))
        node_2_neighbors = set(self._graph.neighbors(node_2))

        if len(node_1_neighbors) == 0 or len(node_2_neighbors) == 0:
            score = 0.0

        elif len(node_1_neighbors.intersection(node_2_neighbors)) == 0:
            score = 0.0

        else:
            n_intersection = set(node_1_neighbors.intersection(node_2_neighbors))
            n_union = set(node_1_neighbors.union(node_2_neighbors))
            score = float(len(n_intersection))/len(n_union)

        return score

    # !TODO: Integrate this with graph!
    def AdamicAdar(self, node_1, node_2):
        ''' Function takes a CSF graph object and list of edges calculates the Adamic Adar for the nodes given the
        structure of the graph.
        :param graph: CSF graph object
        :param node_1: a node of the graph
        :param node_2: one node of a graph
        :return: AdamicAdar score of the nodes
        '''

        node_1_neighbors = set(self._graph.neighbors(node_1))
        node_2_neighbors = set(self._graph.neighbors(node_2))

        if len(node_1_neighbors) == 0 or len(node_2_neighbors) == 0:
            score = 0.0

        elif len(node_1_neighbors.intersection(node_2_neighbors)) == 0:
            score = 0.0

        else:
            score = 0.0
            n_intersection = node_1_neighbors.intersection(node_2_neighbors)

            for c in n_intersection:
                score += 1/np.log(self._graph.degree(c))
        return score

    def degree(self, node: int) -> int:
        """Return degree of given node.

        Parameters
        ---------------------
        node: int,
            The node whose neigbours are to be identified.
