from typing import List, Dict
import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
from dict_hash import sha256, Hashable  # type: ignore
from .numba_graph import NumbaGraph
from .random_walks import random_walk, random_walk_with_traps

#######################################
# Class to wrapp the NumbaGraph code  #
#######################################


class Graph(Hashable):
    def __init__(self, *args, **kwargs):
        """Create new instance of Graph."""
        self._graph = NumbaGraph(*args, **kwargs)

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
        if not self._graph.random_walk_preprocessing:
            raise ValueError(
                "Given graph was not properly setup for random walk.")

        if self._graph.has_traps:
            return tf.ragged.constant(random_walk_with_traps(self._graph, number, length))
        return tf.constant(random_walk(self._graph, number, length))

    def neighbors(self, node: int) -> List:
        """Return neighbors of given node.

        Parameters
        ---------------------
        node: int,
            The node whose neigbours are to be identified.

        Returns
        ---------------------
        List of neigbours of given node.
        """
        return self._graph.neighbors(node)

    @property
    def edges_indices(self) -> np.ndarray:
        """Return edges indices.

        Returns
        ---------------------
        Numpy array of edges indices.
        """
        return np.array(self._graph._edges_indices)

    @property
    def nodes_indices(self) -> np.ndarray:
        """Return edges indices.

        Returns
        ---------------------
        Numpy array of edges indices.
        """
        return np.array(self._graph._nodes_reverse_mapping)

    @property
    def sources_indices(self) -> np.ndarray:
        """Return sources of all edges.

        Returns
        ---------------------
        Numpy array of surces.
        """
        return self.edges_indices[:, 0]

    @property
    def destinations_indices(self) -> np.ndarray:
        """Return destinations of all edges.

        Returns
        ---------------------
        Numpy array of destinations.
        """
        return self.edges_indices[:, 1]

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
    def DegreeProduct(graph: Graph, node_1, node_2):
        ''' Function takes a CSF graph object and list of edges calculates the Degree Product or Preferential Attachment for these
        nodes given the structure of the graph.
        :param graph: Graph  object
        :param node_1: one node of graph
        :param node_2: one node of a graph
        :return: Degree Product score of the nodes
        '''

        return graph.degree(node_1) * graph.degree(node_2)

    # !TODO: Integrate this with graph!
    def CommonNeighbors(graph: Graph, node_1, node_2):
        ''' Function takes a CSF graph object and list of edges calculates the Common Neighbors for these nodes given the
        structure of the graph.
        :param graph: Graph object
        :param node_1: one node of graph
        :param node_2: one node of a graph
        :return: Common Neighbors score of the nodes
        '''

        node_1_neighbors = set(graph.neighbors(node_1))
        node_2_neighbors = set(graph.neighbors(node_2))

        if len(node_1_neighbors) == 0 or len(node_2_neighbors) == 0:
            score = 0.0

        elif len(node_1_neighbors.intersection(node_2_neighbors)) == 0:
            score = 0.0

        else:
            n_intersection = node_1_neighbors.intersection(node_2_neighbors)
            score = float(len(n_intersection))

        return score

    # !TODO: Integrate this with graph!
    def Jaccard(graph: Graph, node_1, node_2):
        ''' Function takes a CFS graph object and list of edges calculates the Jaccard for these nodes given the
        structure of the graph.
        :param graph: CFS graph object
        :param node_1: one node of graph
        :param node_2: one node of a graph
        :return: The Jaccad score of two nodes
        '''

        node_1_neighbors = set(graph.neighbors(node_1))
        node_2_neighbors = set(graph.neighbors(node_2))

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
    def AdamicAdar(graph: Graph, node_1, node_2):
        ''' Function takes a CSF graph object and list of edges calculates the Adamic Adar for the nodes given the
        structure of the graph.
        :param graph: CSF graph object
        :param node_1: a node of the graph
        :param node_2: one node of a graph
        :return: AdamicAdar score of the nodes
        '''

        node_1_neighbors = set(graph.neighbors(node_1))
        node_2_neighbors = set(graph.neighbors(node_2))

        if len(node_1_neighbors) == 0 or len(node_2_neighbors) == 0:
            score = 0.0

        elif len(node_1_neighbors.intersection(node_2_neighbors)) == 0:
            score = 0.0

        else:
            score = 0.0
            n_intersection = node_1_neighbors.intersection(node_2_neighbors)

            for c in n_intersection:
                score += 1/np.log(graph.degree(c))
        return score

    def degree(self, node: int) -> int:
        """Return degree of given node.

        Parameters
        ---------------------
        node: int,
            The node whose neigbours are to be identified.

        Returns
        ---------------------
        Number of neighbors of given node.
        """
        return self._graph.degree(node)

    def consistent_hash(self) -> str:
        """Return hash for the current instance of the graph."""
        return sha256({
            "random_walk_preprocessing": self._graph.random_walk_preprocessing,
            **(
                {
                    "edges_alias": self._graph._edges_alias,
                    "nodes_alias": self._graph._nodes_alias
                }
                if self._graph.random_walk_preprocessing else {}
            ),
            "edges": self.edges_indices,
            "nodes": self.nodes_indices
        })
