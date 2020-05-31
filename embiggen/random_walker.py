import numpy as np
from numba import njit
from multiprocessing import cpu_count, Pool
from typing import List, Tuple
from .graph import Graph
from tqdm.auto import tqdm, trange

import tensorflow as tf


class RandomWalker:

    def __init__(self, verbose: bool = True, workers: int = -1):
        """Create new RandomWalker object.

        Parameters
        -----------------------
        verbose: bool = True,
            Wethever to show or not the loading bar.
            By default True.
        workers: int = -1
            Number of processes to use. Use a number lower or equal to 0 to
            use all available processes. Default is -1.
        """
        self._verbose = verbose
        self._workers = workers if workers > 0 else cpu_count()

    def _graph_walk(self, walks: np.ndarray, graph: Graph, walk_length: int) -> List[np.array]:
        """Generate a random walk for each node in the graph.

        Parameters
        ----------
        graph:Graph
            The graph on which the random walks will be done.
        walk_length:int
            The length of the walks in edges traversed. The result will be
            a list of arrays of (walk_length + 1) nodes.

        Returns
        -------
        A list of arrays of (walk_length + 1) nodes.
        """
        for walk, start_node in zip(walks, graph.nodes_indices):
            walk[0] = prev = start_node
            walk[1] = curr = graph.extract_random_node_neighbour(start_node)
            i = hash_edges[prev, curr]
            for index in range(2, walk_length):
                walk[index] = tmp = graph.extract_random_edge_neighbour(
                    prev, curr
                )
                prev = curr
                curr = tmp

        for start_node in graph.nodes_indices:
            edge = fucntion(start_node)
            for index in range(walk_length):
                walk[indexe] = edge = graph.extract_from(edge)

    def walk(self,
             graph: Graph,
             walk_length: int,
             num_walks: int
             ) -> tf.RaggedTensor:
        """Return a list of graph walks

        Parameters
        ----------
        graph:Graph
            The graph on which the random walks will be done.
        walk_length:int
            The length of the walks in edges traversed.

        Returns
        -------
        A Ragged tensor of n graph walks with shape:
        (num_walks, graph.nodes_number, walk_length)
        """
        all_walks = np.empty((
            graph.nodes_number*num_walks,
            walk_length
        ), dtype=np.int64)

        for i in trange(num_walks, disable=not self._verbose):
            walks = all_walks[i*graph.nodes_number:(i+1)*graph.nodes_number]
            self._graph_walk(walks, graph, walk_length)
        return tf.constant(all_walks)