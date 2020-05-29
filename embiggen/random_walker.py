import numpy as np
from numba import njit
from multiprocessing import cpu_count, Pool
from typing import List
from .graph import ProbabilisticGraph
from tqdm.auto import tqdm

import tensorflow as tf


class RandomWalker:

    def __init__(self, workers: int = -1):
        """Create new RandomWalker object.

        Parameters
        -----------------------
        workers: int = -1
            Number of processes to use. Use a number lower or equal to 0 to
            use all available processes. Default is -1.
        """
        self._workers = workers if workers <= 0 else cpu_count()

    @njit
    def _walk(self,
              graph: ProbabilisticGraph,
              walk_length: int,
              start_node: int
              ) -> np.array:
        """Do a random walk of length walk_length starting from start_node.

        Parameters
        ----------
        graph:ProbabilisticGraph
            The graph on which the random walk will be done.
        walk_length:int
            The length of the walk in edges traversed. The result will be
            an array of (walk_length + 1) nodes.
        start_node:int
            From which node the random walk will start.

        Returns
        -------
        An array of (walk_length + 1) nodes.
        """
        walk = np.zeros(walk_length + 1)
        for index in range(walk_length):
            walk[index] = start_node
            start_node = graph.extract_random_neighbour(start_node)
        walk[index + 1] = start_node
        return walk

    @njit
    def _graph_walk(self,
                    graph: ProbabilisticGraph,
                    walk_length: int
                    ) -> List[np.array]:
        """Generate a random walk for each node in the graph.

        Parameters
        ----------
        graph:ProbabilisticGraph
            The graph on which the random walks will be done.
        walk_length:int
            The length of the walks in edges traversed. The result will be
            a list of arrays of (walk_length + 1) nodes.

        Returns
        -------
        A list of arrays of (walk_length + 1) nodes.
        """
        return [
            self._walk(graph, walk_length, node)
            for node in graph.nodes_indices
        ]

    def walk(self,
             graph: ProbabilisticGraph,
             walk_length:
             int, num_walks: int
             ) -> tf.RaggedTensor:
        """Return a list of graph walks

        Parameters
        ----------
        graph:ProbabilisticGraph
            The graph on which the random walks will be done.
        walk_length:int
            The length of the walks in edges traversed.

        Returns
        -------
        A Ragged tensor of n graph walks with shape:
        (num_walks, graph.nodes_number, walk_length)
        """
        # TODO There is no need for the tensor to be rugged.
        with Pool(min(self._workers, num_walks)) as pool:
            walks_tensor = tf.ragged.constant(np.fromiter(tqdm(
                pool.imap_unordered(
                    self._graph_walk,
                    (
                        (graph, walk_length)
                        for _ in range(num_walks)
                    )
                ),
                total=num_walks,
                desc='Performing walks'
            )))
            pool.close()
            pool.join()
        return walks_tensor
