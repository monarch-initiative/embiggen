import numpy as np
from numba import njit, jitclass, types
from multiprocessing import cpu_count, Pool
from typing import List, Tuple
from .graph import Graph
from tqdm.auto import tqdm, trange

import tensorflow as tf


@jitclass([
    ('_verbose', types.boolean)
])
class RandomWalker:

    def __init__(self, verbose: bool = True):
        """Create new RandomWalker object.

        Parameters
        -----------------------
        verbose: bool = True,
            Wethever to show or not the loading bar.
            By default True.
        """
        self._verbose = verbose

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

        for i in range(num_walks):
            for start_node in range(graph.nodes_number):
                k = i*graph.nodes_number+start_node
                all_walks[k][0] = src = start_node
                all_walks[k][1] = dst = graph.extract_random_node_neighbour(start_node)
                edge = graph.get_edge_id(src, dst)
                for index in range(2, walk_length):
                    all_walks[k][index] = edge = graph.extract_random_edge_neighbour(edge)
        return all_walks
