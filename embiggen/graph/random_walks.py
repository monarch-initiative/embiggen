from numba import njit, prange  # type: ignore
from typing import List
import numpy as np  # type: ignore
from .numba_graph import NumbaGraph

# This function is out of the class because otherwise we would not be able
# to activate the parallel=True flag.
@njit(parallel=True)
def random_walk(graph: NumbaGraph, number: int, length: int) -> np.ndarray:
    """Return a list of graph walks

    Parameters
    ----------
    graph: NumbaGraph,
        The graph on which the random walks will be done.
    number: int,
        Number of walks to execute.
    length:int,
        The length of the walks in edges traversed.

    Returns
    -------
    Numpy array with all the walks containing the numeric IDs of nodes.
    """
    # or alternatively a numpy array.
    all_walks = np.empty((graph.nodes_number, number, length), dtype=np.int64)

    # We can use prange to parallelize the walks and the iterations on the
    # graph nodes.
    for src in prange(graph.nodes_number):  # pylint: disable=not-an-iterable
        for i in prange(number):  # pylint: disable=not-an-iterable
            walk = all_walks[src][i]
            walk[0] = src
            walk[1], edge = graph.extract_random_node_neighbour(walk[0])
            for index in range(2, length):
                edge = graph.extract_random_edge_neighbour(edge)
                walk[index] = graph.get_edge_destination(edge)
    return all_walks


# This function is out of the class because otherwise we would not be able
# to activate the parallel=True flag.
# In this random walk we take into account the possibility of encountering
# traps within the execution of the code.
@njit(parallel=True)
def random_walk_with_traps(graph: NumbaGraph, number: int, length: int) -> List[List[List[int]]]:
    """Return a list of graph walks

    Parameters
    ----------
    graph: NumbaGraph,
        The graph on which the random walks will be done.
    number: int,
        Number of walks to execute.
    length:int,
        The length of the walks in edges traversed.

    Returns
    -------
    List of list of walks containing the numeric IDs of nodes.
    """
    nodes_number = graph.nodes_number
    all_walks = [[[0] for _ in range(number)] for _ in range(nodes_number)]

    # We can use prange to parallelize the walks and the iterations on the
    # graph nodes.
    for src in prange(nodes_number):  # pylint: disable=not-an-iterable
        for i in prange(number):  # pylint: disable=not-an-iterable
            walk = all_walks[src][i]
            walk[0] = src
            # Check if the current node has neighbors
            if graph.is_node_trap(walk[0]):
                # If the node has no neighbors and is therefore a trap,
                # we need to interrupt the walk as we cannot proceed further.
                continue
            dst, edge = graph.extract_random_node_neighbour(walk[0])
            walk.append(dst)
            for _ in range(2, length):
                # If the previous destination was a trap, we need to stop the
                # loop.
                if graph.is_edge_trap(edge):
                    break
                edge = graph.extract_random_edge_neighbour(edge)
                walk.append(graph.get_edge_destination(edge))
    return all_walks
