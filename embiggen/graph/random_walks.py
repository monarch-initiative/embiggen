from numba import njit, prange  # type: ignore
from typing import List, Callable
import numpy as np  # type: ignore
from .numba_graph import NumbaGraph
from .graph_types import numpy_nodes_type
from ..utils import numba_log


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
    all_walks = np.empty(
        (graph.nodes_number * number, length),
        dtype=numpy_nodes_type
    )

    # Cache the functions to call to have a minor speed-up
    extract_node_neighbour = graph.extract_node_neighbour
    extract_edge_neighbour = graph.extract_edge_neighbour

    # We can use prange to parallelize the walks and the iterations on the
    # graph nodes.
    for i in prange(number):  # pylint: disable=not-an-iterable
        for src in prange(graph.nodes_number):  # pylint: disable=not-an-iterable
            walk = all_walks[i*graph.nodes_number + src]
            walk[0] = src
            walk[1], edge = extract_node_neighbour(walk[0])
            for index in range(2, length):
                walk[index], edge = extract_edge_neighbour(edge)
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
    all_walks = [[0] for _ in range(number*nodes_number)]

    # Cache the functions to call to have a minor speed-up
    is_node_trap = graph.is_node_trap
    extract_node_neighbour = graph.extract_node_neighbour
    extract_edge_neighbour = graph.extract_edge_neighbour

    # We can use prange to parallelize the walks and the iterations on the
    # graph nodes.
    for i in prange(number):  # pylint: disable=not-an-iterable
        for src in prange(nodes_number):  # pylint: disable=not-an-iterable
            walk = all_walks[i*nodes_number + src]
            walk[0] = src
            # Check if the current node has neighbors
            if is_node_trap(walk[0]):
                # If the node has no neighbors and is therefore a trap,
                # we need to interrupt the walk as we cannot proceed further.
                continue
            dst, edge = extract_node_neighbour(walk[0])
            walk.append(dst)
            for _ in range(2, length):
                # If the previous destination was a trap, we need to stop the
                # loop.
                if is_node_trap(dst):
                    break
                dst, edge = extract_edge_neighbour(edge)
                walk.append(dst)
    return all_walks
