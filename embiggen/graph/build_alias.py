from .alias_method import alias_setup
from numba import typed, njit, prange  # type: ignore
import numpy as np  # type: ignore
from typing import List, Tuple, Set, Dict
from ..utils import numba_log
from .graph_types import (
    numpy_indices_type,
    numpy_probs_type,
    alias_list_type
)


@njit
def build_default_alias_vectors(
    number: int
) -> Tuple[List[int], List[float]]:
    """Return empty alias vectors to be populated in
        build_alias_nodes below

    Parameters
    -----------------------
    number: int,
        Number of aliases to setup for.

    Returns
    -----------------------
    Returns default alias vectors.
    """
    alias = typed.List.empty_list(alias_list_type)
    empty_j = np.empty(0, dtype=numpy_indices_type)
    empty_q = np.empty(0, dtype=numpy_probs_type)
    for _ in range(number):
        alias.append((empty_j, empty_q))
    return alias


@njit
def get_min_max_edge(neighbors: List[int], node: int) -> Tuple[int, int]:
    """Return tuple with minimum and maximum edge for given node.
    This method is used to retrieve the indices needed to extract the 
    neighbours from the destination array.

    Parameters
    ---------------------------
    neighbors: List[int],
        List of offsets neighbours for given node.
    node: int,
        The id of the node to access.

    Returns
    ----------------------------
    Tuple with minimum and maximum edge index for given node.
    """
    return 0 if node == 0 else neighbors[node-1], neighbors[node]


@njit
def get_node_transition_weights(
    node: int,
    weights: List[float],
    node_types: List[int],
    destinations: List[int],
    min_edge: int,
    max_edge: int,
    change_node_type_weight: float
) -> List[int]:
    """Return transition weights vector for current node.

    NB: the returned weights are NOT normalized.

    Parameters
    ----------------------------
    node: int,
        Node for which to compute the transition weights.
    weights: List[float],
        Weights of all the edges, sorted.
    node_types: List[int],
        Types of the nodes.
    destinations: List[int],
        List of the destinations or the given node.
    min_edge: int,
        Edge with minimum index in the order sorted by sources.
    max_edge: int,
        Edge with maximum index in the order sorted by sources.
    change_node_type_weight: float,
        Factor to use to compute the transition between different node types.

    Returns
    -----------------------------
    Vector of weights describing the transition of the node through each edge.
    """
    # If weights are given
    if weights.size:
        # We retrieve the weights relative to these transitions
        transition_weights = weights[min_edge:max_edge]
    else:
        # Otherwise we start with a vector of ones.
        transition_weights = np.ones(max_edge-min_edge, dtype=numpy_probs_type)

    # If the node types were given:
    if node_types.size:
        # if the destination node type matches the neighbour
        # destination node type (we are not changing the node type)
        # we weigth using the provided change_node_type_weight weight.
        mask = node_types[node] == node_types[destinations]
        transition_weights[mask] /= change_node_type_weight

    return transition_weights


@njit(parallel=True)
def build_alias_nodes(
    neighbors: List[int],
    weights: List[float],
    destinations: List[int],
    node_types: List[int],
    change_node_type_weight: float
) -> List[Tuple[List[int], List[float]]]:
    """Return aliases for nodes to use for alias method for 
       selecting from discrete distribution.

    Parameters
    -----------------------
    neighbors:  List[int],
        List of neighbouring edges represented as offsets.
    weights: List[float],
        List of weights for each edge.

    Returns
    -----------------------
    Lists of lists representing node aliases 
    """

    alias = build_default_alias_vectors(len(neighbors))

    for i in prange(len(neighbors)):  # pylint: disable=not-an-iterable
        node = np.int64(i)
        min_edge, max_edge = get_min_max_edge(neighbors, node)

        # Do not call the alias setup if the node is a trap.
        # Because that node will have no neighbors and thus the necessity
        # of setupping the alias method to efficently extract the neighbour.
        if min_edge == max_edge:
            continue

        neighboring_nodes = destinations[min_edge:max_edge],

        # Compute the probabilities relative to the node weights.
        probs = get_node_transition_weights(
            node,
            weights,
            node_types,
            neighboring_nodes,
            min_edge,
            max_edge,
            change_node_type_weight
        )

        alias[node] = alias_setup(probs/probs.sum())
    return alias


@njit(parallel=True)
def build_alias_edges(
    edges_set: Set[Tuple[int, int]],
    neighbors: List[List[int]],
    node_types: List[int],
    edge_types: List[int],
    weights: List[float],
    sources: List[int],
    destinations: List[int],
    traps: List[bool],
    return_weight: float,
    explore_weight: float,
    change_node_type_weight: float,
    change_edge_type_weight: float,
) -> List[Tuple[List[int], List[float]]]:
    """Return aliases for edges to use for alias method for 
       selecting from discrete distribution.

    Parameters
    -----------------------
    neighbors: List[List[int]],
        List of neighbouring edges for each node.
    node_types: List[int],
        List of node types.
    edge_types: List[int],
        List of edge types.
    weights: List[float],
        List of edge weights.
    sources: List[int],
        List of source nodes for each edge.
    destinations: List[int],
        List of destination nodes for each edge.
    traps: List[bool],
        List of trap nodes boolean mask.
    return_weight: float,
        hyperparameter for breadth first random walk - 1/p
    explore_weight: float,
        hyperparameter for depth first random walk - 1/q
    change_node_type_weight: float,
        hyperparameter for changing node type during random walk
    change_edge_type_weight: float,
        hyperparameter for changing edge type during random walk

    Returns
    -----------------------
    Lists of lists representing edges aliases.
    """

    alias = build_default_alias_vectors(len(sources))

    for i in prange(len(sources)):  # pylint: disable=not-an-iterable
        k = np.int64(i)
        src = sources[k]
        dst = destinations[k]
        min_edge, max_edge = get_min_max_edge(neighbors, dst)

        total_neighbors = max_edge - min_edge

        # Do not call the alias setup if the edge is a traps.
        # Because that edge will have no neighbors and thus the necessity
        # of setupping the alias method to efficently extract the neighbour.
        if total_neighbors == 0:
            continue

        if len(weights) == 0:
            # If the weights are uniform we can just assign a numpy array
            # filled up of ones.
            probs = np.ones(max_edge-min_edge, dtype=numpy_probs_type)
        else:
            # We get the weight for the edge from the destination to
            # the neighbour.
            probs = weights[min_edge:max_edge]

        neighbors_destinations = destinations[min_edge:max_edge]

        if len(node_types):
            # if the destination node type matches the neighbour
            # destination node type (we are not changing the node type)
            # we weigth using the provided change_node_type_weight weight.
            mask = node_types[dst] == node_types[neighbors_destinations]
            probs[mask] /= change_node_type_weight

        if len(edge_types):
            # Similarly if the neighbour edge type matches the previous
            # edge type (we are not changing the edge type)
            # we weigth using the provided change_edge_type_weight weight.
            mask = edge_types[k] == edge_types[min_edge:max_edge]
            probs[mask] /= change_edge_type_weight

        # If the neigbour matches with the source, hence this is
        # a backward loop like the following:
        # SRC -> DST
        #  ▲     /
        #   \___/
        #
        # We weight the edge weight with the given return weight.
        is_looping_back = neighbors_destinations == src
        probs[is_looping_back] *= return_weight

        for index in prange(total_neighbors):  # pylint: disable=not-an-iterable
            # If the edge between the neighbour and the original edge is
            # looping backward we continue.
            if is_looping_back[index]:
                continue
            # If the backward loop does not exist, we multiply the weight
            # of the edge by the weight for moving forward and explore more.
            ndst = neighbors_destinations[index]
            if traps[ndst] or (ndst, src) not in edges_set:
                probs[index] *= explore_weight

        # Finally we assign the obtained alias method probabilities.
        alias[k] = alias_setup(probs/probs.sum())
    return alias
