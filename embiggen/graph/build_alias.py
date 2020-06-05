from .alias_method import alias_setup
from numba import typed, njit, prange  # type: ignore
import numpy as np  # type: ignore
from typing import List, Tuple, Set
from .graph_types import (
    numpy_vector_alias_indices_type,
    numpy_vector_alias_probs_type,
    alias_list_type
)


@njit
def build_default_alias_vectors(number: int) -> Tuple[List[int], List[float]]:
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
    empty_j = np.empty(0, dtype=numpy_vector_alias_indices_type)
    empty_q = np.empty(0, dtype=numpy_vector_alias_probs_type)
    for _ in number:
        alias.append((empty_j, empty_q))
    return alias


@njit(parallel=True)
def build_alias_nodes(
    nodes_neighboring_edges: List[List[int]],
    weights: List[float]
) -> List[Tuple[List[int], List[float]]]:
    """Return aliases for nodes to use for alias method for 
       selecting from discrete distribution.

    Parameters
    -----------------------
    nodes_neighboring_edges:  List[List[int]],
        List of neighbouring edges for each node.
    weights: List[float],
        List of weights for each edge.

    Returns
    -----------------------
    Lists of lists representing node aliases 
    """
    number = len(nodes_neighboring_edges)
    alias = build_default_alias_vectors(number)

    for i in prange(number):  # pylint: disable=not-an-iterable
        src = np.int64(i)
        neighboring_edges = nodes_neighboring_edges[src]
        neighboring_edges_number = len(neighboring_edges)

        # Do not call the alias setup if the node is a trap.
        # Because that node will have no neighbors and thus the necessity
        # of setupping the alias method to efficently extract the neighbour.
        if neighboring_edges_number == 0:
            continue

        probs = np.zeros(
            neighboring_edges_number,
            dtype=numpy_vector_alias_probs_type
        )
        for j, neighboring_edge in enumerate(neighboring_edges):
            probs[j] = weights[neighboring_edge]

        alias[src] = alias_setup(probs/probs.sum())
    return alias


@njit(parallel=True)
def build_alias_edges(
    edges_set: Set[Tuple[int, int]],
    nodes_neighboring_edges: List[List[int]],
    node_types: List[int],
    edge_types: List[int],
    weights: List[float],
    default_weight: float,
    sources: List[int],
    destinations: List[int],
    return_weight: float,
    explore_weight: float,
    change_node_type_weight: float,
    change_edge_type_weight: float,
) -> List[Tuple[List[int], List[float]]]:
    """Return aliases for edges to use for alias method for 
       selecting from discrete distribution.

    Parameters
    -----------------------
    edges_set: Set[Tuple[int, int]],
        Set of unique edges.
    nodes_neighboring_edges: List[List[int]],
        List of neighbouring edges for each node.
    node_types: List[int],
        List of node types.
    edge_types: List[int],
        List of edge types.
    weights: List[float],
        List of edge weights.
    default_weight: float,
        Default weight to use if weights is None.
    sources: List[int],
        List of source nodes for each edge.
    destinations: List[int],
        List of destination nodes for each edge.
    return_weight: float,
        hyperparameter for breadth first random walk - 1/p
    explore_weight: float,
        hyperparameter for depth first random walk - 1/q
    change_node_type_weight: float,
        hyperparameter for changing node type during random walk
    change_edge_type_weight: float,
        hyperparameter for changing edge type during random walk

    Raises
    -----------------------
    ValueError,
        If given node types list has not the same length of the given nodes
        neighbouring edges list.
    ValueError,
        If given edge types have not the same length of the given weights list.
    ValueError,
        If given source nodes have not the same length of the given of the
        given weights list.
    ValueError,
        If return_weight is not a strictly positive real number.
    ValupeError,
        If explore_weight is not a strictly positive real number.
    ValueError,      
        If change_node_type_weight is not a strictly positive real number.
    ValueError,
        If change_edge_type_weight is not a strictly positive real number.

    Returns
    -----------------------
    Lists of lists representing edges aliases.
    """

    if len(node_types) != len(nodes_neighboring_edges):
        raise ValueError(
            "Given node types list has not the same length of the given "
            "nodes neighbouring edges list."
        )
    if edge_types is not None and len(edge_types) != len(sources):
        raise ValueError(
            "Given edge types list has not the same length of the given "
            "sources list!"
        )
    if weights is not None and len(sources) != len(weights):
        raise ValueError(
            "Given source nodes list has not the same length of the given "
            "weights list!"
        )

    if return_weight <= 0:
        raise ValueError("Given return weight is not a positive number")

    if explore_weight <= 0:
        raise ValueError("Given explore weight is not a positive number")

    if change_node_type_weight <= 0:
        raise ValueError(
            "Given change_node_type_weigh is not a positive number")

    if change_edge_type_weight <= 0:
        raise ValueError(
            "Given change_edge_type_weight is not a positive number")

    number = len(sources)
    alias = build_default_alias_vectors(number)

    for i in prange(len(sources)):  # pylint: disable=not-an-iterable
        k = np.int64(i)
        src = sources[k]
        dst = destinations[k]
        neighboring_edges = nodes_neighboring_edges[dst]
        neighboring_edges_number = len(neighboring_edges)

        # Do not call the alias setup if the edge is a trap.
        # Because that edge will have no neighbors and thus the necessity
        # of setupping the alias method to efficently extract the neighbour.
        if neighboring_edges_number == 0:
            continue

        probs = np.zeros(
            neighboring_edges_number,
            dtype=numpy_vector_alias_probs_type
        )

        for index, edge in enumerate(neighboring_edges):
            # We get the weight for the edge from the destination to
            # the neighbour.
            weight = default_weight if weights is None else weights[edge]
            # Then we retrieve the neigh_dst node type.
            # And if the destination node type matches the neighbour
            # destination node type (we are not changing the node type)
            # we weigth using the provided change_node_type_weight weight.
            ndst = destinations[edge]
            if node_types is not None and node_types[dst] == node_types[ndst]:
                weight /= change_node_type_weight
            # Similarly if the neighbour edge type matches the previous
            # edge type (we are not changing the edge type)
            # we weigth using the provided change_edge_type_weight weight.
            if edge_types is not None and edge_types[k] == edge_types[edge]:
                weight /= change_edge_type_weight
            # If the neigbour matches with the source, hence this is
            # a backward loop like the following:
            # SRC -> DST
            #  ▲     /
            #   \___/
            #
            # We weight the edge weight with the given return weight.
            if ndst == src:
                weight = weight * return_weight
            # If the backward loop does not exist, we multiply the weight
            # of the edge by the weight for moving forward and explore more.
            elif (ndst, src) not in edges_set:
                weight = weight * explore_weight
            # Then we store these results into the probability vector.
            probs[index] = weight
        alias[k] = alias_setup(probs/probs.sum())
    return alias
