from typing import List, Tuple, Dict, Set
import numpy as np  # type: ignore
from numba.experimental import jitclass, njit, prange  # type: ignore
from numba import typed, types  # type: ignore
from .numba_graph import NumbaGraph
from .graph_types import (
    nodes_mapping_type,
    numpy_nodes_type,
    numba_vector_nodes_type
)


@njit(parallel=True)
def process_edges(
    sources_names: List[str],
    destinations_names: List[str],
    mapping: Dict[str, int]
) -> Tuple[List[int], List[int], Set[Tuple[int, int]]]:
    """Return triple with mapped sources, destinations and edges set.

    Parameters
    --------------------------
    sources_names: List[str],
        List of the sources names.
    destinations_names: List[str],
        List of the destinations names.
    mapping: Dict[str, int],
        Dictionary mapping of the nodes.

    Returns
    --------------------------
    Triple with:
        - The mapped sources using given mapping, as a list of integers.
        - The mapped destinations using given mapping, as a list of integers.
        - The mapped edge sets using given mapping, as a set of tuples.
    """
    edges_number = len(sources_names)
    sources = np.empty(edges_number, dtype=numpy_nodes_type)
    destinations = np.empty(edges_number, dtype=numpy_nodes_type)
    edges_set = set()

    for i in prange(edges_number):
        # Store the sources into the sources vector
        sources[i] = mapping[str(sources_names[i])]
        # Store the destinations into the destinations vector
        destinations[i] = mapping[str(destinations_names[i])]
        # Adding the edge to the set
        edges_set.add((sources[i], destinations[i]))

    return sources, destinations, edges_set


@jitclass([
    ('_destinations', numba_vector_nodes_type),
    ('_sources', numba_vector_nodes_type),
    ('_nodes_mapping', types.DictType(*nodes_mapping_type)),
    ('_reverse_nodes_mapping', types.ListType(types.string)),
    ('_preprocess', types.boolean),
])
class NumbaDirectedGraph:

    def __init__(
        self,
        nodes: List[str],
        sources_names: List[str],
        destinations_names: List[str],
        preprocess: bool = True,
        **kwargs
    ):
        """Crate a new instance of a NumbaDirectedGraph with given edges.

        Parameters
        -------------------------
        nodes: List[str],
            List of node names.
        sources_names: List[int],
            List of the source nodes in edges of the graph.
        destinations_names: List[int],
            List of the destination nodes in edges of the graph.
        preprocess: bool = True,
            Wethever to preprocess this graph for random walk.

        Raises
        -------------------------
        ValueError,
            If given sources length does not match destinations length.
        """

        if len(sources_names) != len(destinations_names):
            raise ValueError(
                "Given sources length does not match destinations length."
            )

        self._preprocess = preprocess

        # Creating mapping and reverse mapping of nodes and integer ID.
        # The map looks like the following:
        # {
        #   "node_1_id": 0,
        #   "node_2_id": 1
        # }
        #
        # The reverse mapping is just a list of the nodes.
        #
        self._nodes_mapping = typed.Dict.empty(*nodes_mapping_type)
        self._reverse_nodes_mapping = typed.List.empty_list(types.string)
        for i, node in enumerate(nodes):
            self._nodes_mapping[str(node)] = np.uint32(i)
            self._reverse_nodes_mapping.append(str(node))

        # Transform the lists of names into IDs
        self._sources, self._destinations, edges_set = process_edges(
            sources_names, destinations_names, self._nodes_mapping
        )

        if not self._preprocess:
            return

        self._graph = NumbaGraph(
            nodes_number=len(self._nodes_mapping),
            sources=self._sources,
            destinations=self._destinations,
            edges_set=edges_set,
            **kwargs
        )