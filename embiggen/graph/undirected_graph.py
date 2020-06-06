from typing import List, Tuple, Dict, Set
import numpy as np  # type: ignore
from numba.experimental import jitclass  # type: ignore
from numba import typed, types, njit, prange  # type: ignore
from .directed_graph import DirectedGraph


class UndirectedGraph(DirectedGraph):

    def __init__(
        self,
        sources_names: np.ndarray,
        destinations_names: np.ndarray,
        node_types: np.ndarray = None,
        edge_types: np.ndarray = None,
        weights: np.ndarray = None,
        **kwargs
    ):
        """Crate a new instance of a NumbaDirectedGraph with given edges.

        Parameters
        -------------------------
        sources_names: List[int],
            List of the source nodes in edges of the graph.
        destinations_names: List[int],
            List of the destination nodes in edges of the graph.

        Raises
        -------------------------
        ValueError,
            If given sources length does not match destinations length.
        """

        if len(sources_names) != len(destinations_names):
            raise ValueError(
                "Given sources length does not match destinations length."
            )

        # Counting self-loops
        loops_mask = sources_names == destinations_names
        total_loops = loops_mask.sum()

        total_orig_edges = len(sources_names)
        total_edges = (total_orig_edges-total_loops)*2 + total_loops

        full_sources = np.empty(total_edges, dtype=sources_names.dtype)
        full_destinations = np.empty(total_edges, dtype=destinations_names.dtype)
        
        full_sources[:total_orig_edges] = sources_names
        full_sources[total_orig_edges:] = sources_names[~loops_mask]

        full_destinations[:total_orig_edges] = destinations_names
        full_destinations[total_orig_edges:] = destinations_names[~loops_mask]

        if node_types is not None:
            full_node_types = np.empty(total_edges, dtype=np.uint16)
            full_node_types[:total_orig_edges] = node_types
            full_node_types[total_orig_edges:] = node_types[~loops_mask]
            node_types = full_node_types

        if edge_types is not None:
            full_edge_types = np.empty(total_edges, dtype=np.uint16)
            full_edge_types[:total_orig_edges] = edge_types
            full_edge_types[total_orig_edges:] = edge_types[~loops_mask]
            edge_types = full_edge_types

        if len(weights) > 0:
            full_weights = np.empty(total_edges, dtype=np.uint16)
            full_weights[:total_orig_edges] = weights
            full_weights[total_orig_edges:] = weights[~loops_mask]
            weights = full_weights

        super().__init__(
            sources_names=full_sources,
            destinations_names=full_destinations,
            node_types=node_types,
            edge_types=edge_types,
            weights=weights,
            **kwargs
        )
