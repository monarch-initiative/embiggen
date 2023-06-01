"""Module provding a very abstract edge feature to be used in the edge prediction and edge-label prediction models.

The main goal of this feature is to provide a trasparent interface to query the features of specific edges
in a graph. Most commonly, rasterizing a complete graph set of edge features would be too memory intensive,
and this abstract class provides a standardized way to query the features of a specific edge.
"""
from embiggen.utils.abstract_feature import AbstractFeature
from ensmallen import Graph
from typing import Dict, List
import numpy as np


class AbstractEdgeFeature(AbstractFeature):

    def __init__(self):
        """Create a new abstract edge feature."""
        super().__init__()

    def fit(self, graph: Graph):
        """Fit the edge feature to the given graph.
        
        Parameters
        -----------------------
        graph: Graph,
            The graph to use as base for the topological metrics.
        """
        raise NotImplementedError(
            "The method fit was not implemented."
        )
    
    def is_fit(self) -> bool:
        """Return whether the edge feature is fit."""
        raise NotImplementedError(
            "The method is_fit was not implemented."
        )

    @classmethod
    def get_feature_dictionary_keys(cls) -> List[str]:
        """Return the list of keys to be used in the feature dictionary."""
        raise NotImplementedError(
            "The method get_feature_dictionary_keys was not implemented."
        )
    
    def get_feature_dictionary_shapes(self) -> Dict[str, List[int]]:
        """Return the dictionary of shapes to be used in the feature dictionary."""
        raise NotImplementedError(
            "The method get_feature_dictionary_shapes was not implemented."
        )

    def get_edge_feature_from_edge_node_ids(self, support: Graph, sources: np.ndarray, destinations: np.ndarray) -> Dict[str, np.ndarray]:
        """Return the edge feature for the given edge.
        
        Parameters
        -----------------------
        support: Graph,
            The graph to use as base for the topological metrics.
        sources: np.ndarray,
            The source node ids.
        destinations: np.ndarray,
            The destination node ids.
        """
        raise NotImplementedError(
            "The method get_edge_feature_from_edge_node_ids was not implemented."
        )
    
    def get_edge_feature_from_graph(self, graph: Graph, support: Graph) -> Dict[str, np.ndarray]:
        """Return the edge feature for the given graph.
        
        Parameters
        -----------------------
        graph: Graph,
            The graph for which to compute the edge feature.
        support: Graph,
            The graph to use as base for the topological metrics.
        """
        raise NotImplementedError(
            "The method get_edge_feature_from_graph was not implemented."
        )