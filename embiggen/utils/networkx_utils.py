"""Submodule with utilities relative to NetworkX."""
from ensmallen import Graph
import networkx as nx
import numpy as np

def convert_ensmallen_graph_to_networkx_graph(
    graph: Graph
) -> nx.Graph:
    """Return networkX graph derived from the provided Ensmallen Graph.
    
    Parameters
    -----------
    graph: Graph
        The graph to be converted.
    """
    if graph.is_directed():
        result_graph = nx.DiGraph()
    else:
        result_graph = nx.Graph()

    result_graph.add_nodes_from(graph.get_node_ids())

    if graph.has_edge_weights():
        result_graph.add_weighted_edges_from([
            (src_name, dst_name, edge_weight)
            for (src_name, dst_name), edge_weight in zip(
                graph.get_directed_edge_node_ids(),
                graph.get_edge_weights()
            )
        ])
    else:
        result_graph.add_edges_from(
            graph.get_edge_node_ids(directed=graph.is_directed())
        )

    return result_graph