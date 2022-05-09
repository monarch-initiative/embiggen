"""Submodule providing methods to check for graph oddities."""
import warnings
from ensmallen import Graph


def detect_graph_node_embedding_oddities(graph: Graph):
    """Raises warnings if there are topological oddities in this graph.

    Parameters
    ----------------
    graph: Graph
        The graph to run checks on.

    Raises
    ----------------
    ValueError
        If the provided graph does not have nodes.
    ValueError
        If the provided graph does not have edges.
    """
    if not graph.has_nodes():
        raise ValueError("The provided graph does not have nodes.")

    if not graph.has_edges():
        raise ValueError("The provided graph does not have edges.")
    
    if graph.has_singleton_nodes():
        warnings.warn(
            (
                "Please be advised that this graph contains {} singleton nodes. "
                "Consider that node embedding algorithms that only use topological information "
                "such as CBOW, GloVe, SPINE and SkipGram are not able to provide meaningful "
                "embeddings for these nodes, and their embedding will be generally "
                "far away from any other node. It is also possible that all singleton nodes "
                "will receive a relatively similar node embedding. "
                "Consider dropping them by using the `graph.remove_singleton_nodes()` method."
            ).format(graph.get_singleton_nodes_number())
        )
    
    if graph.has_singleton_nodes_with_selfloops():
        warnings.warn(
            (
                "Please be advised that this graph contains {} singleton nodes with self-loops. "
                "Consider that node embedding algorithms that only use topological information "
                "such as CBOW, GloVe, SPINE and SkipGram are not able to provide meaningful "
                "embeddings for these nodes, and their embedding will be generally "
                "far away from any other node. It is also possible that all singleton nodes "
                "will receive a relatively similar node embedding. "
                "Consider dropping them by using the `graph.remove_singleton_nodes_with_selfloops()` method."
            ).format(graph.get_singleton_nodes_with_selfloops_number())
        )