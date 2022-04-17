"""Submodule providing methods to enforce the graph sorting."""
from ensmallen import Graph


def enforce_sorted_graph(graph: Graph):
    """Checks whether the graph has nodes sorted properly.

    Parameters
    ----------------
    graph: Graph
        The graph to run checks on.

    Raises
    ----------------
    ValueError
        If the graph does not have graph sorted by decreasing
        node degree.
    """
    if not graph.has_nodes_sorted_by_decreasing_outbound_node_degree():
        raise ValueError(
            "The given graph does not have the nodes sorted by decreasing "
            "order, therefore the negative sampling (which follows a zipfian "
            "distribution) would not approximate well the Softmax.\n"
            "In order to sort the given graph in such a way that the node IDs "
            "are sorted by decreasing outbound node degrees, you can use "
            "the Graph method "
            "`graph.sort_by_decreasing_outbound_node_degree()`"
        )
