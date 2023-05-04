"""Submodule with utilities relative to NetworkX."""
from tqdm.auto import tqdm, trange
from ensmallen import Graph, GraphBuilder
import networkx as nx


def convert_ensmallen_graph_to_networkx_graph(
    graph: Graph,
    numeric_node_ids: bool = False
) -> nx.Graph:
    """Return NetworkX graph derived from the provided Ensmallen Graph.

    Parameters
    -----------
    graph: Graph
        The graph to be converted from ensmallen to NetworkX.
    numeric_node_ids: bool = False
        Whether to use numeric node IDs or string node IDs.
        By default, we use the string node IDs as they are more
        interpretable.
    """
    if graph.is_directed():
        result_graph = nx.DiGraph(name=graph.get_name())
    else:
        result_graph = nx.Graph(name=graph.get_name())

    for node_id in trange(
        graph.get_number_of_nodes(),
        leave=False,
        dynamic_ncols=True,
        desc="Parsing nodes"
    ):
        result_graph.add_node(
            node_id if numeric_node_ids else graph.get_node_name_from_node_id(node_id),
            node_types=graph.get_unchecked_node_type_names_from_node_id(
                node_id
            ),
        )

    for edge_id in trange(
        graph.get_number_of_directed_edges(),
        leave=False,
        dynamic_ncols=True,
        desc="Parsing edges"
    ):
        result_graph.add_edge(
            *(
                graph.get_node_ids_from_edge_id(edge_id)
                if numeric_node_ids else
                graph.get_node_names_from_edge_id(edge_id)
            ),
            **(
                dict(
                    weight=graph.get_unchecked_edge_weight_from_edge_id(edge_id),
                )
                if graph.has_edge_weights() else dict()
            ),
            **(
                dict(
                    edge_type=graph.get_unchecked_edge_type_name_from_edge_id(edge_id),
                )
                if graph.has_edge_types() else dict()
            ),
        )

    return result_graph


def convert_networkx_graph_to_ensmallen_graph(
    graph: nx.Graph
) -> Graph:
    """Return Ensmallen Graph derived from the provided NetworkX Graph.

    Parameters
    -----------
    graph: nx.Graph
        The graph to be converted from NetworkX to Ensmallen.
    """
    builder = GraphBuilder()
    builder.set_directed(graph.is_directed())
    builder.set_name(graph.name or "NetworkX graph")

    for node_name in tqdm(
        graph.nodes(),
        total=graph.number_of_nodes(),
        leave=False,
        dynamic_ncols=True,
        desc="Parsing nodes"
    ):
        node_types = graph.nodes[node_name].get("node_types", None)
        if (
            node_types is not None and
            not isinstance(node_types, list)
        ):
            node_types = [node_types]
        builder.add_node(node_name, node_type=node_types)

    for source, destination in tqdm(
        graph.edges(),
        total=graph.number_of_edges(),
        leave=False,
        dynamic_ncols=True,
        desc="Parsing edges"
    ):
        edge_data = graph.get_edge_data(source, destination)
        builder.add_edge(
            source,
            destination,
            edge_type=edge_data.get("edge_data", None),
            weight=edge_data.get("weight", None)
        )

    return builder.build()
