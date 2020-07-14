from ensmallen_graph import EnsmallenGraph


class GraphVisualizations:
    
    def __init__(self, graph: EnsmallenGraph):
        

    def plot_embedding(
        graph: EnsmallenGraph,
        tsne_embedding: np.ndarray,
        k: int = 10,
        axes: Axes = None
    ):
        if axes is None:
            _, axes = plt.subplots(figsize=(5, 5))

        if graph.node_types_mapping is None:
            node_types = np.zeros(graph.get_nodes_number(), dtype=np.uint8)
            common_node_types_names = ["No node type provided"]
        else:
            nodes, node_types = graph.get_top_k_nodes_by_node_type(k)
            tsne_embedding = tsne_embedding[nodes]
            common_node_types_names = list(
                np.array(graph.node_types_reverse_mapping)[np.unique(node_types)])

        colors = list(TABLEAU_COLORS.keys())[:len(common_node_types_names)]

        scatter = axes.scatter(
            *tsne_embedding.T,
            s=0.25,
            c=node_types,
            cmap=ListedColormap(colors)
        )
        axes.legend(
            handles=scatter.legend_elements()[0],
            labels=common_node_types_names
        )
        return axes
