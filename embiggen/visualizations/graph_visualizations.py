"""Module with embedding visualization tools."""
from ensmallen_graph import EnsmallenGraph  # pylint: disable=no-name-in-module
import numpy as np
from typing import Dict
from matplotlib._color_data import TABLEAU_COLORS
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
from ..transformers import GraphTransformer, NodeTransformer


class GraphVisualizations:
    """Tools to visualize the graph embeddings."""

    def __init__(
        self,
        method: str = "hadamard",
        random_state: int = 42,
        size: float = 0.1,
        verbose: bool = True
    ):
        """Create new GraphVisualizations object."""
        self._graph_transformer = GraphTransformer(method=method)
        self._node_transformer = NodeTransformer()
        self._random_state = random_state
        self._verbose = verbose
        self._mapping = None
        self._size = size
        self._random = np.random.RandomState(  # pylint: disable=no-member
            seed=random_state
        )

    def tsne(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Return TSNE embedding of given array."""
        try:
            from tsnecuda import TSNE
        except ModuleNotFoundError:
            from MulticoreTSNE import MulticoreTSNE as TSNE
            if "n_jobs" not in kwargs:
                kwargs["n_jobs"] = cpu_count()
        return TSNE(
            verbose=self._verbose,
            **kwargs
        ).fit_transform(X)

    def fit(self, embedding: np.ndarray, mapping: Dict[str, int]):
        """Build embeddings for visualization porposes."""
        self._graph_transformer.fit(embedding)
        self._node_transformer.fit(embedding)
        self._mapping = mapping

    def _plot_node_types(
        self,
        graph: EnsmallenGraph,
        node_tsne: np.ndarray,
        axes: Axes
    ):
        """Plot common node types of provided graph.

        Parameters
        ------------------------------
        graph: EnsmallenGraph,
            The graph to visualize.
        node_tsne: np.ndarray,
            The node tsne embedding.
        axes: Axes,
            Axes to use to plot.
        """
        if graph.node_types_mapping is None:
            node_types = np.zeros(graph.get_nodes_number(), dtype=np.uint8)
            common_node_types_names = ["No node type provided"]
        else:
            nodes, node_types = graph.get_top_k_nodes_by_node_type(10)
            node_tsne = node_tsne[nodes]
            common_node_types_names = list(
                np.array(graph.node_types_reverse_mapping)[np.unique(node_types)])

        colors = list(TABLEAU_COLORS.keys())[:len(common_node_types_names)]

        scatter = axes.scatter(
            *node_tsne.T,
            s=self._size,
            c=node_types,
            cmap=ListedColormap(colors)
        )
        axes.legend(
            handles=scatter.legend_elements()[0],
            labels=common_node_types_names
        )
        axes.set_title("Node types")

    def _plot_node_degrees(
        self,
        graph: EnsmallenGraph,
        node_tsne: np.ndarray,
        fig: Figure,
        axes: Axes
    ):
        """Plot common node types of provided graph.

        Parameters
        ------------------------------
        graph: EnsmallenGraph,
            The graph to visualize.
        node_tsne: np.ndarray,
            The node tsne embedding.
        fig: Figure,
            Figure to use to plot.
        axes: Axes,
            Axes to use to plot.
        """
        degrees = graph.degrees()
        two_median = np.median(degrees)*2
        degrees[degrees > two_median] = min(two_median, degrees.max())
        scatter = axes.scatter(
            *node_tsne.T,
            c=degrees,
            s=self._size,
            cmap=plt.cm.get_cmap('RdYlBu')
        )
        fig.colorbar(scatter, ax=axes)
        axes.set_title("Node degrees")

    def _plot_edge_types(
        self,
        graph: EnsmallenGraph,
        edge_tsne: np.ndarray,
        axes: Axes
    ):
        """Plot common edge types of provided graph.

        Parameters
        ------------------------------
        graph: EnsmallenGraph,
            The graph to visualize.
        edge_tsne: np.ndarray,
            The edge tsne embedding.
        axes: Axes,
            Axes to use to plot.
        """
        if graph.edge_types_mapping is None:
            edge_types = np.zeros(graph.get_edges_number(), dtype=np.uint8)
            common_edge_types_names = ["No edge type provided"]
        else:
            edges, edge_types = graph.get_top_k_edges_by_edge_type(10)
            edge_tsne = edge_tsne[edges]
            common_edge_types_names = list(
                np.array(graph.edge_types_reverse_mapping)[np.unique(edge_types)])

        colors = list(TABLEAU_COLORS.keys())[:len(common_edge_types_names)]

        scatter = axes.scatter(
            *edge_tsne.T,
            s=self._size,
            c=edge_types,
            cmap=ListedColormap(colors)
        )
        axes.legend(
            handles=scatter.legend_elements()[0],
            labels=common_edge_types_names
        )
        axes.set_title("Edge types")

    def _plot_edge_weights(
        self,
        graph: EnsmallenGraph,
        edge_tsne: np.ndarray,
        fig: Figure,
        axes: Axes
    ):
        """Plot common edge types of provided graph.

        Parameters
        ------------------------------
        graph: EnsmallenGraph,
            The graph to visualize.
        edge_tsne: np.ndarray,
            The edge tsne embedding.
        fig: Figure,
            Figure to use to plot.
        axes: Axes,
            Axes to use to plot.
        """
        scatter = axes.scatter(
            *edge_tsne.T,
            c=graph.weights,
            s=self._size,
            cmap=plt.cm.get_cmap('RdYlBu')
        )
        fig.colorbar(scatter, ax=axes)
        axes.set_title("Edge weights")

    def visualize(self, graph: EnsmallenGraph, tsne_kwargs: Dict = None):
        """Visualize given graph."""
        if tsne_kwargs is None:
            tsne_kwargs = {}
        # Compute the original graph edge embedding
        edge_embedding = self._graph_transformer.transform(graph)
        # Computing the node embedding
        nodes = np.array([
            self._mapping[node]
            for node in graph.nodes_reverse_mapping
        ])
        # Computing the TSNE embedding
        if self._verbose:
            print("Computing node TSNE embedding.")
        nodes_tsne = self.tsne(
            self._node_transformer.transform(nodes),
            **tsne_kwargs
        )
        if self._verbose:
            print("Computing edge TSNE embedding.")
        edge_tsne = self.tsne(
            edge_embedding,
            **tsne_kwargs
        )
        # Creating the figure and axes
        fig, axes = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=(10, 10)
        )
        axes = axes.flatten()
        for ax in axes:
            # for major ticks
            ax.set_xticks([])
            ax.set_yticks([])
            # for minor ticks
            ax.set_xticks([], minor=True)
            ax.set_yticks([], minor=True)
        (
            node_type_axes,
            node_degree_axes,
            edge_type_axes,
            edge_weight_axes
        ) = axes
        # Starting to visualize the various embeddings
        # Plotting the node types
        self._plot_node_types(
            graph,
            nodes_tsne,
            node_type_axes
        )
        # Plotting the edge types
        self._plot_node_degrees(
            graph,
            nodes_tsne,
            fig,
            node_degree_axes
        )
        # Plotting the edge types
        self._plot_edge_types(
            graph,
            edge_tsne,
            edge_type_axes
        )
        # Plotting the edge weights
        self._plot_edge_weights(
            graph,
            edge_tsne,
            fig,
            edge_weight_axes
        )

        fig.tight_layout()

        return fig, axes
