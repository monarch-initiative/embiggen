"""Module with embedding visualization tools."""
from multiprocessing import cpu_count
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from ensmallen_graph import EnsmallenGraph  # pylint: disable=no-name-in-module
from matplotlib._color_data import TABLEAU_COLORS
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from ..transformers import GraphTransformer, NodeTransformer


class GraphVisualizations:
    """Tools to visualize the graph embeddings."""

    def __init__(
        self,
        method: str = "hadamard",
        random_state: int = 42,
        verbose: bool = True
    ):
        """Create new GraphVisualizations object."""
        self._graph_transformer = GraphTransformer(method=method)
        self._node_transformer = NodeTransformer()
        self._random_state = random_state
        self._verbose = verbose
        self._node_mapping = self._node_embedding = self._edge_embedding = None
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
            if "random_state" not in kwargs:
                kwargs["random_state"] = self._random_state
        return TSNE(
            verbose=self._verbose,
            **kwargs
        ).fit_transform(X)

    def fit_transform_nodes(
        self,
        graph: EnsmallenGraph,
        embedding: np.ndarray,
        node_mapping: Dict[str, int],
        **kwargs: Dict
    ):
        """Executes fitting for plotting node embeddings."""
        self._node_transformer.fit(embedding)
        self._node_embedding = self.tsne(
            self._node_transformer.transform(np.fromiter((
                node_mapping[node]
                for node in graph.nodes_reverse_mapping
            ), dtype=np.int)),
            **kwargs
        )

    def fit_transform_edges(
        self,
        graph: EnsmallenGraph,
        embedding: np.ndarray,
        **kwargs: Dict
    ):
        """Executes fitting for plotting edge embeddings."""
        self._graph_transformer.fit(embedding)
        self._edge_embedding = self.tsne(
            self._graph_transformer.transform(graph),
            **kwargs
        )

    def plot_node_types(
        self,
        graph: EnsmallenGraph,
        k: int = 10,
        s: float = 0.01,
        alpha: float = 0.5,
        figure: Figure = None,
        axes: Axes = None,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """Plot common node types of provided graph.

        Parameters
        ------------------------------
        graph: EnsmallenGraph,
            The graph to visualize.
        k: int = 10,
            Number of node types to visualize.
        s: float = 0.01,
            Size of the scatter.
        alpha: float = 0.5,
            Alpha level for the scatter plot.
        figure: Figure = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Axes = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.

        Returns
        ------------------------------
        Figure and Axis of the plot.
        """
        if self._node_embedding is None:
            raise ValueError(
                "Node fitting must be executed before plot."
            )

        if figure is None or axes is None:
            figure, axes = plt.subplots(**kwargs)

        if graph.node_types_reverse_mapping is None:
            node_types = np.zeros(graph.get_nodes_number(), dtype=np.uint8)
            common_node_types_names = ["No node type provided"]
            node_tsne = self._node_embedding
        else:
            nodes, node_types = graph.get_top_k_nodes_by_node_type(k)
            node_tsne = self._node_embedding[nodes]
            common_node_types_names = np.array(
                graph.node_types_reverse_node_mapping
            )[np.unique(node_types)].tolist()

        colors = list(TABLEAU_COLORS.keys())[:len(common_node_types_names)]

        # Shuffling points to avoid having artificial clusters
        # caused by positions.
        index = np.arange(node_types.size)
        np.random.shuffle(index)
        node_types = node_types[index]
        node_tsne = node_tsne[index]

        scatter = axes.scatter(
            *node_tsne.T,
            s=s,
            alpha=alpha,
            c=node_types,
            cmap=ListedColormap(colors)
        )
        axes.legend(
            handles=scatter.legend_elements()[0],
            labels=common_node_types_names,
            loc="right"
        )
        axes.set_xticks([])
        axes.set_xticks([], minor=True)
        axes.set_yticks([])
        axes.set_yticks([], minor=True)
        axes.set_title("Node types")
        return figure, axes

    def plot_node_degrees(
        self,
        graph: EnsmallenGraph,
        s: float = 0.01,
        alpha: float = 0.5,
        figure: Figure = None,
        axes: Axes = None,
        **kwargs: Dict
    ):
        """Plot common node types of provided graph.

        Parameters
        ------------------------------
        graph: EnsmallenGraph,
            The graph to visualize.
        s: float = 0.01,
            Size of the scatter.
        alpha: float = 0.5,
            Alpha level for the scatter plot.
        figure: Figure = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Axes = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.

        Returns
        ------------------------------
        Figure and Axis of the plot.
        """
        if self._node_embedding is None:
            raise ValueError(
                "Node fitting must be executed before plot."
            )

        if figure is None or axes is None:
            figure, axes = plt.subplots(**kwargs)

        degrees = graph.degrees()
        two_median = np.median(degrees)*2
        degrees[degrees > two_median] = min(two_median, degrees.max())

        # Shuffling points to avoid having artificial clusters
        # caused by positions.
        index = np.arange(degrees.size)
        np.random.shuffle(index)
        degrees = degrees[index]
        node_tsne = self._node_embedding[index]

        scatter = axes.scatter(
            *node_tsne.T,
            c=degrees,
            s=s,
            alpha=alpha,
            cmap=plt.cm.get_cmap('RdYlBu')
        )
        figure.colorbar(scatter, ax=axes)
        axes.set_xticks([])
        axes.set_xticks([], minor=True)
        axes.set_yticks([])
        axes.set_yticks([], minor=True)
        axes.set_title("Node degrees")
        return figure, axes

    def plot_edge_types(
        self,
        graph: EnsmallenGraph,
        k: int = 10,
        s: float = 0.01,
        alpha: float = 0.5,
        figure: Figure = None,
        axes: Axes = None,
        **kwargs: Dict
    ):
        """Plot common edge types of provided graph.

        Parameters
        ------------------------------
        graph: EnsmallenGraph,
            The graph to visualize.
        k: int = 10,
            Number of edge types to visualize.
        s: float = 0.01,
            Size of the scatter.
        alpha: float = 0.5,
            Alpha level for the scatter plot.
        figure: Figure = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Axes = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.

        Returns
        ------------------------------
        Figure and Axis of the plot.
        """
        if self._edge_embedding is None:
            raise ValueError(
                "Edge fitting must be executed before plot."
            )

        if figure is None or axes is None:
            figure, axes = plt.subplots(**kwargs)

        if graph.edge_types_reverse_mapping is None:
            edge_types = np.zeros(graph.get_edges_number(), dtype=np.uint8)
            common_edge_types_names = ["No edge type provided"]
            edge_tsne = self._edge_embedding
        else:
            edges, edge_types = graph.get_top_k_edges_by_edge_type(k)
            edge_tsne = self._edge_embedding[edges]
            common_edge_types_names = np.array(
                graph.edge_types_reverse_node_mapping
            )[np.unique(edge_types)].tolist()

        colors = list(TABLEAU_COLORS.keys())[:len(common_edge_types_names)]

        # Shuffling points to avoid having artificial clusters
        # caused by positions.
        index = np.arange(edge_types.size)
        np.random.shuffle(index)
        edge_types = edge_types[index]
        edge_tsne = edge_tsne[index]

        scatter = axes.scatter(
            *edge_tsne.T,
            s=s,
            alpha=alpha,
            c=edge_types,
            cmap=ListedColormap(colors)
        )
        axes.legend(
            handles=scatter.legend_elements()[0],
            labels=common_edge_types_names,
            loc="right"
        )
        axes.set_xticks([])
        axes.set_xticks([], minor=True)
        axes.set_yticks([])
        axes.set_yticks([], minor=True)
        axes.set_title("Edge types")
        return figure, axes

    def plot_edge_weights(
        self,
        graph: EnsmallenGraph,
        s: float = 0.01,
        alpha: float = 0.5,
        figure: Figure = None,
        axes: Axes = None,
        **kwargs: Dict
    ):
        """Plot common edge types of provided graph.

        Parameters
        ------------------------------
        graph: EnsmallenGraph,
            The graph to visualize.
        s: float = 0.01,
            Size of the scatter.
        alpha: float = 0.5,
            Alpha level for the scatter plot.
        figure: Figure = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Axes = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.

        Returns
        ------------------------------
        Figure and Axis of the plot.
        """
        if self._edge_embedding is None:
            raise ValueError(
                "Edge fitting must be executed before plot."
            )

        if figure is None or axes is None:
            figure, axes = plt.subplots(**kwargs)

        # Shuffling points to avoid having artificial clusters
        # caused by positions.
        index = np.arange(graph.get_edges_number())
        np.random.shuffle(index)
        edge_embedding = self._edge_embedding[index]
        weights = graph.weights[index]

        scatter = axes.scatter(
            *edge_embedding.T,
            c=weights,
            s=s,
            alpha=alpha,
            cmap=plt.cm.get_cmap('RdYlBu')
        )
        figure.colorbar(scatter, ax=axes)
        axes.set_xticks([])
        axes.set_xticks([], minor=True)
        axes.set_yticks([])
        axes.set_yticks([], minor=True)
        axes.set_title("Edge weights")
        return figure, axes
