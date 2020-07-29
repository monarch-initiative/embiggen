"""Module with embedding visualization tools."""
from multiprocessing import cpu_count
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from ensmallen_graph import EnsmallenGraph  # pylint: disable=no-name-in-module
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..transformers import GraphTransformer, NodeTransformer


class GraphVisualizations:
    """Tools to visualize the graph embeddings."""

    def __init__(self, method: str = "hadamard"):
        """Create new GraphVisualizations object.

        Parameters
        -----------------------
        method: str = "hadamard",
            Edge embedding method.
        """
        self._graph_transformer = GraphTransformer(method=method)
        self._node_transformer = NodeTransformer()
        self._node_mapping = self._node_embedding = self._edge_embedding = None

    def tsne(self, X: np.ndarray, **kwargs:Dict) -> np.ndarray:
        """Return TSNE embedding of given array.
        
        Depending on what is available, we use tsnecuda or MulticoreTSNE.

        Parameters
        -----------------------
        X: np.ndarray,
            The data to embed.
        **kwargs: Dict,
            Parameters to pass directly to TSNE.

        Returns
        -----------------------
        The obtained TSNE embedding.
        """
        try:
            from tsnecuda import TSNE
        except ModuleNotFoundError:
            from MulticoreTSNE import MulticoreTSNE as TSNE
            if "n_jobs" not in kwargs:
                kwargs["n_jobs"] = cpu_count()
        return TSNE(**kwargs).fit_transform(X)

    def fit_transform_nodes(
        self,
        graph: EnsmallenGraph,
        embedding: np.ndarray,
        node_mapping: Dict[str, int],
        **kwargs: Dict
    ):
        """Executes fitting for plotting node embeddings.
        
        Parameters
        -------------------------
        graph: EnsmallenGraph,
            Graph from where to extract the nodes.
        embedding: np.ndarray,
            Embedding obtained from SkipGram, CBOW or GloVe.
        node_mapping: Dict[str, int],
            Nodes mapping to use to map eventual subgraphs.
        **kwargs: Dict,
            Data to pass directly to TSNE.
        """
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
        """Executes fitting for plotting edge embeddings.
        
        Parameters
        -------------------------
        graph: EnsmallenGraph,
            Graph from where to extract the edges.
        embedding: np.ndarray,
            Embedding obtained from SkipGram, CBOW or GloVe.
        **kwargs: Dict,
            Data to pass directly to TSNE.            
        """
        self._graph_transformer.fit(embedding)
        self._edge_embedding = self.tsne(
            self._graph_transformer.transform(graph),
            **kwargs
        )

    def plot_node_types(
        self,
        graph: EnsmallenGraph,
        k: int = 10,
        figure: Figure = None,
        axes: Axes = None,
        scatter_kwargs: Dict = None,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """Plot common node types of provided graph.

        Parameters
        ------------------------------
        graph: EnsmallenGraph,
            The graph to visualize.
        k: int = 10,
            Number of node types to visualize.
        figure: Figure = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Dict = None,
            Kwargs to pass to the scatter plot call.
        axes: Axes = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.

        Raises
        ------------------------------
        ValueError,
            If edge fitting was not yet executed.
        ValueError,
            If given k is greater than maximum supported value (10).

        Returns
        ------------------------------
        Figure and Axis of the plot.
        """
        if self._node_embedding is None:
            raise ValueError(
                "Node fitting must be executed before plot."
            )

        if k > 10:
            raise ValueError(
                "Values of k greater than 10 are not supported!"
            )

        if figure is None or axes is None:
            figure, axes = plt.subplots(**kwargs)

        if scatter_kwargs is None:
            scatter_kwargs = dict(
                s=1,
                marker=".",
                alpha=0.9
            )

        if graph.node_types_reverse_mapping is None:
            node_types = np.zeros(graph.get_nodes_number(), dtype=np.uint8)
            common_node_types_names = ["No node type provided"]
            node_tsne = self._node_embedding
        else:
            nodes, node_types = graph.get_top_k_nodes_by_node_type(k)
            node_tsne = self._node_embedding[nodes]
            common_node_types_names = np.array(
                graph.node_types_reverse_mapping
            )[np.unique(node_types)].tolist()

        # Shuffling points to avoid having artificial clusters
        # caused by positions.
        index = np.arange(node_types.size)
        np.random.shuffle(index)
        node_types = node_types[index]
        node_tsne = node_tsne[index]

        scatter = axes.scatter(
            *node_tsne.T,
            c=node_types,
            cmap=plt.get_cmap("tab10"),
            **scatter_kwargs,
        )
        legend = axes.legend(
            handles=scatter.legend_elements()[0],
            labels=common_node_types_names,
            bbox_to_anchor=(1.05, 1.0),
            loc='upper left'
        )
        for legend_handle in legend.legendHandles:
            legend_handle._legmarker.set_alpha(  # pylint: disable=protected-access
                1
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
        figure: Figure = None,
        axes: Axes = None,
        scatter_kwargs: Dict = None,
        **kwargs: Dict
    ):
        """Plot common node types of provided graph.

        Parameters
        ------------------------------
        graph: EnsmallenGraph,
            The graph to visualize.
        figure: Figure = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Dict = None,
            Kwargs to pass to the scatter plot call.
        axes: Axes = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.

        Raises
        ------------------------------
        ValueError,
            If edge fitting was not yet executed.

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

        if scatter_kwargs is None:
            scatter_kwargs = dict(
                s=1,
                marker=".",
                alpha=0.9
            )

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
            cmap=plt.cm.get_cmap('RdYlBu'),
            **scatter_kwargs,
        )
        color_bar = figure.colorbar(scatter, ax=axes)
        color_bar.set_alpha(1)
        color_bar.draw_all()
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
        figure: Figure = None,
        axes: Axes = None,
        scatter_kwargs: Dict = None,
        **kwargs: Dict
    ):
        """Plot common edge types of provided graph.

        Parameters
        ------------------------------
        graph: EnsmallenGraph,
            The graph to visualize.
        k: int = 10,
            Number of edge types to visualize.
        figure: Figure = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Dict = None,
            Kwargs to pass to the scatter plot call.
        axes: Axes = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.

        Raises
        ------------------------------
        ValueError,
            If edge fitting was not yet executed.
        ValueError,
            If given k is greater than maximum supported value (10).

        Returns
        ------------------------------
        Figure and Axis of the plot.
        """
        if self._edge_embedding is None:
            raise ValueError(
                "Edge fitting must be executed before plot."
            )

        if k > 10:
            raise ValueError(
                "Values of k greater than 10 are not supported!"
            )

        if figure is None or axes is None:
            figure, axes = plt.subplots(**kwargs)

        if scatter_kwargs is None:
            scatter_kwargs = dict(
                s=1,
                marker=".",
                alpha=0.9
            )

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

        # Shuffling points to avoid having artificial clusters
        # caused by positions.
        index = np.arange(edge_types.size)
        np.random.shuffle(index)
        edge_types = edge_types[index]
        edge_tsne = edge_tsne[index]

        scatter = axes.scatter(
            *edge_tsne.T,
            c=edge_types,
            cmap=plt.get_cmap("tab10"),
            **scatter_kwargs,
        )
        legend = axes.legend(
            handles=scatter.legend_elements()[0],
            labels=common_edge_types_names,
            bbox_to_anchor=(1.05, 1.0),
            loc='upper left'
        )
        for legend_handle in legend.legendHandles:
            legend_handle._legmarker.set_alpha(  # pylint: disable=protected-access
                1
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
        figure: Figure = None,
        axes: Axes = None,
        scatter_kwargs: Dict = None,
        **kwargs: Dict
    ):
        """Plot common edge types of provided graph.

        Parameters
        ------------------------------
        graph: EnsmallenGraph,
            The graph to visualize.
        figure: Figure = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Dict = None,
            Kwargs to pass to the scatter plot call.
        axes: Axes = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.

        Raises
        ------------------------------
        ValueError,
            If edge fitting was not yet executed.

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

        if scatter_kwargs is None:
            scatter_kwargs = dict(
                s=1,
                marker=".",
                alpha=0.9
            )

        # Shuffling points to avoid having artificial clusters
        # caused by positions.
        index = np.arange(graph.get_edges_number())
        np.random.shuffle(index)
        edge_embedding = self._edge_embedding[index]
        weights = graph.weights[index]

        scatter = axes.scatter(
            *edge_embedding.T,
            c=weights,
            cmap=plt.cm.get_cmap('RdYlBu'),
            **scatter_kwargs,
        )
        color_bar = figure.colorbar(scatter, ax=axes)
        color_bar.set_alpha(1)
        color_bar.draw_all()
        axes.set_xticks([])
        axes.set_xticks([], minor=True)
        axes.set_yticks([])
        axes.set_yticks([], minor=True)
        axes.set_title("Edge weights")
        return figure, axes
