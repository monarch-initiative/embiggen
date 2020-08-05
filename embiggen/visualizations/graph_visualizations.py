"""Module with embedding visualization tools."""
from collections import Counter, OrderedDict
from multiprocessing import cpu_count
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from ensmallen_graph import EnsmallenGraph  # pylint: disable=no-name-in-module
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend_handler import HandlerBase
from sanitize_ml_labels import sanitize_ml_labels

from ..transformers import GraphTransformer, NodeTransformer


class GraphVisualizations:
    """Tools to visualize the graph embeddings."""

    DEFAULT_SCATTER_KWARGS = dict(
        s=1,
        marker=".",
        alpha=0.7
    )

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
        self._method = method

    def tsne(self, X: np.ndarray, **kwargs: Dict) -> np.ndarray:
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

    def _shuffle(self, *args: List[np.ndarray]) -> List[np.ndarray]:
        """Return given arrats shuffled synchronously."""
        # Shuffling points to avoid having artificial clusters
        # caused by positions.
        index = np.arange(args[0].shape[0])
        np.random.shuffle(index)
        return [
            arg[index]
            for arg in args
        ]

    def _clear_axes(self, axes: Axes, title: str):
        """Reset the axes ticks and set the given title."""
        axes.set_xticks([])
        axes.set_xticks([], minor=True)
        axes.set_yticks([])
        axes.set_yticks([], minor=True)
        axes.set_title(title)

    def _set_legend(
        self,
        axes: Axes,
        labels: List[str],
        handles: List[HandlerBase]
    ):
        """Set the legend with the given values and handles transparency.

        Parameters
        ----------------------------
        axes: Axes,
            The axes on which to put the legend.
        labels: List[str],
            Labels to put in the legend.
        handles: List,
            Handles to display in the legend (the curresponding matplotlib
            objects).
        """
        legend = axes.legend(
            handles=handles,
            labels=sanitize_ml_labels(labels),
            loc='best'
        )
        # Setting alpha level in the legend to avoid having a transparent
        # legend scatter dots.
        for legend_handle in legend.legendHandles:
            legend_handle._legmarker.set_alpha(  # pylint: disable=protected-access
                1
            )

    def _to_dense(self, types: List, reverse_mapping: List[str]) -> Tuple:
        """Return types and labels converted to dense space.

        Parameters
        --------------------------
        types: List,
            The list of types (either edge types or node types).
        reverse_mapping: List[str],
            The list for the reverse mapping.

        Returns
        --------------------------
        Tuple with the types and labels.
        """
        dense_map = OrderedDict([
            (t, i)
            for i, (t, _) in enumerate(Counter(types).most_common())
        ])
        types = np.array([
            dense_map[edge_type]
            for edge_type in types
        ])
        labels = [reverse_mapping[key] for key in dense_map]
        return types, labels

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

    def plot_nodes(
        self,
        figure: Figure = None,
        axes: Axes = None,
        scatter_kwargs: Dict = None,
        **kwargs: Dict
    ):
        """Plot nodes of provided graph.

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
            scatter_kwargs = GraphVisualizations.DEFAULT_SCATTER_KWARGS

        axes.scatter(*self._node_embedding.T, **scatter_kwargs)
        self._clear_axes(axes, "Nodes embedding")
        return figure, axes

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
            scatter_kwargs = GraphVisualizations.DEFAULT_SCATTER_KWARGS

        nodes, node_types = graph.get_top_k_nodes_by_node_type(k)
        node_embeddding = self._node_embedding[nodes]
        node_types, labels = self._to_dense(
            node_types,
            graph.node_types_reverse_mapping
        )
        node_embeddding, node_types = self._shuffle(
            node_embeddding,
            node_types
        )

        scatter = axes.scatter(
            *node_embeddding.T,
            c=node_types,
            cmap=plt.get_cmap("tab10"),
            **scatter_kwargs,
        )

        self._set_legend(
            axes,
            labels,
            scatter.legend_elements()[0]
        )
        self._clear_axes(axes, "Nodes types")
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
            scatter_kwargs = GraphVisualizations.DEFAULT_SCATTER_KWARGS

        degrees = graph.degrees()
        two_median = np.median(degrees)*3
        degrees[degrees > two_median] = min(two_median, degrees.max())

        node_embeddding, degrees = self._shuffle(self._node_embedding, degrees)

        scatter = axes.scatter(
            *node_embeddding.T,
            c=degrees,
            cmap=plt.cm.get_cmap('RdYlBu'),
            **scatter_kwargs,
        )
        color_bar = figure.colorbar(scatter, ax=axes)
        color_bar.set_alpha(1)
        color_bar.draw_all()
        self._clear_axes(axes, "Nodes degrees")
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
            scatter_kwargs = GraphVisualizations.DEFAULT_SCATTER_KWARGS

        edges, edge_types = graph.get_top_k_edges_by_edge_type(k)
        edge_tsne = self._edge_embedding[edges]
        edge_types, labels = self._to_dense(
            edge_types,
            graph.edge_types_reverse_mapping
        )
        edge_tsne, edge_types = self._shuffle(edge_tsne, edge_types)

        scatter = axes.scatter(
            *edge_tsne.T,
            c=edge_types,
            cmap=plt.get_cmap("tab10"),
            **scatter_kwargs,
        )

        self._set_legend(
            axes,
            labels,
            scatter.legend_elements()[0]
        )
        self._clear_axes(axes, "Edge types")
        return figure, axes

    def plot_edges(
        self,
        figure: Figure = None,
        axes: Axes = None,
        scatter_kwargs: Dict = None,
        **kwargs: Dict
    ):
        """Plot edge embedding of provided graph.

        Parameters
        ------------------------------
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
            scatter_kwargs = GraphVisualizations.DEFAULT_SCATTER_KWARGS

        axes.scatter(*self._edge_embedding.T, **scatter_kwargs)
        self._clear_axes(
            axes,
            "Edge embeddings with method {method}".format(
                method=self._method
            )
        )
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
            scatter_kwargs = GraphVisualizations.DEFAULT_SCATTER_KWARGS

        edge_embedding, weights = self._shuffle(
            self._edge_embedding,
            graph.weights
        )

        scatter = axes.scatter(
            *edge_embedding.T,
            c=weights,
            cmap=plt.cm.get_cmap('RdYlBu'),
            **scatter_kwargs,
        )
        color_bar = figure.colorbar(scatter, ax=axes)
        color_bar.set_alpha(1)
        color_bar.draw_all()
        self._clear_axes(axes, "Edge weights")
        return figure, axes
