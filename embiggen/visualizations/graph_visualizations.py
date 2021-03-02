"""Module with embedding visualization tools."""
from multiprocessing import cpu_count
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ensmallen_graph import EnsmallenGraph  # pylint: disable=no-name-in-module
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend_handler import HandlerBase
from sanitize_ml_labels import sanitize_ml_labels
from sklearn.decomposition import PCA

from ..transformers import GraphTransformer, NodeTransformer


class GraphVisualization:
    """Tools to visualize the graph embeddings."""

    DEFAULT_SCATTER_KWARGS = dict(
        s=1,
        marker=".",
        alpha=0.7,
    )
    DEFAULT_SUBPLOT_KWARGS = dict(
        figsize=(10, 10),
        dpi=200
    )

    def __init__(
        self,
        graph: EnsmallenGraph,
        decomposition_method: str = "TSNE",
        n_components: int = 2,
        method: str = "Hadamard"
    ):
        """Create new GraphVisualization object.

        Parameters
        --------------------------
        graph: EnsmallenGraph,
            The graph to visualize.
        decomposition_method: str = "TSNE",
            The decomposition method to use.
            The supported methods are TSNE and PCA.
        n_components: int = 2,
            Number of components to reduce the image to.
            Currently, we only support 2D decompositions but we plan
            to add support for also 3D decompositions.
        method: str = "Hadamard",
            Edge embedding method.
            Can either be 'Hadamard', 'Sum', 'Average', 'L1', 'AbsoluteL1', 'L2' or 'Concatenate'.

        Raises
        ---------------------------
        ValueError,
            If the target decomposition size is not supported.
        ModuleNotFoundError,
            If TSNE decomposition has been required and no module supporting
            it is installed.
        """
        self._graph = graph
        self._graph_transformer = GraphTransformer(method=method)
        self._node_transformer = NodeTransformer()
        self._node_mapping = self._node_embedding = self._edge_embedding = None

        if n_components != 2:
            raise ValueError(
                "We currently only support 2D decomposition visualization."
            )

        self._n_components = n_components

        if decomposition_method == "TSNE":
            try:
                # We try to use CUDA tsne if available, but this does not
                # currently support 3D decomposition. If the user has required a
                # 3D decomposition we need to switch to the MulticoreTSNE version.
                if n_components != 2:
                    raise ModuleNotFoundError()
                from tsnecuda import TSNE as CUDATSNE  # pylint: disable=import-error,import-outside-toplevel
                self._decomposition_method = CUDATSNE(
                    n_components=2,
                    verbose=True
                )
            except ModuleNotFoundError:
                try:
                    from MulticoreTSNE import MulticoreTSNE  # pylint: disable=import-outside-toplevel
                    self._decomposition_method = MulticoreTSNE(
                        n_components=n_components,
                        n_jobs=cpu_count(),
                        verbose=True
                    )
                except ModuleNotFoundError:
                    try:
                        from sklearn.manifold import TSNE  # pylint: disable=import-outside-toplevel
                        self._decomposition_method = TSNE(
                            n_components=n_components,
                            n_jobs=cpu_count(),
                            verbose=True
                        )
                    except:
                        raise ModuleNotFoundError(
                            "You do not have installed a supported TSNE "
                            "decomposition algorithm. Depending on your use case, "
                            "we suggest you install tsne-cuda if your graph is "
                            "very big (in the millions of nodes) if you have access "
                            "to a compatible GPU system.\n"
                            "Alternatively, we suggest (and support) MulticoreTSNE, "
                            "which tends to be easier to install, and is significantly "
                            "faster than the Sklearn implementation.\n"
                            "Alternatively, we suggest (and support) MulticoreTSNE, "
                            "which tends to be easier to install, and is significantly "
                            "faster than the Sklearn implementation.\n"
                            "If you intend to do 3D decompositions, "
                            "remember that tsne-cuda, at the time of writing, "
                            "does not support them."
                        )
        elif decomposition_method == "PCA":
            self._decomposition_method = PCA(
                n_components=n_components
            )
        else:
            raise ValueError(
                "We currently only support PCA and TSNE decomposition methods."
            )

    def decompose(self, X: np.ndarray) -> np.ndarray:
        """Return requested decomposition of given array.

        Parameters
        -----------------------
        X: np.ndarray,
            The data to embed.

        Raises
        -----------------------
        ValueError,
            If the given vector has less components than the required
            decomposition target.

        Returns
        -----------------------
        The obtained decomposition.
        """
        if X.shape[1] == self._n_components:
            return X
        if X.shape[1] < self._n_components:
            raise ValueError(
                "The vector to decompose has less components than "
                "the decomposition target."
            )
        return self._decomposition_method.fit_transform(X)

    def _shuffle(self, *args: List[np.ndarray]) -> List[np.ndarray]:
        """Return given arrays shuffled synchronously.

        The reason to shuffle the points is mainly that this avoids for
        'fake' clusters to appear simply by stacking the points by class
        artifically according to how the points are sorted.
        """
        index = np.arange(args[0].shape[0])
        np.random.shuffle(index)
        return [arg[index] for arg in args]

    def _clear_axes(
        self,
        figure: Figure,
        axes: Axes,
        title: str
    ):
        """Reset the axes ticks and set the given title."""
        axes.set_xticks([])
        axes.set_xticks([], minor=True)
        axes.set_yticks([])
        axes.set_yticks([], minor=True)
        axes.set_title(title)
        figure.tight_layout()

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

    def fit_transform_nodes(self, embedding: pd.DataFrame):
        """Executes fitting for plotting node embeddings.

        Parameters
        -------------------------
        embedding: pd.DataFrame,
            Embedding of the graph nodes.
        """
        self._node_transformer.fit(embedding)
        self._node_embedding = self.decompose(
            self._node_transformer.transform(self._graph.get_node_names())
        )

    def fit_transform_edges(self, embedding: np.ndarray):
        """Executes fitting for plotting edge embeddings.

        Parameters
        -------------------------
        embedding: np.ndarray,
            Embedding obtained from SkipGram, CBOW or GloVe.
        """
        self._graph_transformer.fit(embedding)
        self._edge_embedding = self.decompose(
            self._graph_transformer.transform(self._graph),
        )

    def plot_nodes(
        self,
        graph: EnsmallenGraph,
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
            figure, axes = plt.subplots(
                **(kwargs if kwargs else GraphVisualization.DEFAULT_SUBPLOT_KWARGS))

        if scatter_kwargs is None:
            scatter_kwargs = GraphVisualization.DEFAULT_SCATTER_KWARGS

        axes.scatter(*self._node_embedding.T, **scatter_kwargs)
        self._clear_axes(figure, axes, "Nodes embedding - {}".format(graph))
        return figure, axes

    def plot_node_types(
        self,
        graph: EnsmallenGraph,
        k: int = 10,
        figure: Figure = None,
        axes: Axes = None,
        scatter_kwargs: Dict = None,
        other_label: str = "Other",
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
        other_label: str = "Other",
            Label to use for edges below the top k threshold.
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
            figure, axes = plt.subplots(
                **(kwargs if kwargs else GraphVisualization.DEFAULT_SUBPLOT_KWARGS)
            )

        if scatter_kwargs is None:
            scatter_kwargs = GraphVisualization.DEFAULT_SCATTER_KWARGS

        top_node_types = set(list(zip(*sorted(
            graph.get_node_type_counts().items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]))[0])

        node_types = graph.get_node_types()
        node_labels = graph.get_node_type_names()

        for i, node_type in enumerate(node_types):
            if node_type not in top_node_types:
                node_types[i] = len(top_node_types)

        for node_type in range(graph.get_node_types_number()):
            if node_type not in top_node_types:
                node_labels[node_type] = other_label

        node_embeddding, node_types = self._shuffle(
            self._node_embedding,
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
            node_labels,
            scatter.legend_elements()[0]
        )
        self._clear_axes(
            figure, axes, "Nodes types - {}".format(graph.get_name()))
        return figure, axes

    def plot_node_degrees(
        self,
        graph: EnsmallenGraph,
        figure: Figure = None,
        axes: Axes = None,
        scatter_kwargs: Dict = None,
        **kwargs: Dict
    ):
        """Plot node degrees heatmap.

        Parameters
        ------------------------------
        graph: EnsmallenGraph,
            The graph to visualize.
        figure: Figure = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Axes = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Dict = None,
            Kwargs to pass to the scatter plot call.
        **kwargs: Dict,
            Additional kwargs for the subplots.

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
            figure, axes = plt.subplots(
                **(kwargs if kwargs else GraphVisualization.DEFAULT_SUBPLOT_KWARGS))

        if scatter_kwargs is None:
            scatter_kwargs = GraphVisualization.DEFAULT_SCATTER_KWARGS

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
        self._clear_axes(
            figure, axes, "Nodes degrees - {}".format(graph.get_name()))
        return figure, axes

    def plot_edge_types(
        self,
        graph: EnsmallenGraph,
        k: int = 10,
        figure: Figure = None,
        axes: Axes = None,
        scatter_kwargs: Dict = None,
        other_label: str = "Other",
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
        other_label: str = "Other",
            Label to use for edges below the top k threshold.
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
            figure, axes = plt.subplots(
                **(kwargs if kwargs else GraphVisualization.DEFAULT_SUBPLOT_KWARGS))

        if scatter_kwargs is None:
            scatter_kwargs = GraphVisualization.DEFAULT_SCATTER_KWARGS

        top_edge_types = set(list(zip(*sorted(
            graph.get_edge_type_counts().items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]))[0])

        edge_types = graph.get_edge_types()
        edge_labels = graph.get_edge_type_names()

        for i, edge_type in enumerate(edge_types):
            if edge_type not in top_edge_types:
                edge_types[i] = len(top_edge_types)

        for edge_type in range(graph.get_edge_types_number()):
            if edge_type not in top_edge_types:
                edge_labels[edge_type] = other_label

        edge_tsne, edge_types = self._shuffle(self._edge_embedding, edge_types)

        scatter = axes.scatter(
            *edge_tsne.T,
            c=edge_types,
            cmap=plt.get_cmap("tab10"),
            **scatter_kwargs,
        )

        self._set_legend(
            axes,
            edge_labels,
            scatter.legend_elements()[0]
        )
        self._clear_axes(
            figure, axes, "Edge types - {}".format(graph.get_name()))
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
            figure, axes = plt.subplots(
                **(kwargs if kwargs else GraphVisualization.DEFAULT_SUBPLOT_KWARGS))

        if scatter_kwargs is None:
            scatter_kwargs = GraphVisualization.DEFAULT_SCATTER_KWARGS

        axes.scatter(*self._edge_embedding.T, **scatter_kwargs)
        self._clear_axes(
            figure,
            axes,
            "Edge embeddings with method {method}".format(
                method=self._graph_transformer.method
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
            figure, axes = plt.subplots(
                **(kwargs if kwargs else GraphVisualization.DEFAULT_SUBPLOT_KWARGS))

        if scatter_kwargs is None:
            scatter_kwargs = GraphVisualization.DEFAULT_SCATTER_KWARGS

        edge_embedding, weights = self._shuffle(
            self._edge_embedding,
            graph.get_weights()
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
        self._clear_axes(
            figure, axes, "Edge weights - {}".format(graph.get_name()))
        return figure, axes
