"""Module with embedding visualization tools."""
from multiprocessing import cpu_count
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ensmallen_graph import EnsmallenGraph  # pylint: disable=no-name-in-module
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend_handler import HandlerBase
from matplotlib.colors import LogNorm
from sanitize_ml_labels import sanitize_ml_labels
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

from ..transformers import GraphTransformer, NodeTransformer


class GraphVisualization:
    """Tools to visualize the graph embeddings."""

    DEFAULT_SCATTER_KWARGS = dict(
        s=3,
        marker=".",
        alpha=0.9,
    )
    DEFAULT_SUBPLOT_KWARGS = dict(
        figsize=(10, 10),
        dpi=120
    )

    def __init__(
        self,
        graph: EnsmallenGraph,
        decomposition_method: str = "TSNE",
        n_components: int = 2,
        node_embedding_method: str = None,
        edge_embedding_method: str = "Hadamard",
        subsample_points: int = 50_000
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
        node_embedding_method: str = None,
            Name of the node embedding method used.
            If provided, it is added to the images titles.
        edge_embedding_method: str = "Hadamard",
            Edge embedding method.
            Can either be 'Hadamard', 'Sum', 'Average', 'L1', 'AbsoluteL1', 'L2' or 'Concatenate'.
        subsample_points: int = 50_000,
            Number of points to subsample.
            Some graphs have a number of nodes and edges in the millions.
            Using non-CUDA versions of TSNE, the dimensionality reduction
            procedure can take a considerable amount of time.
            For this porpose, we include the possibility to subsample the
            points to the given number.
            The subsampling is done in a way that takes into consideration
            the node types and/or edge types (the subsampling is applied
            separately to the two different sets) by using a Stratified Shuffle
            Split if there are node types or edge types.
            Otherwise, a normal train test split is used.

        Raises
        ---------------------------
        ValueError,
            If the target decomposition size is not supported.
        ModuleNotFoundError,
            If TSNE decomposition has been required and no module supporting
            it is installed.
        """
        self._graph = graph
        self._graph_transformer = GraphTransformer(
            method=edge_embedding_method
        )
        self._node_transformer = NodeTransformer()
        self._node_embedding_method = node_embedding_method
        self._node_mapping = self._node_embedding = self._edge_embedding = None
        self._subsampled_node_ids = None
        self._subsampled_edge_ids = None
        self._subsample_points = subsample_points

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

    def _shuffle(self, *args: List[Union[np.ndarray, pd.DataFrame, None]]) -> List[np.ndarray]:
        """Return given arrays shuffled synchronously.

        The reason to shuffle the points is mainly that this avoids for
        'fake' clusters to appear simply by stacking the points by class
        artifically according to how the points are sorted.
        """
        index = np.arange(args[0].shape[0])
        np.random.shuffle(index)
        return [
            arg[index] if isinstance(arg, np.ndarray)
            else arg.iloc[index] if isinstance(arg, pd.DataFrame)
            else None
            for arg in args
        ]

    def _clear_axes(
        self,
        figure: Figure,
        axes: Axes,
        title: str
    ):
        """Reset the axes ticks and set the given title."""
        axes.set_axis_off()
        if self._node_embedding_method is not None:
            title = "{} - {}".format(
                title,
                self._node_embedding_method
            )
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
        # Retrieve the nodes
        node_names = np.array(self._graph.get_node_names())
        # If necessary, we proceed with the subsampling
        if self._graph.get_nodes_number() > self._subsample_points:
            # If there are node types, we use a stratified
            # node sampling so that all the nodes types may be displayed.
            if self._graph.has_node_types():
                Splitter = StratifiedShuffleSplit
            else:
                # Otherwise there is no need to stratify.
                Splitter = ShuffleSplit
            # We compute the indices
            self._subsampled_node_ids, _ = next(Splitter(
                n_splits=1,
                train_size=self._subsample_points
            ).split(node_names, self._graph.get_node_types()))
            # And sample the nodes
            node_names = node_names[self._subsampled_node_ids]

        self._node_transformer.fit(embedding)
        self._node_embedding = pd.DataFrame(
            self.decompose(
                self._node_transformer.transform(node_names)
            ),
            index=node_names
        )

    def fit_transform_edges(self, embedding: np.ndarray):
        """Executes fitting for plotting edge embeddings.

        Parameters
        -------------------------
        embedding: np.ndarray,
            Embedding obtained from SkipGram, CBOW or GloVe.
        """
        # Retrieve the edges
        edge_names = np.array(self._graph.get_edge_names())
        # If necessary, we proceed with the subsampling
        if self._graph.get_edges_number() > self._subsample_points:
            # If there are edge types, we use a stratified
            # edge sampling so that all the edges types may be displayed.
            if self._graph.has_edge_types():
                Splitter = StratifiedShuffleSplit
            else:
                # Otherwise there is no need to stratify.
                Splitter = ShuffleSplit
            # We compute the indices
            self._subsampled_edge_ids, _ = next(Splitter(
                n_splits=1,
                train_size=self._subsample_points
            ).split(edge_names, self._graph.get_edge_types()))
            # And sample the edges
            edge_names = edge_names[self._subsampled_edge_ids]

        self._graph_transformer.fit(embedding)
        self._edge_embedding = pd.DataFrame(
            self.decompose(
                self._graph_transformer.transform(edge_names),
            ),
            index=edge_names
        )

    def _plot_scatter(
        self,
        title: str,
        points: np.ndarray,
        colors: List[int] = None,
        labels: List[str] = None,
        figure: Figure = None,
        axes: Axes = None,
        scatter_kwargs: Dict = None,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """Plot nodes of provided graph.

        Parameters
        ------------------------------
        title: str,
            Title to use for the plot.
        points: np.ndarray,
            Points to plot.
        colors: List[int] = None,
            List of the colors to use for the scatter plot.
        labels: List[str] = None,
            Labels for the different colors.
        figure: Figure = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Axes = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Dict = None,
            Kwargs to pass to the scatter plot call.
        **kwargs: Dict,
            Arguments to pass to the subplots.

        Returns
        ------------------------------
        Figure and Axis of the plot.
        """
        if figure is None or axes is None:
            figure, axes = plt.subplots(
                **(kwargs if kwargs else GraphVisualization.DEFAULT_SUBPLOT_KWARGS)
            )

        scatter_kwargs = {
            **({} if scatter_kwargs is None else scatter_kwargs),
            **GraphVisualization.DEFAULT_SCATTER_KWARGS
        }

        points, colors = self._shuffle(
            points,
            colors
        )

        scatter = axes.scatter(
            *points.T,
            c=colors,
            **scatter_kwargs
        )

        if labels is not None:
            self._set_legend(
                axes,
                labels,
                scatter.legend_elements()[0]
            )

        self._clear_axes(
            figure,
            axes,
            "{title} - {graph_name}".format(
                title=title,
                graph_name=self._graph.get_name()
            )
        )

        return figure, axes, scatter

    def _plot_types(
        self,
        title: str,
        points: np.ndarray,
        types: List[int],
        type_labels: List[str],
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
        title: str,
            Title to use for the plot.
        points: np.ndarray,
            Points to plot.
        types: List[int],
            Types of the provided points.
        type_labels: List[str],
            List of the labels for the provided types.
        k: int = 10,
            Number of node types to visualize.
        figure: Figure = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Axes = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Dict = None,
            Kwargs to pass to the scatter plot call.
        other_label: str = "Other",
            Label to use for edges below the top k threshold.
        **kwargs: Dict,
            Arguments to pass to the subplots.

        Raises
        ------------------------------
        ValueError,
            If edge fitting was not yet executed.
        ValueError,
            If given k is greater than maximum supported value (10).
        ValueError,
            If the number of given type labels does not match the number
            of given type counts.

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

        if not isinstance(types, np.ndarray):
            raise ValueError(
                "Expecting types to be a numpy array."
            )

        number_of_types = len(type_labels)

        if number_of_types != types.max() + 1:
            raise ValueError(
                "Expecting types to be from a dense range."
            )

        counts = np.bincount(types, minlength=number_of_types)
        top_counts = np.argsort(counts)[::-1][:k]

        type_labels = list(type_labels[top_counts])

        if k < number_of_types:
            for i, element_type in enumerate(types):
                if element_type not in top_counts:
                    types[i] = k+1

            type_labels.append(other_label)

        figure, axis, _ = self._plot_scatter(
            title=title,
            points=points,
            colors=types,
            labels=type_labels,
            figure=figure,
            axes=axes,
            scatter_kwargs=scatter_kwargs,
            **kwargs
        )

        return figure, axis

    def plot_nodes(
        self,
        figure: Figure = None,
        axes: Axes = None,
        scatter_kwargs: Dict = None,
        **kwargs: Dict
    ) -> Tuple[Figure, Axes]:
        """Plot nodes of provided graph.

        Parameters
        ------------------------------
        figure: Figure = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Axes = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Dict = None,
            Kwargs to pass to the scatter plot call.
        **kwargs: Dict,
            Arguments to pass to the subplots.

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

        figure, axis, _ = self._plot_scatter(
            "Nodes embedding",
            self._node_embedding,
            figure=figure,
            axes=axes,
            scatter_kwargs=scatter_kwargs,
            **kwargs
        )

        return figure, axis

    def plot_edges(
        self,
        figure: Figure = None,
        axes: Axes = None,
        scatter_kwargs: Dict = None,
        **kwargs: Dict
    ) -> Tuple[Figure, Axes]:
        """Plot edge embedding of provided graph.

        Parameters
        ------------------------------
        figure: Figure = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Axes = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Dict = None,
            Kwargs to pass to the scatter plot call.
        **kwargs: Dict,
            Arguments to pass to the subplots.

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

        figure, axis, _ = self._plot_scatter(
            "Edges embedding",
            self._edge_embedding,
            figure=figure,
            axes=axes,
            scatter_kwargs=scatter_kwargs,
            **kwargs
        )

        return figure, axis

    def plot_node_types(
        self,
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
        if not self._graph.has_node_types():
            raise ValueError(
                "The graph does not have node types!"
            )

        if self._node_embedding is None:
            raise ValueError(
                "Node fitting must be executed before plot."
            )

        node_types = self._graph.get_node_types()
        if self._subsampled_node_ids is not None:
            node_types = node_types[self._subsampled_node_ids]

        return self._plot_types(
            "Node types",
            self._node_embedding.values,
            types=node_types,
            type_labels=np.array(self._graph.get_node_type_names()),
            k=k,
            figure=figure,
            axes=axes,
            scatter_kwargs=scatter_kwargs,
            other_label=other_label,
            **kwargs
        )

    def plot_node_degrees(
        self,
        figure: Figure = None,
        axes: Axes = None,
        scatter_kwargs: Dict = None,
        **kwargs: Dict
    ):
        """Plot node degrees heatmap.

        Parameters
        ------------------------------
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

        degrees = self._graph.degrees()
        if self._subsampled_node_ids is not None:
            degrees = degrees[self._subsampled_node_ids]

        figure, axes, scatter = self._plot_scatter(
            "Node degrees",
            self._node_embedding.values,
            colors=degrees,
            figure=figure,
            axes=axes,
            scatter_kwargs={
                **({} if scatter_kwargs is None else scatter_kwargs),
                "cmap": plt.cm.get_cmap('RdYlBu'),
                "norm": LogNorm()
            },
            **kwargs
        )

        color_bar = figure.colorbar(scatter, ax=axes)
        color_bar.set_alpha(1)
        color_bar.draw_all()
        return figure, axes

    def plot_edge_types(
        self,
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
            If the graph does not have edge types.
        ValueError,
            If edge fitting was not yet executed.
        ValueError,
            If given k is greater than maximum supported value (10).

        Returns
        ------------------------------
        Figure and Axis of the plot.
        """
        if not self._graph.has_edge_types():
            raise ValueError(
                "The graph does not have edge types!"
            )

        if self._edge_embedding is None:
            raise ValueError(
                "Edge fitting was not yet executed!"
            )

        edge_types = self._graph.get_edge_types()
        if self._subsampled_node_ids is not None:
            edge_types = edge_types[self._subsampled_edge_ids]

        return self._plot_types(
            "Edge types",
            self._edge_embedding.values,
            types=edge_types,
            type_labels=np.array(self._graph.get_edge_type_names()),
            k=k,
            figure=figure,
            axes=axes,
            scatter_kwargs=scatter_kwargs,
            other_label=other_label,
            **kwargs
        )

    def plot_edge_weights(
        self,
        figure: Figure = None,
        axes: Axes = None,
        scatter_kwargs: Dict = None,
        **kwargs: Dict
    ):
        """Plot common edge types of provided graph.

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
        if not self._graph.has_weights():
            raise ValueError(
                "The graph does not have edge weights!"
            )

        if self._edge_embedding is None:
            raise ValueError(
                "Edge fitting must be executed before plot."
            )

        weights = self._graph.get_weights()
        if self._subsampled_node_ids is not None:
            weights = weights[self._subsampled_node_ids]

        figure, axes, scatter = self._plot_scatter(
            "Edge weights",
            self._node_embedding.values,
            colors=weights,
            figure=figure,
            axes=axes,
            scatter_kwargs={
                **({} if scatter_kwargs is None else scatter_kwargs),
                "cmap": plt.cm.get_cmap('RdYlBu')
            },
            **kwargs
        )

        color_bar = figure.colorbar(scatter, ax=axes)
        color_bar.set_alpha(1)
        color_bar.draw_all()
        return figure, axes
