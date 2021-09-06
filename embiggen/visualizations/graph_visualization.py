"""Module with embedding visualization tools."""
from multiprocessing import cpu_count
from typing import Dict, List, Tuple, Union, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ddd_subplots import subplots as subplots_3d
from ensmallen import Graph  # pylint: disable=no-name-in-module
from matplotlib.axes import Axes
from matplotlib.collections import Collection
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.figure import Figure
from matplotlib.legend_handler import HandlerBase, HandlerTuple
from sanitize_ml_labels import sanitize_ml_labels
from sklearn.decomposition import PCA
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler

from ..transformers import GraphTransformer, NodeTransformer


class GraphVisualization:
    """Tools to visualize the graph embeddings."""

    DEFAULT_SCATTER_KWARGS = dict(
        s=5,
        alpha=0.7
    )
    DEFAULT_SUBPLOT_KWARGS = dict(
        figsize=(7, 7),
        dpi=200
    )

    def __init__(
        self,
        graph: Graph,
        decomposition_method: str = "TSNE",
        scaler_method: "Scaler" = RobustScaler,
        n_components: int = 2,
        node_embedding_method: str = None,
        edge_embedding_method: str = "Hadamard",
        subsample_points: int = 20_000,
        random_state: int = 42,
        decomposition_kwargs: Dict = None
    ):
        """Create new GraphVisualization object.

        Parameters
        --------------------------
        graph: Graph,
            The graph to visualize.
        decomposition_method: str = "TSNE",
            The decomposition method to use.
            The supported methods are TSNE and PCA.
        scaler_method: "Scaler" = RobustScaler,
            The scaler object to use to normalize the embedding.
            By default we use a Robust Scaler.
            Pass None to not use any scaler.
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
        subsample_points: int = 20_000,
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
            If None is given, no subsampling is executed.
        random_state: int = 42,
            The random state to reproduce the visualizations.
        decomposition_kwargs: Dict = None,
            Kwargs to forward to the selected decomposition method.

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
        self._random_state = random_state

        if decomposition_kwargs is None:
            decomposition_kwargs = {}

        if n_components not in {2, 3}:
            raise ValueError(
                "We currently only support 2D and 3D decomposition visualization."
            )

        self._n_components = n_components
        self._scaler_method = None if scaler_method is None else scaler_method()

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
                    random_seed=random_state,
                    verbose=True,
                    **decomposition_kwargs
                )
            except ModuleNotFoundError:
                try:
                    from MulticoreTSNE import \
                        MulticoreTSNE  # pylint: disable=import-outside-toplevel
                    self._decomposition_method = MulticoreTSNE(
                        n_components=n_components,
                        n_jobs=cpu_count(),
                        random_state=random_state,
                        verbose=True,
                        **decomposition_kwargs
                    )
                except ModuleNotFoundError:
                    try:
                        from sklearn.manifold import \
                            TSNE  # pylint: disable=import-outside-toplevel
                        self._decomposition_method = TSNE(
                            n_components=n_components,
                            n_jobs=cpu_count(),
                            random_state=random_state,
                            verbose=True,
                            **decomposition_kwargs
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
                n_components=n_components,
                random_state=random_state,
                **decomposition_kwargs
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

    def _shuffle(
        self,
        *args: List[Union[np.ndarray, pd.DataFrame, None]]
    ) -> List[np.ndarray]:
        """Return given arrays shuffled synchronously.

        The reason to shuffle the points is mainly that this avoids for
        'fake' clusters to appear simply by stacking the points by class
        artifically according to how the points are sorted.

        Parameters
        ------------------------
        *args: List[Union[np.ndarray, pd.DataFrame, None]],
            The lists to shuffle.

        Returns
        ------------------------
        Shuffled data using given random state.
        """
        index = np.arange(args[0].shape[0])
        random_state = np.random.RandomState(  # pylint: disable=no-member
            seed=self._random_state
        )
        random_state.shuffle(index)
        return [
            arg[index] if isinstance(arg, np.ndarray)
            else arg.iloc[index] if isinstance(arg, pd.DataFrame)
            else None
            for arg in args
        ]

    def _set_legend(
        self,
        axes: Axes,
        labels: List[str],
        handles: List[HandlerBase],
        legend_title: str
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
        legend_title: str,
            Title for the legend.
        """
        legend = axes.legend(
            handles=handles,
            labels=sanitize_ml_labels(labels),
            loc='best',
            title=legend_title,
            **(
                dict(handler_map={tuple: HandlerTuple(ndivide=None)})
                if isinstance(handles[0], tuple)
                else {}
            )
        )
        # Setting alpha level in the legend to avoid having a transparent
        # legend scatter dots.
        for legend_handle in legend.legendHandles:
            legend_handle._legmarker.set_alpha(  # pylint: disable=protected-access
                1
            )

    def fit_transform_nodes(
        self,
        node_embedding: pd.DataFrame
    ):
        """Executes fitting for plotting node embeddings.

        Parameters
        -------------------------
        node_embedding: pd.DataFrame,
            Embedding of the graph nodes.
        """
        # Retrieve the nodes
        node_names = np.array(self._graph.get_node_names())
        # If necessary, we proceed with the subsampling
        if self._subsample_points is not None and self._graph.get_nodes_number() > self._subsample_points:
            # If there are node types, we use a stratified
            # node sampling so that all the nodes types may be displayed.
            if self._graph.has_node_types() and not self._graph.has_singleton_node_types():
                Splitter = StratifiedShuffleSplit
            else:
                # Otherwise there is no need to stratify.
                Splitter = ShuffleSplit
            # We compute the indices
            self._subsampled_node_ids, _ = next(Splitter(
                n_splits=1,
                train_size=self._subsample_points,
                random_state=self._random_state
            ).split(node_names, self._flatten_multi_label_and_unknown_node_types()))
            # And sample the nodes
            node_names = node_names[self._subsampled_node_ids]

        if self._scaler_method is not None:
            node_embedding = pd.DataFrame(
                self._scaler_method.fit_transform(node_embedding),
                columns=node_embedding.columns,
                index=node_embedding.index,
            )
        self._node_transformer.fit(node_embedding)
        self._node_embedding = pd.DataFrame(
            self.decompose(
                self._node_transformer.transform(node_names)
            ),
            index=node_names
        )

    def fit_transform_edges(
        self,
        node_embedding: Optional[pd.DataFrame] = None,
        edge_embedding: Optional[pd.DataFrame] = None,
    ):
        """Executes fitting for plotting edge embeddings.

        Parameters
        -------------------------
        node_embedding: Optional[pd.DataFrame] = None,
            Node embedding obtained from SkipGram, CBOW or GloVe or others.
        node_embedding: Optional[pd.DataFrame] = None,
            Edge embedding.

        Raises
        -------------------------
        ValueError,
            If neither the node embedding nor the edge embedding have
            been provided. You need to provide exactly one of the two.
        ValueError,
            If the shape of the given node embedding does not match
            the number of nodes in the graph.
        ValueError,
            If the shape of the given node embedding does not match
            the number of edges in the graph.   
        """
        if node_embedding is None and edge_embedding is None:
            raise ValueError(
                "You need to provide either the node embedding or the "
                "edge embedding."
            )
        if node_embedding is not None and edge_embedding is not None:
            raise ValueError(
                "You need to provide either the node embedding or the "
                "edge embedding. You cannot provide both at once."
            )
        if node_embedding is not None and node_embedding.shape[0] != self._graph.get_nodes_number():
            raise ValueError(
                ("The number of rows provided with the given node embedding {} "
                 "does not match the number of nodes in the graph {}.").format(
                    node_embedding.shape[0],
                    self._graph.get_nodes_number()
                )
            )
        if edge_embedding is not None and edge_embedding.shape[0] != self._graph.get_directed_edges_number():
            raise ValueError(
                ("The number of rows provided with the given edge embedding {} "
                 "does not match the number of directed edges in the graph {}.").format(
                    edge_embedding.shape[0],
                    self._graph.get_directed_edges_number()
                )
            )

        # Retrieve the edges
        edge_names = np.array(self._graph.get_edge_node_names(directed=True))
        # If necessary, we proceed with the subsampling
        if self._subsample_points is not None and len(edge_names) > self._subsample_points:
            # If there are edge types, we use a stratified
            # edge sampling so that all the edges types may be displayed.
            if self._graph.has_edge_types() and not self._graph.has_singleton_edge_types():
                Splitter = StratifiedShuffleSplit
            else:
                # Otherwise there is no need to stratify.
                Splitter = ShuffleSplit
            # We compute the indices
            self._subsampled_edge_ids, _ = next(Splitter(
                n_splits=1,
                train_size=self._subsample_points,
                random_state=self._random_state
            ).split(edge_names, self._flatten_unknown_edge_types()))
            # And sample the edges
            edge_names = edge_names[self._subsampled_edge_ids]
            if edge_embedding is not None:
                edge_embedding = edge_embedding[self._subsampled_edge_ids]

        if node_embedding is not None:
            if self._scaler_method is not None:
                node_embedding = pd.DataFrame(
                    self._scaler_method.fit_transform(node_embedding),
                    columns=node_embedding.columns,
                    index=node_embedding.index,
                )
            self._graph_transformer.fit(node_embedding)
            edge_embedding = self._graph_transformer.transform(edge_names)
        self._edge_embedding = pd.DataFrame(
            self.decompose(edge_embedding),
            index=edge_names
        )

    def _plot_scatter(
        self,
        title: str,
        points: np.ndarray,
        colors: List[int] = None,
        edgecolors: List[int] = None,
        labels: List[str] = None,
        legend_title: str = "",
        show_title: bool = True,
        show_legend: bool = True,
        figure: Figure = None,
        axes: Axes = None,
        scatter_kwargs: Dict = None,
        train_indices: np.ndarray = None,
        test_indices: np.ndarray = None,
        train_marker: str = "o",
        test_marker: str = "X",
        **kwargs
    ) -> Tuple[Figure, Axes, Tuple[Collection]]:
        """Plot nodes of provided graph.

        Parameters
        ------------------------------
        title: str,
            Title to use for the plot.
        points: np.ndarray,
            Points to plot.
        colors: List[int] = None,
            List of the colors to use for the scatter plot.
        edgecolors: List[int] = None,
            List of the edge colors to use for the scatter plot.
        labels: List[str] = None,
            Labels for the different colors.
        legend_title: str = "",
            Title for the legend.
        show_title: bool = True,
            Whether to show the figure title.
        show_legend: bool = True,
            Whether to show the legend.
        figure: Figure = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Axes = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Dict = None,
            Kwargs to pass to the scatter plot call.
        train_indices: np.ndarray = None,
            Indices to draw using the training marker.
            If None, all points are drawn using the training marker.
        test_indices: np.ndarray = None,
            Indices to draw using the test marker.
            If None, while providing the train indices, we only plot the
            training points.
        train_marker: str = "o",
            The marker to use to draw the training points.
        test_marker: str = "X",
            The marker to use to draw the test points.
        **kwargs: Dict,
            Arguments to pass to the subplots.

        Raises
        ------------------------------
        ValueError,
            If given train and test indices overlap.

        Returns
        ------------------------------
        Figure and Axis of the plot, followed by tuple of collections.
        """
        if train_indices is not None and test_indices is not None:
            if np.isin(train_indices, test_indices).any():
                raise ValueError(
                    "The train and test indices overlap."
                )

        if figure is None or axes is None:
            if self._n_components == 2:
                figure, axes = plt.subplots(**{
                    **GraphVisualization.DEFAULT_SUBPLOT_KWARGS,
                    **kwargs
                })
            else:
                figure, axes = subplots_3d(**{
                    **GraphVisualization.DEFAULT_SUBPLOT_KWARGS,
                    **kwargs
                })

        scatter_kwargs = {
            **GraphVisualization.DEFAULT_SCATTER_KWARGS,
            **(
                dict(linewidths=0)
                if edgecolors is None
                else dict(linewidths=0.5)
            ),
            **({} if scatter_kwargs is None else scatter_kwargs),
        }

        train_test_mask = np.zeros((points.shape[0]))

        if train_indices is not None:
            train_test_mask[train_indices] = 1

        if test_indices is not None:
            train_test_mask[test_indices] = 2

        points, colors, edgecolors, train_test_mask = self._shuffle(
            points,
            colors,
            edgecolors,
            train_test_mask
        )

        legend_elements = []
        collections = []

        color_names = np.array([
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:gray",
            "tab:olive",
            "tab:cyan",
        ])

        if colors is not None:
            cmap = scatter_kwargs.pop(
                "cmap",
                ListedColormap(color_names[:int(colors.max() + 1)])
            )

        if train_indices is None and test_indices is None:
            scatter = axes.scatter(
                *points.T,
                c=colors,
                edgecolors=None if edgecolors is None else cmap(edgecolors),
                marker=train_marker,
                cmap=cmap,
                **scatter_kwargs
            )
            collections.append(scatter)
            legend_elements += scatter.legend_elements()[0]

        if train_indices is not None:
            train_mask = train_test_mask == 1
            train_scatter = axes.scatter(
                *points[train_mask].T,
                c=colors[train_mask],
                edgecolors=None if edgecolors is None else cmap(
                    edgecolors[train_mask]
                ),
                marker=train_marker,
                cmap=cmap,
                **scatter_kwargs
            )
            collections.append(train_scatter)
            legend_elements.append(train_scatter.legend_elements()[0])

        if test_indices is not None:
            test_mask = train_test_mask == 2
            test_scatter = axes.scatter(
                *points[test_mask].T,
                c=colors[test_mask],
                edgecolors=None if edgecolors is None else cmap(
                    edgecolors[test_mask]),
                marker=test_marker,
                cmap=cmap,
                **scatter_kwargs
            )
            collections.append(test_scatter)
            legend_elements.append(test_scatter.legend_elements()[0])

        rectangle_to_fill_legend = matplotlib.patches.Rectangle(
            (0, 0), 1, 1,
            fill=False,
            edgecolor='none',
            visible=False
        )

        if all(
            e is not None
            for e in (colors, train_indices, test_indices, labels)
        ):
            unique_train_colors = np.unique(colors[train_mask])
            unique_test_colors = np.unique(colors[test_mask])
            new_legend_elements = []
            train_element_index = 0
            test_element_index = 0
            for color in np.unique(colors):
                new_tuple = []
                if color in unique_train_colors:
                    new_tuple.append(legend_elements[0][train_element_index])
                    train_element_index += 1
                else:
                    new_tuple.append(rectangle_to_fill_legend)
                if color in unique_test_colors:
                    new_tuple.append(legend_elements[1][test_element_index])
                    test_element_index += 1
                else:
                    new_tuple.append(rectangle_to_fill_legend)

                new_legend_elements.append(tuple(new_tuple))
            legend_elements = new_legend_elements

        if show_legend and labels is not None:
            self._set_legend(
                axes,
                labels,
                legend_elements,
                legend_title
            )

        if self._n_components == 2:
            axes.set_axis_off()

        title = "{} - {}".format(
            title,
            self._graph.get_name(),
        )

        if self._node_embedding_method is not None:
            title = "{} - {}".format(
                title,
                self._node_embedding_method
            )

        if show_title:
            axes.set_title(title)
        figure.tight_layout()

        return figure, axes, collections

    def _plot_types(
        self,
        title: str,
        points: np.ndarray,
        types: List[int],
        type_labels: List[str],
        legend_title: str,
        show_title: bool = True,
        show_legend: bool = True,
        predictions: List[int] = None,
        k: int = 9,
        figure: Figure = None,
        axes: Axes = None,
        scatter_kwargs: Dict = None,
        other_label: str = "Other",
        train_indices: np.ndarray = None,
        test_indices: np.ndarray = None,
        train_marker: str = "o",
        test_marker: str = "X",
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
        legend_title: str,
            Title for the legend.
        show_title: bool = True,
            Whether to show the figure title.
        show_legend: bool = True,
            Whether to show the legend.
        predictions: List[int] = None,
            List of the labels predicted.
            If None, no prediction is visualized.
        k: int = 9,
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
        train_indices: np.ndarray = None,
            Indices to draw using the training marker.
            If None, all points are drawn using the training marker.
        test_indices: np.ndarray = None,
            Indices to draw using the test marker.
            If None, while providing the train indices,
        train_marker: str = "o",
            The marker to use to draw the training points.
        test_marker: str = "X",
            The marker to use to draw the test points.
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
        if k > 9:
            raise ValueError(
                "Values of k greater than 9 are not supported!"
            )

        # if not isinstance(types, np.ndarray):
        #     raise ValueError(
        #         "Expecting types to be a numpy array."
        #     )
        types = np.array(types)

        number_of_types = np.unique(types).size
        type_labels = np.array(type_labels)

        counts = np.bincount(types, minlength=number_of_types)
        top_counts = [
            index
            for index, _ in sorted(
                enumerate(zip(counts, type_labels)),
                key=lambda x: x[1],
                reverse=True
            )[:k]
        ]

        type_labels = list(type_labels[top_counts])

        for i, element_type in enumerate(types):
            if element_type not in top_counts:
                types[i] = k
            else:
                types[i] = top_counts.index(element_type)

        if predictions is not None:
            predictions = predictions.copy()
            for i, element_type in enumerate(predictions):
                if element_type not in top_counts:
                    predictions[i] = k
                else:
                    predictions[i] = top_counts.index(element_type)

        if k < number_of_types:
            type_labels.append(other_label)

        figure, axis, _ = self._plot_scatter(
            title=title,
            points=points,
            colors=types,
            edgecolors=predictions,
            labels=type_labels,
            legend_title=legend_title,
            show_title=show_title,
            show_legend=show_legend,
            figure=figure,
            axes=axes,
            scatter_kwargs=scatter_kwargs,
            train_indices=train_indices,
            test_indices=test_indices,
            train_marker=train_marker,
            test_marker=test_marker,
            **kwargs
        )

        return figure, axis

    def plot_nodes(
        self,
        figure: Figure = None,
        axes: Axes = None,
        scatter_kwargs: Dict = None,
        train_indices: np.ndarray = None,
        test_indices: np.ndarray = None,
        train_marker: str = "o",
        test_marker: str = "X",
        show_title: bool = True,
        show_legend: bool = True,
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
        train_indices: np.ndarray = None,
            Indices to draw using the training marker.
            If None, all points are drawn using the training marker.
        test_indices: np.ndarray = None,
            Indices to draw using the test marker.
            If None, while providing the train indices,
        train_marker: str = "o",
            The marker to use to draw the training points.
        test_marker: str = "X",
            The marker to use to draw the test points.
        show_title: bool = True,
            Whether to show the figure title.
        show_legend: bool = True,
            Whether to show the legend.
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
            train_indices=train_indices,
            test_indices=test_indices,
            train_marker=train_marker,
            test_marker=test_marker,
            show_title=show_title,
            show_legend=show_legend,
            **kwargs
        )

        return figure, axis

    def plot_edges(
        self,
        figure: Figure = None,
        axes: Axes = None,
        scatter_kwargs: Dict = None,
        train_indices: np.ndarray = None,
        test_indices: np.ndarray = None,
        train_marker: str = "o",
        test_marker: str = "X",
        show_title: bool = True,
        show_legend: bool = True,
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
        train_indices: np.ndarray = None,
            Indices to draw using the training marker.
            If None, all points are drawn using the training marker.
        test_indices: np.ndarray = None,
            Indices to draw using the test marker.
            If None, while providing the train indices,
        train_marker: str = "o",
            The marker to use to draw the training points.
        test_marker: str = "X",
            The marker to use to draw the test points.
        show_title: bool = True,
            Whether to show the figure title.
        show_legend: bool = True,
            Whether to show the legend.
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
            train_indices=train_indices,
            test_indices=test_indices,
            train_marker=train_marker,
            test_marker=test_marker,
            show_title=show_title,
            show_legend=show_legend,
            **kwargs
        )

        return figure, axis

    def _flatten_multi_label_and_unknown_node_types(self) -> np.ndarray:
        # The following is needed to normalize the multiple types
        node_types_counts = self._graph.get_node_type_id_counts_hashmap()
        node_types_number = self._graph.get_node_types_number()
        # When we have multiple node types for a given node, we set it to
        # the most common node type of the set.
        return np.array([
            sorted(
                node_type_ids,
                key=lambda node_type: node_types_counts[node_type],
                reverse=True
            )[0]
            if node_type_ids is not None
            else
            node_types_number
            for node_type_ids in self._graph.get_node_type_ids()
        ])

    def _flatten_unknown_edge_types(self) -> np.ndarray:
        # The following is needed to normalize the multiple types
        edge_types_number = self._graph.get_edge_types_number()
        # When we have multiple node types for a given node, we set it to
        # the most common node type of the set.
        return np.array([
            edge_type_id
            if edge_type_id is not None
            else
            edge_types_number
            for edge_type_id in self._graph.get_edge_type_ids()
        ])

    def plot_node_types(
        self,
        node_type_predictions: List[int] = None,
        k: int = 9,
        figure: Figure = None,
        axes: Axes = None,
        scatter_kwargs: Dict = None,
        legend_title: str = "Node types",
        other_label: str = "Other",
        train_indices: np.ndarray = None,
        test_indices: np.ndarray = None,
        train_marker: str = "o",
        test_marker: str = "X",
        show_title: bool = True,
        show_legend: bool = True,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """Plot common node types of provided graph.

        Parameters
        ------------------------------
        node_type_predictions: List[int] = None,
            Predictions of the node types.
        k: int = 9,
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
        train_indices: np.ndarray = None,
            Indices to draw using the training marker.
            If None, all points are drawn using the training marker.
        test_indices: np.ndarray = None,
            Indices to draw using the test marker.
            If None, while providing the train indices,
        train_marker: str = "o",
            The marker to use to draw the training points.
        test_marker: str = "X",
            The marker to use to draw the test points.
        show_title: bool = True,
            Whether to show the figure title.
        show_legend: bool = True,
            Whether to show the legend.
        **kwargs: Dict,
            Arguments to pass to the subplots.

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

        node_types = self._flatten_multi_label_and_unknown_node_types()
        if self._subsampled_node_ids is not None:
            node_types = node_types[self._subsampled_node_ids]

        node_type_names = self._graph.get_unique_node_type_names()

        if self._graph.has_unknown_node_types():
            node_type_names.append("Unknown")

        node_type_names = np.array(node_type_names)

        return self._plot_types(
            "Node types",
            self._node_embedding.values,
            types=node_types,
            type_labels=node_type_names,
            legend_title=legend_title,
            predictions=node_type_predictions,
            k=k,
            figure=figure,
            axes=axes,
            scatter_kwargs=scatter_kwargs,
            other_label=other_label,
            train_indices=train_indices,
            test_indices=test_indices,
            train_marker=train_marker,
            test_marker=test_marker,
            show_title=show_title,
            show_legend=show_legend,
            **kwargs
        )

    def plot_connected_components(
        self,
        k: int = 9,
        figure: Figure = None,
        axes: Axes = None,
        scatter_kwargs: Dict = None,
        other_label: str = "Other",
        legend_title: str = "Component sizes",
        train_indices: np.ndarray = None,
        test_indices: np.ndarray = None,
        train_marker: str = "o",
        test_marker: str = "X",
        show_title: bool = True,
        show_legend: bool = True,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """Plot common node types of provided graph.

        Parameters
        ------------------------------
        k: int = 9,
            Number of components to visualize.
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
        legend_title: str = "Component sizes",
            Title for the legend.
        train_indices: np.ndarray = None,
            Indices to draw using the training marker.
            If None, all points are drawn using the training marker.
        test_indices: np.ndarray = None,
            Indices to draw using the test marker.
            If None, while providing the train indices,
        train_marker: str = "o",
            The marker to use to draw the training points.
        test_marker: str = "X",
            The marker to use to draw the test points.
        show_title: bool = True,
            Whether to show the figure title.
        show_legend: bool = True,
            Whether to show the legend.
        **kwargs: Dict,
            Arguments to pass to the subplots.

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

        components, components_number, _, _ = self._graph.connected_components()
        sizes = np.bincount(components, minlength=components_number)

        if self._subsampled_node_ids is not None:
            components = components[self._subsampled_node_ids]

        return self._plot_types(
            "Components",
            self._node_embedding.values,
            types=components,
            type_labels=np.array([
                "Size {}".format(size)
                for size in sizes
            ]),
            legend_title=legend_title,
            show_title=show_title,
            show_legend=show_legend,
            k=k,
            figure=figure,
            axes=axes,
            scatter_kwargs=scatter_kwargs,
            other_label=other_label,
            train_indices=train_indices,
            test_indices=test_indices,
            train_marker=train_marker,
            test_marker=test_marker,
            **kwargs
        )

    def plot_node_degrees(
        self,
        figure: Figure = None,
        axes: Axes = None,
        scatter_kwargs: Dict = None,
        train_indices: np.ndarray = None,
        test_indices: np.ndarray = None,
        train_marker: str = "o",
        test_marker: str = "X",
        use_log_scale: bool = True,
        show_title: bool = True,
        show_legend: bool = True,
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
        train_indices: np.ndarray = None,
            Indices to draw using the training marker.
            If None, all points are drawn using the training marker.
        test_indices: np.ndarray = None,
            Indices to draw using the test marker.
            If None, while providing the train indices, 
        train_marker: str = "o",
            The marker to use to draw the training points.
        test_marker: str = "X",
            The marker to use to draw the test points.
        use_log_scale: bool = True,
            Whether to use log scale.
        show_title: bool = True,
            Whether to show the figure title.
        show_legend: bool = True,
            Whether to show the legend.
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

        degrees = self._graph.get_node_degrees()
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
                **({"norm": LogNorm()} if use_log_scale else {})
            },
            train_indices=train_indices,
            test_indices=test_indices,
            train_marker=train_marker,
            test_marker=test_marker,
            show_title=show_title,
            show_legend=show_legend,
            **kwargs
        )

        color_bar = figure.colorbar(scatter[0], ax=axes)
        color_bar.set_alpha(1)
        color_bar.draw_all()
        return figure, axes

    def plot_edge_types(
        self,
        edge_type_predictions: List[int] = None,
        k: int = 9,
        figure: Figure = None,
        axes: Axes = None,
        scatter_kwargs: Dict = None,
        other_label: str = "Other",
        legend_title: str = "Edge types",
        train_indices: np.ndarray = None,
        test_indices: np.ndarray = None,
        train_marker: str = "o",
        test_marker: str = "X",
        show_title: bool = True,
        show_legend: bool = True,
        **kwargs: Dict
    ):
        """Plot common edge types of provided graph.

        Parameters
        ------------------------------
        edge_type_predictions: List[int] = None,
            Predictions of the edge types.
        k: int = 9,
            Number of edge types to visualize.
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
        legend_title: str = "Edge types",
            Title for the legend.
        train_indices: np.ndarray = None,
            Indices to draw using the training marker.
            If None, all points are drawn using the training marker.
        test_indices: np.ndarray = None,
            Indices to draw using the test marker.
            If None, while providing the train indices,
        train_marker: str = "o",
            The marker to use to draw the training points.
        test_marker: str = "X",
            The marker to use to draw the test points.
        show_title: bool = True,
            Whether to show the figure title.
        show_legend: bool = True,
            Whether to show the legend.
        **kwargs: Dict,
            Additional kwargs for the subplots.

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

        edge_type_number = self._graph.get_edge_types_number()
        edge_types = np.array([
            edge_type_id
            if edge_type_id is not None
            else edge_type_number
            for edge_type_id in self._graph.get_edge_type_ids()
        ])

        if self._subsampled_edge_ids is not None:
            edge_types = edge_types[self._subsampled_edge_ids]

        edge_type_names = self._graph.get_unique_edge_type_names()

        if self._graph.has_unknown_edge_types():
            edge_type_names.append("Unknown")

        edge_type_names = np.array(edge_type_names)

        return self._plot_types(
            "Edge types",
            self._edge_embedding.values,
            types=edge_types,
            type_labels=edge_type_names,
            legend_title=legend_title,
            predictions=edge_type_predictions,
            k=k,
            figure=figure,
            axes=axes,
            scatter_kwargs=scatter_kwargs,
            other_label=other_label,
            train_indices=train_indices,
            test_indices=test_indices,
            train_marker=train_marker,
            test_marker=test_marker,
            show_title=show_title,
            show_legend=show_legend,
            **kwargs
        )

    def plot_edge_weights(
        self,
        figure: Figure = None,
        axes: Axes = None,
        scatter_kwargs: Dict = None,
        train_indices: np.ndarray = None,
        test_indices: np.ndarray = None,
        train_marker: str = "o",
        test_marker: str = "X",
        show_title: bool = True,
        show_legend: bool = True,
        **kwargs: Dict
    ):
        """Plot common edge types of provided graph.

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
        train_indices: np.ndarray = None,
            Indices to draw using the training marker.
            If None, all points are drawn using the training marker.
        test_indices: np.ndarray = None,
            Indices to draw using the test marker.
            If None, while providing the train indices, 
        train_marker: str = "o",
            The marker to use to draw the training points.
        test_marker: str = "X",
            The marker to use to draw the test points.
        show_title: bool = True,
            Whether to show the figure title.
        show_legend: bool = True,
            Whether to show the legend.
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
        if not self._graph.has_edge_weights():
            raise ValueError(
                "The graph does not have edge weights!"
            )

        if self._edge_embedding is None:
            raise ValueError(
                "Edge fitting must be executed before plot."
            )

        weights = self._graph.get_edge_weights()
        if self._subsampled_edge_ids is not None:
            weights = weights[self._subsampled_edge_ids]

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
            train_indices=train_indices,
            test_indices=test_indices,
            train_marker=train_marker,
            test_marker=test_marker,
            show_title=show_title,
            show_legend=show_legend,
            **kwargs
        )

        color_bar = figure.colorbar(scatter[0], ax=axes)
        color_bar.set_alpha(1)
        color_bar.draw_all()
        return figure, axes
