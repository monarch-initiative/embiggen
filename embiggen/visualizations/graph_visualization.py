"""Module with embedding visualization tools."""
from multiprocessing import cpu_count
from typing import Dict, List, Tuple, Union, Optional, Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ddd_subplots import subplots as subplots_3d, rotate, display_video_at_path
from ensmallen import Graph  # pylint: disable=no-name-in-module
from matplotlib.collections import Collection
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend_handler import HandlerBase, HandlerTuple
from matplotlib import collections as mc
from pandas.core.frame import DataFrame
from sanitize_ml_labels import sanitize_ml_labels
from sklearn.decomposition import PCA
from tqdm.auto import trange, tqdm
import warnings
import itertools

from ..transformers import GraphTransformer, NodeTransformer


class GraphVisualization:
    """Tools to visualize the graph embeddings."""

    DEFAULT_SCATTER_KWARGS = dict(
        s=5,
        alpha=0.7
    )
    DEFAULT_EDGES_SCATTER_KWARGS = dict(
        alpha=0.5
    )
    DEFAULT_SUBPLOT_KWARGS = dict(
        figsize=(7, 7),
        dpi=200
    )

    def __init__(
        self,
        graph: Graph,
        decomposition_method: str = "TSNE",
        n_components: int = 2,
        rotate: bool = False,
        video_format: str = "webm",
        duration: int = 10,
        fps: int = 24,
        node_embedding_method_name: str = "auto",
        edge_embedding_method: str = "Concatenate",
        number_of_subsampled_nodes: int = 20_000,
        number_of_subsampled_edges: int = 20_000,
        number_of_subsampled_negative_edges: int = 20_000,
        random_state: int = 42,
        decomposition_kwargs: Optional[Dict] = None
    ):
        """Create new GraphVisualization object.

        Parameters
        --------------------------
        graph: Graph,
            The graph to visualize.
        decomposition_method: str = "TSNE",
            The decomposition method to use.
            The supported methods are UMAP, TSNE and PCA.
        n_components: int = 2,
            Number of components to reduce the image to.
            Currently we support 2D, 3D and 4D visualizations.
        rotate: bool = False,
            Whether to create a rotating animation.
        video_format: str = "webm"
            What video format to use for the animations.
        duration: int = 15,
            Duration of the animation in seconds.
        fps: int = 24,
            Number of frames per second in animations.
        node_embedding_method_name: str = "auto",
            Name of the node embedding method used.
            If "auto" is used, then we try to infer the type of
            node embedding algorithm used, which in some cases is
            recognizable automatically.
        edge_embedding_method: str = "Concatenate",
            Edge embedding method.
            Can either be 'Hadamard', 'Sum', 'Average', 'L1', 'AbsoluteL1', 'L2' or 'Concatenate'.
        number_of_subsampled_nodes: int = 20_000,
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
        number_of_subsampled_edges: int = 20_000,
            Number of edges to subsample.
            The same considerations described for the subsampled nodes number
            also apply for the edges number.
            Not subsampling the edges in most graphs is a poor life choice.
        number_of_subsampled_negative_edges: int = 20_000,
            Number of edges to subsample.
            The same considerations described for the subsampled nodes number
            also apply for the edges number.
            Not subsampling the edges in most graphs is a poor life choice.
        random_state: int = 42,
            The random state to reproduce the visualizations.
        decomposition_kwargs: Optional[Dict] = None,
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
        self._rotate = rotate
        self._graph_name = self._graph.get_name()
        self._edge_embedding_method = edge_embedding_method

        self._node_embedding_method_name = node_embedding_method_name
        self._node_mapping = self._node_embedding = self._edge_embedding = self._negative_edge_embedding = None
        self._subsampled_node_ids = None
        self._subsampled_edge_ids = None
        self._subsampled_negative_edge_node_ids = None
        self._has_autodetermined_node_embedding_name = False

        # Check if the number of subsamples are unreasonable.
        if any(
            isinstance(number_of_subsamples, int) and number_of_subsamples == 0
            for number_of_subsamples in (
                number_of_subsampled_nodes,
                number_of_subsampled_edges,
                number_of_subsampled_negative_edges
            )
        ):
            raise ValueError(
                "One of the number of subsamples provided is zero."
            )
        if any(
            number_of_subsamples is None or number_of_subsamples > 100_000
            for number_of_subsamples in (
                number_of_subsampled_nodes,
                number_of_subsampled_edges,
                number_of_subsampled_negative_edges
            )
        ):
            warnings.warn(
                "One of the number of subsamples requested is either None "
                "(so no subsampling is executed) or it is higher than "
                "100k values. Note that all the available decomposition "
                "algorithms supported do not scale too well on "
                "datasets of this size and their visualization may "
                "just produce Gaussian spheres, even though the data "
                "is informative."
            )
        self._number_of_subsampled_nodes = number_of_subsampled_nodes
        self._number_of_subsampled_edges = number_of_subsampled_edges
        self._number_of_subsampled_negative_edges = number_of_subsampled_negative_edges
        self._random_state = random_state
        self._video_format = video_format
        self._duration = duration
        self._fps = fps

        if decomposition_kwargs is None:
            decomposition_kwargs = {}

        self._n_components = n_components
        self._decomposition_method = decomposition_method
        self._decomposition_kwargs = decomposition_kwargs

    def get_decomposition_method(self) -> Callable:
        if self._decomposition_method == "UMAP":
            # The UMAP package graph is not automatically installed
            # with the Embiggen package because it has multiple possible
            # installation options that are left to the user.
            # It can be, generally speaking, installed using:
            #
            # ```bash
            # pip install umap-learn
            # ````
            from umap import UMAP
            return UMAP(**{
                **dict(
                    n_components=self._n_components,
                    random_state=self._random_state,
                    transform_seed=self._random_state,
                    n_jobs=cpu_count(),
                    tqdm_kwds=dict(
                        desc="Computing UMAP",
                        leave=False,
                        dynamic_ncols=True
                    ),
                    verbose=True,
                ),
                **self._decomposition_kwargs
            }).fit_transform
        elif self._decomposition_method == "TSNE":
            try:
                # We try to use CUDA tsne if available, but this does not
                # currently support 3D decomposition. If the user has required a
                # 3D decomposition we need to switch to the MulticoreTSNE version.
                # Additionally, in the case that the desired decomposition
                # uses some not available parameters, such as a cosine distance
                # metric, we will capture that use case as a NotImplementedError.
                if self._n_components != 2:
                    raise NotImplementedError()
                from tsnecuda import TSNE as CUDATSNE  # pylint: disable=import-error,import-outside-toplevel
                return CUDATSNE(**{
                    **dict(
                        n_components=2,
                        random_seed=self._random_state,
                        verbose=True,
                    ),
                    **self._decomposition_kwargs
                }).fit_transform
            except (ModuleNotFoundError, NotImplementedError):
                try:
                    from MulticoreTSNE import \
                        MulticoreTSNE  # pylint: disable=import-outside-toplevel
                    return MulticoreTSNE(**{
                        **dict(
                            n_components=self._n_components,
                            n_jobs=cpu_count(),
                            metric="cosine",
                            random_state=self._random_state,
                            verbose=True,
                        ),
                        **self._decomposition_kwargs
                    }).fit_transform
                except (ModuleNotFoundError, OSError, RuntimeError):
                    try:
                        from sklearn.manifold import \
                            TSNE  # pylint: disable=import-outside-toplevel
                        return TSNE(**{
                            **dict(
                                n_components=self._n_components,
                                n_jobs=cpu_count(),
                                random_state=self._random_state,
                                verbose=True,
                                metric="cosine",
                                method="exact" if self._n_components == 4 else "barnes_hut",
                                square_distances=True,
                            ),
                            **self._decomposition_kwargs
                        }).fit_transform
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
        elif self._decomposition_method == "PCA":
            return PCA(**{
                **dict(
                    n_components=self._n_components,
                    random_state=self._random_state,
                ),
                **self._decomposition_kwargs
            }).fit_transform
        else:
            raise ValueError(
                "We currently only support PCA and TSNE decomposition methods."
            )

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
        *args: List[Union[np.ndarray, pd.DataFrame, None]]
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
        if self._decomposition_method == "TSNE" and X.shape[1] > 50:
            X = PCA(
                n_components=50,
                random_state=self._random_state
            ).fit_transform(X)
        return self.get_decomposition_method()(X)

    def _set_legend(
        self,
        axes: Axes,
        labels: List[str],
        handles: List[HandlerBase],
        legend_title: str,
        loc: str = 'best',
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
        loc: str = 'best'
            Position for the legend.
        """
        legend = axes.legend(
            handles=handles,
            labels=sanitize_ml_labels(labels),
            loc=loc,
            title=legend_title,
            **(
                dict(handler_map={tuple: HandlerTuple(ndivide=None)})
                if isinstance(handles[0], tuple)
                else {}
            )
        )

        # Setting maximum alpha to the visualization
        # to avoid transparency in the dots.
        for lh in legend.legendHandles:
            lh.set_alpha(1)

    def automatically_detect_node_embedding_method(self, node_embedding: np.ndarray) -> Optional[str]:
        """Detect node embedding method using heuristics, where possible."""
        # Rules to detect SPINE embedding
        if node_embedding.dtype == "uint8" and node_embedding.min() == 0:
            return "SPINE"
        # Rules to detect TFIDF/BERT embedding
        if node_embedding.dtype == "float16" and node_embedding.shape[1] == 768:
            return "TFIDF-weighted BERT"
        return None

    def fit_transform_nodes(
        self,
        node_embedding: Union[pd.DataFrame, np.ndarray]
    ):
        """Executes fitting for plotting node embeddings.

        Parameters
        -------------------------
        node_embedding: Union[pd.DataFrame, np.ndarray]
            Embedding of the graph nodes.
        """
        if node_embedding.shape[0] != self._graph.get_nodes_number():
            raise ValueError(
                ("The number of rows provided with the given node embedding {} "
                 "does not match the number of nodes in the graph {}.").format(
                    node_embedding.shape[0],
                    self._graph.get_nodes_number()
                )
            )
        if self._node_embedding_method_name == "auto" or self._has_autodetermined_node_embedding_name:
            self._has_autodetermined_node_embedding_name = True
            self._node_embedding_method_name = self.automatically_detect_node_embedding_method(
                node_embedding.values
                if isinstance(node_embedding, pd.DataFrame)
                else node_embedding
            )

        # If necessary, we proceed with the subsampling
        if self._number_of_subsampled_nodes is not None and self._graph.get_nodes_number() > self._number_of_subsampled_nodes:
            # Otherwise there is no need to stratify.
            node_indices = self._subsampled_node_ids = np.random.randint(
                low=0,
                high=self._graph.get_nodes_number(),
                size=self._number_of_subsampled_nodes
            )

            if not isinstance(node_embedding, np.ndarray):
                node_indices = [
                    self._graph.get_node_name_from_node_id(node_id)
                    for node_id in self._subsampled_node_ids
                ]
        else:
            if isinstance(node_embedding, np.ndarray):
                node_indices = self._graph.get_node_ids()
            else:
                node_indices = node_embedding.index

        node_transformer = NodeTransformer(
            # If the node embedding provided is a numpy array, we assume
            # that the provided embedding is aligned with the graph.
            aligned_node_mapping=isinstance(node_embedding, np.ndarray)
        )
        node_transformer.fit(node_embedding)
        self._node_embedding = self.decompose(node_transformer.transform(node_indices))

    def _get_positive_edges_embedding(
        self,
        node_embedding: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Executes fitting for plotting edge embeddings.

        Parameters
        -------------------------
        node_embedding: Union[pd.DataFrame, np.ndarray]
            Node embedding obtained from SkipGram, CBOW or GloVe or others.
        """
        if node_embedding.shape[0] != self._graph.get_nodes_number():
            raise ValueError(
                ("The number of rows provided with the given node embedding {} "
                 "does not match the number of nodes in the graph {}.").format(
                    node_embedding.shape[0],
                    self._graph.get_nodes_number()
                )
            )
        if self._node_embedding_method_name == "auto" or self._has_autodetermined_node_embedding_name:
            self._has_autodetermined_node_embedding_name = True
            self._node_embedding_method_name = self.automatically_detect_node_embedding_method(
                node_embedding.values
                if isinstance(node_embedding, pd.DataFrame)
                else node_embedding
            )
        # If necessary, we proceed with the subsampling
        if self._number_of_subsampled_edges is not None and self._graph.get_number_of_directed_edges() > self._number_of_subsampled_edges:
            # If there are edge types, we use a stratified
            # edge sampling so that all the edges types may be displayed.
            self._subsampled_edge_ids = np.random.randint(
                low=0,
                high=self._graph.get_number_of_directed_edges(),
                size=self._number_of_subsampled_edges
            )
            if isinstance(node_embedding, np.ndarray):
                edge_indices = [
                    self._graph.get_node_ids_from_edge_id(edge_id)
                    for edge_id in tqdm(
                        self._subsampled_edge_ids,
                        desc="Retrieving edge node ids",
                        leave=False,
                        dynamic_ncols=True
                    )
                ]
            else:
                edge_indices = [
                    self._graph.get_node_names_from_edge_id(edge_id)
                    for edge_id in tqdm(
                        self._subsampled_edge_ids,
                        desc="Retrieving edge node names",
                        leave=False,
                        dynamic_ncols=True
                    )
                ]
        else:
            if isinstance(node_embedding, np.ndarray):
                edge_indices = self._graph.get_directed_edge_node_ids()
            else:
                edge_indices = self._graph.get_directed_edge_node_names()

        graph_transformer = GraphTransformer(
            method=self._edge_embedding_method,
            # If the node embedding provided is a numpy array, we assume
            # that the provided embedding is aligned with the graph.
            aligned_node_mapping=isinstance(node_embedding, np.ndarray)
        )
        graph_transformer.fit(node_embedding)
        return graph_transformer.transform(edge_indices)

    def fit_transform_edges(
        self,
        node_embedding: Union[pd.DataFrame, np.ndarray]
    ):
        """Executes fitting for plotting edge embeddings.

        Parameters
        -------------------------
        node_embedding: Union[pd.DataFrame, np.ndarray]
            Node embedding obtained from SkipGram, CBOW or GloVe or others.
        """
        self._edge_embedding = self.decompose(
            self._get_positive_edges_embedding(
                node_embedding
            )
        )

    def _get_negative_edge_embedding(
        self,
        node_embedding: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Executes aggregation of negative edge embeddings.

        Parameters
        -------------------------
        node_embedding: Union[pd.DataFrame, np.ndarray]
            Node embedding obtained from SkipGram, CBOW or GloVe or others.
        """
        if node_embedding.shape[0] != self._graph.get_nodes_number():
            raise ValueError(
                ("The number of rows provided with the given node embedding {} "
                 "does not match the number of nodes in the graph {}.").format(
                    node_embedding.shape[0],
                    self._graph.get_nodes_number()
                )
            )
        # If necessary, we proceed with the subsampling
        edge_indices = self._subsampled_negative_edge_node_ids = np.random.randint(
            low=0,
            high=self._graph.get_nodes_number(),
            size=(self._number_of_subsampled_negative_edges, 2)
        )

        if not isinstance(node_embedding, np.ndarray):
            edge_indices = np.array([
                (
                    self._graph.get_node_name_from_node_id(src_node_id),
                    self._graph.get_node_name_from_node_id(dst_node_id),
                )
                for (src_node_id, dst_node_id) in tqdm(
                    self._subsampled_negative_edge_node_ids,
                    desc="Retrieving negative edge node names",
                    leave=False,
                    dynamic_ncols=True
                )
            ])

        graph_transformer = GraphTransformer(
            method=self._edge_embedding_method,
            # If the node embedding provided is a numpy array, we assume
            # that the provided embedding is aligned with the graph.
            aligned_node_mapping=isinstance(node_embedding, np.ndarray)
        )
        graph_transformer.fit(node_embedding)
        return graph_transformer.transform(edge_indices)

    def fit_transform_negative_and_positive_edges(
        self,
        node_embedding: Union[pd.DataFrame, np.ndarray]
    ):
        """Executes fitting for plotting negative edge embeddings.

        Parameters
        -------------------------
        node_embedding: Union[pd.DataFrame, np.ndarray]
            Node embedding obtained from SkipGram, CBOW or GloVe or others.
        """
        positive_edge_embedding = self._get_positive_edges_embedding(node_embedding)
        negative_edge_embedding = self._get_negative_edge_embedding(node_embedding)
        raw_edge_embedding = np.vstack([
            positive_edge_embedding,
            negative_edge_embedding
        ])
        edge_embedding = self.decompose(raw_edge_embedding)
        self._edge_embedding = edge_embedding[:positive_edge_embedding.shape[0]]
        self._negative_edge_embedding = edge_embedding[positive_edge_embedding.shape[0]:]

    def _get_figure_and_axes(
        self,
        figure: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        **kwargs: Dict
    ) -> Tuple[Figure, Axes]:
        """Return tuple with figure and axes built using provided kwargs and defaults.

        Parameters
        ---------------------------
        figure: Optional[Figure] = None
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        **kwargs: Dict
            Dictionary of parameters to pass to the instantiation of the new figure and axes if one was not initially provided.

        Raises
        ---------------------------
        ValueError
            If the figure object is None but the axes is some or viceversa.

        Returns
        ---------------------------
        Tuple with the figure and axes.
        """
        if figure is not None and axes is not None:
            return (figure, axes)
        if figure is None and axes is not None or figure is not None and axes is None:
            raise ValueError((
                "Either both figure and axes objects must be None "
                "and thefore new ones must be created or neither of "
                "them can be None."
            ))
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
        return figure, axes

    def _get_complete_title(self, initial_title: str) -> str:
        """Return the complete title for the figure.

        Parameters
        -------------------
        initial_title: str
            Initial title to incorporate.
        """
        title = "{} - {}".format(
            initial_title,
            self._graph_name,
        )

        if self._node_embedding_method_name is not None and self._node_embedding_method_name != "auto":
            title = "{} - {} ({})".format(
                title,
                self._node_embedding_method_name,
                self._decomposition_method,
            )

        return title

    def _plot_scatter(
        self,
        points: np.ndarray,
        title: str,
        colors: Optional[List[int]] = None,
        edgecolors: Optional[List[int]] = None,
        labels: List[str] = None,
        legend_title: str = "",
        show_title: bool = True,
        show_legend: bool = True,
        loc: str = "best",
        figure: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        scatter_kwargs: Optional[Dict] = None,
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
        train_marker: str = "o",
        test_marker: str = "X",
        apply_tight_layout: bool = True,
        return_collections: bool = False,
        **kwargs
    ) -> Tuple[Figure, Axes, Tuple[Collection]]:
        """Plot nodes of provided graph.

        Parameters
        ------------------------------
        points: np.ndarray,
            Points to plot.
        title: str,
            Title to use for the plot.
        colors: Optional[List[int]] = None,
            List of the colors to use for the scatter plot.
        edgecolors: Optional[List[int]] = None,
            List of the edge colors to use for the scatter plot.
        labels: List[str] = None,
            Labels for the different colors.
        legend_title: str = "",
            Title for the legend.
        show_title: bool = True,
            Whether to show the figure title.
        show_legend: bool = True,
            Whether to show the legend.
        loc: str = 'best'
            Position for the legend.
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Optional[Dict] = None,
            Kwargs to pass to the scatter plot call.
        train_indices: Optional[np.ndarray] = None,
            Indices to draw using the training marker.
            If None, all points are drawn using the training marker.
        test_indices: Optional[np.ndarray] = None,
            Indices to draw using the test marker.
            If None, while providing the train indices, we only plot the
            training points.
        train_marker: str = "o",
            The marker to use to draw the training points.
        test_marker: str = "X",
            The marker to use to draw the test points.
        apply_tight_layout: bool = True,
            Whether to apply the tight layout on the matplotlib
            Figure object.
        return_collections: bool = False,
            Whether to return the scatter plot collections.
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

        figure, axes = self._get_figure_and_axes(
            figure=figure,
            axes=axes,
            **kwargs
        )

        scatter_kwargs = {
            **GraphVisualization.DEFAULT_SCATTER_KWARGS,
            **(
                dict(linewidths=0)
                if edgecolors is None
                else dict(linewidths=0.5)
            ),
            **({} if scatter_kwargs is None else scatter_kwargs),
        }

        train_test_mask = np.zeros((points.shape[0], ))

        if train_indices is not None:
            train_test_mask[train_indices] = 1

        if test_indices is not None:
            train_test_mask[test_indices] = 2

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
        else:
            cmap = None

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
                legend_title,
                loc=loc
            )

        if self._n_components == 2:
            axes.set_axis_off()

        if show_title:
            axes.set_title(title)

        if apply_tight_layout:
            figure.tight_layout()

        if return_collections and not self._rotate:
            return figure, axes, collections
        return figure, axes

    def _wrapped_plot_scatter(self, **kwargs):
        if self._rotate:
            # These backups are needed for two reasons:
            # 1) Processes in python necessarily copy the instance objects for each process
            #    and this can cause a considerable memery peak to occour.
            # 2) Some of the objects considered are not picklable, such as, at the time of writing
            #    the lambdas used in the graph transformer or the graph object itself.
            graph_backup = self._graph
            node_embedding = self._node_embedding
            edge_embedding = self._edge_embedding
            negative_edge_embedding = self._negative_edge_embedding
            self._node_embedding = None
            self._edge_embedding = None
            self._negative_edge_embedding = None
            self._graph = None
            try:
                kwargs["loc"] = "lower right"
                path = "{}.{}".format(
                    kwargs["title"].lower().replace(" ", ""),
                    self._video_format
                )
                rotate(
                    self._plot_scatter,
                    path=path,
                    duration=self._duration,
                    fps=self._fps,
                    verbose=True,
                    **kwargs
                )
            except (Exception, KeyboardInterrupt) as e:
                self._node_embedding = node_embedding
                self._edge_embedding = edge_embedding
                self._negative_edge_embedding = negative_edge_embedding
                self._graph = graph_backup
                raise e
            self._node_embedding = node_embedding
            self._edge_embedding = edge_embedding
            self._negative_edge_embedding = negative_edge_embedding
            self._graph = graph_backup
            return display_video_at_path(path)
        return self._plot_scatter(**kwargs)

    def _plot_types(
        self,
        points: np.ndarray,
        title: str,
        types: List[int],
        type_labels: List[str],
        legend_title: str,
        show_title: bool = True,
        show_legend: bool = True,
        loc: str = "best",
        predictions: Optional[List[int]] = None,
        k: int = 9,
        figure: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        scatter_kwargs: Optional[Dict] = None,
        other_label: str = "Other",
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
        train_marker: str = "o",
        test_marker: str = "X",
        **kwargs
    ) -> Optional[Tuple[Figure, Axes]]:
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
        loc: str = 'best'
            Position for the legend.
        predictions: Optional[List[int]] = None,
            List of the labels predicted.
            If None, no prediction is visualized.
        k: int = 9,
            Number of node types to visualize.
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Optional[Dict] = None,
            Kwargs to pass to the scatter plot call.
        other_label: str = "Other",
            Label to use for edges below the top k threshold.
        train_indices: Optional[np.ndarray] = None,
            Indices to draw using the training marker.
            If None, all points are drawn using the training marker.
        test_indices: Optional[np.ndarray] = None,
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

        if not isinstance(type_labels, np.ndarray):
            raise ValueError(
                (
                    "The parameter type_labels was expected to be a numpy array, "
                    "but an object of type `{}` was provided."
                ).format(type(type_labels))
            )

        if not isinstance(types, np.ndarray):
            raise ValueError(
                (
                    "The parameter types was expected to be a numpy array, "
                    "but an object of type `{}` was provided."
                ).format(type(types))
            )

        counts = np.bincount(types)
        number_of_types = len(counts)
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
            type_labels.append(other_label.format(number_of_types - k))

        return self._wrapped_plot_scatter(**{
            **dict(
                points=points,
                title=title,
                colors=types,
                edgecolors=predictions,
                labels=type_labels,
                legend_title=legend_title,
                show_title=show_title,
                show_legend=show_legend,
                loc=loc,
                figure=figure,
                axes=axes,
                scatter_kwargs=scatter_kwargs,
                train_indices=train_indices,
                test_indices=test_indices,
                train_marker=train_marker,
                test_marker=test_marker,
            ),
            **kwargs
        })

    def plot_edge_segments(
        self,
        figure: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        scatter_kwargs: Optional[Dict] = None,
        **kwargs: Dict
    ) -> Tuple[Figure, Axes]:
        """Plot edge segments between the nodes of the graph.

        Parameters
        ------------------------
        figure: Optional[Figure] = None
            The figure object to plot over.
            If None, a new figure is created automatically and returned.
        axes: Optional[Axes] = None
            The axes object to plot over.
            If None, a new axes is created automatically and returned.
        scatter_kwargs: Optional[Dict] = None
            Dictionary of parameters to pass to the scattering of the edges.
        **kwargs: Dict
            Dictionary of parameters to pass to the instantiation of the new figure and axes if one was not initially provided.

        Returns
        ------------------------
        Tuple with either the provided or created figure and axes.
        """
        if self._node_embedding is None:
            raise ValueError(
                "Node fitting must be executed before plot."
            )

        figure, axes = self._get_figure_and_axes(
            figure=figure,
            axes=axes,
            **kwargs
        )

        if self._subsampled_node_ids is not None:
            edge_node_ids = self._graph.get_edge_ids_from_node_ids(
                node_ids=self._subsampled_node_ids,
                add_selfloops_where_missing=False,
                complete=False,
            )
        else:
            edge_node_ids = self._graph.get_edge_node_ids(
                directed=False
            )

        lines_collection = mc.LineCollection(
            self._node_embedding[edge_node_ids],
            linewidths=1,
            zorder=0,
            **{
                **GraphVisualization.DEFAULT_EDGES_SCATTER_KWARGS,
                **(
                    {} if scatter_kwargs is None else scatter_kwargs
                )
            }
        )
        axes.add_collection(lines_collection)

        return figure, axes

    def plot_nodes(
        self,
        figure: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        scatter_kwargs: Optional[Dict] = None,
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
        train_marker: str = "o",
        test_marker: str = "X",
        show_title: bool = True,
        show_legend: bool = True,
        loc: str = "best",
        annotate_nodes: Union[str, bool] = "auto",
        show_edges: bool = False,
        edge_scatter_kwargs: Optional[Dict] = None,
        **kwargs: Dict
    ) -> Tuple[Figure, Axes]:
        """Plot nodes of provided graph.

        Parameters
        ------------------------------
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Optional[Dict] = None,
            Kwargs to pass to the scatter plot call.
        train_indices: Optional[np.ndarray] = None,
            Indices to draw using the training marker.
            If None, all points are drawn using the training marker.
        test_indices: Optional[np.ndarray] = None,
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
        loc: str = 'best'
            Position for the legend.
        annotate_nodes: Union[str, bool] = "auto",
            Whether to show the node name when scattering them.
            The default behaviour, "auto", means that it will
            enable this feature automatically when the graph has
            less than 100 nodes.
        show_edges: bool = False,
            Whether to show edges between the different nodes
            shown in the scatter plot.
        edge_scatter_kwargs: Optional[Dict] = None,
            Arguments to provide to the scatter plot of the edges
            if they were required.
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

        if annotate_nodes == "auto":
            annotate_nodes = self._graph.get_nodes_number() < 100 and not self._rotate

        if show_edges:
            figure, axes = self.plot_edge_segments(
                figure,
                axes,
                scatter_kwargs=edge_scatter_kwargs,
                **kwargs
            )

        returned_values = self._wrapped_plot_scatter(
            points=self._node_embedding,
            title=self._get_complete_title("Nodes embedding"),
            figure=figure,
            axes=axes,
            scatter_kwargs=scatter_kwargs,
            train_indices=train_indices,
            test_indices=test_indices,
            train_marker=train_marker,
            test_marker=test_marker,
            show_title=show_title,
            show_legend=show_legend,
            loc=loc,
            **kwargs
        )

        if annotate_nodes:
            figure, axes = returned_values
            returned_values = self.annotate_nodes(
                figure=figure,
                axes=axes,
                points=self._node_embedding,
            )

        return returned_values

    def annotate_nodes(
        self,
        figure: Figure,
        axes: Axes,
        points: np.ndarray
    ) -> Tuple[Figure, Axes]:
        if self._subsampled_node_ids is not None:
            node_names = [
                self._graph.get_node_name_from_node_id(node_id)
                for node_id in self._subsampled_node_ids
            ]
        else:
            node_names = self._graph.get_node_names()
        for i, txt in enumerate(node_names):
            axes.annotate(txt, points[i])
        return (figure, axes)

    def plot_edges(
        self,
        figure: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        scatter_kwargs: Optional[Dict] = None,
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
        train_marker: str = "o",
        test_marker: str = "X",
        show_title: bool = True,
        show_legend: bool = True,
        loc: str = "best",
        **kwargs: Dict
    ) -> Tuple[Figure, Axes]:
        """Plot edge embedding of provided graph.

        Parameters
        ------------------------------
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Optional[Dict] = None,
            Kwargs to pass to the scatter plot call.
        train_indices: Optional[np.ndarray] = None,
            Indices to draw using the training marker.
            If None, all points are drawn using the training marker.
        test_indices: Optional[np.ndarray] = None,
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
        loc: str = 'best'
            Position for the legend.
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

        return self._wrapped_plot_scatter(
            points=self._edge_embedding,
            title=self._get_complete_title("Edges embedding"),
            figure=figure,
            axes=axes,
            scatter_kwargs=scatter_kwargs,
            train_indices=train_indices,
            test_indices=test_indices,
            train_marker=train_marker,
            test_marker=test_marker,
            show_title=show_title,
            show_legend=show_legend,
            loc=loc,
            **kwargs
        )

    def plot_positive_and_negative_edges(
        self,
        figure: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        scatter_kwargs: Optional[Dict] = None,
        show_title: bool = True,
        show_legend: bool = True,
        loc: str = "best",
        **kwargs: Dict
    ) -> Tuple[Figure, Axes]:
        """Plot edge embedding of provided graph.

        Parameters
        ------------------------------
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Optional[Dict] = None,
            Kwargs to pass to the scatter plot call.
        show_title: bool = True,
            Whether to show the figure title.
        show_legend: bool = True,
            Whether to show the legend.
        loc: str = 'best'
            Position for the legend.
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
        if self._edge_embedding is None or self._negative_edge_embedding is None:
            raise ValueError(
                "Positive and negative edge fitting must be executed before plot."
            )

        points = np.vstack([
            self._edge_embedding,
            self._negative_edge_embedding
        ])

        types = np.concatenate([
            np.ones(self._edge_embedding.shape[0], dtype="int64"),
            np.zeros(
                self._negative_edge_embedding.shape[0], dtype="int64"),
        ])

        points, types = self._shuffle(points, types)

        return self._plot_types(
            points=points,
            title=self._get_complete_title("Positive & negative edges"),
            types=types,
            type_labels=np.array([
                "Positive edges",
                "Negative edges"
            ]),
            legend_title="Edges",
            figure=figure,
            axes=axes,
            scatter_kwargs=scatter_kwargs,
            show_title=show_title,
            show_legend=show_legend,
            loc=loc,
            **kwargs
        )

    def _get_flatten_multi_label_and_unknown_node_types(
        self,
        subsampled_node_ids: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Returns flattened node type IDs adjusted for the current instance.

        Implementative details
        ---------------------------------
        If the subsampled node IDs are provided, only those nodes will be taken into account.

        Parameters
        ---------------------------------
        subsampled_node_ids: np.ndarray = None
            If provided, only samples these node IDs.
        """
        # The following is needed to normalize the multiple types
        node_types_counts = self._graph.get_node_type_id_counts_hashmap()
        top_10_node_types = {
            node_type: 50 - i
            for i, node_type in enumerate(sorted(
                node_types_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:50])
        }
        node_types_counts = {
            node_type: top_10_node_types.get(node_type, 0)
            for node_type in node_types_counts
        }
        node_types_number = self._graph.get_node_types_number()
        unknown_node_types_id = node_types_number

        # According to whether the subsampled node IDs were given,
        # we iterate on them or on the complete set of nodes of the graph.
        if subsampled_node_ids is None:
            nodes_iterator = trange(
                self._graph.get_nodes_number(),
                desc="Computing flattened multi-label and unknown node types",
                leave=False,
                dynamic_ncols=True
            )
        else:
            nodes_iterator = tqdm(
                subsampled_node_ids,
                desc="Computing subsampled flattened multi-label and unknown node types",
                leave=False,
                dynamic_ncols=True
            )

        # When we have multiple node types for a given node, we set it to
        # the most common node type of the set.
        return np.fromiter(
            (
                unknown_node_types_id
                if node_type_ids is None
                else
                sorted(
                    node_type_ids,
                    key=lambda node_type: node_types_counts[node_type],
                    reverse=True
                )[0]
                for node_type_ids in (
                    self._graph.get_node_type_ids_from_node_id(node_id)
                    for node_id in nodes_iterator
                )
            ),
            dtype=np.uint32
        )

    def _get_flatten_unknown_edge_types(
        self,
        subsampled_edge_id: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Returns flattened edge type IDs adjusted for the current instance.

        Implementative details
        ---------------------------------
        If the subsampled edge IDs are provided, only those edges will be taken into account.

        Parameters
        ---------------------------------
        subsampled_edge_ids: np.ndarray = None
            If provided, only samples these edge IDs.
        """
        # The following is needed to normalize the multiple types
        edge_types_number = self._graph.get_edge_types_number()
        unknown_edge_types_id = edge_types_number
        # According to whether the subsampled node IDs were given,
        # we iterate on them or on the complete set of nodes of the graph.
        if subsampled_edge_id is None:
            edges_iterator = trange(
                self._graph.get_number_of_directed_edges(),
                desc="Computing flattened unknown edge types",
                leave=False,
                dynamic_ncols=True
            )
        else:
            edges_iterator = tqdm(
                subsampled_edge_id,
                desc="Computing subsampled flattened unknown edge types",
                leave=False,
                dynamic_ncols=True
            )
        # When we have multiple node types for a given node, we set it to
        # the most common node type of the set.
        return np.fromiter(
            (
                unknown_edge_types_id
                if edge_type_id is None
                else
                edge_type_id
                for edge_type_id in (
                    edge_type_id
                    for edge_type_id in (
                        self._graph.get_edge_type_id_from_edge_id(edge_id)
                        for edge_id in edges_iterator
                    )
                )
            ),
            dtype=np.uint32
        )

    def plot_node_types(
        self,
        node_type_predictions: Optional[List[int]] = None,
        k: int = 9,
        figure: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        scatter_kwargs: Optional[Dict] = None,
        legend_title: str = "Node types",
        other_label: str = "Other {} node types",
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
        train_marker: str = "o",
        test_marker: str = "X",
        show_title: bool = True,
        show_legend: bool = True,
        loc: str = "best",
        show_edges: bool = False,
        edge_scatter_kwargs: Optional[Dict] = None,
        annotate_nodes: Union[str, bool] = "auto",
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """Plot common node types of provided graph.

        Parameters
        ------------------------------
        node_type_predictions: Optional[List[int]] = None,
            Predictions of the node types.
        k: int = 9,
            Number of node types to visualize.
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Optional[Dict] = None,
            Kwargs to pass to the scatter plot call.
        other_label: str = "Other {} node types"
            Label to use for edges below the top k threshold.
        train_indices: Optional[np.ndarray] = None,
            Indices to draw using the training marker.
            If None, all points are drawn using the training marker.
        test_indices: Optional[np.ndarray] = None,
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
        loc: str = 'best'
            Position for the legend.
        show_edges: bool = False,
            Whether to show edges between the different nodes
            shown in the scatter plot.
        edge_scatter_kwargs: Optional[Dict] = None,
            Arguments to provide to the scatter plot of the edges
            if they were required.
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

        if show_edges:
            figure, axes = self.plot_edge_segments(
                figure,
                axes,
                scatter_kwargs=edge_scatter_kwargs,
                **kwargs
            )

        if annotate_nodes == "auto":
            annotate_nodes = self._graph.get_nodes_number() < 100 and not self._rotate

        node_types = self._get_flatten_multi_label_and_unknown_node_types(
            self._subsampled_node_ids
        )

        node_type_names_iter = (
            self._graph.get_node_type_name_from_node_type_id(node_id)
            for node_id in trange(
                self._graph.get_node_types_number(),
                desc="Retrieving graph node types",
                leave=False,
                dynamic_ncols=True
            )
        )

        if self._graph.has_unknown_node_types():
            node_type_names_iter = itertools.chain(
                node_type_names_iter,
                iter(("Unknown",))
            )

        node_type_names = np.array(
            list(node_type_names_iter),
            dtype=str,
        )

        returned_values = self._plot_types(
            self._node_embedding,
            self._get_complete_title("Node types"),
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
            loc=loc,
            **kwargs
        )

        if annotate_nodes:
            figure, axes = returned_values
            returned_values = self.annotate_nodes(
                figure=figure,
                axes=axes,
                points=self._node_embedding,
            )

        return returned_values

    def plot_connected_components(
        self,
        k: int = 9,
        figure: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        scatter_kwargs: Optional[Dict] = None,
        other_label: str = "Other {} components",
        legend_title: str = "Component sizes",
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
        train_marker: str = "o",
        test_marker: str = "X",
        show_title: bool = True,
        show_legend: bool = True,
        loc: str = "best",
        annotate_nodes: Union[str, bool] = "auto",
        show_edges: bool = False,
        edge_scatter_kwargs: Optional[Dict] = None,
        **kwargs
    ) -> Tuple[Figure, Axes]:
        """Plot common node types of provided graph.

        Parameters
        ------------------------------
        k: int = 9,
            Number of components to visualize.
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Optional[Dict] = None,
            Kwargs to pass to the scatter plot call.
        other_label: str = "Other {} components",
            Label to use for edges below the top k threshold.
        legend_title: str = "Component sizes",
            Title for the legend.
        train_indices: Optional[np.ndarray] = None,
            Indices to draw using the training marker.
            If None, all points are drawn using the training marker.
        test_indices: Optional[np.ndarray] = None,
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
        loc: str = 'best'
            Position for the legend.
        show_edges: bool = False,
            Whether to show edges between the different nodes
            shown in the scatter plot.
        edge_scatter_kwargs: Optional[Dict] = None,
            Arguments to provide to the scatter plot of the edges
            if they were required.
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

        if show_edges:
            figure, axes = self.plot_edge_segments(
                figure,
                axes,
                scatter_kwargs=edge_scatter_kwargs,
                **kwargs
            )

        if annotate_nodes == "auto":
            annotate_nodes = self._graph.get_nodes_number() < 100 and not self._rotate

        components, components_number, _, _ = self._graph.get_connected_components()
        sizes = np.bincount(components, minlength=components_number)

        if self._subsampled_node_ids is not None:
            components = components[self._subsampled_node_ids]

        returned_values = self._plot_types(
            self._node_embedding,
            self._get_complete_title("Components"),
            types=components,
            type_labels=np.array(
                [
                    "Size {}".format(size)
                    for size in sizes
                ],
                dtype=str
            ),
            legend_title=legend_title,
            show_title=show_title,
            show_legend=show_legend,
            loc=loc,
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

        if annotate_nodes:
            figure, axes = returned_values
            returned_values = self.annotate_nodes(
                figure=figure,
                axes=axes,
                points=self._node_embedding,
            )

        return returned_values

    def plot_node_degrees(
        self,
        figure: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        scatter_kwargs: Optional[Dict] = None,
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
        train_marker: str = "o",
        test_marker: str = "X",
        use_log_scale: bool = True,
        show_title: bool = True,
        show_legend: bool = True,
        loc: str = "best",
        annotate_nodes: Union[str, bool] = "auto",
        show_edges: bool = False,
        edge_scatter_kwargs: Optional[Dict] = None,
        **kwargs: Dict
    ):
        """Plot node degrees heatmap.

        Parameters
        ------------------------------
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Optional[Dict] = None,
            Kwargs to pass to the scatter plot call.
        train_indices: Optional[np.ndarray] = None,
            Indices to draw using the training marker.
            If None, all points are drawn using the training marker.
        test_indices: Optional[np.ndarray] = None,
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
        loc: str = 'best'
            Position for the legend.
        show_edges: bool = False,
            Whether to show edges between the different nodes
            shown in the scatter plot.
        edge_scatter_kwargs: Optional[Dict] = None,
            Arguments to provide to the scatter plot of the edges
            if they were required.
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

        if self._subsampled_node_ids is None:
            degrees = self._graph.get_node_degrees()
        else:
            degrees = np.fromiter(
                (
                    self._graph.get_node_degree_from_node_id(node_id)
                    for node_id in self._subsampled_node_ids
                ),
                dtype=np.uint32
            )

        if annotate_nodes == "auto":
            annotate_nodes = self._graph.get_nodes_number() < 100 and not self._rotate

        if show_edges:
            figure, axes = self.plot_edge_segments(
                figure,
                axes,
                scatter_kwargs=edge_scatter_kwargs,
                **kwargs
            )

        returned_values = self._wrapped_plot_scatter(
            points=self._node_embedding,
            title=self._get_complete_title("Node degrees"),
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
            loc=loc,
            return_collections=True,
            **kwargs
        )

        if not self._rotate:
            figure, axes, scatter = returned_values
            color_bar = figure.colorbar(scatter[0], ax=axes)
            color_bar.set_alpha(1)
            color_bar.draw_all()

        if annotate_nodes:
            returned_values = self.annotate_nodes(
                figure=figure,
                axes=axes,
                points=self._node_embedding,
            )

        return returned_values

    def plot_edge_types(
        self,
        edge_type_predictions: Optional[List[int]] = None,
        k: int = 9,
        figure: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        scatter_kwargs: Optional[Dict] = None,
        other_label: str = "Other {} edge types",
        legend_title: str = "Edge types",
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
        train_marker: str = "o",
        test_marker: str = "X",
        show_title: bool = True,
        show_legend: bool = True,
        loc: str = "best",
        **kwargs: Dict
    ):
        """Plot common edge types of provided graph.

        Parameters
        ------------------------------
        edge_type_predictions: Optional[List[int]] = None,
            Predictions of the edge types.
        k: int = 9,
            Number of edge types to visualize.
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Optional[Dict] = None,
            Kwargs to pass to the scatter plot call.
        other_label: str = "Other {} edge types",
            Label to use for edges below the top k threshold.
        legend_title: str = "Edge types",
            Title for the legend.
        train_indices: Optional[np.ndarray] = None,
            Indices to draw using the training marker.
            If None, all points are drawn using the training marker.
        test_indices: Optional[np.ndarray] = None,
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
        loc: str = 'best'
            Position for the legend.
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

        edge_types = self._get_flatten_unknown_edge_types(
            self._subsampled_edge_ids
        )

        edge_type_names_iter = (
            self._graph.get_edge_type_name_from_edge_type_id(edge_id)
            for edge_id in trange(
                self._graph.get_edge_types_number(),
                desc="Retrieving graph edge types",
                leave=False,
                dynamic_ncols=True
            )
        )

        if self._graph.has_unknown_edge_types():
            edge_type_names_iter = itertools.chain(
                edge_type_names_iter,
                iter(("Unknown",))
            )

        edge_type_names = np.array(list(edge_type_names_iter), dtype=str)

        return self._plot_types(
            self._edge_embedding,
            self._get_complete_title(
                "Edge types - {}".format(self._edge_embedding_method)),
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
            loc=loc,
            **kwargs
        )

    def plot_edge_weights(
        self,
        figure: Optional[Figure] = None,
        axes: Optional[Axes] = None,
        scatter_kwargs: Optional[Dict] = None,
        train_indices: Optional[np.ndarray] = None,
        test_indices: Optional[np.ndarray] = None,
        train_marker: str = "o",
        test_marker: str = "X",
        show_title: bool = True,
        show_legend: bool = True,
        loc: str = "best",
        **kwargs: Dict
    ):
        """Plot common edge types of provided graph.

        Parameters
        ------------------------------
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        scatter_kwargs: Optional[Dict] = None,
            Kwargs to pass to the scatter plot call.
        train_indices: Optional[np.ndarray] = None,
            Indices to draw using the training marker.
            If None, all points are drawn using the training marker.
        test_indices: Optional[np.ndarray] = None,
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
        loc: str = 'best'
            Position for the legend.
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

        if self._subsampled_edge_ids is None:
            weights = self._graph.get_edge_weights()
        else:
            weights = np.fromiter(
                (
                    self._graph.get_edge_weight_from_edge_id(edge_id)
                    for edge_id in self._subsampled_edge_ids
                ),
                dtype=np.uint32
            )

        returned_values = self._wrapped_plot_scatter(
            points=self._edge_embedding,
            title=self._get_complete_title(
                "Edge weights - {}".format(self._edge_embedding_method)),
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
            loc=loc,
            return_collections=True,
            **kwargs
        )

        if not self._rotate:
            figure, axes, scatter = returned_values
            color_bar = figure.colorbar(scatter[0], ax=axes)
            color_bar.set_alpha(1)
            color_bar.draw_all()
            returned_values = figure, axes

    def plot_dot(self, engine: str = "circo"):
        """Return dot plot of the current graph.

        Parameters
        ------------------------------
        engine: str = "circo",
            The engine to use to visualize the graph.

        Raises
        ------------------------------
        ModuleNotFoundError,
            If graphviz is not installed.
        """
        try:
            import graphviz
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "In order to run the graph Dot visualization, "
                "the graphviz library must be installed. This "
                "library is not an explicit dependency of "
                "Embiggen because it may be hard to install "
                "on some systems and cause the Embiggen library "
                "to fail the installation.\n"
                "In order to install graphviz, try running "
                "`pip install graphviz`."
            )
        return graphviz.Source(
            self._graph.to_dot(),
            engine=engine
        )

    def _annotate_top_nodes(
        self,
        axis: Axes,
        k: int = 5
    ):
        """Add textual label with position of central node to plots.

        Parameters
        ----------------------
        axis: Axis
            The axis where to plot the annotations
        k: int = 5
            The number of top degree nodes to visualize
            By default, 5.
        """
        sorted_node_degrees = np.sort(self._graph.get_node_degrees())
        for i, node_id in enumerate(self._graph.get_top_k_central_node_ids(k)):
            node_degree = self._graph.get_node_degree_from_node_id(node_id)
            axis.annotate(
                self._graph.get_node_name_from_node_id(node_id),
                (
                    np.where(sorted_node_degrees == node_degree)[0][0],
                    node_degree
                ),
                xytext=(40*(1 if i % 2 else -1), -5),
                textcoords='offset points',
                ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2',
                          fc='tab:orange', alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                                color='tab:red')
            )

    def plot_node_degree_distribution(
        self,
        fig: Optional[Figure] = None,
        axis: Optional[Figure] = None,
        apply_tight_layout: bool = True
    ) -> Tuple[Figure, Axes]:
        """Plot the given graph node degree distribution.

        Parameters
        ------------------------------
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        apply_tight_layout: bool = True,
            Whether to apply the tight layout on the matplotlib
            Figure object.
        """
        if axis is None:
            fig, axis = plt.subplots(figsize=(5, 5))
        axis.plot(np.sort(self._graph.get_node_degrees()), '.')
        axis.set_ylabel("Node degree")
        axis.set_xlabel("Nodes sorted by degree")
        axis.set_title("Degrees distribution for {}".format(self._graph_name))
        self._annotate_top_nodes(axis)
        if apply_tight_layout:
            fig.tight_layout()
        return fig, axis

    def plot_edge_weight_distribution(
        self,
        fig: Optional[Figure] = None,
        axis: Optional[Figure] = None,
        apply_tight_layout: bool = True
    ) -> Tuple[Figure, Axes]:
        """Plot the given graph node degree distribution.

        Parameters
        ------------------------------
        figure: Optional[Figure] = None,
            Figure to use to plot. If None, a new one is created using the
            provided kwargs.
        axes: Optional[Axes] = None,
            Axes to use to plot. If None, a new one is created using the
            provided kwargs.
        apply_tight_layout: bool = True,
            Whether to apply the tight layout on the matplotlib
            Figure object.
        """
        if axis is None:
            fig, axis = plt.subplots(figsize=(5, 5))
        axis.hist(self._graph.get_edge_weights(), bins=50)
        axis.set_ylabel("Number of edges")
        axis.set_xlabel("Sorted weights")
        axis.set_title("Weights distribution for {}".format(self._graph_name))
        if apply_tight_layout:
            fig.tight_layout()
        return fig, axis
