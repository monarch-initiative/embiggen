"""Sub-module with methods to compute node-embedding with a one-liner."""
from typing import Dict, List, Union, Tuple

import inspect
import pandas as pd
import tensorflow as tf
from cache_decorator import Cache
from ensmallen_graph import EnsmallenGraph


from ..embedders import GraphCBOW, GraphGloVe, GraphSkipGram, Embedder

SUPPORTED_NODE_EMBEDDING_METHODS = {
    "CBOW": GraphCBOW,
    "GloVe": GraphGloVe,
    "SkipGram": GraphSkipGram
}


def get_available_node_embedding_methods() -> List[str]:
    """Return list of supported node embedding methods."""
    global SUPPORTED_NODE_EMBEDDING_METHODS
    return list(SUPPORTED_NODE_EMBEDDING_METHODS.keys())


def get_node_embedding_method(node_embedding_method_name: str) -> Embedder:
    """Return node embedding method curresponding to given name."""
    global SUPPORTED_NODE_EMBEDDING_METHODS
    return SUPPORTED_NODE_EMBEDDING_METHODS[node_embedding_method_name]


def is_node_embedding_method_supported(node_embedding_method_name: str) -> bool:
    """Return boolean value representing if given node embedding method is supported.

    Parameters
    --------------------
    node_embedding_method_name: str,
        Name of the node embedding method.

    Returns
    --------------------
    Whether the given node embedding method is supported.
    """
    return node_embedding_method_name in get_available_node_embedding_methods()


@Cache(
    cache_path=[
        "node_embeddings/{node_embedding_method_name}/{graph_name}/{_hash}_embedding.csv.xz",
        "node_embeddings/{node_embedding_method_name}/{graph_name}/{_hash}_training_history.csv.xz",
    ],
    args_to_ignore=["devices", "verbose"]
)
def _compute_node_embedding(
    graph: EnsmallenGraph,
    graph_name: str,  # pylint: disable=unused-argument
    node_embedding_method_name: str,
    fit_kwargs: Dict,
    verbose: bool = True,
    devices: Union[List[str], str] = None,
    **kwargs: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return embedding computed with required node embedding method.

    Specifically, this method also caches the embedding automatically.

    Parameters
    --------------------------
    graph: EnsmallenGraph,
        The graph to embed.
    graph_name: str,
        The name of the graph.
    node_embedding_method_name: str,
        The name of the node embedding method to use.
    fit_kwargs: Dict,
        Arguments to pass to the fit call.
    verbose: bool = True,
        Whether to show loading bars.
    devices: Union[List[str], str] = None,
        The devices to use.
        If None, all GPU devices available are used.
    **kwargs: Dict,
        Arguments to pass to the node embedding method constructor.
        Read the documentation of the selected method.

    Returns
    --------------------------
    Tuple with node embedding and training history.
    """
    # Since the verbose kwarg may be provided also on the fit_kwargs
    # we normalize the parameter to avoid collisions.
    verbose = fit_kwargs.pop("verbose", verbose)
    strategy = tf.distribute.MirroredStrategy(devices=devices)
    with strategy.scope():
        # Creating the node embedding model
        model = get_node_embedding_method(node_embedding_method_name)(
            graph,
            support_mirror_strategy=True,
            **kwargs
        )
        # Fitting the node embedding model
        history = model.fit(
            verbose=verbose,
            **fit_kwargs
        )
        # Extracting computed embedding
        node_embedding = model.get_embedding_dataframe()
    return node_embedding, history


def compute_node_embedding(
    graph: EnsmallenGraph,
    node_embedding_method_name: str,
    devices: Union[List[str], str] = None,
    fit_kwargs: Dict = None,
    verbose: bool = True,
    automatically_drop_unsupported_parameters: bool = False,
    automatically_enable_time_memory_tradeoffs: bool = True,
    **kwargs: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return embedding computed using SkipGram on given graph.

    Parameters
    --------------------------
    graph: EnsmallenGraph,
        Graph to embed.
    node_embedding_method_name: str,
        The name of the node embedding method to use.
    devices: Union[List[str], str] = None,
        The devices to use.
        If None, all GPU devices available are used.
    fit_kwargs: Dict = None,
        Arguments to pass to the fit call.
    verbose: bool = True,
        Whether to show loading bars.
    automatically_drop_unsupported_parameters: bool = False,
        If required, we filter out the unsupported parameters.
        This may be useful when running a suite of experiments with a set of
        parameters and you do not want to bother in dropping the parameters
        that are only supported in a subset of methods.
    automatically_enable_time_memory_tradeoffs: bool = True,
        Whether to activate the time memory tradeoffs automatically.
        Often case, this is something you want enabled on your graph object.
        Since, generally, it is a good idea to enable these while
        computing a node embedding we enable these by default.
    **kwargs: Dict,
        Arguments to pass to the node embedding method constructor.
        Read the documentation of the selected method to learn
        which methods are supported by the selected constructor.

    Returns
    --------------------------
    Tuple with node embedding and training history.
    """
    if not is_node_embedding_method_supported(node_embedding_method_name):
        raise ValueError(
            (
                "The given node embedding method `{}` is not supported. "
                "The supported node embedding methods are `{}`."
            ).format(
                node_embedding_method_name,
                get_available_node_embedding_methods()
            )
        )

    # If the fit kwargs are not given we normalize them to an empty dictionary.
    if fit_kwargs is None:
        fit_kwargs = {}

    # If required, we filter out the unsupported parameters.
    # This may be useful when running a suite of experiments with a set of
    # parameters and you do not want to bother in dropping the parameters
    # that are only supported in a subset of methods.
    if automatically_drop_unsupported_parameters and kwargs:
        # Get the list of supported parameters
        supported_parameter = inspect.signature(
            get_node_embedding_method(node_embedding_method_name).__init__
        ).parameters
        # Filter out the unsupported parameters
        kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in supported_parameter
        }

    # If required we enable the time memory tradeoffs.
    if automatically_enable_time_memory_tradeoffs:
        graph.enable(
            vector_destinations=True,
            vector_outbounds=True
        )

    # If devices are given as a single device we adapt this into a list.
    if isinstance(devices, str):
        devices = [devices]

    # Call the wrapper with cache.
    return _compute_node_embedding(
        graph,
        graph_name=graph.get_name(),
        node_embedding_method_name=node_embedding_method_name,
        fit_kwargs=fit_kwargs,
        verbose=verbose,
        devices=devices,
        **kwargs
    )
