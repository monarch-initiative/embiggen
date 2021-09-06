"""Sub-module with methods to compute node-embedding with a one-liner."""
import inspect
import warnings
from typing import Dict, List, Tuple, Union

import pandas as pd
import tensorflow as tf
from cache_decorator import Cache
from ensmallen import Graph

from ..embedders import (Embedder, GraphCBOW, GraphGloVe, GraphSkipGram,
                         Siamese, SimplE, TransE, TransH, TransR)
from ..utils import has_gpus, has_nvidia_drivers, has_rocm_drivers

SUPPORTED_NODE_EMBEDDING_METHODS = {
    "CBOW": GraphCBOW,
    "GloVe": GraphGloVe,
    "SkipGram": GraphSkipGram,
    "Siamese": Siamese,
    "TransE": TransE,
    "SimplE": SimplE,
    "TransH": TransH,
    "TransR": TransR,
}

REQUIRE_ZIPFIAN = [
    "CBOW",
    "SkipGram"
]

RANDOM_WALK_BASED_MODELS = [
    "CBOW",
    "GloVe",
    "SkipGram"
]

LINK_PREDICTION_BASED_MODELS = [
    "Siamese",
    "TransR",
    "TransE",
    "TransH",
    "SimplE"
]

assert set(RANDOM_WALK_BASED_MODELS +
           LINK_PREDICTION_BASED_MODELS) == set(SUPPORTED_NODE_EMBEDDING_METHODS)


def get_available_node_embedding_methods() -> List[str]:
    """Return list of supported node embedding methods."""
    return list(SUPPORTED_NODE_EMBEDDING_METHODS.keys())


def get_node_embedding_method(node_embedding_method_name: str) -> Embedder:
    """Return node embedding method curresponding to given name."""
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


def _train_model(
    graph: Graph,
    node_embedding_method_name: str,
    fit_kwargs: Dict,
    verbose: bool,
    support_mirrored_strategy: bool,
    **kwargs: Dict
) -> Tuple[Union[pd.DataFrame, Tuple[pd.DataFrame]], pd.DataFrame]:
    """Return embedding computed with required node embedding method.

    Parameters
    --------------------------
    graph: Graph,
        The graph to embed.
    node_embedding_method_name: str,
        The name of the node embedding method to use.
    fit_kwargs: Dict,
        Arguments to pass to the fit call.
    verbose: bool = True,
        Whether to show loading bars.
    use_mirrored_strategy: bool = True,
        Whether to use mirrored strategy.
    **kwargs: Dict,
        Arguments to pass to the node embedding method constructor.
        Read the documentation of the selected method.

    Returns
    --------------------------
    Tuple with node embedding and training history.
    """
    # Creating the node embedding model
    model = get_node_embedding_method(node_embedding_method_name)(
        graph,
        support_mirrored_strategy=support_mirrored_strategy,
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


@Cache(
    cache_path=[
        "node_embeddings/{node_embedding_method_name}/{graph_name}/{_hash}_embedding.pkl.gz",
        "node_embeddings/{node_embedding_method_name}/{graph_name}/{_hash}_training_history.csv.xz",
    ],
    args_to_ignore=["devices", "use_mirrored_strategy", "verbose"]
)
def _compute_node_embedding(
    graph: Graph,
    graph_name: str,  # pylint: disable=unused-argument
    node_embedding_method_name: str,
    fit_kwargs: Dict,
    verbose: bool = True,
    use_mirrored_strategy: bool = True,
    devices: Union[List[str], str] = None,
    **kwargs: Dict
) -> Tuple[Union[pd.DataFrame, Tuple[pd.DataFrame]], pd.DataFrame]:
    """Return embedding computed with required node embedding method.

    Specifically, this method also caches the embedding automatically.

    Parameters
    --------------------------
    graph: Graph,
        The graph to embed.
    graph_name: str,
        The name of the graph.
    node_embedding_method_name: str,
        The name of the node embedding method to use.
    fit_kwargs: Dict,
        Arguments to pass to the fit call.
    verbose: bool = True,
        Whether to show loading bars.
    use_mirrored_strategy: bool = True,
        Whether to use mirrored strategy.
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
    kwargs = dict(
        graph=graph,
        node_embedding_method_name=node_embedding_method_name,
        fit_kwargs=fit_kwargs,
        verbose=verbose,
        support_mirrored_strategy=use_mirrored_strategy,
        **kwargs
    )
    if use_mirrored_strategy:
        strategy = tf.distribute.MirroredStrategy(devices=devices)
        with strategy.scope():
            return _train_model(**kwargs)
    return _train_model(**kwargs)


def compute_node_embedding(
    graph: Graph,
    node_embedding_method_name: str,
    use_mirrored_strategy: bool = True,
    devices: Union[List[str], str] = None,
    fit_kwargs: Dict = None,
    verbose: Union[bool, int] = True,
    automatically_drop_unsupported_parameters: bool = False,
    automatically_enable_time_memory_tradeoffs: bool = True,
    automatically_sort_by_decreasing_outbound_node_degree: bool = True,
    **kwargs: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return embedding computed using SkipGram on given graph.

    Parameters
    --------------------------
    graph: Graph,
        Graph to embed.
    node_embedding_method_name: str,
        The name of the node embedding method to use.
    use_mirrored_strategy: bool = True,
        Whether to use mirror strategy to distribute the
        computation across multiple devices.
        Note that this will be automatically disabled if the
        set of devices detected is only composed of one,
        since using MirroredStrategy adds a significant overhead
        and may endup limiting the device usage.
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
    automatically_sort_by_decreasing_outbound_node_degree: bool = True,
        Whether to automatically sort the nodes by the outbound node degree.
        This is necessary in order to run SkipGram efficiently with the NCE loss.
        It will ONLY be executed if the requested model is SkipGram.
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

    # To avoid some nighmares we check availability of GPUs.
    if not has_gpus():
        # If there are no GPUs, mirrored strategy makes no sense.
        if use_mirrored_strategy:
            use_mirrored_strategy = False
            warnings.warn(
                "It does not make sense to use mirrored strategy "
                "when GPUs are not available.\n"
                "The parameter has been disabled."
            )
        # We check for drivers to try and give a more explainatory
        # warning about the absence of GPUs.
        if has_nvidia_drivers():
            warnings.warn(
                "It was not possible to detect GPUs but the system "
                "has NVIDIA drivers installed.\n"
                "It is very likely there is some mis-configuration "
                "with your TensorFlow instance.\n"
                "The model will train a LOT faster if you figure "
                "out what may be the cause of this issue on your "
                "system: sometimes a simple reboot will do a lot of good."
            )
        elif has_rocm_drivers():
            warnings.warn(
                "It was not possible to detect GPUs but the system "
                "has ROCM drivers installed.\n"
                "It is very likely there is some mis-configuration "
                "with your TensorFlow instance.\n"
                "The model will train a LOT faster if you figure "
                "out what may be the cause of this issue on your "
                "system: sometimes a simple reboot will do a lot of good."
            )
        else:
            warnings.warn(
                "It was neither possible to detect GPUs nor GPU drivers "
                "of any kind on your system (neither CUDA or ROCM).\n"
                "The model will proceed with trainining, but it will be "
                "significantly slower than what would be possible "
                "with GPU acceleration."
            )

    # If the fit kwargs are not given we normalize them to an empty dictionary.
    if fit_kwargs is None:
        fit_kwargs = {}

    # If the model requested is SkipGram and the given graph does not have sorted
    # node IDs according to decreasing outbound node degrees, we create the new graph
    # that has the node IDs sorted.
    if automatically_sort_by_decreasing_outbound_node_degree and node_embedding_method_name in REQUIRE_ZIPFIAN and not graph.has_nodes_sorted_by_decreasing_outbound_node_degree():
        graph = graph.sort_by_decreasing_outbound_node_degree()

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
        if node_embedding_method_name in RANDOM_WALK_BASED_MODELS:
            graph.enable(
                vector_sources=False,
                vector_destinations=True,
                vector_cumulative_node_degrees=True
            )
        if node_embedding_method_name in LINK_PREDICTION_BASED_MODELS:
            graph.enable(
                vector_sources=True,
                vector_destinations=True,
                vector_cumulative_node_degrees=False
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
        use_mirrored_strategy=use_mirrored_strategy,
        devices=devices,
        **kwargs
    )
