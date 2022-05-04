"""Sub-module with methods to compute node-embedding with a one-liner."""
import inspect
from typing import Dict, List, Tuple, Union

import pandas as pd
from cache_decorator import Cache
from ensmallen import Graph
from ensmallen.datasets import get_dataset

from ..utils import execute_gpu_checks, get_available_gpus_number, has_gpus

from ..embedders import SUPPORTED_NODE_EMBEDDING_METHODS
from ..embedders.ensmallen_embedders import EnsmallenEmbedder

REQUIRE_ZIPFIAN = [
    "cbow",
    "skipgram"
]

RANDOM_WALK_BASED_MODELS = [
    "cbow",
    "glove",
    "skipgram",
    "spine",
    "weightedspine"
]

EDGE_PREDICTION_BASED_MODELS = [
    "siamese",
    "transr",
    "transe",
    "transh",
    "simple"
]


def get_available_node_embedding_methods() -> List[str]:
    """Return list of supported node embedding methods."""
    return list(SUPPORTED_NODE_EMBEDDING_METHODS.keys())


def get_node_embedding_method(
    node_embedding_method_name: str,
    use_only_cpu: bool
):
    """Return node embedding method curresponding to given name.

    Parameters
    ---------------------
    node_embedding_method_name: str
        The name of the embedding method to retrieve
    use_only_cpu: bool
        Whether to retrieve CPU or GPU versions of the model, when available.
    """
    model = SUPPORTED_NODE_EMBEDDING_METHODS[node_embedding_method_name]
    # If there is a further choice to be made for this model.
    if isinstance(model, dict):
        model = model["cpu"] if use_only_cpu else model["gpu"]
    return model


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
        "node_embeddings/{node_embedding_method_name}/tensorflow/{graph_name}/{_hash}_embedding.pkl.gz",
        "node_embeddings/{node_embedding_method_name}/tensorflow/{graph_name}/{_hash}_training_history.csv.xz",
    ],
    args_to_ignore=["devices", "use_mirrored_strategy", "verbose"]
)
def _compute_node_tensorflow_embedding(
    graph: Graph,
    graph_name: str,  # pylint: disable=unused-argument
    node_embedding_method_name: str,
    fit_kwargs: Dict,
    verbose: bool,
    use_mirrored_strategy: bool,
    devices: Union[List[str], str],
    **kwargs: Dict
) -> Tuple[Union[pd.DataFrame, Tuple[pd.DataFrame]], pd.DataFrame]:
    """Wrapper for `compute_node_embedding` method."""
    # Since the verbose kwarg may be provided also on the fit_kwargs
    # we normalize the parameter to avoid collisions.
    verbose = fit_kwargs.pop("verbose", verbose)

    node_embedding_model = get_node_embedding_method(
        node_embedding_method_name,
        False
    )
    # Otherwise it is a TensorFlow-based model.
    import tensorflow as tf
    if tf.config.list_physical_devices('GPU') and use_mirrored_strategy:
        strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
            devices
        )
    else:
        strategy = tf.distribute.get_strategy()

    with strategy.scope():
        # Creating the node embedding model
        model = node_embedding_model(
            graph=graph,
            **kwargs
        )

    # Fitting the node embedding model
    history = model.fit(
        verbose=verbose,
        **fit_kwargs
    )
    return model.get_embedding_dataframe(), history


@Cache(
    cache_path="node_embeddings/{node_embedding_method_name}/ensmallen/{graph_name}/{_hash}_embedding.csv.gz",
    args_to_ignore=["verbose"]
)
def _compute_node_ensmallen_embedding(
    graph: Graph,
    graph_name: str,  # pylint: disable=unused-argument
    node_embedding_method_name: str,
    fit_kwargs: Dict,
    verbose: bool,
    **kwargs: Dict
) -> Tuple[Union[pd.DataFrame, Tuple[pd.DataFrame]], pd.DataFrame]:
    """Wrapper for `compute_node_embedding` method."""
    # Since the verbose kwarg may be provided also on the fit_kwargs
    # we normalize the parameter to avoid collisions.
    verbose = fit_kwargs.pop("verbose", verbose)

    node_embedding_model = get_node_embedding_method(
        node_embedding_method_name,
        True
    )

    # Create the Ensmallen-based model.
    model = node_embedding_model(
        verbose=verbose,
        **kwargs
    )
    return model.fit_transform(
        graph,
        **fit_kwargs
    )


def compute_node_embedding(
    graph: Union[Graph, str],
    node_embedding_method_name: str,
    use_only_cpu: bool = True,
    use_mirrored_strategy: Union[bool, str] = "auto",
    devices: Union[List[str], str] = None,
    fit_kwargs: Dict = None,
    verbose: Union[bool, int] = True,
    automatically_drop_unsupported_parameters: bool = False,
    automatically_enable_time_memory_tradeoffs: bool = True,
    **kwargs: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return embedding computed using SkipGram on given graph.

    Parameters
    --------------------------
    graph: Union[Graph, str],
        Graph to embed.
        If a graph name is provided, we try to retrieve
        it using the Ensmallen automatic retrieval.
    node_embedding_method_name: str,
        The name of the node embedding method to use.
    use_only_cpu: bool = True,
        Whether to only use CPU.
        Do note that for CBOW and SkipGram models,
        this will switch the implementation from the
        TensorFlow implementation and will use our Rust Ensmallen one.
    use_mirrored_strategy: Union[bool, str] = "auto"
        Whether to use mirror strategy to distribute the
        computation across multiple devices.
        This is automatically enabled if more than one
        GPU is detected and the flag `use_only_gpu` was
        not provided, or if the list of devices to use
        was provided and it includes at least a GPU.
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
    lower_node_embedding_method_name = node_embedding_method_name.lower()
    if not is_node_embedding_method_supported(lower_node_embedding_method_name):
        raise ValueError(
            (
                "The given node embedding method `{}` is not supported. "
                "The supported node embedding methods are `{}`."
            ).format(
                node_embedding_method_name,
                get_available_node_embedding_methods()
            )
        )
    
    if isinstance(graph, str):
        graph = get_dataset(graph)()

    # If devices are given as a single device we adapt this into a list.
    if isinstance(devices, str):
        devices = [devices]

    # If in the list of provided devices there is a GPU specified,
    # and there are more than one GPU, we need to use the MirroredStrategy
    # to distribute its computation.
    if devices and any(
        "GPU" in device
        for device in devices
    ) and get_available_gpus_number() > 1:
        use_mirrored_strategy = True

    # If the use only CPU parameter is set to auto,
    # we automatically enable it when no GPUs are
    # available.
    if use_only_cpu == "auto" and not has_gpus():
        use_only_cpu = True

    if not use_only_cpu and use_mirrored_strategy == "auto" and get_available_gpus_number() > 1:
        use_mirrored_strategy = True

    if use_only_cpu and use_mirrored_strategy == True:
        raise ValueError(
            "It does not make sense to require to use only CPU "
            "and require to use the mirrored strategy for GPUs."
        )

    # If the fit kwargs are not given we normalize them to an empty dictionary.
    if fit_kwargs is None:
        fit_kwargs = {}

    # If required, we filter out the unsupported parameters.
    # This may be useful when running a suite of experiments with a set of
    # parameters and you do not want to bother in dropping the parameters
    # that are only supported in a subset of methods.
    if automatically_drop_unsupported_parameters:
        if kwargs:
            # Get the list of supported parameters
            supported_parameter = inspect.signature(
                get_node_embedding_method(
                    lower_node_embedding_method_name,
                    use_only_cpu
                ).__init__
            ).parameters
            # Filter out the unsupported parameters
            kwargs = {
                key: value
                for key, value in kwargs.items()
                if key in supported_parameter
            }
        if fit_kwargs:
            # Get the list of supported parameters
            model = get_node_embedding_method(
                lower_node_embedding_method_name,
                use_only_cpu
            )
            if issubclass(model, EnsmallenEmbedder):
                supported_parameter = inspect.signature(
                    model.fit_transform
                ).parameters
            else:
                supported_parameter = inspect.signature(
                    model.fit
                ).parameters
            # Filter out the unsupported parameters
            fit_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key in supported_parameter
            }

    # If required we enable the time memory tradeoffs.
    if automatically_enable_time_memory_tradeoffs:
        if lower_node_embedding_method_name in RANDOM_WALK_BASED_MODELS:
            graph.enable(
                vector_sources=False,
                vector_destinations=True,
                vector_cumulative_node_degrees=True
            )
        if lower_node_embedding_method_name in EDGE_PREDICTION_BASED_MODELS:
            graph.enable(
                vector_sources=True,
                vector_destinations=True,
                vector_cumulative_node_degrees=False
            )

    if issubclass(get_node_embedding_method(
        lower_node_embedding_method_name,
        use_only_cpu
    ), EnsmallenEmbedder):
        # Call the wrapper with cache.
        # Do note that this model does not return any history.
        return _compute_node_ensmallen_embedding(
            graph,
            graph_name=graph.get_name(),
            node_embedding_method_name=lower_node_embedding_method_name,
            fit_kwargs=fit_kwargs,
            verbose=verbose,
            **kwargs
        ), None
    return _compute_node_tensorflow_embedding(
        graph,
        graph_name=graph.get_name(),
        node_embedding_method_name=lower_node_embedding_method_name,
        fit_kwargs=fit_kwargs,
        verbose=verbose,
        use_mirrored_strategy=use_mirrored_strategy,
        devices=devices,
        **kwargs
    )
