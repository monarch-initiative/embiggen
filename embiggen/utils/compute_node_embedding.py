"""Sub-module with methods to compute node-embedding with a one-liner."""
from typing import Dict, List, Union

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
    cache_path="node_embeddings/{node_embedding_method_name}/{graph_name}/{_hash}.csv.xz",
    args_to_ignore=["devices"]
)
def _compute_node_embedding(
    graph: EnsmallenGraph,
    graph_name: str,  # pylint: disable=unused-argument
    node_embedding_method_name: str,
    fit_kwargs: Dict,
    devices: Union[List[str], str] = None,
    **kwargs: Dict
) -> pd.DataFrame:
    """Return embedding computed with required node embedding method.

    Specifically, this method also caches the embedding automatically.

    Parameters
    ---------------------
    graph: EnsmallenGraph,
        The graph to embed.
    graph_name: str,
        The name of the graph.
    node_embedding_method_name: str,
        The name of the node embedding method to use.
    fit_kwargs: Dict,
        Arguments to pass to the fit call.
    devices: Union[List[str], str] = None,
        The devices to use.
        If None, all GPU devices available are used.
    **kwargs: Dict,
        Arguments to pass to the node embedding method constructor.
        Read the documentation of the selected method.
    """

    strategy = tf.distribute.MirroredStrategy(devices=devices)
    with strategy.scope():
        # Creating the node embedding model
        model = get_node_embedding_method(node_embedding_method_name)(
            graph,
            support_mirror_strategy=True,
            **kwargs
        )
        # Fitting the node embedding model
        model.fit(**fit_kwargs)
        # Extracting computed embedding
        node_embedding = model.get_embedding_dataframe()
    return node_embedding


def compute_node_embedding(
    graph: EnsmallenGraph,
    node_embedding_method_name: str,
    devices: Union[List[str], str] = None,
    fit_kwargs: Dict = None,
    **kwargs: Dict
) -> pd.DataFrame:
    """Return embedding computed using SkipGram on given graph.

    Parameters
    ----------------------
    graph: EnsmallenGraph,
        Graph to embed.
    node_embedding_method_name: str,
        The name of the node embedding method to use.
    devices: Union[List[str], str] = None,
        The devices to use.
        If None, all GPU devices available are used.
    fit_kwargs: Dict = None,
        Arguments to pass to the fit call.
    **kwargs: Dict,
        Arguments to pass to the node embedding method constructor.
        Read the documentation of the selected method.

    Returns
    ---------------------
    Pandas dataframe with the computed embedding.
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
    if fit_kwargs is None:
        fit_kwargs = {}

    # If devices are given as a single device we adapt this into a list.
    if isinstance(devices, str):
        devices = [devices]
    return _compute_node_embedding(
        graph,
        graph_name=graph.get_name(),
        node_embedding_method_name=node_embedding_method_name,
        fit_kwargs=fit_kwargs,
        devices=devices,
        **kwargs
    )
