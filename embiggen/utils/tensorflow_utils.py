"""Submodule with utilities on TensorFlow versions."""
import tensorflow as tf
from packaging import version
from validate_version_code import validate_version_code
from ensmallen import Graph

def tensorflow_version_is_higher_or_equal_than(tensorflow_version: str) -> bool:
    """Returns boolean if the TensorFlow version is higher than provided one.

    Parameters
    ----------------------
    tensorflow_version: str,
        The version of TensorFlow to check against.

    Raises
    ----------------------
    ValueError,
        If the provided version code is not a valid one.

    Returns
    ----------------------
    Boolean representing if installed TensorFlow version is higher than given one.
    """
    if not validate_version_code(tensorflow_version):
        raise ValueError(
            (
                "The provided TensorFlow version code `{}` "
                "is not a valid version code."
            ).format(tensorflow_version)
        )
    return version.parse(tf.__version__) >= version.parse(tensorflow_version)


def tensorflow_version_is_less_or_equal_than(tensorflow_version: str) -> bool:
    """Returns boolean if the TensorFlow version is less or equal than provided one.

    Parameters
    ----------------------
    tensorflow_version: str,
        The version of TensorFlow to check against.

    Raises
    ----------------------
    ValueError,
        If the provided version code is not a valid one.

    Returns
    ----------------------
    Boolean representing if installed TensorFlow version is less or equal than given one.
    """
    if not validate_version_code(tensorflow_version):
        raise ValueError(
            (
                "The provided TensorFlow version code `{}` "
                "is not a valid version code."
            ).format(tensorflow_version)
        )
    return version.parse(tf.__version__) <= version.parse(tensorflow_version)


def must_have_tensorflow_version_higher_or_equal_than(
    tensorflow_version: str,
    feature_name: str = None
):
    """Returns boolean if the TensorFlow version is higher than provided one.

    Parameters
    ----------------------
    tensorflow_version: str,
        The version of TensorFlow to check against.
    feature_name: str = None,
        The name of the feature that the requested version of TensorFlow
        has and previous version do not have.

    Raises
    ----------------------
    ValueError,
        If the provided version code is not a valid one.
    ValueError,
        If the installed TensorFlow version is lower than requested one.
    """
    if not tensorflow_version_is_higher_or_equal_than(tensorflow_version):
        feature_message = ""
        if feature_name is not None:
            feature_message = "\nSpecifically, the feature requested is called `{}`.".format(
                feature_name
            )
        raise ValueError(
            (
                "The required minimum TensorFlow version is `{}`, but "
                "the installed TensorFlow version is `{}`.{}"
            ).format(
                tf.__version__,
                tensorflow_version,
                feature_message
            )
        )


def graph_to_sparse_tensor(
    graph: Graph,
    use_weights: bool
) -> tf.SparseTensor:
    """Returns provided graph as sparse Tensor.

    Parameters
    -------------------
    graph: Graph,
        The graph to convert.
    use_weights: bool,
        Whether to load the graph weights.

    Raises
    -------------------
    ValueError,
        If the weights are requested but the graph does not contain any.
    ValueError,
        If the graph contains singletons.
    ValueError,
        If the graph is a multigraph.

    Returns
    -------------------
    SparseTensor with (weighted) adjacency matrix.
    """
    if use_weights and not graph.has_edge_weights():
        raise ValueError(
            "Weighted were requested but the provided graph "
            "does not contain any edge weight."
        )

    if graph.has_singleton_nodes():
        raise ValueError(
            "The GCN model does not support operations on graph containing "
            "singletons. You need to either drop the singleton nodes or "
            "add to them a singleton."
        )

    if graph.is_multigraph():
        raise ValueError(
            "The GCN model does not support operations on a multigraph. "
            "You need to drop the parallel edges in order to execute this "
            "model."
        )

    return tf.SparseTensor(
        graph.get_edge_node_ids(directed=True),
        (
            graph.get_edge_weights()
            if use_weights
            else tf.ones(graph.get_number_of_directed_edges())
        ),
        (graph.get_nodes_number(), graph.get_nodes_number()),
    )
