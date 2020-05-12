import numpy as np  # type:ignore
import tensorflow as tf  # type:ignore

from typing import List, Tuple, Union


def calculate_total_probs(j: np.ndarray, q: np.ndarray) -> np.array:
    """Use the alias method to calculate the total probabilities of the discrete events.

    Args:
        j: A vector of node aliases (e.g. array([2, 3, 0, 2])).
        q: A vector of alias-method probabilities (e.g. array([0.5, 0.5, 1. , 0.5])).

    Returns:
        An array of floats representing the total probabilities of the discrete event (e.g. array([0.125, 0.125,
        0.5  , 0.25 ])).
    """

    n = len(j)
    probs = np.zeros(n)

    for i in range(n):
        p = q[i]
        probs[i] += p

        if p < 1.0:
            alias_index = j[i]
            probs[alias_index] += 1 - p

    s = np.sum(probs)

    return probs / s


def gets_tensor_length(tensor: Union[tf.data.Dataset, tf.RaggedTensor]) -> int:
    """Returns the length of a tensorflow object.

    Args:
        tensor: A tf.Dataset or tf.RaggedTensor object.

    Returns:
        An integer representing the length of the tensor.
    """

    if isinstance(tensor, tf.RaggedTensor):
        return tensor.shape[0]
    else:
        return len([edge for edge in tensor])


def searches_tensor(test_edge:  Union[Tuple, List], tensor: Union[tf.data.Dataset, tf.RaggedTensor]) -> bool:
    """Searches a tensorflow object for an edge, which is represented as a tensorflow Variable object.

    NOTE. This method works for tf.Dataset and tf.RaggedTensor objects.

    Args:
        test_edge: A tf.Variable object.
        tensor: A tf.Dataset or tf.RaggedTensor object.

    Returns:
        An bool indicating whether or not the test_edge was found in the tensor flow object.
    """

    # check if edge object contains ints or string
    if all(x for x in test_edge if isinstance(x, str)):
        tf_edge = tf.Variable(test_edge, dtype=tf.string)
    else:
        tf_edge = tf.Variable(test_edge, dtype=tf.int32)

    # find edge in tensor stack
    search_result = [edge for edge in tensor if all(node for node in (edge == tf_edge).numpy())]

    # if edge is in tensor stack, length should be greater than 1
    return len(search_result) > 0
