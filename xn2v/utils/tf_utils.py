import tensorflow as tf  # type:ignore

from typing import List, Tuple, Union


def gets_tensor_length(tensor: Union[tf.data.Dataset, tf.RaggedTensor]) -> int:
    """Returns the length of a tensorflow object.

    Args:
        tensor: A tf.Dataset or tf.RaggedTensor object.

    Returns:
        An integer representing the length of the tensor.

    Raises:
        TypeError: If an object other than a tf.data.Dataset or tf.RaggedTensor is passed.
    """

    if isinstance(tensor, tf.RaggedTensor):
        return tensor.shape[0]
    elif isinstance(tensor, tf.data.Dataset):
        return len([edge for edge in tensor])
    else:
        raise TypeError('tensor object must be of type tf.data.Dataset or tf.RaggedTensor')


def searches_tensor(test_edge:  Union[Tuple, List], tensor: Union[tf.data.Dataset, tf.RaggedTensor]) -> bool:
    """Searches a tensorflow object for an edge, which is represented as a tensorflow Variable object.

    NOTE. This method works for tf.Dataset and tf.RaggedTensor objects.

    Args:
        test_edge: A tf.Variable object.
        tensor: A tf.Dataset or tf.RaggedTensor object.

    Returns:
        An bool indicating whether or not the test_edge was found in the tensor flow object.

    Raises:
        ValueError: If the object passed to tf.Variable contains mixed types (i.e. (1, "a")).
    """

    # check if edge object contains ints or string
    if all(x for x in test_edge if isinstance(x, str)):
        tf_edge = tf.Variable(test_edge, dtype=tf.string)
    elif all(x for x in test_edge if isinstance(x, int)):
        tf_edge = tf.Variable(test_edge, dtype=tf.int32)
    else:
        raise ValueError('tensor object cannot contain mixed data types')

    # find edge in tensor stack
    search_result = [edge for edge in tensor if all(node for node in (edge == tf_edge).numpy())]

    # if edge is in tensor stack, length should be greater than 1
    return len(search_result) > 0
