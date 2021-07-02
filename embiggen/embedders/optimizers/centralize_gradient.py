"""Submodule implementing method to centralize gradient over zero mean."""
from typing import List, Tuple
import tensorflow as tf


def centralize_gradient(gradient: tf.Tensor) -> tf.Tensor:
    """Centralize over zero mean the provided gradient.

    Parameters
    -----------------------
    gradient: tf.Tensor,
        Gradient to centralize over zero mean.
    """
    if len(gradient.shape) > 1:
        gradient -= tf.sparse.reduce_sum(
            gradient,
            axis=list(range(len(gradient.shape) - 1)),
            keepdims=True,
            output_is_sparse=True
        )
    return gradient


def centralize_gradients(gradients_and_vars: List[Tuple[tf.Tensor, tf.Tensor]]) -> List[Tuple[tf.Tensor, tf.Tensor]]:
    """Centralize over zero mean all the provided gradients.

    Parameters
    -----------------------
    gradients_and_vars: List[Tuple[tf.Tensor, tf.Tensor]],
        List of gradients and relative variables.
    """
    return [
        (centralize_gradient(gradient), var)
        for gradient, var in gradients_and_vars
    ]
