"""Submodule implementing method to centralize gradient over zero mean.

This operation is described in the following paper: https://arxiv.org/pdf/2004.01461.pdf

The code is based on implementations provided by:
- https://ai.plainenglish.io/gradient-centralization-in-keras-9e4e34a8b895
- https://keras.io/examples/vision/gradient_centralization/
- https://github.com/Rishit-dagli/Gradient-Centralization-TensorFlow
"""
from typing import List

import tensorflow as tf
import tensorflow.keras.backend as K  # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.optimizers import Optimizer  # pylint: disable=import-error,no-name-in-module


def get_centralized_gradient(gradient: tf.Tensor) -> tf.Tensor:
    """Centralize over zero mean the provided gradient.

    Parameters
    -----------------------
    gradient: tf.Tensor,
        Gradient to centralize over zero mean.

    References
    -----------------------
    [Gradient Centralization: A New Optimization Technique for Deep Neural Networks](https://arxiv.org/pdf/2004.01461.pdf)
    """
    if len(gradient.shape) > 1:
        gradient -= tf.reduce_mean(
            gradient,
            axis=list(range(len(gradient.shape) - 1)),
            keepdims=True
        )
    return gradient


def get_centralized_gradients(
    optimizer: Optimizer,
    loss: tf.Tensor,
    params: List
) -> List[tf.Tensor]:
    """Centralize over zero mean all the provided gradients.

    Parameters
    -----------------------
    optimizer: Optimizer,
        The optimizer instance (e.g. Adam, Nadam, ...)
    loss: tf.Tensor,
        The loss function of this model.
    params: List,
        Parameters relevant to the computation of the gradient.

    References
    -----------------------
    [Gradient Centralization: A New Optimization Technique for Deep Neural Networks](https://arxiv.org/pdf/2004.01461.pdf)
    """
    # We compute the gradients with zero mean
    gradients = [
        get_centralized_gradient(gradient)
        for gradient in K.gradients(loss, params)
    ]

    # We check that no operation without gradient
    # appeared in the computation graph.
    if None in gradients:
        raise ValueError(
            'An operation has `None` for gradient. '
            'Please make sure that all of your ops have a '
            'gradient defined (i.e. are differentiable). '
            'Common ops without gradient: '
            'K.argmax, K.round, K.eval.'
        )
    # If the optimizer has the clipnorm feature,
    # and the optimizer has a greater than zero clipping norm
    # we execute the operation.
    if hasattr(optimizer, 'clipnorm') and optimizer.clipnorm > 0:
       # First we compute the gradients norm.
        norm = K.sqrt(sum([
            K.sum(K.square(g))
            for g in gradients
        ]))
        # The normalize the various computed gradients
        gradients = [
            tf.keras.optimizers.clip_norm(
                gradient,
                optimizer.clipnorm,
                norm
            )
            for gradient in gradients
        ]

    # Similarly, if the optimizer has the clipvalue feature,
    # and the optimizer has a greater than zero clipping value
    # we execute the operation
    if hasattr(optimizer, 'clipvalue') and optimizer.clipvalue > 0:
        gradients = [
            K.clip(
                gradient,
                -optimizer.clipvalue,
                optimizer.clipvalue
            )
            for gradient in gradients
        ]
    return gradients


def apply_centralized_gradients(optimizers: Optimizer):
    """Replaces the get_gradients method with the one centralizing the gradient."""
    optimizers.get_gradients = get_centralized_gradients
