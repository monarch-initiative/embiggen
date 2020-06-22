#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Embedding Utility Functions.

Manipulates or Processes Embeddings
* get_embedding
* calculates_cosine_similarity
Reads and Writes Embedding Data
* write_embeddings
* load_embeddings
"""

# import needed libraries
import pickle

import numpy as np  # type: ignore
import os
import os.path
import tensorflow as tf  # type: ignore

from typing import Dict, List, Union, Any


# TODO: consider updating writes_embeddings to not require id2word when writing embedding data


def get_embedding(x: Union[int, np.ndarray], embedding: Union[np.ndarray, tf.Variable], device: str = 'cpu')  \
        -> Union[np.ndarray, tf.Tensor]:
    """Get the embedding corresponding to the data points in x. Note, we ensure that this code is carried out on
    the CPU because some ops are not compatible with the GPU.

    Args:
        x: A integer representing a node or word index.
        embedding: A 2D tensor with shape (samples, sequence_length), where each entry is a sequence of integers.
        device: A string that indicates whether to run computations on (default=cpu).
    Returns:
        embedding: Corresponding embeddings, with shape (batch_size, embedding_dimension)
    Raises:
        ValueError: If the embedding variable is None.
    """

    if embedding is None:
        raise ValueError('No embedding data found (i.e. embedding is None)')
    with tf.device(device):
        return tf.nn.embedding_lookup(embedding, x)


def calculate_cosine_similarity(x_embed: tf.Tensor, embedding: Union[np.ndarray, tf.Variable], device: str = 'cpu') \
        -> Union[np.ndarray, tf.Tensor]:
    """Computes the cosine similarity between a provided embedding and all other embedding vectors.
        Args:
            x_embed: A Tensor containing word embeddings.
            embedding: A 2D tensor with shape (samples, sequence_length), where each entry is a sequence of integers.
            device: A string that indicates whether to run computations on (default=cpu).
        Returns:
            cosine_sim_op: A tensor of the cosine similarities between input data embedding and all other embeddings.
        """

    # !TODO: Consider implementing this simply in scipy

    with tf.device(device):
        x_embed_cast = tf.cast(x_embed, tf.float32)
        x_embed_norm = x_embed_cast / \
            tf.sqrt(tf.reduce_sum(tf.square(x_embed_cast)))
        x_embed_sqrt = tf.sqrt(tf.reduce_sum(
            tf.square(embedding), 1, keepdims=True), tf.float32)
        embedding_norm = embedding / x_embed_sqrt

        # calculate cosine similarity
        cosine_sim_op = tf.matmul(
            x_embed_norm, embedding_norm, transpose_b=True)

        return cosine_sim_op