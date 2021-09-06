"""GraphCBOW model for graph embedding."""
from typing import Dict, Union

import numpy as np
import pandas as pd
from ensmallen import Graph
from tensorflow.keras.optimizers import \
    Optimizer   # pylint: disable=import-error,no-name-in-module

from .cbow import CBOW
from .node2vec import Node2Vec


class GraphCBOW(Node2Vec):
    """GraphCBOW model for graph embedding.

    The GraphCBOW model for graoh embedding receives a list of contexts and tries
    to predict the central word. The model makes use of an NCE loss layer
    during the training process to generate the negatives.
    """

    def __init__(
        self,
        graph: Graph,
        embedding_size: int = 100,
        embedding: Union[np.ndarray, pd.DataFrame] = None,
        extra_features: Union[np.ndarray, pd.DataFrame] = None,
        optimizer: Union[str, Optimizer] = None,
        negative_samples: int = 10,
        walk_length: int = 128,
        batch_size: int = 256,
        iterations: int = 16,
        window_size: int = 4,
        return_weight: float = 1.0,
        explore_weight: float = 1.0,
        change_node_type_weight: float = 1.0,
        change_edge_type_weight: float = 1.0,
        max_neighbours: int = None,
        elapsed_epochs: int = 0,
        support_mirrored_strategy: bool = False,
        random_state: int = 42,
        dense_node_mapping: Dict[int, int] = None,
        use_gradient_centralization: bool = True,
    ):
        """Create new sequence Embedder model.

        Parameters
        -------------------------------------------
        graph: Graph,
            Graph to be embedded.
        word2vec_model: Word2Vec,
            Word2Vec model to use.
        embedding_size: int = 100,
            Dimension of the embedding.
        embedding: Union[np.ndarray, pd.DataFrame] = None,
            The seed embedding to be used.
            Note that it is not possible to provide at once both
            the embedding and either the vocabulary size or the embedding size.
        extra_features: Union[np.ndarray, pd.DataFrame] = None,
            Optional extra features to be used during the computation
            of the embedding. The features must be available for all the
            elements considered for the embedding.
        optimizer: Union[str, Optimizer] = None,
            The optimizer to be used during the training of the model.
            By default, if None is provided, Nadam with learning rate
            set at 0.01 is used.
        window_size: int = 4,
            Window size for the local context.
            On the borders the window size is trimmed.
        negative_samples: int = 10,
            The number of negative classes to randomly sample per batch.
            This single sample of negative classes is evaluated for each element in the batch.
        walk_length: int = 128,
            Maximal length of the walks.
        batch_size: int = 256,
            Number of nodes to include in a single batch.
        iterations: int = 16,
            Number of iterations of the single walks.
        window_size: int = 4,
            Window size for the local context.
            On the borders the window size is trimmed.
        return_weight: float = 1.0,
            Weight on the probability of returning to the same node the walk just came from
            Having this higher tends the walks to be
            more like a Breadth-First Search.
            Having this very high  (> 2) makes search very local.
            Equal to the inverse of p in the Node2Vec paper.
        explore_weight: float = 1.0,
            Weight on the probability of visiting a neighbor node
            to the one we're coming from in the random walk
            Having this higher tends the walks to be
            more like a Depth-First Search.
            Having this very high makes search more outward.
            Having this very low makes search very local.
            Equal to the inverse of q in the Node2Vec paper.
        change_node_type_weight: float = 1.0,
            Weight on the probability of visiting a neighbor node of a
            different type than the previous node. This only applies to
            colored graphs, otherwise it has no impact.
        change_edge_type_weight: float = 1.0,
            Weight on the probability of visiting a neighbor edge of a
            different type than the previous edge. This only applies to
            multigraphs, otherwise it has no impact.
        max_neighbours: int = None,
            Number of maximum neighbours to consider when using approximated walks.
            By default, None, we execute exact random walks.
            This is mainly useful for graphs containing nodes with extremely high degrees.
        elapsed_epochs: int = 0,
            Number of elapsed epochs to init state of generator.
        support_mirrored_strategy: bool = False,
            Wethever to patch support for mirror strategy.
            At the time of writing, TensorFlow's MirrorStrategy does not support
            input values different from floats, therefore to support it we need
            to convert the unsigned int 32 values that represent the indices of
            the embedding layers we receive from Ensmallen to floats.
            This will generally slow down performance, but in the context of
            exploiting multiple GPUs it may be unnoticeable.
        random_state: int = 42,
            The random state to reproduce the training sequence.
        dense_node_mapping: Dict[int, int] = None,
            Mapping to use for converting sparse walk space into a dense space.
            This object can be created using the method (available from the
            graph object created using Graph)
            called `get_dense_node_mapping` that returns a mapping from
            the non trap nodes (those from where a walk could start) and
            maps these nodes into a dense range of values.
        use_gradient_centralization: bool = True,
            Whether to wrap the provided optimizer into a normalized
            one that centralizes the gradient.
            It is automatically enabled if the current version of
            TensorFlow supports gradient transformers.
            More detail here: https://arxiv.org/pdf/2004.01461.pdf
        """
        if not graph.has_nodes_sorted_by_decreasing_outbound_node_degree():
            raise ValueError(
                "The given graph does not have the nodes sorted by decreasing "
                "order, therefore the NCE loss sampling (which follows a zipfian "
                "distribution) would not approximate well the Softmax.\n"
                "In order to sort the given graph in such a way that the node IDs "
                "are sorted by decreasing outbound node degrees, you can use "
                "the Graph method "
                "`graph.sort_by_decreasing_outbound_node_degree()`"
            )
        super().__init__(
            graph=graph,
            word2vec_model=CBOW,
            embedding_size=embedding_size,
            embedding=embedding,
            extra_features=extra_features,
            optimizer=optimizer,
            negative_samples=negative_samples,
            walk_length=walk_length,
            batch_size=batch_size,
            iterations=iterations,
            window_size=window_size,
            return_weight=return_weight,
            explore_weight=explore_weight,
            change_node_type_weight=change_node_type_weight,
            change_edge_type_weight=change_edge_type_weight,
            max_neighbours=max_neighbours,
            elapsed_epochs=elapsed_epochs,
            support_mirrored_strategy=support_mirrored_strategy,
            random_state=random_state,
            dense_node_mapping=dense_node_mapping,
            use_gradient_centralization=use_gradient_centralization
        )
