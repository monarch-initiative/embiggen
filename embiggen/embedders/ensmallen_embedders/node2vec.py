"""Module providing abstract Node2Vec implementation."""
from typing import Optional
from ensmallen import Graph
import numpy as np
from ensmallen import models
from .ensmallen_embedder import EnsmallenEmbedder


class Node2Vec(EnsmallenEmbedder):
    """Abstract class for Node2Vec algorithms."""

    MODELS = {
        "cbow": models.CBOW,
        "skipgram": models.SkipGram
    }

    def __init__(
        self,
        model_name: str,
        embedding_size: int = 100,
        epochs: int = 10,
        clipping_value: float = 6.0,
        number_of_negative_samples: int = 5,
        log_sigmoid: Optional[bool] = True,
        siamese: Optional[bool] = False,
        walk_length: int = 128,
        iterations: int = 1,
        window_size: int = 4,
        return_weight: float = 1.0,
        explore_weight: float = 1.0,
        change_node_type_weight: float = 1.0,
        change_edge_type_weight: float = 1.0,
        max_neighbours: Optional[int] = 100,
        learning_rate: float = 0.01,
        learning_rate_decay: float = 0.9,
        normalize_by_degree: bool = False,
        random_state: int = 42,
        verbose: bool = True
    ):
        """Create new abstract Node2Vec method.

        Parameters
        --------------------------
        model_name: str
            The ensmallen graph embedding model to use.
            Can either be the SkipGram of CBOW models.
        embedding_size: int = 100
            Dimension of the embedding.
        epochs: int = 10
            Number of epochs to train the model for.
        window_size: int = 4
            Window size for the local context.
            On the borders the window size is trimmed.
        clipping_value: float = 6.0
            Value at which we clip the dot product, mostly for numerical stability issues.
            By default, `6.0`, where the loss is already close to zero.
        number_of_negative_samples: int = 5
            The number of negative classes to randomly sample per batch.
            This single sample of negative classes is evaluated for each element in the batch.
        walk_length: int = 128
            Maximal length of the walks.
        iterations: int = 1
            Number of iterations of the single walks.
        window_size: int = 4
            Window size for the local context.
            On the borders the window size is trimmed.
        return_weight: float = 1.0
            Weight on the probability of returning to the same node the walk just came from
            Having this higher tends the walks to be
            more like a Breadth-First Search.
            Having this very high  (> 2) makes search very local.
            Equal to the inverse of p in the Node2Vec paper.
        explore_weight: float = 1.0
            Weight on the probability of visiting a neighbor node
            to the one we're coming from in the random walk
            Having this higher tends the walks to be
            more like a Depth-First Search.
            Having this very high makes search more outward.
            Having this very low makes search very local.
            Equal to the inverse of q in the Node2Vec paper.
        change_node_type_weight: float = 1.0
            Weight on the probability of visiting a neighbor node of a
            different type than the previous node. This only applies to
            colored graphs, otherwise it has no impact.
        change_edge_type_weight: float = 1.0
            Weight on the probability of visiting a neighbor edge of a
            different type than the previous edge. This only applies to
            multigraphs, otherwise it has no impact.
        max_neighbours: Optional[int] = 100
            Number of maximum neighbours to consider when using approximated walks.
            By default, None, we execute exact random walks.
            This is mainly useful for graphs containing nodes with high degrees.
        learning_rate: float = 0.01
            The learning rate to use to train the Node2Vec model.
        learning_rate_decay: float = 0.9
            Factor to reduce the learning rate for at each epoch. By default 0.9.
        normalize_by_degree: bool = False
            Whether to normalize the random walk by the node degree
            of the destination node degrees.
        random_state: int = 42
            The random state to reproduce the training sequence.
        verbose: bool = True
            Whether to show loading bars
        """
        if model_name.lower() not in ("cbow", "skipgram"):
            raise ValueError(
                (
                    "The provided model name {} is not supported. "
                    "The supported models are `CBOW` and `SkipGram`."
                ).format(model_name)
            )
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._learning_rate_decay = learning_rate_decay

        self._model = Node2Vec.MODELS[model_name.lower()](
            embedding_size=embedding_size,
            window_size=window_size,
            clipping_value=clipping_value,
            number_of_negative_samples=number_of_negative_samples,
            log_sigmoid=log_sigmoid,
            siamese=siamese,
            walk_length=walk_length,
            return_weight=return_weight,
            explore_weight=explore_weight,
            change_edge_type_weight=change_edge_type_weight,
            change_node_type_weight=change_node_type_weight,
            max_neighbours=max_neighbours,
            random_state=random_state,
            iterations=iterations,
            normalize_by_degree=normalize_by_degree
        )

        super().__init__(
            embedding_size=embedding_size,
            verbose=verbose
        )

    def _fit_transform_graph(self, graph: Graph) -> np.ndarray:
        return self._model.fit_transform(
            graph, 
            epochs=self._epochs,
            learning_rate=self._learning_rate,
            learning_rate_decay=self._learning_rate_decay,
            verbose=self._verbose
        )