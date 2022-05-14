"""Module providing abstract Node2Vec implementation."""
from typing import Optional, Union, Dict, Any
from ensmallen import Graph
import numpy as np
import pandas as pd
from ensmallen import models
from ...utils import AbstractEmbeddingModel, abstract_class


@abstract_class
class Node2VecEnsmallen(AbstractEmbeddingModel):
    """Abstract class for Node2Vec algorithms."""

    MODELS = {
        "cbow": models.CBOW,
        "kgcbow": models.KGCBOW,
        "skipgram": models.SkipGram
    }

    def __init__(
        self,
        model_name: str,
        embedding_size: int = 100,
        epochs: int = 10,
        clipping_value: float = 6.0,
        number_of_negative_samples: int = 5,
        log_sigmoid: bool = True,
        siamese: bool = False,
        walk_length: int = 128,
        iterations: int = 1,
        window_size: int = 4,
        return_weight: float = 1.0,
        explore_weight: float = 1.0,
        change_node_type_weight: float = 1.0,
        change_edge_type_weight: float = 1.0,
        max_neighbours: int = 100,
        learning_rate: float = 0.01,
        learning_rate_decay: float = 0.9,
        normalize_by_degree: bool = False,
        stochastic_downsample_by_degree: bool = False,
        normalize_learning_rate_by_degree: bool = False,
        use_zipfian_sampling: bool = True,
        random_state: int = 42,
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
        max_neighbours: int = 100
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
        stochastic_downsample_by_degree: bool = False
            Randomly skip samples with probability proportional to the degree of the central node. By default false.
        normalize_learning_rate_by_degree: bool = False
            Divide the learning rate by the degree of the central node. By default false.
        use_zipfian_sampling: bool = True
            Sample negatives proportionally to their degree. By default true.
        random_state: int = 42
            The random state to reproduce the training sequence.
        """
        if model_name.lower() not in ("cbow", "skipgram"):
            raise ValueError(
                (
                    "The provided model name {} is not supported. "
                    "The supported models are `CBOW`, `KGCBOW` and `SkipGram`."
                ).format(model_name)
            )

        self._epochs = epochs
        self._learning_rate = learning_rate
        self._learning_rate_decay = learning_rate_decay
        self._embedding_size = embedding_size
        self._window_size = window_size
        self._clipping_value = clipping_value
        self._number_of_negative_samples = number_of_negative_samples
        self._log_sigmoid = log_sigmoid
        self._siamese = siamese
        self._walk_length = walk_length
        self._return_weight = return_weight
        self._explore_weight = explore_weight
        self._change_edge_type_weight = change_edge_type_weight
        self._change_node_type_weight = change_node_type_weight
        self._max_neighbours = max_neighbours
        self._random_state = random_state
        self._iterations = iterations
        self._normalize_by_degree = normalize_by_degree
        self._stochastic_downsample_by_degree = stochastic_downsample_by_degree
        self._normalize_learning_rate_by_degree = normalize_learning_rate_by_degree
        self._use_zipfian_sampling = use_zipfian_sampling

        self._model = Node2VecEnsmallen.MODELS[model_name.lower()](
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
            normalize_by_degree=normalize_by_degree,
            stochastic_downsample_by_degree=stochastic_downsample_by_degree,
            normalize_learning_rate_by_degree=normalize_learning_rate_by_degree,
            use_zipfian_sampling=use_zipfian_sampling,
        )

        super().__init__(embedding_size=embedding_size)

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters of the model."""
        return {
            **super().parameters(),
            **dict(
                epochs=self._epochs,
                learning_rate=self._learning_rate,
                learning_rate_decay=self._learning_rate_decay,
                embedding_size=self._embedding_size,
                window_size=self._window_size,
                clipping_value=self._clipping_value,
                number_of_negative_samples=self._number_of_negative_samples,
                log_sigmoid=self._log_sigmoid,
                siamese=self._siamese,
                walk_length=self._walk_length,
                return_weight=self._return_weight,
                explore_weight=self._explore_weight,
                change_edge_type_weight=self._change_edge_type_weight,
                change_node_type_weight=self._change_node_type_weight,
                max_neighbours=self._max_neighbours,
                random_state=self._random_state,
                iterations=self._iterations,
                normalize_by_degree=self._normalize_by_degree,
                stochastic_downsample_by_degree=self._stochastic_downsample_by_degree,
                normalize_learning_rate_by_degree=self._normalize_learning_rate_by_degree,
                use_zipfian_sampling=self._use_zipfian_sampling
            )
        }

    @staticmethod
    def task_name() -> str:
        return "Node Embedding"

    @staticmethod
    def library_name() -> str:
        return "Ensmallen"

    def requires_nodes_sorted_by_decreasing_node_degree(self) -> bool:
        return False

    def is_topological(self) -> bool:
        return True

    def _fit_transform(
        self,
        graph: Graph,
        return_dataframe: bool = True,
        verbose: bool = True
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Return node embedding."""
        node_embedding = self._model.fit_transform(
            graph,
            epochs=self._epochs,
            learning_rate=self._learning_rate,
            learning_rate_decay=self._learning_rate_decay,
            verbose=verbose
        )
        if return_dataframe:
            return pd.DataFrame(
                node_embedding,
                index=graph.get_node_names()
            )
        return node_embedding

    @staticmethod
    def requires_edge_weights() -> bool:
        return False

    @staticmethod
    def requires_positive_edge_weights() -> bool:
        return True