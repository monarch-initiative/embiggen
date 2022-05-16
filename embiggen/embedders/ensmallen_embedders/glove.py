"""Module providing GloVe model implementation."""
from typing import Optional, Dict, Any, Union
from ensmallen import Graph
from ensmallen import models
import numpy as np
import pandas as pd
from ...utils import AbstractEmbeddingModel


class GloVeEnsmallen(AbstractEmbeddingModel):
    """Class providing GloVe implemeted in Rust from Ensmallen."""

    def __init__(
        self,
        embedding_size: int = 100,
        epochs: int = 10,
        clipping_value: float = 6.0,
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
    ):
        """Create new abstract Node2Vec method.

        Parameters
        --------------------------
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
            The learning rate to use to train the Node2Vec model. By default 0.01.
        learning_rate_decay: float = 0.9
            Factor to reduce the learning rate for at each epoch. By default 0.9.
        normalize_by_degree: bool = False
            Whether to normalize the random walk by the node degree
            of the destination node degrees.
        random_state: int = 42
            The random state to reproduce the training sequence.
        """

        self._epochs=epochs
        self._learning_rate=learning_rate,
        self._learning_rate_decay=learning_rate_decay,

        self._model_kwargs = dict(
            embedding_size=embedding_size,
            clipping_value=clipping_value,
            walk_length=walk_length,
            iterations=iterations,
            window_size=window_size,
            return_weight=return_weight,
            explore_weight=explore_weight,
            change_edge_type_weight=change_edge_type_weight,
            change_node_type_weight=change_node_type_weight,
            max_neighbours=max_neighbours,
            normalize_by_degree=normalize_by_degree,
            random_state=random_state,
        )

        self._model = models.GloVe(**self._model_kwargs)

        super().__init__(
            embedding_size=embedding_size
        )

    @staticmethod
    def smoke_test_parameters() -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            embedding_size=5,
            epochs=1,
            window_size=1,
            walk_length=4,
            iterations=1,
            max_neighbours= 10,
            number_of_negative_samples=1
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters of the model."""
        return {
            **super().parameters(),
            **self._model_kwargs,
            **dict(
                epochs=self._epochs,
                learning_rate=self._learning_rate,
                learning_rate_decay=self._learning_rate_decay,
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
    
    @staticmethod
    def model_name() -> str:
        """Returns name of the model."""
        return "GloVe"

    @staticmethod
    def requires_node_types() -> bool:
        return False

    @staticmethod
    def requires_edge_types() -> bool:
        return False