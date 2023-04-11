"""Node2Vec wrapper for FastNode2Vec numba-based node embedding library."""
from typing import Dict, Union, Any

import numpy as np
import pandas as pd
from ensmallen import Graph

from embiggen.utils.abstract_models import AbstractEmbeddingModel, EmbeddingResult
from fastnode2vec import Graph as FNGraph
from fastnode2vec import Node2Vec
from multiprocessing import cpu_count
from time import time


class Node2VecFastNode2Vec(AbstractEmbeddingModel):
    """Node2Vec wrapper for FastNode2Vec numba-based node embedding library."""

    def __init__(
        self,
        embedding_size: int = 100,
        epochs: int = 30,
        walk_length: int = 128,
        iterations: int = 10,
        window_size: int = 5,
        learning_rate: float = 0.01,
        return_weight: float = 0.25,
        explore_weight: float = 4.0,
        number_of_workers: Union[int, str] = "auto",
        verbose: bool = False,
        random_state: int = 42,
        ring_bell: bool = False,
        enable_cache: bool = False
    ):
        """Create new wrapper for Node2Vec model from FastNode2Vec library.

        Parameters
        -------------------------
        embedding_size: int = 100
            The dimension of the embedding to compute.
        epochs: int = 100
            The number of epochs to use to train the model for.
        learning_rate: float = 0.01
            Learning rate of the model.
        verbose: bool = False
            Whether to show the loading bar.
        random_state: int = 42
            Random seed to use while training the model
        ring_bell: bool = False,
            Whether to play a sound when embedding completes.
        enable_cache: bool = False
            Whether to enable the cache, that is to
            store the computed embedding.
        """
        self._walk_length = walk_length
        self._iterations = iterations
        self._window_size = window_size
        self._return_weight = return_weight
        self._explore_weight = explore_weight
        self._epochs = epochs
        self._verbose = verbose
        self._learning_rate = learning_rate
        self._time_required_by_last_embedding = None
        if number_of_workers == "auto":
            number_of_workers = cpu_count()
        self._number_of_workers = number_of_workers

        super().__init__(
            embedding_size=embedding_size,
            enable_cache=enable_cache,
            ring_bell=ring_bell,
            random_state=random_state
        )

    @classmethod
    def smoke_test_parameters(cls) -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            **super().smoke_test_parameters(),
            window_size=1,
            walk_length=2,
            iterations=1
        )

    def parameters(self) -> Dict[str, Any]:
        return dict(
            **super().parameters(),
            **dict(
                epochs=self._epochs,
                walk_length=self._walk_length,
                window_size=self._window_size,
                iterations=self._iterations,
                return_weight=self._return_weight,
                explore_weight=self._explore_weight,
            )
        )

    @classmethod
    def model_name(cls) -> str:
        """Returns name of the model"""
        return "Node2Vec"

    @classmethod
    def library_name(cls) -> str:
        return "FastNode2Vec"

    @classmethod
    def task_name(cls) -> str:
        return "Node Embedding"

    def get_time_required_by_last_embedding(self) -> float:
        """Returns the time required by last embedding."""
        if self._time_required_by_last_embedding is None:
            raise ValueError("You have not yet run an embedding.")
        return self._time_required_by_last_embedding

    def _fit_transform(
        self,
        graph: Graph,
        return_dataframe: bool = True,
    ) -> Union[np.ndarray, pd.DataFrame, Dict[str, np.ndarray], Dict[str, pd.DataFrame]]:
        """Return node embedding"""

        if graph.has_edge_weights():
            edges_iterator = (
                (
                    *graph.get_node_names_from_edge_id(edge_id),
                    graph.get_edge_weight_from_edge_id(edge_id)
                )
                for edge_id in range(graph.get_number_of_directed_edges())
            )
        else:
            edges_iterator = (
                graph.get_node_names_from_edge_id(edge_id)
                for edge_id in range(graph.get_number_of_directed_edges())
            )

        fn_graph: FNGraph = FNGraph(
            edges_iterator,
            directed=True,
            weighted=graph.has_edge_weights(),
            n_edges=graph.get_number_of_directed_edges()
        )

        start = time()

        model: Node2Vec = Node2Vec(
            graph=fn_graph,
            dim=self._embedding_size,
            walk_length=self._walk_length,
            context=self._window_size,
            p=1.0/self._return_weight,
            q=1.0/self._explore_weight,
            workers=self._number_of_workers,
            batch_walks=self._iterations,
            seed=self._random_state,
        )

        model.train(
            epochs=self._epochs * self._iterations,
            progress_bar=self._verbose
        )

        self._time_required_by_last_embedding = time() - start

        # This library does not provide node embedding
        # for disconnected nodes, so we need to patch it up
        # if the graph contains disconnected nodes.
        if graph.has_singleton_nodes():
            node_embedding = np.zeros(
                shape=(graph.get_number_of_nodes(), self._embedding_size),
            )
            for node_id in range(graph.get_number_of_nodes()):
                node_name = graph.get_node_name_from_node_id(node_id)
                if node_name in model.wv:
                    node_embedding[node_id] = model.wv[node_name]
        else:
            node_embedding = model.wv[graph.get_node_names()]

        if return_dataframe:
            node_embedding = pd.DataFrame(
                node_embedding,
                index=graph.get_node_names()
            )

        return EmbeddingResult(
            embedding_method_name=self.model_name(),
            node_embeddings=node_embedding
        )

    @classmethod
    def requires_nodes_sorted_by_decreasing_node_degree(cls) -> bool:
        return False

    @classmethod
    def is_topological(cls) -> bool:
        return True

    @classmethod
    def can_use_edge_weights(cls) -> bool:
        return True

    @classmethod
    def requires_edge_weights(cls) -> bool:
        return False

    @classmethod
    def can_use_node_types(cls) -> bool:
        """Returns whether the model can use node types."""
        return False

    @classmethod
    def can_use_edge_types(cls) -> bool:
        """Returns whether the model can use edge types."""
        return False

    @classmethod
    def is_stocastic(cls) -> bool:
        """Returns whether the model is stocastic and has therefore a random state."""
        return True
