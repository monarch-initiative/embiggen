"""Node2Vec wrapper for PecanPy numba-based node embedding library."""
from typing import Dict, Union, Any

import numpy as np
import pandas as pd
from ensmallen import Graph

from embiggen.utils.abstract_models import AbstractEmbeddingModel, EmbeddingResult
from pecanpy.node2vec import SparseOTF
from multiprocessing import cpu_count
from time import time

class Node2VecPecanPy(AbstractEmbeddingModel):
    """Node2Vec wrapper for PecanPy numba-based node embedding library."""

    def __init__(
        self,
        embedding_size: int = 100,
        epochs: int = 30,
        walk_length: int = 128,
        iterations: int = 10,
        window_size: int = 5,
        batch_size: int = 128,
        learning_rate: float = 0.01,
        return_weight: float = 0.25,
        explore_weight: float = 4.0,
        number_of_workers: Union[int, str] = "auto",
        verbose: bool = False,
        random_state: int = 42,
        ring_bell: bool = False,
        enable_cache: bool = False
    ):
        """Create new wrapper for Node2Vec model from PecanPy library.
        
        Parameters
        -------------------------
        embedding_size: int = 100
            The dimension of the embedding to compute.
        epochs: int = 100
            The number of epochs to use to train the model for.
        batch_size: int = 2**10
            Size of the training batch.
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
        self._batch_size = batch_size
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
                batch_size=self._batch_size,
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
        return "PecanPy"

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

        model: SparseOTF = SparseOTF(
            p=1.0/self._return_weight,
            q=1.0/self._explore_weight,
            workers=self._number_of_workers,
            verbose=self._verbose,
        )

        # Instead of using `SparseOTF` methods,
        # we manipolate directly the object attributes
        # in order to avoid a memory peak and making
        # the conversion of the Ensmallen graph object
        # into the `SparseOTF` more seamless.
        
        # The `indptr` attribute contains the indices
        # of the rows in the CSR representation, which
        # are the comulative node degrees.
        # This is shifted by a value to the right, and the
        # first value is set to zero.
        model.indptr = np.zeros(graph.get_number_of_nodes() + 1, dtype=np.int64)
        model.indptr[1:] = graph.get_cumulative_node_degrees().astype(np.int64)

        # In the `indices` attribute we need to store the destinations.
        model.indices = graph.get_directed_destination_node_ids()

        # We also need to set the `IDlst` attribute, which are
        # the node IDs.
        model.IDlst = graph.get_node_ids()

        # In model data we need to store the edge weights
        # if are present. If the graph is weighted, we use
        # the graph edge weights. Otherwise we set all weights
        # to one, as the library PecanPy does.
        if graph.has_edge_weights():
            model.data = graph.get_directed_edge_weights().astype(np.float64)
        else:
            model.data = np.ones_like(model.indices, dtype=np.float64)

        start = time()

        node_embedding = model.embed(
            dim=self._embedding_size,
            num_walks=self._iterations,
            walk_length=self._walk_length,
            window_size=self._window_size,
            epochs=self._epochs,
            verbose=self._verbose
        )

        self._time_required_by_last_embedding = time() - start

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