"""Module providing SkipGram model implementation."""
from ensmallen import Graph
from .node2vec import Node2Vec
import numpy as np


class GraphSkipGram(Node2Vec):
    """Class providing SkipGram implemeted in Rust from Ensmallen."""

    def _fit_transform_graph(self, graph: Graph) -> np.ndarray:
        return graph.compute_skipgram_embedding(
            embedding_size=self._embedding_size,
            epochs=self._epochs,
            walk_length=self._walk_length,
            return_weight=self._return_weight,
            explore_weight=self._explore_weight,
            change_edge_type_weight=self._change_edge_type_weight,
            change_node_type_weight=self._change_node_type_weight,
            iterations=self._iterations,
            max_neighbours=self._max_neighbours,
            normalize_by_degree=self._normalize_by_degree,
            window_size=self._window_size,
            number_of_negative_samples=self._number_of_negative_samples,
            learning_rate=self._learning_rate,
            random_state=self._random_state,
            verbose=self._verbose,
        )
