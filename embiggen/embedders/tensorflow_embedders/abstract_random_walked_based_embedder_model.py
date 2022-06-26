"""Submodule providing abstract random walk based embedder model."""
from typing import Dict, Any
from embiggen.utils.abstract_models import abstract_class
from embiggen.embedders.tensorflow_embedders.tensorflow_embedder import TensorFlowEmbedder


@abstract_class
class AbstractRandomWalkBasedEmbedderModel(TensorFlowEmbedder):

    def __init__(
        self,
        window_size: int = 4,
        walk_length: int = 128,
        iterations: int = 1,
        return_weight: float = 1.0,
        explore_weight: float = 1.0,
        change_node_type_weight: float = 1.0,
        change_edge_type_weight: float = 1.0,
        max_neighbours: int = 100,
        normalize_by_degree: bool = False,
        random_state: int = 42,
        **kwargs: Dict
    ):
        """Create new GloVe-based TensorFlowEmbedder object.

        Parameters
        -------------------------------
        window_size: int = 4
            Window size for the local context.
            On the borders the window size is trimmed.
        walk_length: int = 128
            Maximal length of the walks.
        iterations: int = 1
            Number of iterations of the single walks.
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
        random_state: int = 42
            The random state to reproduce the training sequence.
        """
        self._window_size = window_size
        self._walk_length = walk_length
        self._return_weight = return_weight
        self._explore_weight = explore_weight
        self._change_edge_type_weight = change_edge_type_weight
        self._change_node_type_weight = change_node_type_weight
        self._max_neighbours = max_neighbours
        self._iterations = iterations
        self._normalize_by_degree = normalize_by_degree

        super().__init__(random_state=random_state, **kwargs)

    @classmethod
    def smoke_test_parameters(cls) -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            **TensorFlowEmbedder.smoke_test_parameters(),
            window_size=1,
            walk_length=4,
            iterations=1,
            max_neighbours=10,
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters of the model."""
        return {
            **super().parameters(),
            **dict(
                window_size=self._window_size,
                walk_length=self._walk_length,
                return_weight=self._return_weight,
                explore_weight=self._explore_weight,
                change_edge_type_weight=self._change_edge_type_weight,
                change_node_type_weight=self._change_node_type_weight,
                max_neighbours=self._max_neighbours,
                iterations=self._iterations,
                normalize_by_degree=self._normalize_by_degree,
            )
        }

    @classmethod
    def is_topological(cls) -> bool:
        return True

    @classmethod
    def requires_edge_weights(cls) -> bool:
        return False

    @classmethod
    def requires_positive_edge_weights(cls) -> bool:
        return True

    @classmethod
    def requires_node_types(cls) -> bool:
        return False

    @classmethod
    def requires_edge_types(cls) -> bool:
        return False

    @classmethod
    def can_use_edge_weights(cls) -> bool:
        """Returns whether the model can optionally use edge weights."""
        return True

    def is_using_edge_weights(self) -> bool:
        """Returns whether the model is parametrized to use edge weights."""
        return True

    @classmethod
    def can_use_edge_weights(cls) -> bool:
        """Returns whether the model can optionally use edge weights."""
        return True

    def is_using_edge_weights(self) -> bool:
        """Returns whether the model is parametrized to use edge weights."""
        return True

    @classmethod
    def can_use_node_types(cls) -> bool:
        """Returns whether the model can optionally use node types."""
        return True

    def is_using_node_types(self) -> bool:
        """Returns whether the model is parametrized to use node types."""
        return self._change_node_type_weight != 1.0

    @classmethod
    def can_use_edge_types(cls) -> bool:
        """Returns whether the model can optionally use edge types."""
        return True

    def is_using_edge_types(self) -> bool:
        """Returns whether the model is parametrized to use edge types."""
        return self._change_edge_type_weight != 1.0
