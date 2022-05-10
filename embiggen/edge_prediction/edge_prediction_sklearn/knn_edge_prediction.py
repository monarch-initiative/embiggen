"""Submodule wrapping KNN for edge prediction."""
from typing import Dict, Any
from sklearn.neighbors import KNeighborsClassifier
from .sklearn_edge_prediction_adapter import SklearnEdgePredictionAdapter


class KNNEdgePrediction(SklearnEdgePredictionAdapter):
    """Create wrapper over Sklearn KNN classifier for edge prediction."""

    def __init__(
        self,
        n_neighbors=5,
        weights='distance',
        algorithm='auto',
        leaf_size=30,
        p=2,
        metric='minkowski',
        metric_params=None,
        n_jobs=-1,
        edge_embedding_method: str = "Concatenate",
        unbalance_rate: float = 1.0,
        sample_only_edges_with_heterogeneous_node_types: bool = False,
        random_state: int = 42
    ):
        """Create the KNN for Edge Prediction."""
        self._n_neighbors = n_neighbors
        self._weights = weights
        self._algorithm = algorithm
        self._leaf_size = leaf_size
        self._p = p
        self._metric = metric
        self._metric_params = metric_params
        self._n_jobs = n_jobs
        super().__init__(
            KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                algorithm=algorithm,
                leaf_size=leaf_size, metric=metric, p=p,
                metric_params=metric_params,
                n_jobs=n_jobs
            ),
            edge_embedding_method,
            unbalance_rate,
            sample_only_edges_with_heterogeneous_node_types,
            random_state
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters used for this model."""
        return {
            **super().parameters(),
            **dict(
                n_neighbors = self._n_neighbors,
                weights = self._weights,
                algorithm = self._algorithm,
                leaf_size = self._leaf_size,
                p = self._p,
                metric = self._metric,
                metric_params = self._metric_params,
                n_jobs = self._n_jobs,
            )
        }
