"""Submodule wrapping KNN for edge label prediction."""
from typing import Dict, Any
from sklearn.neighbors import KNeighborsClassifier
from multiprocessing import cpu_count
from .sklearn_edge_label_prediction_adapter import SklearnEdgeLabelPredictionAdapter


class KNNEdgeLabelPrediction(SklearnEdgeLabelPredictionAdapter):
    """Create wrapper over Sklearn KNN classifier for edge label prediction."""

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
        random_state: int = 42
    ):
        """Create the KNN for Edge Label Prediction."""
        self._n_neighbors = n_neighbors
        self._weights = weights
        self._algorithm = algorithm
        self._leaf_size = leaf_size
        self._p = p
        self._metric = metric
        self._metric_params = metric_params
        self._n_jobs = cpu_count() if n_jobs == -1 else n_jobs
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

    @staticmethod
    def model_name() -> str:
        return "KNeighbors Classifier"