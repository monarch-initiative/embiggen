"""Submodule wrapping K-Neighbour for Edge-label prediction."""
from typing import Dict, Any
from sklearn.neighbors import KNeighborsClassifier
from embiggen.edge_label_prediction.edge_label_prediction_sklearn.sklearn_edge_label_prediction_adapter import (
    SklearnEdgeLabelPredictionAdapter,
)
from embiggen.utils import normalize_kwargs


class KNeighborsClassifierEdgeLabelPrediction(SklearnEdgeLabelPredictionAdapter):
    """Create wrapper over Sklearn K-Neighbour classifier for Edge-label prediction."""

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        algorithm: str = "auto",
        leaf_size: int = 30,
        p: int = 2,
        metric: str = "minkowski",
        metric_params: Dict[str, Any] = None,
        n_jobs: int = -1,
        edge_embedding_method: str = "Concatenate",
        use_edge_metrics: bool = False,
        random_state: int = 42,
    ):
        """Create the Decision Tree for Edge-label prediction."""
        self._kwargs = normalize_kwargs(
            self,
            dict(
                n_neighbors=n_neighbors,
                weights=weights,
                algorithm=algorithm,
                leaf_size=leaf_size,
                p=p,
                metric=metric,
                metric_params=metric_params,
                n_jobs=n_jobs,
            )
        )

        super().__init__(
            KNeighborsClassifier(**self._kwargs),
            edge_embedding_method=edge_embedding_method,
            use_edge_metrics=use_edge_metrics,
            random_state=random_state
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters used for this model."""
        return {**super().parameters(), **self._kwargs}

    @classmethod
    def smoke_test_parameters(cls) -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(leaf_size=1000, n_neighbors=1)

    @classmethod
    def model_name(cls) -> str:
        return "K-Neighbour Classifier"
