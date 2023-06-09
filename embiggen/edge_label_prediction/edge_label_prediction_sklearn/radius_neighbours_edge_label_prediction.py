"""Submodule wrapping Radius Neighbour for Edge prediction."""
from typing import Dict, Any, Optional
from sklearn.neighbors import RadiusNeighborsClassifier
from embiggen.edge_label_prediction.edge_label_prediction_sklearn.sklearn_edge_label_prediction_adapter import (
    SklearnEdgeLabelPredictionAdapter,
)
from embiggen.utils import normalize_kwargs


class RadiusNeighborsClassifierEdgePrediction(SklearnEdgeLabelPredictionAdapter):
    """Create wrapper over Sklearn Radius Neighbour classifier for Edge prediction."""

    def __init__(
        self,
        radius: float = 1.0,
        weights: str = "uniform",
        algorithm: str = "auto",
        leaf_size: int = 30,
        p: int = 2,
        metric: str = "minkowski",
        outlier_label: Optional[str] = "most_frequent",
        metric_params: Dict[str, Any] = None,
        n_jobs: int = -1,
        edge_embedding_method: str = "Concatenate",
        use_edge_metrics: bool = False,
        random_state: int = 42,
    ):
        """Create the Decision Tree for Edge prediction."""
        self._kwargs = normalize_kwargs(
            self,
            dict(
                radius=radius,
                weights=weights,
                algorithm=algorithm,
                leaf_size=leaf_size,
                p=p,
                metric=metric,
                outlier_label=outlier_label,
                metric_params=metric_params,
                n_jobs=n_jobs,
            )
        )

        super().__init__(
            RadiusNeighborsClassifier(**self._kwargs),
            edge_embedding_method=edge_embedding_method,
            use_edge_metrics=use_edge_metrics,
            random_state=random_state,
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters used for this model."""
        return {**super().parameters(), **self._kwargs}

    @classmethod
    def smoke_test_parameters(cls) -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(leaf_size=1000)

    @classmethod
    def model_name(cls) -> str:
        return "Radius Neighbour Classifier"
