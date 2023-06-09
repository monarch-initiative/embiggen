"""Submodule wrapping Radius Neighbour for Node-label prediction."""
from typing import Dict, Any, Optional
from sklearn.neighbors import RadiusNeighborsClassifier
from embiggen.node_label_prediction.node_label_prediction_sklearn.sklearn_node_label_prediction_adapter import (
    SklearnNodeLabelPredictionAdapter,
)
from embiggen.utils import normalize_kwargs


class RadiusNeighborsClassifierNodeLabelPrediction(SklearnNodeLabelPredictionAdapter):
    """Create wrapper over Sklearn Radius Neighbour classifier for Node-label prediction."""

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
    ):
        """Create the Decision Tree for Node-label prediction."""
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

        super().__init__(RadiusNeighborsClassifier(**self._kwargs))

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

    @classmethod
    def supports_multilabel_prediction(cls) -> bool:
        """Returns whether the model supports multilabel prediction."""
        return True

    @classmethod
    def is_stocastic(cls) -> bool:
        """Returns whether the model is stocastic and has therefore a random state."""
        return False