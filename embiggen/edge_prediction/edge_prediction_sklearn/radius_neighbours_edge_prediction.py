"""Submodule wrapping Radius Neighbour for Edge Prediction prediction."""
from typing import Dict, Any, Optional
from sklearn.neighbors import RadiusNeighborsClassifier
from embiggen.edge_prediction.edge_prediction_sklearn.sklearn_edge_prediction_adapter import SklearnEdgePredictionAdapter
from embiggen.utils import normalize_kwargs


class RadiusNeighborsClassifierEdgeLabelPrediction(SklearnEdgePredictionAdapter):
    """Create wrapper over Sklearn Radius Neighbour classifier for Edge Prediction prediction."""

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
        training_unbalance_rate: float = 1.0,
        training_sample_only_edges_with_heterogeneous_node_types: bool = False,
        use_edge_metrics: bool = False,
        use_scale_free_distribution: bool = True,
        prediction_batch_size: int = 2**12,
        random_state: int = 42,
    ):
        """Create the Decision Tree for Edge Prediction prediction."""
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
            training_unbalance_rate=training_unbalance_rate,
            use_edge_metrics=use_edge_metrics,
            use_scale_free_distribution=use_scale_free_distribution,
            training_sample_only_edges_with_heterogeneous_node_types=training_sample_only_edges_with_heterogeneous_node_types,
            prediction_batch_size=prediction_batch_size,
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
