"""Submodule wrapping Ridge Classifier for Edge prediction."""
from typing import Dict, Any, Union, Optional
from sklearn.linear_model import RidgeClassifier
from embiggen.edge_prediction.edge_prediction_sklearn.sklearn_edge_prediction_adapter import (
    SklearnEdgePredictionAdapter,
)
from embiggen.utils import normalize_kwargs


class RidgeClassifierEdgePrediction(SklearnEdgePredictionAdapter):
    """Create wrapper over Sklearn Ridge Classifier classifier for Edge prediction."""

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        copy_X: bool = True,
        max_iter: Optional[int] = None,
        tol: float = 1e-4,
        class_weight: Union[Dict, str] = "balanced",
        solver: str = "auto",
        positive: bool = False,
        edge_embedding_method: str = "Concatenate",
        training_unbalance_rate: float = 1.0,
        training_sample_only_edges_with_heterogeneous_node_types: bool = False,
        use_edge_metrics: bool = False,
        use_scale_free_distribution: bool = True,
        prediction_batch_size: int = 2**12,
        random_state: int = 42,
    ):
        """Create the Ridge Classifier for Edge prediction."""
        self._kwargs = normalize_kwargs(
            self,
            dict(
                alpha=alpha,
                fit_intercept=fit_intercept,
                copy_X=copy_X,
                max_iter=max_iter,
                tol=tol,
                class_weight=class_weight,
                solver=solver,
                positive=positive,
            )
        )

        super().__init__(
            RidgeClassifier(**self._kwargs, random_state=random_state),
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
        return dict(max_iter=2)

    @classmethod
    def model_name(cls) -> str:
        return "Ridge Classifier"
