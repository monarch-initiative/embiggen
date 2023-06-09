"""Submodule wrapping Ridge Classifier Cross Validator for Edge prediction."""
from typing import Dict, Any, Tuple, Union, Optional
from sklearn.linear_model import RidgeClassifierCV
from embiggen.edge_prediction.edge_prediction_sklearn.sklearn_edge_prediction_adapter import (
    SklearnEdgePredictionAdapter,
)
from embiggen.utils import normalize_kwargs


class RidgeClassifierCVEdgePrediction(SklearnEdgePredictionAdapter):
    """Create wrapper over Sklearn Ridge Classifier Cross Validator for Edge prediction."""

    def __init__(
        self,
        alphas: Tuple[float] = (0.1, 1.0, 10.0),
        fit_intercept: bool = True,
        scoring: Optional[str] = "f1_macro",
        cv: int=10,
        class_weight: Union[Dict, str] = "balanced",
        store_cv_values: bool = False,
        edge_embedding_method: str = "Concatenate",
        training_unbalance_rate: float = 1.0,
        training_sample_only_edges_with_heterogeneous_node_types: bool = False,
        use_edge_metrics: bool = False,
        use_scale_free_distribution: bool = True,
        prediction_batch_size: int = 2**12,
        random_state: int = 42,
    ):
        """Create the Ridge Classifier Cross Validator for Edge prediction."""
        self._kwargs = normalize_kwargs(
            self,
            dict(
                alphas=alphas,
                fit_intercept=fit_intercept,
                scoring=scoring,
                cv=cv,
                class_weight=class_weight,
                store_cv_values=store_cv_values,
            )
        )

        super().__init__(
            RidgeClassifierCV(**self._kwargs),
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
        return dict(cv=2)

    @classmethod
    def model_name(cls) -> str:
        return "Ridge Classifier Cross Validator"
