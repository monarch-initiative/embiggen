"""Submodule wrapping Ridge Classifier Cross Validator for Edge-label prediction."""
from typing import Dict, Any, Tuple, Union, Optional
from sklearn.linear_model import RidgeClassifierCV
from embiggen.edge_label_prediction.edge_label_prediction_sklearn.sklearn_edge_label_prediction_adapter import (
    SklearnEdgeLabelPredictionAdapter,
)
from embiggen.utils import normalize_kwargs


class RidgeClassifierCVEdgeLabelPrediction(SklearnEdgeLabelPredictionAdapter):
    """Create wrapper over Sklearn Ridge Classifier Cross Validator for Edge-label prediction."""

    def __init__(
        self,
        alphas: Tuple[float] = (0.1, 1.0, 10.0),
        fit_intercept: bool = True,
        scoring: str = "f1_macro",
        cv:int=10,
        class_weight: Union[Dict, str] = "balanced",
        store_cv_values: bool = False,
        edge_embedding_method: str = "Concatenate",
        use_edge_metrics: bool = False,
        random_state: int = 42,
    ):
        """Create the Ridge Classifier Cross Validator for Edge-label prediction."""
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
            use_edge_metrics=use_edge_metrics,
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
