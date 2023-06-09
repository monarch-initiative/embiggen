"""Submodule wrapping Ridge Classifier for Node-label prediction."""
from typing import Dict, Any, Union, Optional
from sklearn.linear_model import RidgeClassifier
from embiggen.node_label_prediction.node_label_prediction_sklearn.sklearn_node_label_prediction_adapter import (
    SklearnNodeLabelPredictionAdapter,
)
from embiggen.utils import normalize_kwargs


class RidgeClassifierNodeLabelPrediction(SklearnNodeLabelPredictionAdapter):
    """Create wrapper over Sklearn Ridge Classifier classifier for Node-label prediction."""

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
        random_state: int = 323,
    ):
        """Create the Ridge Classifier for Node-label prediction."""
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
