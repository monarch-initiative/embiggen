"""Submodule wrapping Linear Support Vector Machine for Node-label prediction."""
from typing import Dict, Any, Union
from sklearn.svm import LinearSVC
import numpy as np
from embiggen.node_label_prediction.node_label_prediction_sklearn.sklearn_node_label_prediction_adapter import (
    SklearnNodeLabelPredictionAdapter,
)
from embiggen.utils import normalize_kwargs


class LinearSVCNodeLabelPrediction(SklearnNodeLabelPredictionAdapter):
    """Create wrapper over Sklearn Linear Support Vector Machine classifier for Node-label prediction."""

    def __init__(
        self,
        penalty: str = "l2",
        loss: str = "squared_hinge",
        dual: bool = True,
        tol: float = 1e-4,
        C: float = 1.0,
        multi_class: str = "ovr",
        fit_intercept: bool = True,
        intercept_scaling: float = 1.0,
        class_weight: Union[Dict, str] = "balanced",
        verbose: int = 0,
        random_state: int = 42,
        max_iter: int = 1000,
    ):
        """Create the Linear Support Vector Machine for Node-label prediction."""
        self._kwargs = normalize_kwargs(
            self,
            dict(
                penalty=penalty,
                loss=loss,
                dual=dual,
                tol=tol,
                C=C,
                multi_class=multi_class,
                fit_intercept=fit_intercept,
                intercept_scaling=intercept_scaling,
                class_weight=class_weight,
                verbose=verbose,
                max_iter=max_iter,
            )
        )

        super().__init__(
            LinearSVC(**self._kwargs, random_state=random_state),
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
        return "Linear Support Vector Classifier"

    @classmethod
    def supports_multilabel_prediction(cls) -> bool:
        """Returns whether the model supports multilabel prediction."""
        return False
