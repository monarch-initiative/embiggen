"""Submodule wrapping Logistic Regression Cross Validator for Edge-label prediction."""
from typing import Dict, Any, Optional
from sklearn.linear_model import LogisticRegressionCV
from embiggen.edge_label_prediction.edge_label_prediction_sklearn.sklearn_edge_label_prediction_adapter import (
    SklearnEdgeLabelPredictionAdapter,
)
from embiggen.utils import normalize_kwargs


class LogisticRegressionCVEdgeLabelPrediction(SklearnEdgeLabelPredictionAdapter):
    """Create wrapper over Sklearn Logistic Regression Cross Validator for Edge-label prediction."""

    def __init__(
        self,
        Cs: int = 10,
        fit_intercept: bool = True,
        cv: int = 10,
        dual: bool = False,
        penalty: str = "l2",
        scoring: str = "f1_macro",
        solver: str = "lbfgs",
        tol: float = 1e-4,
        max_iter: int = 100,
        class_weight: Optional[str] = "balanced",
        n_jobs: int = -1,
        verbose: int = 0,
        refit: bool = True,
        intercept_scaling: float = 1.0,
        multi_class: str = "auto",
        l1_ratios=None,
        edge_embedding_method: str = "Concatenate",
        use_edge_metrics: bool = False,
        random_state: int = 42,
    ):
        """Create the Logistic Regression Cross Validator for Edge-label prediction."""
        self._kwargs = normalize_kwargs(
            self,
            dict(
                Cs=Cs,
                fit_intercept=fit_intercept,
                cv=cv,
                dual=dual,
                penalty=penalty,
                scoring=scoring,
                solver=solver,
                tol=tol,
                max_iter=max_iter,
                class_weight=class_weight,
                n_jobs=n_jobs,
                verbose=verbose,
                refit=refit,
                intercept_scaling=intercept_scaling,
                multi_class=multi_class,
                l1_ratios=l1_ratios,
            )
        )

        super().__init__(
            LogisticRegressionCV(**self._kwargs, random_state=random_state),
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
        return "Logistic Regression Cross Validator"