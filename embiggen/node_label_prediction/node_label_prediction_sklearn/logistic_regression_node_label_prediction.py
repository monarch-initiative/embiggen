"""Submodule wrapping Logistic Regression for node label prediction."""
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from .sklearn_node_label_prediction_adapter import SklearnNodeLabelPredictionAdapter


class LogisticRegressionEdgeLabelPrediction(SklearnNodeLabelPredictionAdapter):
    """Create wrapper over Sklearn Logistic Regression classifier for node label prediction."""

    def __init__(
        self,
        penalty='l2',
        dual=False,
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        solver='lbfgs',
        max_iter=100,
        multi_class='auto',
        verbose=0,
        warm_start=False,
        n_jobs=None,
        l1_ratio=None,
        random_state: int = 42
    ):
        """Create the Logistic Regression for node label Prediction."""
        super().__init__(
            LogisticRegression(
                penalty = penalty,
                dual = dual,
                tol = tol,
                C = C,
                fit_intercept = fit_intercept,
                intercept_scaling = intercept_scaling,
                class_weight = class_weight,
                random_state = random_state,
                solver = solver,
                max_iter = max_iter,
                multi_class = multi_class,
                verbose = verbose,
                warm_start = warm_start,
                n_jobs = n_jobs,
                l1_ratio = l1_ratio,
            ),
            random_state
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters used for this model."""
        return {
            **super().parameters(),
            **dict(
                penalty = self._model_instance.penalty,
                dual = self._model_instance.dual,
                tol = self._model_instance.tol,
                C = self._model_instance.C,
                fit_intercept = self._model_instance.fit_intercept,
                intercept_scaling = self._model_instance.intercept_scaling,
                class_weight = self._model_instance.class_weight,
                random_state = self._model_instance.random_state,
                solver = self._model_instance.solver,
                max_iter = self._model_instance.max_iter,
                multi_class = self._model_instance.multi_class,
                verbose = self._model_instance.verbose,
                warm_start = self._model_instance.warm_start,
                n_jobs = self._model_instance.n_jobs,
                l1_ratio = self._model_instance.l1_ratio
            )
        }
