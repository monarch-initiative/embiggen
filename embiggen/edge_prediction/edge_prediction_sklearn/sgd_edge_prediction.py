"""Submodule wrapping SGD for edge prediction."""
from typing import Dict, Any
from sklearn.linear_model import SGDClassifier
from .sklearn_edge_prediction_adapter import SklearnEdgePredictionAdapter


class SGDEdgePrediction(SklearnEdgePredictionAdapter):
    """Create wrapper over Sklearn SGD classifier for edge prediction."""

    def __init__(
        self,
        loss="hinge",
        penalty='l2',
        alpha=0.0001,
        l1_ratio=0.15,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3,
        shuffle=True,
        verbose=0,
        n_jobs=-1,
        learning_rate="optimal",
        eta0=0.0,
        power_t=0.5,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        class_weight=None,
        warm_start=False,
        average=False,
        edge_embedding_method: str = "Concatenate",
        unbalance_rate: float = 1.0,
        sample_only_edges_with_heterogeneous_node_types: bool = False,
        random_state: int = 42
    ):
        """Create the SGD for Edge Prediction."""
        self._loss = loss,
        self._penalty = penalty,
        self._alpha = alpha,
        self._l1_ratio = l1_ratio,
        self._fit_intercept = fit_intercept,
        self._max_iter = max_iter,
        self._tol = tol,
        self._shuffle = shuffle,
        self._n_jobs = n_jobs,
        self._learning_rate = learning_rate,
        self._eta0 = eta0,
        self._power_t = power_t,
        self._early_stopping = early_stopping,
        self._validation_fraction = validation_fraction,
        self._n_iter_no_change = n_iter_no_change,
        self._class_weight = class_weight,
        self._warm_start = warm_start,
        self._average = average
        super().__init__(
            SGDClassifier(
                loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio,
                fit_intercept=fit_intercept, max_iter=max_iter, tol=tol,
                shuffle=shuffle, verbose=verbose, n_jobs=n_jobs,
                random_state=random_state, learning_rate=learning_rate, eta0=eta0,
                power_t=power_t, early_stopping=early_stopping,
                validation_fraction=validation_fraction,
                n_iter_no_change=n_iter_no_change, class_weight=class_weight,
                warm_start=warm_start, average=average
            ),
            edge_embedding_method,
            unbalance_rate,
            sample_only_edges_with_heterogeneous_node_types,
            random_state
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters used for this model."""
        return {
            **super().parameters(),
            **dict(
                loss=self._loss,
                penalty=self._penalty,
                alpha=self._alpha,
                l1_ratio=self._l1_ratio,
                fit_intercept=self._fit_intercept,
                max_iter=self._max_iter,
                tol=self._tol,
                shuffle=self._shuffle,
                n_jobs=self._n_jobs,
                random_state=self._random_state,
                learning_rate=self._learning_rate,
                eta0=self._eta0,
                power_t=self._power_t,
                early_stopping=self._early_stopping,
                validation_fraction=self._validation_fraction,
                n_iter_no_change=self._n_iter_no_change,
                class_weight=self._class_weight,
                warm_start=self._warm_start,
                average=self._average
            )
        }


    @staticmethod
    def model_name() -> str:
        return "SGD Classifier"