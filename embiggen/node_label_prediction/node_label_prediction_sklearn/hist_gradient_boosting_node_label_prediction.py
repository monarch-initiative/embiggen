"""Submodule wrapping Hist Gradient Boosting for node label prediction."""
from typing import Dict, Any
from sklearn.ensemble import HistGradientBoostingClassifier
from embiggen.node_label_prediction.node_label_prediction_sklearn.sklearn_node_label_prediction_adapter import SklearnNodeLabelPredictionAdapter
from embiggen.utils.normalize_kwargs import normalize_kwargs


class HistGradientBoostingNodeLabelPrediction(SklearnNodeLabelPredictionAdapter):
    """Create wrapper over Sklearn Hist Gradient Boosting classifier for node label prediction."""

    def __init__(
        self,
        loss='log_loss',
        learning_rate=0.1,
        max_iter: int = 100,
        max_leaf_nodes: int = 31,
        max_depth: int = 3,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.,
        max_bins: int = 255,
        categorical_features=None,
        monotonic_cst=None,
        interaction_cst=None,
        warm_start=False,
        early_stopping="auto",
        scoring="loss",
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-7,
        verbose=0,
        class_weight=None,
        random_state: int = 42
    ):
        """Create the Hist Gradient Boosting for node label Prediction."""
        self._kwargs = normalize_kwargs(
            self,
            dict(
                loss=loss,
                learning_rate=learning_rate,
                max_iter=max_iter,
                max_leaf_nodes=max_leaf_nodes,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                l2_regularization=l2_regularization,
                max_bins=max_bins,
                categorical_features=categorical_features,
                monotonic_cst=monotonic_cst,
                interaction_cst=interaction_cst,
                warm_start=warm_start,
                early_stopping=early_stopping,
                scoring=scoring,
                validation_fraction=validation_fraction,
                n_iter_no_change=n_iter_no_change,
                tol=tol,
                verbose=verbose,
                class_weight=class_weight,
            )
        )
        super().__init__(
            HistGradientBoostingClassifier(
                **self._kwargs,
                random_state=random_state
            ),
            random_state
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters used for this model."""
        return dict(
            **super().parameters(),
            **self._kwargs
        )

    @classmethod
    def smoke_test_parameters(cls) -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            max_depth=1,
            max_iter=2
        )

    @classmethod
    def model_name(cls) -> str:
        return "Hist Gradient Boosting Classifier"

    @classmethod
    def supports_multilabel_prediction(cls) -> bool:
        """Returns whether the model supports multilabel prediction."""
        return False
