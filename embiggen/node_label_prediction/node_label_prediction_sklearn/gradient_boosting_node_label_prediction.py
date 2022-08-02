"""Submodule wrapping Gradient Boosting for node label prediction."""
from typing import Dict, Any
from sklearn.ensemble import GradientBoostingClassifier
from embiggen.node_label_prediction.node_label_prediction_sklearn.sklearn_node_label_prediction_adapter import SklearnNodeLabelPredictionAdapter


class GradientBoostingNodeLabelPrediction(SklearnNodeLabelPredictionAdapter):
    """Create wrapper over Sklearn Gradient Boosting classifier for node label prediction."""

    def __init__(
        self,
        loss='deviance',
        learning_rate=0.1,
        n_estimators=100,
        subsample=1.0,
        criterion='friedman_mse',
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.,
        max_depth=3,
        min_impurity_decrease=0.,
        init=None,
        max_features=None,
        verbose=0,
        max_leaf_nodes=None,
        warm_start=False,
        validation_fraction=0.1,
        n_iter_no_change=None,
        tol=1e-4,
        ccp_alpha=0.0,
        random_state: int = 42
    ):
        """Create the Gradient Boosting for node label Prediction."""
        self._loss=loss
        self._learning_rate=learning_rate
        self._n_estimators=n_estimators
        self._criterion=criterion
        self._min_samples_split=min_samples_split
        self._min_samples_leaf=min_samples_leaf
        self._min_weight_fraction_leaf=min_weight_fraction_leaf
        self._max_depth=max_depth
        self._init=init
        self._subsample=subsample
        self._max_features=max_features
        self._max_leaf_nodes=max_leaf_nodes
        self._min_impurity_decrease=min_impurity_decrease
        self._warm_start=warm_start
        self._validation_fraction=validation_fraction
        self._n_iter_no_change=n_iter_no_change
        self._tol=tol
        self._ccp_alpha=ccp_alpha
        super().__init__(
            GradientBoostingClassifier(
                loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
                criterion=criterion, min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_depth=max_depth, init=init, subsample=subsample,
                max_features=max_features,
                random_state=random_state, verbose=verbose,
                max_leaf_nodes=max_leaf_nodes,
                min_impurity_decrease=min_impurity_decrease,
                warm_start=warm_start, validation_fraction=validation_fraction,
                n_iter_no_change=n_iter_no_change, tol=tol, ccp_alpha=ccp_alpha
            ),
            random_state
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters used for this model."""
        return dict(
            **super().parameters(),
            loss=self._loss,
            learning_rate=self._learning_rate,
            n_estimators=self._n_estimators,
            criterion=self._criterion,
            min_samples_split=self._min_samples_split,
            min_samples_leaf=self._min_samples_leaf,
            min_weight_fraction_leaf=self._min_weight_fraction_leaf,
            max_depth=self._max_depth,
            init=self._init,
            subsample=self._subsample,
            max_features=self._max_features,
            max_leaf_nodes=self._max_leaf_nodes,
            min_impurity_decrease=self._min_impurity_decrease,
            warm_start=self._warm_start,
            validation_fraction=self._validation_fraction,
            n_iter_no_change=self._n_iter_no_change,
            tol=self._tol,
            ccp_alpha=self._ccp_alpha
        )

    @classmethod
    def smoke_test_parameters(cls) -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            max_depth=1,
            n_estimators=1
        )

    @classmethod
    def model_name(cls) -> str:
        return "Gradient Boosting Classifier"