"""Submodule wrapping Gradient Boosting for edge prediction."""
from typing import Dict, Any
from sklearn.ensemble import GradientBoostingClassifier
from embiggen.edge_prediction.edge_prediction_sklearn.sklearn_edge_prediction_adapter import SklearnEdgePredictionAdapter
from embiggen.utils.normalize_kwargs import normalize_kwargs


class GradientBoostingEdgePrediction(SklearnEdgePredictionAdapter):
    """Create wrapper over Sklearn Gradient Boosting classifier for edge prediction."""

    def __init__(
        self,
        loss='log_loss',
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
        max_features="sqrt",
        verbose=0,
        max_leaf_nodes=None,
        warm_start=False,
        validation_fraction=0.1,
        n_iter_no_change=None,
        tol=1e-4,
        ccp_alpha=0.0,
        edge_embedding_method: str = "Concatenate",
        training_unbalance_rate: float = 1.0,
        training_sample_only_edges_with_heterogeneous_node_types: bool = False,
        use_edge_metrics: bool = False,
        use_scale_free_distribution: bool = True,
        prediction_batch_size: int = 2**12,
        random_state: int = 42
    ):
        """Create the Gradient Boosting for Edge Prediction."""
        self._tree_kwargs = normalize_kwargs(
            self,
            dict(
                loss=loss,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                criterion=criterion,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_depth=max_depth,
                verbose=verbose,
                init=init,
                subsample=subsample,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                min_impurity_decrease=min_impurity_decrease,
                warm_start=warm_start,
                validation_fraction=validation_fraction,
                n_iter_no_change=n_iter_no_change,
                tol=tol,
                ccp_alpha=ccp_alpha,
            )
        )

        super().__init__(
            GradientBoostingClassifier(
                **self._tree_kwargs
            ),
            edge_embedding_method=edge_embedding_method,
            training_unbalance_rate=training_unbalance_rate,
            use_edge_metrics=use_edge_metrics,
            use_scale_free_distribution=use_scale_free_distribution,
            training_sample_only_edges_with_heterogeneous_node_types=training_sample_only_edges_with_heterogeneous_node_types,
            prediction_batch_size=prediction_batch_size,
            random_state=random_state
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters used for this model."""
        return {
            **super().parameters(),
            **self._tree_kwargs
        }

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
