"""Submodule wrapping Random Forest for edge prediction."""
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier
from embiggen.edge_prediction.edge_prediction_sklearn.decision_tree_edge_prediction import DecisionTreeEdgePrediction
from embiggen.edge_prediction.edge_prediction_sklearn.sklearn_edge_prediction_adapter import SklearnEdgePredictionAdapter


class RandomForestEdgePrediction(SklearnEdgePredictionAdapter):
    """Create wrapper over Sklearn Random Forest classifier for edge prediction."""

    def __init__(
        self,
        n_estimators: int = 1000,
        criterion: str = "gini",
        max_depth: int = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_weight_fraction_leaf: float = 0.,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.,
        bootstrap=True,
        oob_score=False,
        n_jobs=-1,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
        edge_embedding_method: str = "Concatenate",
        training_unbalance_rate: float = 1.0,
        training_sample_only_edges_with_heterogeneous_node_types: bool = False,
        use_edge_metrics: bool = False,
        use_scale_free_distribution: bool = True,
        prediction_batch_size: int = 2**12,
        random_state: int = 42
    ):
        """Create the Random Forest for Edge  Prediction."""
        self._n_estimators = n_estimators
        self._criterion = criterion
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._min_weight_fraction_leaf = min_weight_fraction_leaf
        self._max_features = max_features
        self._max_leaf_nodes = max_leaf_nodes
        self._min_impurity_decrease = min_impurity_decrease
        self._bootstrap = bootstrap
        self._oob_score = oob_score
        self._n_jobs = n_jobs
        self._random_state = random_state
        self._verbose = verbose
        self._warm_start = warm_start
        self._ccp_alpha = ccp_alpha
        self._max_samples = max_samples

        super().__init__(
            RandomForestClassifier(
                n_estimators=n_estimators,
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                min_impurity_decrease=min_impurity_decrease,
                bootstrap=bootstrap,
                oob_score=oob_score,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
                warm_start=warm_start,
                ccp_alpha=ccp_alpha,
                max_samples=max_samples
            ),
            edge_embedding_method=edge_embedding_method,
            training_unbalance_rate=training_unbalance_rate,
            use_edge_metrics=use_edge_metrics,
            use_scale_free_distribution=use_scale_free_distribution,
            training_sample_only_edges_with_heterogeneous_node_types=training_sample_only_edges_with_heterogeneous_node_types,
            prediction_batch_size=prediction_batch_size,
            random_state=random_state
        )

    @classmethod
    def smoke_test_parameters(cls) -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            **DecisionTreeEdgePrediction.smoke_test_parameters(),
            n_estimators=1
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters used for this model."""
        return {
            **super().parameters(),
            **dict(
                n_estimators = self._n_estimators,
                criterion = self._criterion,
                max_depth = self._max_depth,
                min_samples_split = self._min_samples_split,
                min_samples_leaf = self._min_samples_leaf,
                min_weight_fraction_leaf = self._min_weight_fraction_leaf,
                max_features = self._max_features,
                max_leaf_nodes = self._max_leaf_nodes,
                min_impurity_decrease = self._min_impurity_decrease,
                bootstrap = self._bootstrap,
                oob_score = self._oob_score,
                n_jobs = self._n_jobs,
                random_state = self._random_state,
                verbose = self._verbose,
                warm_start = self._warm_start,
                ccp_alpha = self._ccp_alpha,
                max_samples = self._max_samples,
            )
        }

    
    @classmethod
    def model_name(cls) -> str:
        return "Random Forest Classifier"