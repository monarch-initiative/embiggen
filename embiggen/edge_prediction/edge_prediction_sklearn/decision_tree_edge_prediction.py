"""Submodule wrapping Decision Tree for edge prediction."""
from typing import Dict, Any
from sklearn.tree import DecisionTreeClassifier
from embiggen.edge_prediction.edge_prediction_sklearn.sklearn_edge_prediction_adapter import SklearnEdgePredictionAdapter


class DecisionTreeEdgePrediction(SklearnEdgePredictionAdapter):
    """Create wrapper over Sklearn Random Forest classifier for edge prediction."""

    def __init__(
        self,
        criterion="gini",
        splitter="best",
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.,
        max_features=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.,
        ccp_alpha=0.0,
        edge_embedding_method: str = "Concatenate",
        training_unbalance_rate: float = 1.0,
        training_sample_only_edges_with_heterogeneous_node_types: bool = False,
        use_edge_metrics: bool = False,
        use_scale_free_distribution: bool = True,
        prediction_batch_size: int = 2**12,
        random_state: int = 42
    ):
        """Create the Decision Tree for Edge Prediction."""
        self._criterion = criterion
        self._splitter = splitter
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._min_weight_fraction_leaf = min_weight_fraction_leaf
        self._max_features = max_features
        self._max_leaf_nodes = max_leaf_nodes
        self._min_impurity_decrease = min_impurity_decrease
        self._random_state = random_state
        self._ccp_alpha = ccp_alpha

        super().__init__(
            DecisionTreeClassifier(
                criterion=criterion,
                splitter=splitter,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                min_impurity_decrease=min_impurity_decrease,
                random_state=random_state,
                ccp_alpha=ccp_alpha,
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
            **dict(
                criterion=self._criterion,
                splitter=self._splitter,
                max_depth=self._max_depth,
                min_samples_split=self._min_samples_split,
                min_samples_leaf=self._min_samples_leaf,
                min_weight_fraction_leaf=self._min_weight_fraction_leaf,
                max_features=self._max_features,
                max_leaf_nodes=self._max_leaf_nodes,
                min_impurity_decrease=self._min_impurity_decrease,
                random_state=self._random_state,
                ccp_alpha=self._ccp_alpha,
            )
        }

    @classmethod
    def model_name(cls) -> str:
        return "Decision Tree Classifier"

    @classmethod
    def smoke_test_parameters(cls) -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            max_depth=1,
        )