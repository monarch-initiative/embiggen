"""Submodule wrapping Decision Tree for edge label prediction."""
from typing import Dict, Any
from sklearn.tree import DecisionTreeClassifier
from embiggen.edge_label_prediction.edge_label_prediction_sklearn.sklearn_edge_label_prediction_adapter import SklearnEdgeLabelPredictionAdapter


class DecisionTreeEdgeLabelPrediction(SklearnEdgeLabelPredictionAdapter):
    """Create wrapper over Sklearn Decision Tree classifier for edge label prediction."""

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
        min_impurity_split=None,
        class_weight=None,
        ccp_alpha=0.0,
        edge_embedding_method: str = "Concatenate",
        random_state: int = 42
    ):
        """Create the Decision Tree for Edge Label Prediction."""
        self._criterion = criterion
        self._splitter = splitter
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._min_weight_fraction_leaf = min_weight_fraction_leaf
        self._max_features = max_features
        self._max_leaf_nodes = max_leaf_nodes
        self._min_impurity_decrease = min_impurity_decrease
        self._min_impurity_split = min_impurity_split
        self._random_state = random_state
        self._class_weight = class_weight
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
                min_impurity_split=min_impurity_split,
                random_state=random_state,
                class_weight=class_weight,
                ccp_alpha=ccp_alpha,
            ),
            edge_embedding_method,
            random_state
        )

    @staticmethod
    def smoke_test_parameters() -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            max_depth=1,
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
                min_impurity_split=self._min_impurity_split,
                random_state=self._random_state,
                class_weight=self._class_weight,
                ccp_alpha=self._ccp_alpha,
            )
        }

    @staticmethod
    def model_name() -> str:
        return "Decision Tree Classifier"