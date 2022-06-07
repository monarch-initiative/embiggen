"""Submodule wrapping Extra Trees for edge label prediction."""
from typing import Dict, Any
from sklearn.ensemble import ExtraTreesClassifier
from multiprocessing import cpu_count
from embiggen.edge_label_prediction.edge_label_prediction_sklearn.decision_tree_edge_label_prediction import DecisionTreeEdgeLabelPrediction
from embiggen.edge_label_prediction.edge_label_prediction_sklearn.sklearn_edge_label_prediction_adapter import SklearnEdgeLabelPredictionAdapter


class ExtraTreesEdgeLabelPrediction(SklearnEdgeLabelPredictionAdapter):
    """Create wrapper over Sklearn Extra Trees classifier for edge label prediction."""

    def __init__(
        self,
        n_estimators: int = 1000,
        criterion: str = "gini",
        max_depth: int = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_weight_fraction_leaf: float = 0.,
        max_features="auto",
        max_leaf_nodes=None,
        min_impurity_decrease=0.,
        min_impurity_split=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=-1,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        edge_embedding_method: str = "Concatenate",
        random_state: int = 42
    ):
        """Create the Extra Trees for Edge Label Prediction."""
        self._n_estimators = n_estimators
        self._criterion = criterion
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._min_weight_fraction_leaf = min_weight_fraction_leaf
        self._max_features = max_features
        self._max_leaf_nodes = max_leaf_nodes
        self._min_impurity_decrease = min_impurity_decrease
        self._min_impurity_split = min_impurity_split
        self._bootstrap = bootstrap
        self._oob_score = oob_score
        self._n_jobs = cpu_count() if n_jobs == -1 else n_jobs
        self._random_state = random_state
        self._verbose = verbose
        self._warm_start = warm_start
        self._class_weight = class_weight
        self._ccp_alpha = ccp_alpha
        self._max_samples = max_samples

        super().__init__(
            ExtraTreesClassifier(
                n_estimators=n_estimators,
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                min_impurity_decrease=min_impurity_decrease,
                min_impurity_split=min_impurity_split,
                bootstrap=bootstrap,
                oob_score=oob_score,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
                warm_start=warm_start,
                class_weight=class_weight,
                ccp_alpha=ccp_alpha,
                max_samples=max_samples
            ),
            edge_embedding_method,
            random_state
        )

    @staticmethod
    def smoke_test_parameters() -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            **DecisionTreeEdgeLabelPrediction.smoke_test_parameters(),
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
                min_impurity_split = self._min_impurity_split,
                bootstrap = self._bootstrap,
                oob_score = self._oob_score,
                n_jobs = self._n_jobs,
                random_state = self._random_state,
                verbose = self._verbose,
                warm_start = self._warm_start,
                class_weight = self._class_weight,
                ccp_alpha = self._ccp_alpha,
                max_samples = self._max_samples,
            )
        }
    
    @staticmethod
    def model_name() -> str:
        return "Extra Trees Classifier"