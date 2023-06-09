"""Node-label prediction model based on XGB."""
from typing import Dict, Any, Optional, Union, List
from xgboost import XGBClassifier
import numpy as np
from embiggen.node_label_prediction.sklearn_like_node_label_prediction_adapter import (
    SklearnLikeNodeLabelPredictionAdapter,
)


class XGBNodeLabelPrediction(SklearnLikeNodeLabelPredictionAdapter):
    """Node-label prediction model based on XGB."""

    def __init__(
        self,
        max_depth: int = 6,
        max_leaves: int = 0,
        max_bin: int = 256,
        grow_policy: Optional[str] = None,
        learning_rate: float = 0.3,
        n_estimators: int = 100,
        verbosity: int = 1,
        objective = None,
        booster: Optional[str] = None,
        tree_method: str = "auto",
        n_jobs: int = -1,
        gamma: Optional[float] = None,
        min_child_weight: float = 1.0,
        max_delta_step: float = 0.0,
        subsample: float = 1.0,
        sampling_method: str = "uniform",
        colsample_bytree: float = 1.0,
        colsample_bylevel: float = 1.0,
        colsample_bynode: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        base_score: Optional[float] = None,
        missing: float = np.nan,
        num_parallel_tree: int = 1,
        monotone_constraints: Optional[Union[Dict[str, int], str]] = None,
        interaction_constraints: Optional[List[List[int]]] = None,
        importance_type: Optional[str] = None,
        gpu_id: Optional[int] = None,
        validate_parameters: bool = False,
        predictor: str = "auto",
        enable_categorical: bool = False,
        max_cat_to_onehot: Optional[int] = None,
        max_cat_threshold: Optional[int] = None,
        eval_metric: Optional[Union[str, List[str]]] = None,
        early_stopping_rounds: Optional[int] = None,
        random_state: int = 42
    ):
        """Build a XGB node-label prediction model."""
        self._kwargs = dict(
            max_depth=max_depth,
            max_leaves=max_leaves,
            max_bin=max_bin,
            grow_policy=grow_policy,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            verbosity=verbosity,
            objective=objective,
            booster=booster,
            tree_method=tree_method,
            n_jobs=n_jobs,
            gamma=gamma,
            min_child_weight=min_child_weight,
            max_delta_step=max_delta_step,
            subsample=subsample,
            sampling_method=sampling_method,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel,
            colsample_bynode=colsample_bynode,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            base_score=base_score,
            missing=missing,
            num_parallel_tree=num_parallel_tree,
            monotone_constraints=monotone_constraints,
            interaction_constraints=interaction_constraints,
            importance_type=importance_type,
            gpu_id=gpu_id,
            validate_parameters=validate_parameters,
            predictor=predictor,
            enable_categorical=enable_categorical,
            max_cat_to_onehot=max_cat_to_onehot,
            max_cat_threshold=max_cat_threshold,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds,
        )

        super().__init__(
            model_instance=XGBClassifier(
                **self._kwargs,
                random_state=random_state,
            ),
            random_state=random_state,
        )

    @classmethod
    def smoke_test_parameters(cls) -> Dict[str, Any]:
        return dict(
            n_estimators=2,
            max_depth=2,
        )

    def parameters(self) -> Dict[str, Any]:
        return dict(
            **super().parameters(),
            **self._kwargs,
        )

    @classmethod
    def model_name(cls) -> str:
        """Return the name of the model."""
        return "XGBoost"

    @classmethod
    def library_name(cls) -> str:
        """Return name of the model."""
        return "XGBoost"

    @classmethod
    def supports_multilabel_prediction(cls) -> bool:
        """Returns whether the model supports multilabel prediction."""
        return False
