"""Edge-label prediction model based on XGB."""
from typing import Dict, Any, Optional, Union, List
from xgboost import XGBClassifier
import numpy as np
from embiggen.edge_label_prediction.sklearn_like_edge_label_prediction_adapter import (
    SklearnLikeEdgeLabelPredictionAdapter,
)


class XGBEdgeLabelPrediction(SklearnLikeEdgeLabelPredictionAdapter):
    """Edge-label prediction model based on XGB."""

    def __init__(
        self,
        max_depth: Optional[int] = None,
        max_leaves: Optional[int] = None,
        max_bin: Optional[int] = None,
        grow_policy: Optional[str] = None,
        learning_rate: Optional[float] = None,
        n_estimators: int = 100,
        verbosity: Optional[int] = None,
        objective=None,
        booster: Optional[str] = None,
        tree_method: Optional[str] = None,
        n_jobs: Optional[int] = None,
        gamma: Optional[float] = None,
        min_child_weight: Optional[float] = None,
        max_delta_step: Optional[float] = None,
        subsample: Optional[float] = None,
        sampling_method: Optional[str] = None,
        colsample_bytree: Optional[float] = None,
        colsample_bylevel: Optional[float] = None,
        colsample_bynode: Optional[float] = None,
        reg_alpha: Optional[float] = None,
        reg_lambda: Optional[float] = None,
        scale_pos_weight: Optional[float] = None,
        base_score: Optional[float] = None,
        missing: float = np.nan,
        num_parallel_tree: Optional[int] = None,
        monotone_constraints: Optional[Union[Dict[str, int], str]] = None,
        interaction_constraints: Optional[str] = None,
        importance_type: Optional[str] = None,
        gpu_id: Optional[int] = None,
        validate_parameters: Optional[bool] = None,
        predictor: Optional[str] = None,
        enable_categorical: bool = False,
        feature_types=None,
        max_cat_to_onehot: Optional[int] = None,
        max_cat_threshold: Optional[int] = None,
        eval_metric: Optional[Union[str, List[str]]] = None,
        early_stopping_rounds: Optional[int] = None,
        callbacks=None,
        edge_embedding_method: str = "Concatenate",
        use_edge_metrics: bool = False,
        random_state: int = 42,
    ):
        """Build a XGB Edge-label prediction model."""
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
            scale_pos_weight=scale_pos_weight,
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
            feature_types=feature_types,
            max_cat_to_onehot=max_cat_to_onehot,
            max_cat_threshold=max_cat_threshold,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds,
            callbacks=callbacks,
        )

        super().__init__(
            model_instance=XGBClassifier(
                **self._kwargs,
                random_state=random_state,
            ),
            edge_embedding_method=edge_embedding_method,
            use_edge_metrics=use_edge_metrics,
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
        return "XGB"

    @classmethod
    def library_name(cls) -> str:
        """Return name of the model."""
        return "XGB"
