"""Edge prediction model based on LightGBM.""" ""
from typing import Dict, Union, Any
from lightgbm import LGBMClassifier
from embiggen.edge_prediction.sklearn_like_edge_prediction_adapter import (
    SklearnLikeEdgePredictionAdapter,
)
from embiggen.utils.normalize_kwargs import normalize_kwargs


class LightGBMEdgePredictionModel(SklearnLikeEdgePredictionAdapter):
    """Edge prediction model based on LightGBM."""

    def __init__(
        self,
        boosting_type: str = "gbdt",
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample_for_bin: int = 200000,
        class_weight: Union[Dict, str] = "balanced",
        min_split_gain: float = 0.0,
        min_child_weight: float = 0.001,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        subsample_freq: int = 0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        n_jobs: int = -1,
        importance_type: str = "split",
        edge_embedding_method: str = "Concatenate",
        training_unbalance_rate: float = 1.0,
        training_sample_only_edges_with_heterogeneous_node_types: bool = False,
        use_edge_metrics: bool = False,
        use_scale_free_distribution: bool = True,
        prediction_batch_size: int = 2**12,
        random_state: int = 42,
        **kwargs: Dict,
    ):
        """Build a LightGBM Edge prediction model.

        Parameters
        --------------------------
        boosting_type: str = "gbdt",
            The type of boosting to use.
        num_leaves: int = 31,
            The number of leaves to use.
        max_depth: int = -1,
            The maximum depth of the tree.
        learning_rate: float = 0.1,
            The learning rate to use.
        n_estimators: int = 100,
            The number of estimators to use.
        subsample_for_bin: int = 200000,
            The number of samples to use for binning.
        class_weight: Union[Dict, str] = "balanced",
            The class weight to use.
        min_split_gain: float = 0.0,
            The minimum split gain to use.
        min_child_weight: float = 0.001,
            The minimum child weight to use.
        min_child_samples: int = 20,
            The minimum number of child samples to use.
        subsample: float = 1.0,
            The subsample to use.
        subsample_freq: int = 0,
            The subsample frequency to use.
        colsample_bytree: float = 1.0,
            The column sample by tree to use.
        reg_alpha: float = 0.0,
            The regularization alpha to use.
        reg_lambda: float = 0.0,
            The regularization lambda to use.
        n_jobs: int = -1,
            The number of jobs to use.
        importance_type: str = "split",
            The importance type to use.
        edge_embedding_method: str = "Concatenate"
            The method to use to compute the edges.
        use_edge_metrics: bool = False
            Whether to use the edge metrics from traditional edge prediction.
            These metrics currently include:
            - Adamic Adar
            - Jaccard Coefficient
            - Resource allocation index
            - Preferential attachment
        random_state: int = 42
            The random state to use to reproduce the training.
        **kwargs: Dict,
            Additional keyword arguments to pass to the model.
        """
        self._kwargs = normalize_kwargs(
            self,
            dict(
                boosting_type=boosting_type,
                num_leaves=num_leaves,
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                subsample_for_bin=subsample_for_bin,
                class_weight=class_weight,
                min_split_gain=min_split_gain,
                min_child_weight=min_child_weight,
                min_child_samples=min_child_samples,
                subsample=subsample,
                subsample_freq=subsample_freq,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                n_jobs=n_jobs,
                importance_type=importance_type,
                **kwargs,
            )
        )
        model_instance = LGBMClassifier(
            **self._kwargs,
            random_state=random_state,
        )

        super().__init__(
            model_instance=model_instance,
            edge_embedding_method=edge_embedding_method,
            training_unbalance_rate=training_unbalance_rate,
            training_sample_only_edges_with_heterogeneous_node_types=training_sample_only_edges_with_heterogeneous_node_types,
            use_edge_metrics=use_edge_metrics,
            use_scale_free_distribution=use_scale_free_distribution,
            prediction_batch_size=prediction_batch_size,
        )

    def parameters(self) -> Dict[str, Any]:
        return dict(
            **super().parameters(),
            **self._kwargs,
        )

    @classmethod
    def smoke_test_parameters(cls) -> Dict[str, Any]:
        return dict(
            n_estimators=2,
            max_depth=2,
        )

    @classmethod
    def model_name(cls) -> str:
        """Return the name of the model."""
        return "LightGBM"

    @classmethod
    def library_name(cls) -> str:
        """Return name of the model."""
        return "LightGBM"