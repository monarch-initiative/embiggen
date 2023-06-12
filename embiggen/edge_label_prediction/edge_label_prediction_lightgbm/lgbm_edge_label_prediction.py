"""Edge-label prediction model based on LightGBM.""" ""
from typing import Dict, Union, Any, List
from lightgbm import LGBMClassifier
from embiggen.edge_label_prediction.sklearn_like_edge_label_prediction_adapter import (
    SklearnLikeEdgeLabelPredictionAdapter,
)
from embiggen.utils.normalize_kwargs import normalize_kwargs


class LightGBMEdgeLabelPrediction(SklearnLikeEdgeLabelPredictionAdapter):
    """Edge-label prediction model based on LightGBM."""

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
        edge_embedding_methods: Union[List[str], str] = "Concatenate",
        use_edge_metrics: bool = False,
        random_state: int = 42,
        **kwargs: Dict,
    ):
        """Build a LightGBM Edge-label prediction model.

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
        edge_embedding_method: Union[List[str], str] = "Concatenate",
            The method(s) to use to compute the edges.
            If multiple edge embedding are provided, they
            will be Concatenated and fed to the model.
            The supported edge embedding methods are:
             * Hadamard: element-wise product
             * Sum: element-wise sum
             * Average: element-wise mean
             * L1: element-wise subtraction
             * AbsoluteL1: element-wise subtraction in absolute value
             * SquaredL2: element-wise subtraction in squared value
             * L2: element-wise squared root of squared subtraction
             * Concatenate: Concatenate of source and destination node features
             * Min: element-wise minimum
             * Max: element-wise maximum
             * L2Distance: vector-wise L2 distance - this yields a scalar
             * CosineSimilarity: vector-wise cosine similarity - this yields a scalar
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
            random_state=random_state,
            edge_embedding_methods=edge_embedding_methods,
            use_edge_metrics=use_edge_metrics,
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