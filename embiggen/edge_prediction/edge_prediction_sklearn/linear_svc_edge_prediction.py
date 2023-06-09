"""Submodule wrapping Linear Support Vector Machine for Edge prediction."""
from typing import Dict, Any, Union
from sklearn.svm import LinearSVC
from embiggen.edge_prediction.edge_prediction_sklearn.sklearn_edge_prediction_adapter import SklearnEdgePredictionAdapter
from embiggen.utils import normalize_kwargs


class LinearSVCEdgeLabelPrediction(SklearnEdgePredictionAdapter):
    """Create wrapper over Sklearn Linear Support Vector Machine classifier for Edge prediction."""

    def __init__(
        self,
        penalty: str = "l2",
        loss: str = "squared_hinge",
        dual: bool = True,
        tol: float = 1e-4,
        C: float = 1.0,
        multi_class: str = "ovr",
        fit_intercept: bool = True,
        intercept_scaling: float = 1.0,
        class_weight: Union[Dict, str] = "balanced",
        verbose: int = 0,
        max_iter: int = 1000,
        edge_embedding_method: str = "Concatenate",
        training_unbalance_rate: float = 1.0,
        training_sample_only_edges_with_heterogeneous_node_types: bool = False,
        use_edge_metrics: bool = False,
        use_scale_free_distribution: bool = True,
        prediction_batch_size: int = 2**12,
        random_state: int = 42
    ):
        """Create the Linear Support Vector Machine for Node-label prediction."""
        self._kwargs = normalize_kwargs(
            self,
            dict(
                penalty=penalty,
                loss=loss,
                dual=dual,
                tol=tol,
                C=C,
                multi_class=multi_class,
                fit_intercept=fit_intercept,
                intercept_scaling=intercept_scaling,
                class_weight=class_weight,
                verbose=verbose,
                max_iter=max_iter,
            )
        )

        super().__init__(
            LinearSVC(**self._kwargs, random_state=random_state),
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
        return {**super().parameters(), **self._kwargs}

    @classmethod
    def smoke_test_parameters(cls) -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(max_iter=2)

    @classmethod
    def model_name(cls) -> str:
        return "Linear Support Vector Classifier"
