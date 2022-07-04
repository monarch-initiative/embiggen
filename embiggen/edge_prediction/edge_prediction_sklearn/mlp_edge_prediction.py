"""Submodule wrapping MLP for edge prediction."""
from typing import Dict, Any
from sklearn.neural_network import MLPClassifier
from embiggen.edge_prediction.edge_prediction_sklearn.sklearn_edge_prediction_adapter import SklearnEdgePredictionAdapter


class MLPEdgePrediction(SklearnEdgePredictionAdapter):
    """Create wrapper over Sklearn MLP classifier for edge prediction."""

    def __init__(
        self,
        hidden_layer_sizes=(100,),
        activation="relu",
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate="constant",
        learning_rate_init=0.001,
        power_t=0.5,
        max_iter=200,
        shuffle=True,
        tol=1e-4,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=10,
        max_fun=15000,
        edge_embedding_method: str = "Concatenate",
        training_unbalance_rate: float = 1.0,
        training_sample_only_edges_with_heterogeneous_node_types: bool = False,
        use_edge_metrics: bool = False,
        use_scale_free_distribution: bool = True,
        prediction_batch_size: int = 2**12,
        random_state: int = 42
    ):
        """Create the MLP for Edge Prediction."""
        self._activation = activation
        self._solver = solver
        self._alpha = alpha
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._learning_rate_init = learning_rate_init
        self._power_t = power_t
        self._max_iter = max_iter
        self._shuffle = shuffle
        self._tol = tol
        self._verbose = verbose
        self._warm_start = warm_start
        self._momentum = momentum
        self._nesterovs_momentum = nesterovs_momentum
        self._early_stopping = early_stopping
        self._validation_fraction = validation_fraction
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._epsilon = epsilon
        self._n_iter_no_change = n_iter_no_change
        self._max_fun = max_fun
        super().__init__(
            MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                solver=solver,
                alpha=alpha,
                batch_size=batch_size,
                learning_rate=learning_rate,
                learning_rate_init=learning_rate_init,
                power_t=power_t,
                max_iter=max_iter,
                shuffle=shuffle,
                random_state=random_state,
                tol=tol,
                verbose=verbose,
                warm_start=warm_start,
                momentum=momentum,
                nesterovs_momentum=nesterovs_momentum,
                early_stopping=early_stopping,
                validation_fraction=validation_fraction,
                beta_1=beta_1,
                beta_2=beta_2,
                epsilon=epsilon,
                n_iter_no_change=n_iter_no_change,
                max_fun=max_fun
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
                activation=self._activation,
                solver=self._solver,
                alpha=self._alpha,
                batch_size=self._batch_size,
                learning_rate=self._learning_rate,
                learning_rate_init=self._learning_rate_init,
                power_t=self._power_t,
                max_iter=self._max_iter,
                shuffle=self._shuffle,
                tol=self._tol,
                verbose=self._verbose,
                warm_start=self._warm_start,
                momentum=self._momentum,
                nesterovs_momentum=self._nesterovs_momentum,
                early_stopping=self._early_stopping,
                validation_fraction=self._validation_fraction,
                beta_1=self._beta_1,
                beta_2=self._beta_2,
                epsilon=self._epsilon,
                n_iter_no_change=self._n_iter_no_change,
                max_fun=self._max_fun,
            )
        }

    @classmethod
    def smoke_test_parameters(cls) -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            hidden_layer_sizes=(1,),
            max_iter=1
        )

    @classmethod
    def model_name(cls) -> str:
        return "MLP Classifier"