"""Module providing Feature Perceptron model for edge prediction."""
from typing import Optional,  Dict, Any, List
from ensmallen import Graph
import numpy as np
from ensmallen import models
from embiggen.edge_prediction.edge_prediction_model import AbstractEdgePredictionModel


class CooccurrenceEnsmallen(AbstractEdgePredictionModel):
    """Edge prediction model based on cooccurrence."""

    def __init__(
        self,
        window_size: int = 10,
        iterations: int = 50,
        random_state: int = 42,
        verbose: bool = True
    ):
        """Create new Feature Percetron object.

        Parameters
        ------------------------
        window_size: int = 10
            Window size for the local context.
            On the borders the window size is trimmed.
        iterations: int = 50
            Number of iterations of the single walks.
        random_state: int = 42
            The random state to reproduce the model initialization and training. By default, 42.
        verbose: bool = True
            Whether to show epochs loading bar.
        """
        super().__init__(random_state=random_state)
        self._window_size = window_size
        self._iterations = iterations
        self._verbose = verbose
        self._model = models.CooccurrenceEdgePrediction(
            window_size=window_size,
            iterations=iterations,
            random_state=random_state
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters used for this model."""
        return dict(
            **super().parameters(),
            window_size=self._window_size,
            iterations=self._iterations,
        )

    def clone(self) -> "CooccurrenceEnsmallen":
        return CooccurrenceEnsmallen(**self.parameters())

    @staticmethod
    def smoke_test_parameters() -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            iterations=1,
            window_size=2
        )

    def _fit(
        self,
        graph: Graph,
        support: Optional[Graph] = None,
        node_features: Optional[List[np.ndarray]] = None,
        node_type_features: Optional[List[np.ndarray]] = None,
        edge_features: Optional[List[np.ndarray]] = None,
    ):
        pass

    def _predict(
        self,
        graph: Graph,
        support: Optional[Graph] = None,
        node_features: Optional[List[np.ndarray]] = None,
        node_type_features: Optional[List[np.ndarray]] = None,
        edge_features: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """Run prediction on the provided graph.

        Parameters
        --------------------
        graph: Graph
            The graph to run predictions on.
        support: Optional[Graph] = None
            The graph describiding the topological structure that
            includes also the above graph. This parameter
            is mostly useful for topological classifiers
            such as Graph Convolutional Networks.
        node_features: Optional[List[np.ndarray]]
            The node features to use.
        node_type_features: Optional[List[np.ndarray]] = None
            The node type features to use.
        edge_features: Optional[List[np.ndarray]] = None
            The edge features to use.
        """
        return self._predict_proba(
            graph=graph,
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_features=edge_features
        ) > 0.5

    def _predict_proba(
        self,
        graph: Graph,
        support: Optional[Graph] = None,
        node_features: Optional[List[np.ndarray]] = None,
        node_type_features: Optional[List[np.ndarray]] = None,
        edge_features: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """Run prediction on the provided graph.

        Parameters
        --------------------
        graph: Graph
            The graph to run predictions on.
        support: Optional[Graph] = None
            The graph describiding the topological structure that
            includes also the above graph. This parameter
            is mostly useful for topological classifiers
            such as Graph Convolutional Networks.
        node_features: Optional[List[np.ndarray]] = None
            The node features to use.
        node_type_features: Optional[List[np.ndarray]] = None
            The node type features to use.
        edge_features: Optional[List[np.ndarray]] = None
            The edge features to use.
        """
        return self._model.predict(
            graph=graph,
            support=support,
        )

    @staticmethod
    def requires_edge_weights() -> bool:
        return False

    @staticmethod
    def requires_positive_edge_weights() -> bool:
        return False

    @staticmethod
    def requires_node_types() -> bool:
        return False

    @staticmethod
    def requires_edge_types() -> bool:
        return False

    @staticmethod
    def can_use_edge_weights() -> bool:
        """Returns whether the model can optionally use edge weights."""
        return False

    @staticmethod
    def can_use_node_types() -> bool:
        """Returns whether the model can optionally use node types."""
        return False


    @staticmethod
    def can_use_edge_types() -> bool:
        """Returns whether the model can optionally use edge types."""
        return False


    @staticmethod
    def model_name() -> str:
        return "Cooccurrence"

    @staticmethod
    def library_name() -> str:
        return "Ensmallen"
