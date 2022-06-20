"""Module providing Perceptron for edge prediction implementation."""
from typing import Optional,  Dict, Any, List
from ensmallen import Graph
import numpy as np
from ensmallen import models
from embiggen.edge_prediction.edge_prediction_model import AbstractEdgePredictionModel


class DecisionTreeEdgePredictionEnsmallen(AbstractEdgePredictionModel):
    """Perceptron model for edge prediction."""

    def __init__(
        self,
        metric: str = "F1Score",
        edge_embedding_method_name: str = "CosineSimilarity",
        number_of_edges_to_sample_per_tree_node: int = 2048,
        number_of_splits_per_tree_node: int = 10,
        sample_only_edges_with_heterogeneous_node_types: bool = False,
        negative_edges_rate: float = 0.5,
        depth: int = 10,
        number_of_epochs: int = 10,
        random_state: int = 42,
        verbose: bool = True
    ):
        """Create new Perceptron object.

        Parameters
        ------------------------
        /// metric: str = "F1Score"
        ///     The metric to use. By default, f1 score.
        /// edge_embedding_method_name: str = "CosineSimilarity"
        ///     The embedding method to use. By default the cosine similarity is used.
        /// number_of_edges_to_sample_per_tree_node: int = 2048
        ///     The number of epochs to train the model for. By default, 2048.
        /// number_of_splits_per_tree_node: int = 10
        ///     The number of samples to include for each mini-batch. By default 10.
        /// sample_only_edges_with_heterogeneous_node_types: bool = False
        ///     Whether to sample negative edges only with source and
        ///     destination nodes that have different node types. By default false.
        /// negative_edges_rate: float = 0.5
        ///     Rate of negative edges over total.
        /// depth: int = 10
        ///     Depth of tree. By default 10.
        /// number_of_epochs: int = 10
        ///     Number of sampling iterations. In each iterations, we sample
        ///     a number of positive and negative edges equal to the number
        ///     of directed edges in the graph.
        ///     By default, we do 10 iterations.
        /// random_state: int = 42
        ///     The random state to reproduce the model initialization and training. By default, 42.
        /// verbose: bool = True
        ///     Whether to show epochs loading bar.
        """
        super().__init__(random_state=random_state)
        self._model_kwargs = dict(
            metric=metric,
            edge_embedding_method_name=edge_embedding_method_name,
            number_of_edges_to_sample_per_tree_node=number_of_edges_to_sample_per_tree_node,
            number_of_splits_per_tree_node=number_of_splits_per_tree_node,
            sample_only_edges_with_heterogeneous_node_types=sample_only_edges_with_heterogeneous_node_types,
            negative_edges_rate=negative_edges_rate,
            depth=depth,
            number_of_epochs=number_of_epochs
        )
        self._verbose = verbose
        self._model = models.EdgePredictionSingleExtraTree(
            **self._model_kwargs,
            random_state=random_state
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters used for this model."""
        return dict(
            **super().parameters(),
            **self._model_kwargs
        )

    def clone(self) -> "DecisionTreeEdgePredictionEnsmallen":
        return DecisionTreeEdgePredictionEnsmallen(**self.parameters())

    @staticmethod
    def smoke_test_parameters() -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            depth=3
        )

    def _fit(
        self,
        graph: Graph,
        support: Optional[Graph] = None,
        node_features: Optional[List[np.ndarray]] = None,
        node_type_features: Optional[List[np.ndarray]] = None,
        edge_features: Optional[List[np.ndarray]] = None,
    ):
        """Run fitting on the provided graph.

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
        node_features = np.hstack(node_features)
        if node_features.dtype != np.float32:
            node_features = node_features.astype(np.float32)
        if not node_features.data.c_contiguous:
            node_features = np.ascontiguousarray(node_features)

        self._model.fit(
            graph=graph,
            node_features=node_features,
            verbose=self._verbose,
            support=support
        )

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
        node_features = np.hstack(node_features)
        if node_features.dtype != np.float32:
            node_features = node_features.astype(np.float32)
        if not node_features.data.c_contiguous:
            node_features = np.ascontiguousarray(node_features)

        return self._model.predict(
            graph=graph,
            node_features=node_features,
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

    def is_using_edge_weights(self) -> bool:
        """Returns whether the model is parametrized to use edge weights."""
        return False

    @staticmethod
    def can_use_node_types() -> bool:
        """Returns whether the model can optionally use node types."""
        return False

    def is_using_node_types(self) -> bool:
        """Returns whether the model is parametrized to use node types."""
        return False

    @staticmethod
    def can_use_edge_types() -> bool:
        """Returns whether the model can optionally use edge types."""
        return False

    def is_using_edge_types(self) -> bool:
        """Returns whether the model is parametrized to use edge types."""
        return False

    @staticmethod
    def model_name() -> str:
        return "Decision Tree Classifier"

    @staticmethod
    def library_name() -> str:
        return "Ensmallen"
