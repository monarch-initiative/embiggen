"""Module providing Feature Perceptron model for edge prediction implementation."""
from typing import Optional,  Dict, Any, List
from ensmallen import Graph
import numpy as np
from ensmallen import models
from embiggen.edge_prediction.edge_prediction_model import AbstractEdgePredictionModel


class FeaturePerceptronEdgePrediction(AbstractEdgePredictionModel):
    """Feature Perceptron model for edge prediction, using only a given node feature.
    
    The peculiarity of this model is that it exclusively receives in input a single,
    relatively simple feature. It is mostly useful to identify bias and try to provide
    a minimum lower bound for the expected performance of better models.
    """

    def __init__(
        self,
        edge_feature_name: str = "Degree",
        number_of_epochs: int = 100,
        number_of_edges_per_mini_batch: int = 1024,
        sample_only_edges_with_heterogeneous_node_types: bool = False,
        learning_rate: float = 0.001,
        random_state: int = 42,
        verbose: bool = True
    ):
        """Create new Feature Percetron object.

        Parameters
        ------------------------
        edge_feature_name: str
            The edge feature to use. By default the source and destination node degrees are used.
            The methods that are currently available are:
            - Degree
            - AdamicAdar
            - JaccardCoefficient
            - ResourceAllocationIndex
            - PreferentialAttachment
        number_of_epochs: int = 100
            The number of epochs to train the model for. By default, 100.
        number_of_edges_per_mini_batch: int = 1024
            The number of samples to include for each mini-batch. By default 1024.
        sample_only_edges_with_heterogeneous_node_types: bool = False
            Whether to sample negative edges only with source and
            destination nodes that have different node types. By default false.
        learning_rate: float = 0.001
            Learning rate to use while training the model. By default 0.001.
        random_state: int = 42
            The random state to reproduce the model initialization and training. By default, 42.
        verbose: bool = True
            Whether to show epochs loading bar.
        """
        super().__init__()
        self._model_kwargs = dict(
            edge_feature_name = edge_feature_name,
            number_of_epochs = number_of_epochs,
            number_of_edges_per_mini_batch = number_of_edges_per_mini_batch,
            sample_only_edges_with_heterogeneous_node_types = sample_only_edges_with_heterogeneous_node_types,
            learning_rate = learning_rate,
            random_state = random_state,
        )
        self._verbose = verbose
        self._model = models.EdgePredictionFeaturePerceptron(**self._model_kwargs)

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters used for this model."""
        return self._model_kwargs

    def clone(self) -> "FeaturePerceptronEdgePrediction":
        return FeaturePerceptronEdgePrediction(**self.parameters())

    @staticmethod
    def smoke_test_parameters() -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            number_of_epochs=1
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
        self._model.fit(
            graph=graph,
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
        return "Feature Perceptron"

    @staticmethod
    def library_name() -> str:
        return "Ensmallen"