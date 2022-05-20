"""Module providing adapter class making edge prediction possible in sklearn models."""
from sklearn.base import ClassifierMixin
from typing import Type, List, Optional, Dict, Any
import numpy as np
import math
import copy
from ensmallen import Graph
from ...utils.sklearn_utils import must_be_an_sklearn_classifier_model
from ...transformers import EdgePredictionTransformer, GraphTransformer
from ..edge_prediction_model import AbstractEdgePredictionModel
from ...utils import abstract_class


@abstract_class
class SklearnEdgePredictionAdapter(AbstractEdgePredictionModel):
    """Class wrapping Sklearn models for running ."""

    def __init__(
        self,
        model_instance: Type[ClassifierMixin],
        edge_embedding_method: str = "Concatenate",
        unbalance_rate: float = 1.0,
        sample_only_edges_with_heterogeneous_node_types: bool = False,
        random_state: int = 42
    ):
        """Create the adapter for Sklearn object.

        Parameters
        ----------------
        model_instance: Type[ClassifierMixin]
            The class instance to be adapted into edge prediction.
        edge_embedding_method: str = "Concatenate"
            The method to use to compute the edges.
        unbalance_rate: float = 1.0
            Unbalance rate for the training non-existing edges.
        sample_only_edges_with_heterogeneous_node_types: bool = False
            Whether to sample negative edges exclusively between nodes with different node types.
            This can be useful when executing a bipartite edge prediction task.
        random_state: int
            The random state to use to reproduce the training.

        Raises
        ----------------
        ValueError
            If the provided model_instance is not a subclass of `ClassifierMixin`.
        """
        super().__init__()
        must_be_an_sklearn_classifier_model(model_instance)
        self._model_instance = model_instance
        self._edge_embedding_method = edge_embedding_method
        self._unbalance_rate = unbalance_rate
        self._random_state = random_state
        self._sample_only_edges_with_heterogeneous_node_types = sample_only_edges_with_heterogeneous_node_types
        # We want to mask the decorator class name
        self.__class__.__name__ = model_instance.__class__.__name__
        self.__class__.__doc__ = model_instance.__class__.__doc__

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters used for this model."""
        return {
            "sample_only_edges_with_heterogeneous_node_types": self._sample_only_edges_with_heterogeneous_node_types,
            "edge_embedding_method": self._edge_embedding_method,
            "unbalance_rate": self._unbalance_rate,
            "random_state": self._random_state
        }

    def clone(self) -> Type["SklearnEdgePredictionAdapter"]:
        """Return copy of self."""
        return copy.deepcopy(self)

    @staticmethod
    def library_name() -> str:
        """Return name of the model."""
        return "scikit-learn"

    def _trasform_graph_into_edge_embedding(
        self,
        graph: Graph,
        node_features: List[np.ndarray],
        node_type_features: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """Transforms the provided data into an Sklearn-compatible numpy array.

        Parameters
        ------------------
        graph: Graph,
            The graph whose edges are to be embedded and predicted.
            It can either be an Graph or a list of lists of edges.
        node_features: List[np.ndarray]
            The node features to be used in the training of the model.
        node_type_features: Optional[List[np.ndarray]] = None,
            The node type features to be used in the training of the model.

        Warns
        ------------------
        If the node features are provided as a numpy array it will not be possible
        to check whether the nodes in the graphs are aligned with the features.

        Raises
        ------------------
        ValueError
            If the two graphs do not share the same node vocabulary.
        """
        if node_type_features is not None:
            raise NotImplementedError(
                "Support for node type features is not currently available for any "
                "of the edge prediction models from the Sklearn library."
            )
        
        gt = GraphTransformer(
            method=self._edge_embedding_method,
            aligned_node_mapping=True
        )

        gt.fit(node_features)

        return gt.transform(
            graph=graph
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
        lpt = EdgePredictionTransformer(
            method=self._edge_embedding_method,
            aligned_node_mapping=True
        )

        lpt.fit(node_features)

        self._model_instance.fit(*lpt.transform(
            positive_graph=graph,
            negative_graph=graph.sample_negative_graph(
                number_of_negative_samples=int(
                    math.ceil(graph.get_edges_number()*self._unbalance_rate)
                ),
                random_state=self._random_state,
                sample_only_edges_with_heterogeneous_node_types=self._sample_only_edges_with_heterogeneous_node_types,
            ),
            shuffle=True,
            random_state=self._random_state
        ))

    def _predict(
        self,
        graph: Graph,
        node_features: Optional[List[np.ndarray]],
        node_type_features: Optional[List[np.ndarray]] = None,
        edge_features: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """Run prediction on the provided graph.

        Parameters
        --------------------
        graph: Graph
            The graph to run predictions on.
        node_features: Optional[List[np.ndarray]]
            The node features to use.
        node_type_features: Optional[List[np.ndarray]] = None
            The node type features to use.
        edge_features: Optional[List[np.ndarray]] = None
            The edge features to use.
        """
        return self._model_instance.predict(self._trasform_graph_into_edge_embedding(
            graph=graph,
            node_features=node_features,
            node_type_features=node_type_features,
        ))

    def _predict_proba(
        self,
        graph: Graph,
        node_features: Optional[List[np.ndarray]],
        node_type_features: Optional[List[np.ndarray]] = None,
        edge_features: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """Run prediction on the provided graph.

        Parameters
        --------------------
        graph: Graph
            The graph to run predictions on.
        node_features: Optional[List[np.ndarray]]
            The node features to use.
        node_type_features: Optional[List[np.ndarray]] = None
            The node type features to use.
        edge_features: Optional[List[np.ndarray]] = None
            The edge features to use.
        """
        return self._model_instance.predict_proba(self._trasform_graph_into_edge_embedding(
            graph=graph,
            node_features=node_features,
            node_type_features=node_type_features,
        ))

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