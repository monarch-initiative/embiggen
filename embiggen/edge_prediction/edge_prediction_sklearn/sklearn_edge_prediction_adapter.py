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

    def clone(self) -> "Self":
        """Return copy of self."""
        return copy.deepcopy(self)

    def name(self) -> str:
        """Return name of the model."""
        return self.__class__.__name__

    def _trasform_graph_into_edge_embedding(
        self,
        graph: Graph,
        node_features: List[np.ndarray],
    ) -> np.ndarray:
        """Transforms the provided data into an Sklearn-compatible numpy array.

        Parameters
        ------------------
        graph: Graph,
            The graph whose edges are to be embedded and predicted.
            It can either be an Graph or a list of lists of edges.
        node_features: List[np.ndarray]
            The node features to be used in the training of the model.

        Warns
        ------------------
        If the node features are provided as a numpy array it will not be possible
        to check whether the nodes in the graphs are aligned with the features.

        Raises
        ------------------
        ValueError
            If the two graphs do not share the same node vocabulary.
        """
        gt = GraphTransformer(
            method=self._edge_embedding_method,
            aligned_node_mapping=False
        )

        gt.fit(node_features)

        return gt.transform(graph=graph)

    def _fit(
        self,
        graph: Graph,
        node_features: Optional[List[np.ndarray]],
        edge_features: Optional[List[np.ndarray]] = None,
    ):
        """Run fitting on the provided graph.

        Parameters
        --------------------
        graph: Graph
            The graph to run predictions on.
        node_features: Optional[List[np.ndarray]]
            The node features to use.
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
            negative_graph=graph.sample_negatives(
                number_of_negative_samples=int(
                    math.ceil(graph.get_edges_number()*self._unbalance_rate)
                ),
                random_state=self._random_state,
                sample_only_edges_with_heterogeneous_node_types=self._sample_only_edges_with_heterogeneous_node_types,
                verbose=False
            ),
            shuffle=True,
            random_state=self._random_state
        ))

    def _predict(
        self,
        graph: Graph,
        node_features: Optional[List[np.ndarray]],
        edge_features: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """Run prediction on the provided graph.

        Parameters
        --------------------
        graph: Graph
            The graph to run predictions on.
        node_features: Optional[List[np.ndarray]]
            The node features to use.
        edge_features: Optional[List[np.ndarray]] = None
            The edge features to use.
        """
        return self._model_instance.predict(self._trasform_graph_into_edge_embedding(
            graph=graph,
            node_features=node_features,
        ))

    def _predict_proba(
        self,
        graph: Graph,
        node_features: Optional[List[np.ndarray]],
        edge_features: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """Run prediction on the provided graph.

        Parameters
        --------------------
        graph: Graph
            The graph to run predictions on.
        node_features: Optional[List[np.ndarray]]
            The node features to use.
        edge_features: Optional[List[np.ndarray]] = None
            The edge features to use.
        """
        return self._model_instance.predict_proba(self._trasform_graph_into_edge_embedding(
            graph=graph,
            node_features=node_features,
        ))
