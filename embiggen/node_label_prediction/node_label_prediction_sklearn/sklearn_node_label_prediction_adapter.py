"""Module providing adapter class making node-label prediction possible in sklearn models."""
from sklearn.base import ClassifierMixin
from typing import Type, List, Dict, Union, Optional, Tuple, Any
import pandas as pd
import numpy as np
import copy
from ensmallen import Graph
from ...transformers import NodeLabelPredictionTransformer, NodeTransformer
from ...utils.sklearn_utils import must_be_an_sklearn_classifier_model
from ..node_label_prediction_model import AbstractNodeLabelPredictionModel
from ...utils import abstract_class


@abstract_class
class SklearnNodeLabelPredictionAdapter(AbstractNodeLabelPredictionModel):
    """Class wrapping Sklearn models for running node-label predictions."""

    def __init__(
        self,
        model_instance: Type[ClassifierMixin],
        random_state: int = 42
    ):
        """Create the adapter for Sklearn object.

        Parameters
        ----------------
        model_instance: Type[ClassifierMixin]
            The class instance to be adapted into node-label prediction.
        random_state: int = 42
            The random state to use to reproduce the training.

        Raises
        ----------------
        ValueError
            If the provided model_instance is not a subclass of `ClassifierMixin`.
        """
        super().__init__()
        must_be_an_sklearn_classifier_model(model_instance)
        self._model_instance = model_instance
        self._random_state = random_state
        # We want to mask the decorator class name
        self.__class__.__name__ = model_instance.__class__.__name__
        self.__class__.__doc__ = model_instance.__class__.__doc__

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters used for this model."""
        return {
            "random_state": self._random_state
        }

    def clone(self) -> Type["SklearnNodeLabelPredictionAdapter"]:
        """Return copy of self."""
        return copy.deepcopy(self)

    def _trasform_graph_into_node_embedding(
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

        Raises
        ------------------
        ValueError
            If the two graphs do not share the same node vocabulary.
        """
        gt = NodeTransformer(aligned_node_mapping=True)
        gt.fit(node_features)
        return gt.transform(graph,)

    def _fit(
        self,
        graph: Graph,
        node_features: np.ndarray,
        edge_features: Optional[np.ndarray] = None,
    ):
        """Execute fitting of the model.

        Parameters
        ------------------
        graph: Graph,
            The graph whose edges are to be embedded and edge types extracted.
            It can either be an Graph or a list of lists of edges.
        node_features: np.ndarray
            The node features to be used in the training of the model.
        edge_features: Optional[np.ndarray] = None
            Optional edge features to be used as input concatenated
            to the obtained edge embedding. The shape must be equal
            to the number of directed edges in the graph.

        Raises
        ------------------
        ValueError
            If the two graphs do not share the same node vocabulary.
        """
        nlpt = NodeLabelPredictionTransformer(
            aligned_node_mapping=True
        )

        nlpt.fit(node_features)

        self._model_instance.fit(*nlpt.transform(
            graph=graph,
            behaviour_for_unknown_node_labels="drop",
            shuffle=True,
            random_state=self._random_state
        ))

    def _predict_proba(
        self,
        graph: Graph,
        node_features: np.ndarray,
        edge_features: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Return evaluations of the model on the edge-label prediction task on the provided data.

        Parameters
        ------------------
        graph: Graph,
            The graph whose edges are to be embedded and predicted.
            It can either be an Graph or a list of lists of edges.
        node_features: np.ndarray
            The node features to be used in the evaluation of the model.
        edge_features: Optional[np.ndarray] = None
            Optional edge features to be used as input concatenated
            to the obtained edge embedding. The shape must be equal
            to the number of directed edges in the provided graph.

        Raises
        ------------------
        ValueError
            If the two graphs do not share the same node vocabulary.
        """
        return self._model_instance.predict_proba(self._trasform_graph_into_node_embedding(
            graph=graph,
            node_features=node_features,
        ))

    def _predict(
        self,
        graph: Graph,
        node_features: np.ndarray,
        edge_features: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Return evaluations of the model on the edge-label prediction task on the provided data.

        Parameters
        ------------------
        graph: Graph,
            The graph whose edges are to be embedded and predicted.
            It can either be an Graph or a list of lists of edges.
        node_features: np.ndarray
            The node features to be used in the evaluation of the model.
        edge_features: Optional[np.ndarray] = None
            Optional edge features to be used as input concatenated
            to the obtained edge embedding. The shape must be equal
            to the number of directed edges in the provided graph.

        Raises
        ------------------
        ValueError
            If the two graphs do not share the same node vocabulary.
        """
        return self._model_instance.predict(self._trasform_graph_into_node_embedding(
            graph=graph,
            node_features=node_features,
        ))

    @staticmethod
    def requires_edge_weights() -> bool:
        return False

    @staticmethod
    def requires_positive_edge_weights() -> bool:
        return False

    @staticmethod
    def requires_edge_types() -> bool:
        return False