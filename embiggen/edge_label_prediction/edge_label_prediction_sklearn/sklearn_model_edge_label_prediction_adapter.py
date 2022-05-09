"""Module providing adapter class making edge-label prediction possible in sklearn models."""
from sklearn.base import ClassifierMixin
from typing import Type, List, Dict, Union, Optional, Tuple
import pandas as pd
import numpy as np
from ensmallen import Graph
from ...utils import must_be_an_sklearn_classifier_model, evaluate_sklearn_classifier
from ...transformers import EdgeLabelPredictionTransformer, GraphTransformer
from ..edge_label_prediction_model import AbstractEdgeLabelPredictionModel


class SklearnModelEdgeLabelPredictionAdapter(AbstractEdgeLabelPredictionModel):
    """Class wrapping Sklearn models for running ."""

    def __init__(
        self,
        model_instance: Type[ClassifierMixin],
        edge_embedding_method: str = "Concatenate",
        random_state: int = 42
    ):
        """Create the adapter for Sklearn object.

        Parameters
        ----------------
        model_instance: Type[ClassifierMixin]
            The class instance to be adapted into edge-label prediction.
        edge_embedding_method: str = "Concatenate"
            The method to use to compute the edges.
        random_state: int = 42
            The random state to use to reproduce the training.

        Raises
        ----------------
        ValueError
            If the provided model_instance is not a subclass of `ClassifierMixin`.
        """
        super().__init__()
        must_be_an_sklearn_classifier_model(model_instance)
        self._random_state = random_state
        self._model_instance = model_instance
        self._edge_embedding_method = edge_embedding_method

    def _trasform_graph_into_edge_embedding(
        self,
        graph: Graph,
        node_features: np.ndarray,
        edge_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Transforms the provided data into an Sklearn-compatible numpy array.

        Parameters
        ------------------
        graph: Graph
            The graph whose edges are to be embedded and predicted.
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
        gt = GraphTransformer(
            method=self._edge_embedding_method,
            aligned_node_mapping=True
        )

        gt.fit(node_features)

        return gt.transform(
            graph=graph,
            edge_features=edge_features
        )

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
        lpt = EdgeLabelPredictionTransformer(
            method=self._edge_embedding_method,
            aligned_node_mapping=True
        )

        lpt.fit(node_features)

        self._model_instance.fit(**lpt.transform(
            graph=graph,
            edge_features=edge_features,
            behaviour_for_unknown_edge_labels="drop",
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
        return self._model_instance.predict_proba(self._trasform_graph_into_edge_embedding(
            graph=graph,
            node_features=node_features,
            edge_features=edge_features,
        ))

    def predict(
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
        return self._model_instance.predict(self._trasform_graph_into_edge_embedding(
            graph=graph,
            node_features=node_features,
            edge_features=edge_features,
        ))
