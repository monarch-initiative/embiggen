"""Module providing adapter class making edge prediction possible in sklearn models."""
from sklearn.base import ClassifierMixin
from typing import Type, List, Dict, Union, Optional, Tuple
import pandas as pd
import numpy as np
from ensmallen import Graph
from ...utils import must_be_an_sklearn_classifier_model, evaluate_sklearn_classifier
from ...transformers import EdgePredictionTransformer, GraphTransformer
from ..edge_prediction_model import AbstractEdgePredictionModel

class SklearnEdgePredictionAdapter(AbstractEdgePredictionModel):
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
            The class instance to be adapted into edge prediction.
        edge_embedding_method: str = "Concatenate"
            The method to use to compute the edges.
        random_state: int
            The random state to use to reproduce the 

        Raises
        ----------------
        ValueError
            If the provided model_instance is not a subclass of `ClassifierMixin`.
        """
        must_be_an_sklearn_classifier_model(model_instance)
        self._model_instance = model_instance
        self._edge_embedding_method = edge_embedding_method
        # We want to mask the decorator class name
        self.__class__.__name__ = model_instance.__class__.__name__
        self.__class__.__doc__ = model_instance.__class__.__doc__

    def _trasform_graph_into_edge_embedding(
        self,
        graph: Union[Graph, List[List[str]], List[List[int]]],
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
    ) -> np.ndarray:
        """Transforms the provided data into an Sklearn-compatible numpy array.

        Parameters
        ------------------
        graph: Union[Graph, List[List[str]], List[List[int]]],
            The graph whose edges are to be embedded and predicted.
            It can either be an Graph or a list of lists of edges.
        node_features: Union[pd.DataFrame, np.ndarray]
            The node features to be used in the training of the model.
        edge_features: Optional[np.ndarray] = None
            Optional edge features to be used as input concatenated
            to the obtained edge embedding. The shape must be equal
            to the sum of the directed edges in the positive and
            negative graphs. In this matrix, first we expect the
            positive graph edge features and secondly the negative graph
            edge features. We will be shuffling the edge features
            alongside the edge embedding to have everything aligned
            correctly.

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

        return gt.transform(
            graph=graph,
            edge_features=edge_features
        )


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

        edge_features, labels = lpt.transform(
            positive_graph=graph,
            negative_graph=graph.sample_negatives(
                number_of_negative_samples=int(
                    math.ceil(edges_number*unbalance_rate)),
                random_state=random_state,
                sample_only_edges_with_heterogeneous_node_types=sample_only_edges_with_heterogeneous_node_types,
                verbose=False
            ),
            shuffle=True,
            random_state=random_state
        )

        self._model_instance.fit(
            edge_features,
            labels
        )

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
            edge_features=edge_features,
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
            edge_features=edge_features,
        ))