"""Module providing adapter class making edge-label prediction possible in sklearn models."""
from sklearn.base import ClassifierMixin
from typing import Type, List, Dict, Union, Optional, Tuple
import pandas as pd
import numpy as np
from ensmallen import Graph
from ...utils import must_be_an_sklearn_classifier_model, evaluate_sklearn_classifier
from ...transformers import EdgeLabelPredictionTransformer, GraphTransformer


class SklearnModelEdgeLabelPredictionAdapter:
    """Class wrapping Sklearn models for running ."""

    def __init__(
        self,
        model_instance: Type[ClassifierMixin],
    ):
        """Create the adapter for Sklearn object.

        Parameters
        ----------------
        model_instance: Type[ClassifierMixin]
            The class instance to be adapted into edge-label prediction.

        Raises
        ----------------
        ValueError
            If the provided model_instance is not a subclass of `ClassifierMixin`.
        """
        must_be_an_sklearn_classifier_model(model_instance)
        self._model_instance = model_instance

    @staticmethod
    def _trasform_graphs_into_edge_embedding(
        graph: Graph,
        node_features: Union[pd.DataFrame, np.ndarray],
        edge_features: Optional[np.ndarray] = None,
        behaviour_for_unknown_edge_labels: Optional[str] = None,
        edge_embedding_method: str = "Concatenate",
        aligned_node_mapping: bool = False,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transforms the provided data into an Sklearn-compatible numpy array.

        Parameters
        ------------------
        graph: Graph,
            The graph whose edges are to be embedded and edge types extracted.
            It can either be an Graph or a list of lists of edges.
        node_features: Union[pd.DataFrame, np.ndarray]
            The node features to be used in the training of the model.
        edge_features: Optional[np.ndarray] = None
            Optional edge features to be used as input concatenated
            to the obtained edge embedding. The shape must be equal
            to the number of directed edges in the graph.
        behaviour_for_unknown_edge_labels: Optional[str] = None
            Behaviour to be followed when encountering edges that do not
            have a known edge type. Possible values are:
            - drop: we drop these edges
            - keep: we keep these edges
            By default, we drop these edges.
            If the behaviour has not been specified and left to None,
            a warning will be raised to notify the user of this uncertainty.
        edge_embedding_method: str = "Concatenate"
            The method to be used to compute the edge embedding
            starting from the provided node features.
        aligned_node_mapping: bool = False,
            This parameter specifies whether the mapping of the embeddings nodes
            matches the internal node mapping of the given graph.
            If these two mappings do not match, the generated edge embedding
            will be meaningless.
        random_state: int = 42
            The random state to reproduce the sampling and training.

        Raises
        ------------------
        ValueError
            If the two graphs do not share the same node vocabulary.
        """
        lpt = EdgeLabelPredictionTransformer(
            method=edge_embedding_method,
            aligned_node_mapping=aligned_node_mapping
        )

        lpt.fit(node_features)

        return lpt.transform(
            graph=graph,
            edge_features=edge_features,
            behaviour_for_unknown_edge_labels=behaviour_for_unknown_edge_labels,
            random_state=random_state
        )

    @staticmethod
    def _trasform_graph_into_edge_embedding(
        graph: Graph,
        node_features: Union[pd.DataFrame, np.ndarray],
        edge_features: Optional[np.ndarray] = None,
        edge_embedding_method: str = "Concatenate",
        aligned_node_mapping: bool = False,
    ) -> np.ndarray:
        """Transforms the provided data into an Sklearn-compatible numpy array.

        Parameters
        ------------------
        graph: Graph
            The graph whose edges are to be embedded and predicted.
            It can either be an Graph or a list of lists of edges.
        node_features: Union[pd.DataFrame, np.ndarray]
            The node features to be used in the training of the model.
        edge_features: Optional[np.ndarray] = None
            Optional edge features to be used as input concatenated
            to the obtained edge embedding. The shape must be equal
            to the number of directed edges in the graph.
        edge_embedding_method: str = "Concatenate"
            The method to be used to compute the edge embedding
            starting from the provided node features.
        aligned_node_mapping: bool = False,
            This parameter specifies whether the mapping of the embeddings nodes
            matches the internal node mapping of the given graph.
            If these two mappings do not match, the generated edge embedding
            will be meaningless.

        Raises
        ------------------
        ValueError
            If the two graphs do not share the same node vocabulary.
        """
        gt = GraphTransformer(
            method=edge_embedding_method,
            aligned_node_mapping=aligned_node_mapping
        )

        gt.fit(node_features)

        return gt.transform(
            graph=graph,
            edge_features=edge_features
        )

    def fit(
        self,
        graph: Graph,
        node_features: Union[pd.DataFrame, np.ndarray],
        edge_features: Optional[np.ndarray] = None,
        behaviour_for_unknown_edge_labels: Optional[str] = None,
        edge_embedding_method: str = "Concatenate",
        aligned_node_mapping: bool = False,
        random_state: int = 42,
        **kwargs
    ):
        """Execute fitting of the model.

        Parameters
        ------------------
        graph: Graph,
            The graph whose edges are to be embedded and edge types extracted.
            It can either be an Graph or a list of lists of edges.
        node_features: Union[pd.DataFrame, np.ndarray]
            The node features to be used in the training of the model.
        edge_features: Optional[np.ndarray] = None
            Optional edge features to be used as input concatenated
            to the obtained edge embedding. The shape must be equal
            to the number of directed edges in the graph.
        behaviour_for_unknown_edge_labels: Optional[str] = None
            Behaviour to be followed when encountering edges that do not
            have a known edge type. Possible values are:
            - drop: we drop these edges
            - keep: we keep these edges
            By default, we drop these edges.
            If the behaviour has not been specified and left to None,
            a warning will be raised to notify the user of this uncertainty.
        edge_embedding_method: str = "Concatenate"
            The method to be used to compute the edge embedding
            starting from the provided node features.
        aligned_node_mapping: bool = False,
            This parameter specifies whether the mapping of the embeddings nodes
            matches the internal node mapping of the given graph.
            If these two mappings do not match, the generated edge embedding
            will be meaningless.
        random_state: int = 42
            The random state to reproduce the sampling and training.
        **kwargs: Dict
            Dictionary of kwargs to be forwarded to sklearn model fitting.

        Raises
        ------------------
        ValueError
            If the two graphs do not share the same node vocabulary.
        """
        self._model_instance.fit(
            *SklearnModelEdgeLabelPredictionAdapter._trasform_graphs_into_edge_embedding(
                graph=graph,
                node_features=node_features,
                edge_features=edge_features,
                behaviour_for_unknown_edge_labels=behaviour_for_unknown_edge_labels,
                edge_embedding_method=edge_embedding_method,
                aligned_node_mapping=aligned_node_mapping,
                random_state=random_state,
            ),
            **kwargs
        )

    def evaluate(
        self,
        graph: Graph,
        node_features: Union[pd.DataFrame, np.ndarray],
        edge_features: Optional[np.ndarray] = None,
        behaviour_for_unknown_edge_labels: Optional[str] = None,
        edge_embedding_method: str = "Concatenate",
        aligned_node_mapping: bool = False,
        random_state: int = 42,
        **kwargs
    ) -> Dict[str, float]:
        """Return evaluations of the model on the edge-label prediction task on the provided data.

        Parameters
        ------------------
        graph: Graph,
            The graph whose edges are to be embedded and edge types extracted.
            It can either be an Graph or a list of lists of edges.
        node_features: Union[pd.DataFrame, np.ndarray]
            The node features to be used in the evaluation of the model.
        edge_features: Optional[np.ndarray] = None
            Optional edge features to be used as input concatenated
            to the obtained edge embedding. The shape must be equal
            to the number of directed edges in the graph.
        behaviour_for_unknown_edge_labels: Optional[str] = None
            Behaviour to be followed when encountering edges that do not
            have a known edge type. Possible values are:
            - drop: we drop these edges
            - keep: we keep these edges
            By default, we drop these edges.
            If the behaviour has not been specified and left to None,
            a warning will be raised to notify the user of this uncertainty.
        edge_embedding_method: str = "Concatenate"
            The method to be used to compute the edge embedding
            starting from the provided node features.
        aligned_node_mapping: bool = False,
            This parameter specifies whether the mapping of the embeddings nodes
            matches the internal node mapping of the given graph.
            If these two mappings do not match, the generated edge embedding
            will be meaningless.
        random_state: int = 42
            The random state to reproduce the sampling and evaluation.
        **kwargs: Dict
            Dictionary of kwargs to be forwarded to sklearn model prediction.

        Raises
        ------------------
        ValueError
            If the two graphs do not share the same node vocabulary.
        """
        return evaluate_sklearn_classifier(
            self._model_instance,
            *SklearnModelEdgeLabelPredictionAdapter._trasform_graphs_into_edge_embedding(
                graph=graph,
                node_features=node_features,
                edge_features=edge_features,
                behaviour_for_unknown_edge_labels=behaviour_for_unknown_edge_labels,
                edge_embedding_method=edge_embedding_method,
                aligned_node_mapping=aligned_node_mapping,
                random_state=random_state,
            ),
            multiclass_or_multilabel=graph.get_edge_types_number() > 2,
            **kwargs
        )

    def predict_proba(
        self,
        graph: Graph,
        node_features: Union[pd.DataFrame, np.ndarray],
        edge_features: Optional[np.ndarray] = None,
        edge_embedding_method: str = "Concatenate",
        aligned_node_mapping: bool = False,
        **kwargs
    ) -> Dict[str, float]:
        """Return evaluations of the model on the edge-label prediction task on the provided data.

        Parameters
        ------------------
        graph: Graph,
            The graph whose edges are to be embedded and predicted.
            It can either be an Graph or a list of lists of edges.
        node_features: Union[pd.DataFrame, np.ndarray]
            The node features to be used in the evaluation of the model.
        edge_features: Optional[np.ndarray] = None
            Optional edge features to be used as input concatenated
            to the obtained edge embedding. The shape must be equal
            to the number of directed edges in the provided graph.
        edge_embedding_method: str = "Concatenate"
            The method to be used to compute the edge embedding
            starting from the provided node features.
        aligned_node_mapping: bool = False,
            This parameter specifies whether the mapping of the embeddings nodes
            matches the internal node mapping of the given graph.
            If these two mappings do not match, the generated edge embedding
            will be meaningless.
        **kwargs: Dict
            Dictionary of kwargs to be forwarded to sklearn model prediction.

        Raises
        ------------------
        ValueError
            If the two graphs do not share the same node vocabulary.
        """
        return self._model_instance.predict_proba(
            SklearnModelEdgeLabelPredictionAdapter._trasform_graph_into_edge_embedding(
                graph=graph,
                node_features=node_features,
                edge_features=edge_features,
                edge_embedding_method=edge_embedding_method,
                aligned_node_mapping=aligned_node_mapping,
            ),
            **kwargs
        )

    def predict(
        self,
        graph: Graph,
        node_features: Union[pd.DataFrame, np.ndarray],
        edge_features: Optional[np.ndarray] = None,
        edge_embedding_method: str = "Concatenate",
        aligned_node_mapping: bool = False,
        **kwargs
    ) -> Dict[str, float]:
        """Return evaluations of the model on the edge-label prediction task on the provided data.

        Parameters
        ------------------
        graph: Graph,
            The graph whose edges are to be embedded and predicted.
            It can either be an Graph or a list of lists of edges.
        node_features: Union[pd.DataFrame, np.ndarray]
            The node features to be used in the evaluation of the model.
        edge_features: Optional[np.ndarray] = None
            Optional edge features to be used as input concatenated
            to the obtained edge embedding. The shape must be equal
            to the number of directed edges in the provided graph.
        edge_embedding_method: str = "Concatenate"
            The method to be used to compute the edge embedding
            starting from the provided node features.
        aligned_node_mapping: bool = False,
            This parameter specifies whether the mapping of the embeddings nodes
            matches the internal node mapping of the given graph.
            If these two mappings do not match, the generated edge embedding
            will be meaningless.
        **kwargs: Dict
            Dictionary of kwargs to be forwarded to sklearn model prediction.

        Raises
        ------------------
        ValueError
            If the two graphs do not share the same node vocabulary.
        """
        return self._model_instance.predict(
            SklearnModelEdgeLabelPredictionAdapter._trasform_graph_into_edge_embedding(
                graph=graph,
                node_features=node_features,
                edge_features=edge_features,
                edge_embedding_method=edge_embedding_method,
                aligned_node_mapping=aligned_node_mapping,
            ),
            **kwargs
        )
