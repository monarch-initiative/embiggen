"""Module providing adapter class making edge prediction possible in sklearn models."""
from sklearn.base import ClassifierMixin
from typing import Type, List, Dict, Union, Optional, Tuple
import pandas as pd
import numpy as np
from ensmallen import Graph
from ...utils import must_be_an_sklearn_classifier_model, evaluate_sklearn_classifier
from ...transformers import EdgePredictionTransformer, GraphTransformer


class SklearnModelEdgePredictionAdapter:
    """Class wrapping Sklearn models for running ."""

    def __init__(
        self,
        model_instance: Type[ClassifierMixin],
    ):
        """Create the adapter for Sklearn object.

        Parameters
        ----------------
        model_instance: Type[ClassifierMixin]
            The class instance to be adapted into edge prediction.

        Raises
        ----------------
        ValueError
            If the provided model_instance is not a subclass of `ClassifierMixin`.
        """
        must_be_an_sklearn_classifier_model(model_instance)
        self._model_instance = model_instance
        # We want to mask the decorator class name
        self.__class__.__name__ = model_instance.__class__.__name__
        self.__class__.__doc__ = model_instance.__class__.__doc__

    @staticmethod
    def _trasform_graphs_into_edge_embedding(
        positive_graph: Union[Graph, List[List[str]], List[List[int]]],
        negative_graph: Union[Graph, List[List[str]], List[List[int]]],
        node_features: Union[pd.DataFrame, np.ndarray],
        edge_features: Optional[np.ndarray] = None,
        edge_embedding_method: str = "Concatenate",
        aligned_node_mapping: bool = False,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transforms the provided data into an Sklearn-compatible numpy array.

        Parameters
        ------------------
        positive_graph: Union[Graph, List[List[str]], List[List[int]]],
            The graph whose edges are to be embedded and labeled as positives.
            It can either be an Graph or a list of lists of edges.
        negative_graph: Union[Graph, List[List[str]], List[List[int]]],
            The graph whose edges are to be embedded and labeled as positives.
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

        Warns
        ------------------
        If the node features are provided as a numpy array it will not be possible
        to check whether the nodes in the graphs are aligned with the features.

        Raises
        ------------------
        ValueError
            If the two graphs do not share the same node vocabulary.
        """
        lpt = EdgePredictionTransformer(
            method=edge_embedding_method,
            aligned_node_mapping=aligned_node_mapping
        )

        lpt.fit(node_features)

        return lpt.transform(
            positive_graph=positive_graph,
            negative_graph=negative_graph,
            edge_features=edge_features,
            random_state=random_state
        )

    @staticmethod
    def _trasform_graph_into_edge_embedding(
        graph: Union[Graph, List[List[str]], List[List[int]]],
        node_features: Union[pd.DataFrame, np.ndarray],
        edge_features: Optional[np.ndarray] = None,
        edge_embedding_method: str = "Concatenate",
        aligned_node_mapping: bool = False,
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
        edge_embedding_method: str = "Concatenate"
            The method to be used to compute the edge embedding
            starting from the provided node features.
        aligned_node_mapping: bool = False,
            This parameter specifies whether the mapping of the embeddings nodes
            matches the internal node mapping of the given graph.
            If these two mappings do not match, the generated edge embedding
            will be meaningless.

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
        positive_graph: Union[Graph, List[List[str]], List[List[int]]],
        negative_graph: Union[Graph, List[List[str]], List[List[int]]],
        node_features: Union[pd.DataFrame, np.ndarray],
        edge_features: Optional[np.ndarray] = None,
        edge_embedding_method: str = "Concatenate",
        aligned_node_mapping: bool = False,
        random_state: int = 42,
        **kwargs
    ):
        """Execute fitting of the model.

        Parameters
        ------------------
        positive_graph: Union[Graph, List[List[str]], List[List[int]]],
            The graph whose edges are to be embedded and labeled as positives.
            It can either be an Graph or a list of lists of edges.
        negative_graph: Union[Graph, List[List[str]], List[List[int]]],
            The graph whose edges are to be embedded and labeled as positives.
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

        Warns
        ------------------
        If the node features are provided as a numpy array it will not be possible
        to check whether the nodes in the graphs are aligned with the features.

        Raises
        ------------------
        ValueError
            If the two graphs do not share the same node vocabulary.
        """
        self._model_instance.fit(
            *SklearnModelEdgePredictionAdapter._trasform_graphs_into_edge_embedding(
                positive_graph=positive_graph,
                negative_graph=negative_graph,
                node_features=node_features,
                edge_features=edge_features,
                edge_embedding_method=edge_embedding_method,
                aligned_node_mapping=aligned_node_mapping,
                random_state=random_state,
            ),
            **kwargs
        )

    def evaluate(
        self,
        positive_graph: Union[Graph, List[List[str]], List[List[int]]],
        negative_graph: Union[Graph, List[List[str]], List[List[int]]],
        node_features: Union[pd.DataFrame, np.ndarray],
        edge_features: Optional[np.ndarray] = None,
        edge_embedding_method: str = "Concatenate",
        aligned_node_mapping: bool = False,
        random_state: int = 42,
        **kwargs
    ) -> Dict[str, float]:
        """Return evaluations of the model on the edge prediction task on the provided data.

        Parameters
        ------------------
        positive_graph: Union[Graph, List[List[str]], List[List[int]]],
            The graph whose edges are to be embedded and labeled as positives.
            It can either be an Graph or a list of lists of edges.
        negative_graph: Union[Graph, List[List[str]], List[List[int]]],
            The graph whose edges are to be embedded and labeled as positives.
            It can either be an Graph or a list of lists of edges.
        node_features: Union[pd.DataFrame, np.ndarray]
            The node features to be used in the evaluation of the model.
        edge_features: Optional[np.ndarray] = None
            Optional edge features to be used as input concatenated
            to the obtained edge embedding. The shape must be equal
            to the sum of the directed edges in the positive and
            negative graphs. In this matrix, first we expect the
            positive graph edge features and secondly the negative graph
            edge features. We will be shuffling the edge features
            alongside the edge embedding to have everything aligned
            correctly.
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

        Warns
        ------------------
        If the node features are provided as a numpy array it will not be possible
        to check whether the nodes in the graphs are aligned with the features.

        Raises
        ------------------
        ValueError
            If the two graphs do not share the same node vocabulary.
        """
        return evaluate_sklearn_classifier(
            self._model_instance,
            *SklearnModelEdgePredictionAdapter._trasform_graphs_into_edge_embedding(
                positive_graph=positive_graph,
                negative_graph=negative_graph,
                node_features=node_features,
                edge_features=edge_features,
                edge_embedding_method=edge_embedding_method,
                aligned_node_mapping=aligned_node_mapping,
                random_state=random_state,
            ),
            multiclass_or_multilabel=False,
            **kwargs
        )

    def predict_proba(
        self,
        graph: Union[Graph, List[List[str]], List[List[int]]],
        node_features: Union[pd.DataFrame, np.ndarray],
        edge_features: Optional[np.ndarray] = None,
        edge_embedding_method: str = "Concatenate",
        aligned_node_mapping: bool = False,
        **kwargs
    ) -> Dict[str, float]:
        """Return evaluations of the model on the edge prediction task on the provided data.

        Parameters
        ------------------
        graph: Union[Graph, List[List[str]], List[List[int]]],
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

        Warns
        ------------------
        If the node features are provided as a numpy array it will not be possible
        to check whether the nodes in the graphs are aligned with the features.

        Raises
        ------------------
        ValueError
            If the two graphs do not share the same node vocabulary.
        """
        return self._model_instance.predict_proba(
            SklearnModelEdgePredictionAdapter._trasform_graph_into_edge_embedding(
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
        graph: Union[Graph, List[List[str]], List[List[int]]],
        node_features: Union[pd.DataFrame, np.ndarray],
        edge_features: Optional[np.ndarray] = None,
        edge_embedding_method: str = "Concatenate",
        aligned_node_mapping: bool = False,
        **kwargs
    ) -> Dict[str, float]:
        """Return evaluations of the model on the edge prediction task on the provided data.

        Parameters
        ------------------
        graph: Union[Graph, List[List[str]], List[List[int]]],
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

        Warns
        ------------------
        If the node features are provided as a numpy array it will not be possible
        to check whether the nodes in the graphs are aligned with the features.

        Raises
        ------------------
        ValueError
            If the two graphs do not share the same node vocabulary.
        """
        return self._model_instance.predict(
            SklearnModelEdgePredictionAdapter._trasform_graph_into_edge_embedding(
                graph=graph,
                node_features=node_features,
                edge_features=edge_features,
                edge_embedding_method=edge_embedding_method,
                aligned_node_mapping=aligned_node_mapping,
            ),
            **kwargs
        )
