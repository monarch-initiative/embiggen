"""Module providing adapter class making node-label prediction possible in sklearn models."""
from sklearn.base import ClassifierMixin
from typing import Type, List, Dict, Union, Optional, Tuple
import pandas as pd
import numpy as np
from ensmallen import Graph
from ...transformers import NodeLabelPredictionTransformer, NodeTransformer
from ...utils import must_be_an_sklearn_classifier_model, evaluate_sklearn_classifier


class SklearnModelNodeLabelPredictionAdapter:
    """Class wrapping Sklearn models for running node-label predictions."""

    def __init__(
        self,
        model_instance: Type[ClassifierMixin],
    ):
        """Create the adapter for Sklearn object.

        Parameters
        ----------------
        model_instance: Type[ClassifierMixin]
            The class instance to be adapted into node-label prediction.

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
    def _trasform_graphs_into_node_embedding(
        graph: Graph,
        node_features: Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]],
        behaviour_for_unknown_node_labels: str = "warn",
        aligned_node_mapping: bool = False,
        shuffle: bool = False,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transforms the provided data into an Sklearn-compatible numpy array.

        Parameters
        ------------------
        graph: Graph,
            The graph to use for this task.
            It can either be an Graph or a list of lists of nodes.
        node_features: Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
            The node features to be used in the training of the model.
        behaviour_for_unknown_node_labels: str = "warn"
            Behaviour to be followed when encountering nodes that do not
            have a known node type. Possible values are:
            - drop: we drop these nodes
            - keep: we keep these nodes
            By default, we drop these nodes.
            If the behaviour has not been specified and left to None,
            a warning will be raised to notify the user of this uncertainty.
        aligned_node_mapping: bool = False,
            This parameter specifies whether the mapping of the embeddings nodes
            matches the internal node mapping of the given graph.
            If these two mappings do not match, the generated edge embedding
            will be meaningless.
        shuffle: bool = False
            Whether to shuffle the labels.
            In some models, this is necessary.
        random_state: int = 42
            The random state to reproduce the sampling and training.

        Raises
        ------------------
        ValueError
            If the two graphs do not share the same node vocabulary.
        """
        nlpt = NodeLabelPredictionTransformer(
            aligned_node_mapping=aligned_node_mapping
        )

        nlpt.fit(node_features)

        return nlpt.transform(
            graph=graph,
            behaviour_for_unknown_node_labels=behaviour_for_unknown_node_labels,
            shuffle=shuffle,
            random_state=random_state
        )

    @staticmethod
    def _trasform_graph_into_node_embedding(
        graph: Graph,
        node_features: Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]],
        aligned_node_mapping: bool = False,
    ) -> np.ndarray:
        """Transforms the provided data into an Sklearn-compatible numpy array.

        Parameters
        ------------------
        graph: Graph,
            The graph whose edges are to be embedded and predicted.
            It can either be an Graph or a list of lists of edges.
        node_features: Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
            The node features to be used in the training of the model.
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
        gt = NodeTransformer(aligned_node_mapping=aligned_node_mapping)
        gt.fit(node_features)
        return gt.transform(graph,)

    def fit(
        self,
        graph: Graph,
        node_features: Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]],
        behaviour_for_unknown_node_labels: str = "warn",
        aligned_node_mapping: bool = False,
        random_state: int = 42,
        **kwargs
    ):
        """Execute fitting of the model.

        Parameters
        ------------------
        graph: Graph,
            The graph to use for this task.
            It can either be an Graph or a list of lists of nodes.
        node_features: Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
            The node features to be used in the training of the model.
        behaviour_for_unknown_node_labels: str = "warn"
            Behaviour to be followed when encountering nodes that do not
            have a known node type. Possible values are:
            - drop: we drop these nodes
            - keep: we keep these nodes
            By default, we drop these nodes.
            If the behaviour has not been specified and left to None,
            a warning will be raised to notify the user of this uncertainty.
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
            *SklearnModelNodeLabelPredictionAdapter._trasform_graphs_into_node_embedding(
                graph=graph,
                node_features=node_features,
                behaviour_for_unknown_node_labels=behaviour_for_unknown_node_labels,
                aligned_node_mapping=aligned_node_mapping,
                random_state=random_state,
            ),
            **kwargs
        )

    def evaluate(
        self,
        graph: Graph,
        node_features: Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]],
        behaviour_for_unknown_node_labels: str = "warn",
        aligned_node_mapping: bool = False,
        random_state: int = 42,
        **kwargs
    ) -> Dict[str, float]:
        """Return evaluations of the model on the node-label prediction task on the provided data.

        Parameters
        ------------------
        graph: Graph,
            The graph to use for this task.
            It can either be an Graph or a list of lists of nodes.
        node_features: Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
            The node features to be used in the evaluation of the model.
        behaviour_for_unknown_node_labels: str = "warn"
            Behaviour to be followed when encountering nodes that do not
            have a known node type. Possible values are:
            - drop: we drop these nodes
            - keep: we keep these nodes
            By default, we drop these nodes.
            If the behaviour has not been specified and left to None,
            a warning will be raised to notify the user of this uncertainty.
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
            *SklearnModelNodeLabelPredictionAdapter._trasform_graphs_into_node_embedding(
                graph=graph,
                node_features=node_features,
                behaviour_for_unknown_node_labels=behaviour_for_unknown_node_labels,
                aligned_node_mapping=aligned_node_mapping,
                shuffle=False,
                random_state=random_state,
            ),
            multiclass_or_multilabel=graph.get_node_types_number() > 2,
            **kwargs
        )

    def predict_proba(
        self,
        graph: Graph,
        node_features: Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]],
        aligned_node_mapping: bool = False,
        **kwargs
    ) -> Dict[str, float]:
        """Return evaluations of the model on the node-label prediction task on the provided data.

        Parameters
        ------------------
        graph: Graph,
            The graph whose edges are to be embedded and predicted.
            It can either be an Graph or a list of lists of edges.
        node_features: Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
            The node features to be used in the evaluation of the model.
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
            SklearnModelNodeLabelPredictionAdapter._trasform_graph_into_edge_embedding(
                graph=graph,
                node_features=node_features,
                aligned_node_mapping=aligned_node_mapping,
            ),
            **kwargs
        )

    def predict(
        self,
        graph: Graph,
        node_features: Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]],
        aligned_node_mapping: bool = False,
        **kwargs
    ) -> Dict[str, float]:
        """Return evaluations of the model on the node-label prediction task on the provided data.

        Parameters
        ------------------
        graph: Graph,
            The graph to use for this task.
            It can either be an Graph or a list of lists of nodes.
        node_features: Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
            The node features to be used in the evaluation of the model.
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
            SklearnModelNodeLabelPredictionAdapter._trasform_graph_into_node_embedding(
                graph=graph,
                node_features=node_features,
                aligned_node_mapping=aligned_node_mapping,
            ),
            **kwargs
        )
