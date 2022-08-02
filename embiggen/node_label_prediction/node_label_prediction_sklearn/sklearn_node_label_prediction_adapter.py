"""Module providing adapter class making node-label prediction possible in sklearn models."""
from sklearn.base import ClassifierMixin
from typing import Type, List, Dict, Optional, Any
import numpy as np
import copy
from ensmallen import Graph
from embiggen.embedding_transformers import NodeLabelPredictionTransformer, NodeTransformer
from embiggen.utils.sklearn_utils import must_be_an_sklearn_classifier_model
from embiggen.node_label_prediction.node_label_prediction_model import AbstractNodeLabelPredictionModel
from embiggen.utils.abstract_models import abstract_class


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
        super().__init__(random_state=random_state)
        must_be_an_sklearn_classifier_model(model_instance)
        self._model_instance = model_instance
        # We want to mask the decorator class name
        self.__class__.__name__ = model_instance.__class__.__name__
        self.__class__.__doc__ = model_instance.__class__.__doc__

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
        gt = NodeTransformer(aligned_mapping=True)
        gt.fit(node_features)
        return gt.transform(graph,)

    def _fit(
        self,
        graph: Graph,
        support: Optional[Graph] = None,
        node_features: Optional[List[np.ndarray]] = None,
        node_type_features: Optional[List[np.ndarray]] = None,
        edge_features: Optional[List[np.ndarray]] = None,
    ):
        """Execute fitting of the model.

        Parameters
        ------------------
        graph: Graph,
            The graph whose edges are to be embedded and edge types extracted.
            It can either be an Graph or a list of lists of edges.
        support: Optional[Graph] = None
            The graph describiding the topological structure that
            includes also the above graph. This parameter
            is mostly useful for topological classifiers
            such as Graph Convolutional Networks.
        node_features: Optional[List[np.ndarray]] = None
            The node features to be used in the training of the model.
        node_type_features: Optional[List[np.ndarray]] = None
            The node type features to be used in the training of the model.
        edge_features: Optional[List[np.ndarray]] = None
            Optional edge features to be used as input concatenated
            to the obtained edge embedding. The shape must be equal
            to the number of directed edges in the graph.

        Raises
        ------------------
        ValueError
            If the two graphs do not share the same node vocabulary.
        """
        nlpt = NodeLabelPredictionTransformer(
            aligned_mapping=True
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
        support: Optional[Graph] = None,
        node_features: Optional[List[np.ndarray]] = None,
        node_type_features: Optional[List[np.ndarray]] = None,
        edge_features: Optional[List[np.ndarray]] = None,
    ) -> Dict[str, float]:
        """Return evaluations of the model on the edge-label prediction task on the provided data.

        Parameters
        ------------------
        graph: Graph,
            The graph whose edges are to be embedded and predicted.
            It can either be an Graph or a list of lists of edges.
        support: Optional[Graph] = None
            The graph describiding the topological structure that
            includes also the above graph. This parameter
            is mostly useful for topological classifiers
            such as Graph Convolutional Networks.
        node_features: Optional[List[np.ndarray]] = None
            The node features to be used in the evaluation of the model.
        node_type_features: Optional[List[np.ndarray]] = None
            The node features to be used in prediction.
        edge_features: Optional[List[np.ndarray]] = None
            Optional edge features to be used as input concatenated
            to the obtained edge embedding. The shape must be equal
            to the number of directed edges in the provided graph.

        Raises
        ------------------
        ValueError
            If the two graphs do not share the same node vocabulary.
        """
        predictions_probabilities = self._model_instance.predict_proba(self._trasform_graph_into_node_embedding(
            graph=graph,
            node_features=node_features,
        ))

        if self.is_multilabel_prediction_task():
            if isinstance(predictions_probabilities, np.ndarray):
                return predictions_probabilities
            if isinstance(predictions_probabilities, list):
                return np.array([
                    class_predictions[:, 1]
                    for class_predictions in predictions_probabilities
                ]).T
            raise NotImplementedError(
                f"The model {self.model_name()} from library {self.library_name()} "
                f"returned an object of type {type(predictions_probabilities)} during "
                "the execution of the predict proba method."
            )

        return predictions_probabilities

    def _predict(
        self,
        graph: Graph,
        support: Optional[Graph] = None,
        node_features: Optional[List[np.ndarray]] = None,
        node_type_features: Optional[List[np.ndarray]] = None,
        edge_features: Optional[List[np.ndarray]] = None,
    ) -> Dict[str, float]:
        """Return evaluations of the model on the edge-label prediction task on the provided data.

        Parameters
        ------------------
        graph: Graph,
            The graph whose edges are to be embedded and predicted.
            It can either be an Graph or a list of lists of edges.
        support: Optional[Graph] = None
            The graph describiding the topological structure that
            includes also the above graph. This parameter
            is mostly useful for topological classifiers
            such as Graph Convolutional Networks.
        node_features: List[np.ndarray]
            The node features to be used in prediction.
        node_type_features: List[np.ndarray]
            The node features to be used in prediction.
        edge_features: Optional[List[np.ndarray]] = None
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

    @classmethod
    def library_name(cls) -> str:
        """Return name of the model."""
        return "scikit-learn"

    @classmethod
    def can_use_edge_weights(cls) -> bool:
        """Returns whether the model can optionally use edge weights."""
        return False
    
    @classmethod
    def can_use_edge_types(cls) -> bool:
        """Returns whether the model can optionally use edge types."""
        return False

