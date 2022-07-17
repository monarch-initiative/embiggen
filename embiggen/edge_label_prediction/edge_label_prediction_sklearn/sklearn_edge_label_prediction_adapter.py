"""Module providing adapter class making edge-label prediction possible in sklearn models."""
from sklearn.base import ClassifierMixin
from typing import Type, List, Dict, Optional, Any
import numpy as np
import copy
from ensmallen import Graph
from embiggen.utils.sklearn_utils import must_be_an_sklearn_classifier_model
from embiggen.embedding_transformers import EdgeLabelPredictionTransformer, GraphTransformer
from embiggen.edge_label_prediction.edge_label_prediction_model import AbstractEdgeLabelPredictionModel


class SklearnEdgeLabelPredictionAdapter(AbstractEdgeLabelPredictionModel):
    """Class wrapping Sklearn models for running edge-label prediction."""

    def __init__(
        self,
        model_instance: Type[ClassifierMixin],
        edge_embedding_method: str = "Concatenate",
        use_edge_metrics: bool = False,
        random_state: int = 42
    ):
        """Create the adapter for Sklearn object.

        Parameters
        ----------------
        model_instance: Type[ClassifierMixin]
            The class instance to be adapted into edge-label prediction.
        edge_embedding_method: str = "Concatenate"
            The method to use to compute the edges.
        use_edge_metrics: bool = False
            Whether to use the edge metrics from traditional edge prediction.
            These metrics currently include:
            - Adamic Adar
            - Jaccard Coefficient
            - Resource allocation index
            - Preferential attachment
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
        self._edge_embedding_method = edge_embedding_method
        self._use_edge_metrics = use_edge_metrics
        # We want to mask the decorator class name
        self.__class__.__name__ = model_instance.__class__.__name__
        self.__class__.__doc__ = model_instance.__class__.__doc__

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters used for this model."""
        return {
            "edge_embedding_method": self._edge_embedding_method,
            "use_edge_metrics": self._use_edge_metrics,
            **super().parameters()
        }

    def clone(self) -> Type["SklearnEdgeLabelPredictionAdapter"]:
        """Return copy of self."""
        return copy.deepcopy(self)

    def _trasform_graph_into_edge_embedding(
        self,
        graph: Graph,
        support: Optional[Graph] = None,
        node_features: Optional[List[np.ndarray]] = None,
        node_type_features: Optional[List[np.ndarray]] = None,
        edge_features: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """Transforms the provided data into an Sklearn-compatible numpy array.

        Parameters
        ------------------
        graph: Graph
            The graph whose edges are to be embedded and predicted.
            It can either be an Graph or a list of lists of edges.
        support: Optional[Graph] = None
            The graph describiding the topological structure that
            includes also the above graph. This parameter
            is mostly useful for topological classifiers
            such as Graph Convolutional Networks.
        node_features: np.ndarray
            The node features to be used in the training of the model.
        node_type_features: np.ndarray
            The node type features to be used.
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
            aligned_mapping=True
        )

        gt.fit(
            node_features,
            node_type_feature=node_type_features
        )

        if self._use_edge_metrics:
            edge_metrics = support.get_all_edge_metrics(
                normalize=True,
                subgraph=graph,
            )
            if edge_features is not None:
                edge_features = np.hstack([
                    *edge_features,
                    edge_metrics
                ])
            else:
                edge_features = edge_metrics

        return gt.transform(
            graph=graph,
            node_types=graph,
            edge_features=edge_features
        )

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
        node_features: np.ndarray
            The node features to be used in the training of the model.
        node_type_features: np.ndarray
            The node type features to be used.
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
            aligned_mapping=True
        )

        lpt.fit(
            node_features,
            node_type_feature=node_type_features
        )

        if self._use_edge_metrics:
            edge_metrics = support.get_all_edge_metrics(
                normalize=True,
                subgraph=graph,
            )
            if edge_features is not None:
                edge_features = np.hstack([
                    *edge_features,
                    edge_metrics
                ])
            else:
                edge_features = edge_metrics

        self._model_instance.fit(*lpt.transform(
            graph=graph,
            edge_features=edge_features,
            behaviour_for_unknown_edge_labels="drop",
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
        node_features: np.ndarray
            The node features to be used.
        node_type_features: np.ndarray
            The node type features to be used.
        edge_features: Optional[np.ndarray] = None
            Optional edge features to be used as input concatenated
            to the obtained edge embedding. The shape must be equal
            to the number of directed edges in the provided graph.

        Raises
        ------------------
        ValueError
            If the two graphs do not share the same node vocabulary.
        """
        prediction_probabilities = self._model_instance.predict_proba(
            self._trasform_graph_into_edge_embedding(
                graph=graph,
                support=support,
                node_features=node_features,
                node_type_features=node_type_features,
                edge_features=edge_features,
            )
        )
        # In the majority but not totality of sklearn models,
        # the predictions of binary models are returned as
        # a couple of vectors for the positive and negative class. 
        if self.is_binary_prediction_task() and prediction_probabilities.shape[1] == 2:
            prediction_probabilities = prediction_probabilities[:, 1]
        return prediction_probabilities


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
        node_features: np.ndarray
            The node features to be used in the evaluation of the model.
        node_type_features: np.ndarray
            The node type features to be used.
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
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_features=edge_features,
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
    def can_use_node_types(cls) -> bool:
        """Returns whether the model can optionally use node types."""
        return False