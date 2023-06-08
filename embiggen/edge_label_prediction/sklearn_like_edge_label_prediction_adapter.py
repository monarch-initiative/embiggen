"""Module providing adapter class making edge-label prediction possible in sklearn models."""
from typing import Type, List, Dict, Optional, Any
import numpy as np
import copy
import compress_pickle
from ensmallen import Graph
from embiggen.utils.abstract_edge_feature import AbstractEdgeFeature
from embiggen.embedding_transformers import (
    EdgeLabelPredictionTransformer,
    GraphTransformer,
)
from embiggen.edge_label_prediction.edge_label_prediction_model import (
    AbstractEdgeLabelPredictionModel,
)


class SklearnLikeEdgeLabelPredictionAdapter(AbstractEdgeLabelPredictionModel):
    """Class wrapping Sklearn models for running edge-label prediction."""

    def __init__(
        self,
        model_instance: "Model",
        edge_embedding_method: str = "Concatenate",
        use_edge_metrics: bool = False,
        random_state: int = 42,
    ):
        """Create the adapter for Sklearn object.

        Parameters
        ----------------
        model_instance: Model
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
        self._model_instance = model_instance
        self._edge_embedding_method = edge_embedding_method
        self._use_edge_metrics = use_edge_metrics

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters used for this model."""
        return {
            "edge_embedding_method": self._edge_embedding_method,
            "use_edge_metrics": self._use_edge_metrics,
            **super().parameters(),
        }

    def clone(self) -> Type["Self"]:
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
            aligned_mapping=True,
            include_both_undirected_edges=False,
        )

        gt.fit(node_features, node_type_feature=node_type_features)

        if edge_features is None:
            edge_features = []

        if not isinstance(edge_features, list):
            edge_features = [edge_features]

        rasterized_edge_features = []

        for edge_feature in edge_features:
            if issubclass(type(edge_feature), AbstractEdgeFeature):
                for feature in edge_feature.get_edge_feature_from_graph(
                    graph=graph,
                    support=support,
                ).values():
                    rasterized_edge_features.append(feature)
            elif isinstance(edge_feature, np.ndarray):
                rasterized_edge_features.append(edge_feature)

        if self._use_edge_metrics:
            rasterized_edge_features.append(
                support.get_all_edge_metrics(
                    normalize=True,
                    subgraph=graph,
                )
            )

        return gt.transform(
            graph=graph, node_types=graph, edge_features=rasterized_edge_features
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
            aligned_mapping=True,
            include_both_undirected_edges=False,
        )

        lpt.fit(
            node_features,
            node_type_feature=node_type_features,
        )

        if edge_features is None:
            edge_features = []

        if not isinstance(edge_features, list):
            edge_features = [edge_features]

        rasterized_edge_features = []

        for edge_feature in edge_features:
            if issubclass(type(edge_feature), AbstractEdgeFeature):
                for feature in edge_feature.get_edge_feature_from_graph(
                    graph=graph,
                    support=support,
                ).values():
                    rasterized_edge_features.append(feature)
            elif isinstance(edge_feature, np.ndarray):
                rasterized_edge_features.append(edge_feature)

        if self._use_edge_metrics:
            rasterized_edge_features.append(
                support.get_all_edge_metrics(
                    normalize=True,
                    subgraph=graph,
                )
            )

        x, y = lpt.transform(
            graph=graph,
            edge_features=rasterized_edge_features,
            behaviour_for_unknown_edge_labels="drop",
        )

        edge_type_counts = graph.get_edge_type_names_counts_hashmap()

        # If the graph is undirected, then there cannot be edge types
        # with non-zero count equal to one. The very least, should be two.
        if not graph.is_directed():
            for count in edge_type_counts.values():
                if count == 1:
                    raise ValueError(
                        "The provided graph is undirected, but there exists an edge type with only one directed edge."
                    )

        number_of_non_zero_edge_types = sum(
            [1 for count in edge_type_counts.values() if count > 0]
        )

        if self.is_binary_prediction_task():
            assert number_of_non_zero_edge_types == 2
        if not self.is_binary_prediction_task():
            assert number_of_non_zero_edge_types > 2
        assert (
            isinstance(y[0], (bool, np.bool_)) == self.is_binary_prediction_task()
        ), f"Thi task boolean status is {self.is_binary_prediction_task()}, but the provided labels are of type {type(y[0])}."
        if not self.is_binary_prediction_task():
            assert y.max() > 1, (
                "Since the current task does not seem to be a binary classification task, "
                "and the edge type counts from the graph is {}, we expected for the maximal "
                "label to be greater than 1, but it is {}. The graph is {}. The graph name is {}. "
                "{}"
            ).format(
                edge_type_counts,
                y.max(),
                "directed" if graph.is_directed() else "undirected",
                graph.get_name(),
                [
                    (
                        graph.get_node_ids_from_edge_id(edge_id),
                        graph.get_edge_type_name_from_edge_id(edge_id),
                    )
                    for edge_id in range(graph.get_number_of_directed_edges())
                ],
            )

        self._model_instance.fit(x, y)

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
        features = self._trasform_graph_into_edge_embedding(
            graph=graph,
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_features=edge_features,
        )

        if hasattr(self._model_instance, "predict_proba"):
            prediction_probabilities = self._model_instance.predict_proba(features)
        else:
            predictions = self._model_instance.predict(features).astype(np.int32)
            prediction_probabilities = np.zeros(
                (predictions.shape[0], len(self._model_instance.classes_)),
                dtype=np.float32,
            )
            prediction_probabilities[np.arange(predictions.size), predictions] = 1

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
        return self._model_instance.predict(
            self._trasform_graph_into_edge_embedding(
                graph=graph,
                support=support,
                node_features=node_features,
                node_type_features=node_type_features,
                edge_features=edge_features,
            )
        )

    @classmethod
    def can_use_edge_weights(cls) -> bool:
        """Returns whether the model can optionally use edge weights."""
        return False

    @classmethod
    def can_use_node_types(cls) -> bool:
        """Returns whether the model can optionally use node types."""
        return False

    @classmethod
    def load(cls, path: str) -> "Self":
        """Load a saved version of the model from the provided path.

        Parameters
        -------------------
        path: str
            Path from where to load the model.
        """
        return compress_pickle.load(path)

    def dump(self, path: str):
        """Dump the current model at the provided path.

        Parameters
        -------------------
        path: str
            Path from where to dump the model.
        """
        compress_pickle.dump(self, path)
