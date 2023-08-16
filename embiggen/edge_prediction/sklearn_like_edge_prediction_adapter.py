"""Module providing adapter class making edge prediction possible in sklearn models."""
from typing import Type, List, Optional, Dict, Any, Union, Tuple
import numpy as np
import math
import compress_pickle
import copy
from ensmallen import Graph
from embiggen.sequences.generic_sequences import EdgePredictionSequence
from embiggen.embedding_transformers import EdgePredictionTransformer, GraphTransformer
from embiggen.edge_prediction.edge_prediction_model import AbstractEdgePredictionModel
from embiggen.utils.abstract_models import abstract_class
from embiggen.utils import AbstractEdgeFeature
from tqdm.auto import tqdm


@abstract_class
class SklearnLikeEdgePredictionAdapter(AbstractEdgePredictionModel):
    """Class wrapping Sklearn models for running ."""

    def __init__(
        self,
        model_instance,
        edge_embedding_methods: Union[List[str], str] = "Concatenate",
        training_unbalance_rate: float = 1.0,
        use_scale_free_distribution: bool = True,
        use_edge_metrics: bool = False,
        prediction_batch_size: int = 2**15,
        random_state: int = 42,
    ):
        """Create the adapter for Sklearn object.

        Parameters
        ----------------
        model_instance: "Model"
            The class instance to be adapted into edge prediction.
        edge_embedding_methods: Union[List[str], str] = "Concatenate",
            The method(s) to use to compute the edges.
            If multiple edge embedding are provided, they
            will be Concatenated and fed to the model.
            The supported edge embedding methods are:
             * Hadamard: element-wise product
             * Sum: element-wise sum
             * Average: element-wise mean
             * L1: element-wise subtraction
             * AbsoluteL1: element-wise subtraction in absolute value
             * SquaredL2: element-wise subtraction in squared value
             * L2: element-wise squared root of squared subtraction
             * Concatenate: Concatenate of source and destination node features
             * Min: element-wise minimum
             * Max: element-wise maximum
             * L2Distance: vector-wise L2 distance - this yields a scalar
             * CosineSimilarity: vector-wise cosine similarity - this yields a scalar
        training_unbalance_rate: float = 1.0
            Unbalance rate for the training non-existing edges.
        use_scale_free_distribution: bool = True
            Whether to sample the negative edges for the TRAINING of the model
            using a zipfian-like distribution that follows the degree distribution
            of the graph. This is generally useful, as these negative edges are less
            trivial to predict then edges sampled uniformely.
            We stringly advise AGAINST using uniform sampling.
        use_edge_metrics: bool = False
            Whether to use the edge metrics from traditional edge prediction.
            These metrics currently include:
            - Adamic Adar
            - Jaccard Coefficient
            - Resource allocation index
            - Preferential attachment
        prediction_batch_size: int = 2**15
            Batch size to use for the predictions.
            Since usually rendering a whole dense graph edge embedding is not
            feaseable in main memory, we chunk it into more digestable smaller
            batches of edges.
        random_state: int
            The random state to use to reproduce the training.

        Raises
        ----------------
        ValueError
            If the provided model_instance is not a subclass of `ClassifierMixin`.
        """
        super().__init__(random_state=random_state)
        self._model_instance = model_instance
        self._edge_embedding_methods = edge_embedding_methods
        self._training_unbalance_rate = training_unbalance_rate
        self._prediction_batch_size = prediction_batch_size
        self._use_edge_metrics = use_edge_metrics
        self._use_scale_free_distribution = use_scale_free_distribution

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters used for this model."""
        return {
            "edge_embedding_methods": self._edge_embedding_methods,
            "training_unbalance_rate": self._training_unbalance_rate,
            "prediction_batch_size": self._prediction_batch_size,
            "use_edge_metrics": self._use_edge_metrics,
            "use_scale_free_distribution": self._use_scale_free_distribution,
            **super().parameters(),
        }

    def clone(self):
        """Return copy of self."""
        return copy.deepcopy(self)

    def _trasform_graph_into_edge_embedding(
        self,
        graph: Union[Graph, Tuple[np.ndarray]],
        node_features: List[np.ndarray],
        support: Graph,
        node_types: Optional[
            Union[Graph, List[Optional[List[str]]], List[Optional[List[int]]]]
        ] = None,
        edge_types: Optional[np.ndarray] = None,
        node_type_features: Optional[List[np.ndarray]] = None,
        edge_type_features: Optional[List[np.ndarray]] = None,
        edge_features: Optional[
            Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]
        ] = None,
    ) -> np.ndarray:
        """Transforms the provided data into an Sklearn-compatible numpy array.

        Parameters
        ------------------
        graph: Graph,
            The graph whose edges are to be embedded and predicted.
            It can either be an Graph or a list of lists of edges.
        node_features: List[np.ndarray]
            The node features to be used in the training of the model.
        support: Optional[Graph] = None
            The graph describiding the topological structure that
            includes also the above graph. This parameter
            is mostly useful for topological classifiers
            such as Graph Convolutional Networks.
        node_types: Optional[Union[Graph, List[Optional[List[str]]], List[Optional[List[int]]]]] = None,
            List of node types whose embedding is to be returned.
            This can be either a list of strings, or a graph, or if the
            aligned_mapping is setted, then this methods also accepts
            a list of ints.
        edge_types: Optional[np.ndarray] = None
            The edge types to use.
        node_type_features: Optional[List[np.ndarray]] = None,
            The node type features to be used in the training of the model.
        edge_type_features: Optional[List[np.ndarray]] = None
            The edge type features to use.
        edge_features: Optional[Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]] = None,
            The edge features to be used in the training of the model.

        Warns
        ------------------
        If the node features are provided as a numpy array it will not be possible
        to check whether the nodes in the graphs are aligned with the features.

        Raises
        ------------------
        ValueError
            If the two graphs do not share the same node vocabulary.
        """
        assert self.is_using_edge_types() == (edge_types is not None)

        graph_transformer = GraphTransformer(
            methods=self._edge_embedding_methods,
            aligned_mapping=True,
            include_both_undirected_edges=False,
        )

        if edge_features is None:
            edge_features = []

        if not isinstance(edge_features, list):
            edge_features = [edge_features]

        for edge_feature in edge_features:
            if not issubclass(type(edge_feature), AbstractEdgeFeature):
                raise NotImplementedError(
                    f"Edge features of type {type(edge_feature)} are not supported."
                    "We currently only support edge features of type AbstractEdgeFeature."
                )

        rasterized_edge_features: List[np.ndarray] = []

        for edge_feature in edge_features:
            if isinstance(graph, Graph):
                for rasterized_edge_feature in edge_feature.get_edge_feature_from_graph(
                    graph=graph,
                    support=support,
                ).values():
                    if not isinstance(rasterized_edge_feature, np.ndarray):
                        raise ValueError(
                            f"The provided edge feature {edge_feature} returned a "
                            f"feature of type {type(rasterized_edge_feature)} "
                            f"when running on a graph of type {type(graph)}."
                            "We currently only support numpy arrays."
                        )
                    rasterized_edge_features.append(rasterized_edge_feature)
            elif isinstance(graph, tuple):
                for (
                    rasterized_edge_feature
                ) in edge_feature.get_edge_feature_from_edge_node_ids(
                    support=support,
                    sources=graph[0],
                    destinations=graph[1],
                ).values():
                    if not isinstance(rasterized_edge_feature, np.ndarray):
                        raise ValueError(
                            f"The provided edge feature {edge_feature} returned a "
                            f"feature of type {type(rasterized_edge_feature)} "
                            f"when running on a graph of type {type(graph)}."
                            "We currently only support numpy arrays."
                        )
                    rasterized_edge_features.append(rasterized_edge_feature)
            else:
                raise NotImplementedError(
                    f"A graph of type {type(graph)} was provided."
                )

        if self._use_edge_metrics:
            if isinstance(graph, Graph):
                edge_metrics = support.get_all_edge_metrics(
                    normalize=True,
                    subgraph=graph,
                )
                if not isinstance(edge_metrics, np.ndarray):
                    raise ValueError(
                        f"The provided graph returned a "
                        f"feature of type {type(edge_metrics)} "
                        f"when running on a graph of type {type(graph)}. "
                        "We currently only support numpy arrays."
                    )
                rasterized_edge_features.append(edge_metrics)
            elif isinstance(graph, tuple):
                edge_metrics = support.get_all_edge_metrics_from_node_ids(
                    source_node_ids=graph[0],
                    destination_node_ids=graph[1],
                    normalize=True,
                )
                if not isinstance(edge_metrics, np.ndarray):
                    raise ValueError(
                        f"The provided graph returned a "
                        f"feature of type {type(edge_metrics)} "
                        f"when running on a graph of type {type(graph)}. "
                        "We currently only support numpy arrays."
                    )
                rasterized_edge_features.append(edge_metrics)
            else:
                raise NotImplementedError(
                    f"A graph of type {type(graph)} was provided."
                )

        graph_transformer.fit(
            node_features,
            node_type_feature=node_type_features,
            edge_type_features=edge_type_features,
        )

        return graph_transformer.transform(
            graph=graph,
            node_types=node_types,
            edge_types=edge_types,
            edge_features=rasterized_edge_features,
        )

    def _fit(
        self,
        graph: Graph,
        support: Optional[Graph] = None,
        node_features: Optional[List[np.ndarray]] = None,
        node_type_features: Optional[List[np.ndarray]] = None,
        edge_type_features: Optional[List[np.ndarray]] = None,
        edge_features: Optional[
            Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]
        ] = None,
    ):
        """Run fitting on the provided graph.

        Parameters
        --------------------
        graph: Graph
            The graph to run predictions on.
        support: Optional[Graph] = None
            The graph describiding the topological structure that
            includes also the above graph. This parameter
            is mostly useful for topological classifiers
            such as Graph Convolutional Networks.
        node_features: Optional[List[np.ndarray]] = None
            The node features to use.
        node_type_features: Optional[List[np.ndarray]] = None
            The node type features to use.
        edge_type_features: Optional[List[np.ndarray]] = None
            The edge type features to use.
        edge_features: Optional[Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]] = None
            The edge features to use.
        """
        lpt = EdgePredictionTransformer(
            methods=self._edge_embedding_methods,
            aligned_mapping=True,
            include_both_undirected_edges=False,
        )

        lpt.fit(
            node_features,
            node_type_feature=node_type_features,
            edge_type_features=edge_type_features,
        )

        if support is None:
            support = graph

        if edge_features is None:
            edge_features: List[Type[AbstractEdgeFeature]] = []

        for edge_feature in edge_features:
            if not issubclass(type(edge_feature), AbstractEdgeFeature):
                raise NotImplementedError(
                    f"Edge features of type {type(edge_feature)} are not supported."
                    "We currently only support edge features of type AbstractEdgeFeature."
                )

        number_of_negative_samples = int(
            math.ceil(
                graph.get_number_of_directed_edges() * self._training_unbalance_rate
            )
        )
        negative_graph = graph.sample_negative_graph(
            number_of_negative_samples=number_of_negative_samples,
            only_from_same_component=True,
            random_state=self._random_state,
            use_scale_free_distribution=self._use_scale_free_distribution,
            sample_edge_types=len(edge_type_features) > 0,
        )

        assert negative_graph.has_edges()

        if negative_graph.has_selfloops():
            assert graph.has_selfloops(), (
                "The negative graph contains self loops, "
                "but the positive graph does not."
            )

        if self._training_unbalance_rate == 1.0:
            number_of_negative_edges = negative_graph.get_number_of_directed_edges()
            number_of_positive_edges = graph.get_number_of_directed_edges()
            self_loop_message = (
                ("The graph contains self loops.")
                if negative_graph.has_selfloops()
                else ("The graph does not contain self loops.")
            )
            assert (
                number_of_negative_edges in (
                    number_of_positive_edges + 1,
                    number_of_positive_edges,
                )
            ), (
                "The negative graph should have the same number of edges as the "
                "positive graph when using a training unbalance rate of 1.0. "
                "We expect the negative graph to have "
                f"{number_of_positive_edges} or {number_of_positive_edges + 1} edges, but found "
                f"{number_of_negative_edges}. {self_loop_message} "
                f"The exact number requested was {number_of_negative_samples}"
            )

        rasterized_edge_features = []

        for edge_feature in edge_features:
            for positive_edge_features, negative_edge_features in zip(
                edge_feature.get_edge_feature_from_graph(
                    graph=graph,
                    support=support,
                ).values(),
                edge_feature.get_edge_feature_from_graph(
                    graph=negative_graph,
                    support=support,
                ).values(),
            ):
                rasterized_edge_features.append(
                    np.vstack((positive_edge_features, negative_edge_features))
                )

        if self._use_edge_metrics:
            rasterized_edge_features.append(
                np.vstack(
                    (
                        support.get_all_edge_metrics(
                            normalize=True,
                            subgraph=graph,
                        ),
                        support.get_all_edge_metrics(
                            normalize=True,
                            subgraph=negative_graph,
                        ),
                    )
                )
            )

        self._model_instance.fit(
            *lpt.transform(
                positive_graph=graph,
                negative_graph=negative_graph,
                edge_features=rasterized_edge_features,
                shuffle=True,
                random_state=self._random_state,
            )
        )

    def _predict(
        self,
        graph: Graph,
        support: Optional[Graph] = None,
        node_features: Optional[List[np.ndarray]] = None,
        node_type_features: Optional[List[np.ndarray]] = None,
        edge_type_features: Optional[List[np.ndarray]] = None,
        edge_features: Optional[
            Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]
        ] = None,
    ) -> np.ndarray:
        """Run prediction on the provided graph.

        Parameters
        --------------------
        graph: Graph
            The graph to run predictions on.
        support: Optional[Graph] = None
            The graph describiding the topological structure that
            includes also the above graph. This parameter
            is mostly useful for topological classifiers
            such as Graph Convolutional Networks.
        node_features: Optional[List[np.ndarray]]
            The node features to use.
        node_type_features: Optional[List[np.ndarray]] = None
            The node type features to use.
        edge_type_features: Optional[List[np.ndarray]] = None
            The edge type features to use.
        edge_features: Optional[Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]] = None
            The edge features to use.
        """
        return (
            self.predict_proba(
                graph=graph,
                support=support,
                node_features=node_features,
                node_type_features=node_type_features,
                edge_type_features=edge_type_features,
                edge_features=edge_features,
            )
            > 0.5
        )

    def _predict_proba(
        self,
        graph: Graph,
        support: Optional[Graph] = None,
        node_features: Optional[List[np.ndarray]] = None,
        node_type_features: Optional[List[np.ndarray]] = None,
        edge_type_features: Optional[List[np.ndarray]] = None,
        edge_features: Optional[
            Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]
        ] = None,
    ) -> np.ndarray:
        """Run prediction on the provided graph.

        Parameters
        --------------------
        graph: Graph
            The graph to run predictions on.
        support: Optional[Graph] = None
            The graph describiding the topological structure that
            includes also the above graph. This parameter
            is mostly useful for topological classifiers
            such as Graph Convolutional Networks.
        node_features: Optional[List[np.ndarray]] = None
            The node features to use.
        node_type_features: Optional[List[np.ndarray]] = None
            The node type features to use.
        edge_type_features: Optional[List[np.ndarray]] = None
            The edge type features to use.
        edge_features: Optional[Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]] = None
            The edge features to use.
        """
        if not graph.has_edges():
            return np.array([])

        sequence = EdgePredictionSequence(
            graph=graph,
            support=support,
            # The node types are handled within the context of the graph transformer
            # and guarantee to be aligned with the node type features and the sampled
            # source and destination nodes.
            return_node_types=False,
            # The same is not possible for the edge types, and therefore we need to
            # handle them separately within the edge prediction sequence.
            return_edge_types=self.is_using_edge_types(),
            use_edge_metrics=False,
            batch_size=self._prediction_batch_size,
        )

        def predict(features):
            if hasattr(self._model_instance, "predict_proba"):
                prediction_probabilities = self._model_instance.predict_proba(features)
            else:
                predictions = self._model_instance.predict(features).astype(np.int32)
                prediction_probabilities = np.zeros(
                    (predictions.shape[0], len(self._model_instance.classes_)),
                    dtype=np.float32,
                )
                prediction_probabilities[np.arange(predictions.size), predictions] = 1

            assert len(prediction_probabilities.shape) in (1, 2)

            # In the majority but not totality of sklearn models,
            # the predictions of binary models are returned as
            # a couple of vectors for the positive and negative class.
            if (
                len(prediction_probabilities.shape) > 1
                and prediction_probabilities.shape[1] > 1
            ):
                prediction_probabilities = prediction_probabilities[:, 1]

            return prediction_probabilities

        return (
            predict(
                self._trasform_graph_into_edge_embedding(
                    graph=(edges[0][0], edges[0][1]),
                    support=support,
                    node_features=node_features,
                    node_types=graph,
                    edge_types=edges[0][2] if self.is_using_edge_types() else None,
                    node_type_features=node_type_features,
                    edge_type_features=edge_type_features,
                    edge_features=edge_features,
                )
            )
            for edges in tqdm(
                (sequence[i] for i in range(len(sequence))),
                total=len(sequence),
                dynamic_ncols=True,
                desc="Running edge predictions",
                leave=False,
            )
        )

    @classmethod
    def can_use_edge_weights(cls) -> bool:
        """Returns whether the model can optionally use edge weights."""
        return False

    @classmethod
    def can_use_node_types(cls) -> bool:
        """Returns whether the model can optionally use node types."""
        return True

    @classmethod
    def requires_node_types(cls) -> bool:
        """Returns whether the model requires node types."""
        return False

    @classmethod
    def can_use_edge_types(cls) -> bool:
        """Returns whether the model can optionally use edge types."""
        return True

    @classmethod
    def requires_edge_types(cls) -> bool:
        """Returns whether the model requires edge types."""
        return False

    @classmethod
    def load(cls, path: str):
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

    @classmethod
    def requires_edge_type_features(cls) -> bool:
        return False

    @classmethod
    def requires_edge_features(cls) -> bool:
        return False

    @classmethod
    def can_use_edge_type_features(cls) -> bool:
        return True

    @classmethod
    def can_use_edge_features(cls) -> bool:
        return True
