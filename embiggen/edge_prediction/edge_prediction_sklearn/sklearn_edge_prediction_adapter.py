"""Module providing adapter class making edge prediction possible in sklearn models."""
from matplotlib import use
from sklearn.base import ClassifierMixin
from typing import Type, List, Optional, Dict, Any, Union
import numpy as np
import math
import copy
from ensmallen import Graph
from embiggen.sequences.generic_sequences import EdgePredictionSequence
from embiggen.utils.sklearn_utils import must_be_an_sklearn_classifier_model
from embiggen.embedding_transformers import EdgePredictionTransformer, GraphTransformer
from embiggen.edge_prediction.edge_prediction_model import AbstractEdgePredictionModel
from embiggen.utils.abstract_models import abstract_class
from tqdm.auto import tqdm


@abstract_class
class SklearnEdgePredictionAdapter(AbstractEdgePredictionModel):
    """Class wrapping Sklearn models for running ."""

    def __init__(
        self,
        model_instance: Type[ClassifierMixin],
        edge_embedding_method: str = "Concatenate",
        training_unbalance_rate: float = 1.0,
        training_sample_only_edges_with_heterogeneous_node_types: bool = False,
        use_scale_free_distribution: bool = True,
        use_edge_metrics: bool = False,
        prediction_batch_size: int = 2**15,
        random_state: int = 42
    ):
        """Create the adapter for Sklearn object.

        Parameters
        ----------------
        model_instance: Type[ClassifierMixin]
            The class instance to be adapted into edge prediction.
        edge_embedding_method: str = "Concatenate"
            The method to use to compute the edges.
        training_unbalance_rate: float = 1.0
            Unbalance rate for the training non-existing edges.
        training_sample_only_edges_with_heterogeneous_node_types: bool = False
            Whether to sample negative edges exclusively between nodes with different node types
            to generate the negative edges used during the training of the model.
            This can be useful when executing a bipartite edge prediction task.
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
        must_be_an_sklearn_classifier_model(model_instance)
        self._model_instance = model_instance
        self._edge_embedding_method = edge_embedding_method
        self._training_unbalance_rate = training_unbalance_rate
        self._prediction_batch_size = prediction_batch_size
        self._use_edge_metrics = use_edge_metrics
        self._use_scale_free_distribution = use_scale_free_distribution
        self._training_sample_only_edges_with_heterogeneous_node_types = training_sample_only_edges_with_heterogeneous_node_types
        # We want to mask the decorator class name
        self.__class__.__name__ = model_instance.__class__.__name__
        self.__class__.__doc__ = model_instance.__class__.__doc__

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters used for this model."""
        return {
            "training_sample_only_edges_with_heterogeneous_node_types": self._training_sample_only_edges_with_heterogeneous_node_types,
            "edge_embedding_method": self._edge_embedding_method,
            "training_unbalance_rate": self._training_unbalance_rate,
            "prediction_batch_size": self._prediction_batch_size,
            "use_edge_metrics": self._use_edge_metrics,
            "use_scale_free_distribution": self._use_scale_free_distribution,
            **super().parameters()
        }

    def clone(self) -> Type["SklearnEdgePredictionAdapter"]:
        """Return copy of self."""
        return copy.deepcopy(self)

    @classmethod
    def library_name(cls) -> str:
        """Return name of the model."""
        return "scikit-learn"

    def _trasform_graph_into_edge_embedding(
        self,
        graph: Union[Graph, np.ndarray],
        node_features: List[np.ndarray],
        support: Optional[Graph] = None,
        node_types: Optional[Union[Graph, List[Optional[List[str]]], List[Optional[List[int]]]]] = None,
        node_type_features: Optional[List[np.ndarray]] = None,
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
        node_type_features: Optional[List[np.ndarray]] = None,
            The node type features to be used in the training of the model.

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
            aligned_mapping=True
        )

        if self._use_edge_metrics:
            if isinstance(graph, Graph):
                edge_features = support.get_all_edge_metrics(
                    normalize=True,
                    subgraph=graph,
                )
            elif isinstance(graph, tuple):
                edge_features = support.get_all_edge_metrics_from_node_ids(
                    *graph,
                    normalize=True,
                )
            else:
                raise NotImplementedError(
                    f"A graph of type {type(graph)} was provided."
                )
        else:
            edge_features = None

        gt.fit(
            node_features,
            node_type_feature=node_type_features
        )

        return gt.transform(
            graph=graph,
            node_types=node_types,
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
        edge_features: Optional[List[np.ndarray]] = None
            The edge features to use.
        """
        lpt = EdgePredictionTransformer(
            method=self._edge_embedding_method,
            aligned_mapping=True
        )

        lpt.fit(
            node_features,
            node_type_feature=node_type_features
        )

        if support is None:
            support = graph

        negative_graph = graph.sample_negative_graph(
            number_of_negative_samples=int(
                math.ceil(graph.get_number_of_edges() *
                          self._training_unbalance_rate)
            ),
            random_state=self._random_state,
            sample_only_edges_with_heterogeneous_node_types=self._training_sample_only_edges_with_heterogeneous_node_types,
            use_scale_free_distribution=self._use_scale_free_distribution
        )

        if self._use_edge_metrics:
            support = support
            edge_features = np.vstack((
                support.get_all_edge_metrics(
                    normalize=True,
                    subgraph=graph,
                ),
                support.get_all_edge_metrics(
                    normalize=True,
                    subgraph=negative_graph,
                )
            ))

        self._model_instance.fit(*lpt.transform(
            positive_graph=graph,
            negative_graph=negative_graph,
            edge_features=edge_features,
            shuffle=True,
            random_state=self._random_state
        ))

    def _predict(
        self,
        graph: Graph,
        support: Optional[Graph] = None,
        node_features: Optional[List[np.ndarray]] = None,
        node_type_features: Optional[List[np.ndarray]] = None,
        edge_features: Optional[List[np.ndarray]] = None,
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
        edge_features: Optional[List[np.ndarray]] = None
            The edge features to use.
        """
        sequence = EdgePredictionSequence(
            graph=graph,
            graph_used_in_training=graph,
            return_node_types=False,
            return_edge_types=False,
            use_edge_metrics=False,
            batch_size=self._prediction_batch_size
        )
        return np.concatenate([
            self._model_instance.predict(self._trasform_graph_into_edge_embedding(
                graph=edges[0],
                support=support,
                node_features=node_features,
                node_types=graph,
                node_type_features=node_type_features,
            ))
            for edges in tqdm(
                (sequence[i] for i in range(len(sequence))),
                total=len(sequence),
                dynamic_ncols=True,
                desc="Running edge predictions",
                leave=False
            )
        ])

    def _predict_proba(
        self,
        graph: Graph,
        support: Optional[Graph] = None,
        node_features: Optional[List[np.ndarray]] = None,
        node_type_features: Optional[List[np.ndarray]] = None,
        edge_features: Optional[List[np.ndarray]] = None,
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
        edge_features: Optional[List[np.ndarray]] = None
            The edge features to use.
        """
        sequence = EdgePredictionSequence(
            graph=graph,
            graph_used_in_training=graph,
            return_node_types=False,
            return_edge_types=False,
            use_edge_metrics=False,
            batch_size=self._prediction_batch_size
        )
        prediction_probabilities = np.concatenate([
            self._model_instance.predict_proba(self._trasform_graph_into_edge_embedding(
                graph=edges[0],
                support=support,
                node_features=node_features,
                node_types=graph,
                node_type_features=node_type_features,
            ))
            for edges in tqdm(
                (sequence[i] for i in range(len(sequence))),
                total=len(sequence),
                dynamic_ncols=True,
                desc="Running edge predictions",
                leave=False
            )
        ])

        # In the majority but not totality of sklearn models,
        # the predictions of binary models are returned as
        # a couple of vectors for the positive and negative class. 
        if len(prediction_probabilities.shape) > 1 and prediction_probabilities.shape[1] > 1:
            prediction_probabilities = prediction_probabilities[:, 1]

        return prediction_probabilities

    @classmethod
    def can_use_edge_weights(cls) -> bool:
        """Returns whether the model can optionally use edge weights."""
        return False

    @classmethod
    def can_use_node_types(cls) -> bool:
        """Returns whether the model can optionally use node types."""
        return False

    @classmethod
    def can_use_edge_types(cls) -> bool:
        """Returns whether the model can optionally use edge types."""
        return False

