"""Module providing Perceptron for edge prediction."""
from typing import Optional,  Dict, Any, List, Union
from ensmallen import Graph
import numpy as np
from ensmallen import models
import compress_json
import json
from embiggen.embedding_transformers import NodeTransformer
from embiggen.edge_prediction.edge_prediction_model import AbstractEdgePredictionModel


class FFGCN(AbstractEdgePredictionModel):
    """Perceptron model for edge prediction."""

    def __init__(
        self,
        units: List[int] = [10, 10],
        learning_rate: float = 0.001,
        first_order_decay_factor: float = 0.9,
        second_order_decay_factor: float = 0.999,
        number_of_steps_per_layer: int = 1000,
        number_of_edges_per_mini_batch: int = 256,
        number_of_oversampling_neighbourhoods_per_node: int = 10,
        sample_only_edges_with_heterogeneous_node_types: bool = False,
        rate_of_neighbours_to_sample: float = 0.5,
        maximum_number_of_neighbours: int = 1000,
        include_node_type_embedding: bool = False,
        include_node_embedding: bool = False,
        avoid_false_negatives: bool = True,
        pre_train: bool = False,
        threshold: float = 3.0,
        prediction_reduce: str = "Mean",
        random_state: int = 42,
        verbose: bool = True
    ):
        """Create new Perceptron object.

        Parameters
        ------------------------
        units: List[int]
            List of units per layer.
        learning_rate: float = 0.001
            The learning rate to use to train the model. By default `0.001`.
        first_order_decay_factor: float = 0.9
            The first order decay to use to train the model. By default `0.9`.
        second_order_decay_factor: float = 0.999
            The second order decay to use to train the model. By default `0.999`.
        number_of_steps_per_layer: int = 1000
            The number of epochs to train the model for. By default, `1000`.
        number_of_edges_per_mini_batch: int = 256
            The number of samples to include for each mini-batch. By default `256`.
        number_of_oversampling_neighbourhoods_per_node: int = 10
            Number of sampling of neighbourhoods per node. By default `10`.
        sample_only_edges_with_heterogeneous_node_types: bool = False
            Whether to sample negative edges only with source and destination nodes that have different node types. By default false.
        rate_of_neighbours_to_sample: float = 0.5
            The rate of neighbours to consider when sub-sampling node-neighbours. This is an important regularization parameter to avoid overfitting. By default `0.5`.
        maximum_number_of_neighbours: int = 1000
            The maximum number of neighbours to consider when sub-sampling node-neighbours. This is an important regularization parameter and avoids excessively computationally complex nodes. By default `1000`.
        include_node_type_embedding: bool = False
            Whether to include a node type embedding layer as part of the first layer.
        include_node_embedding: bool = False
            Whether to include a node embedding layer as part of the first layer.
        avoid_false_negatives: bool = True
            Whether to allow a (generally small) amount of false negatives but with a faster sampling mechanism.
        pre_train: bool = False
            Whether to pre-train the layer without the convolution step. By default `False`.
        threshold: float = 3.0
            Threshold for goodness of the layer. By default, `3.0`.
        prediction_reduce: str = "Mean"
            The reduce method to merge the predictions of the various layers.
            Possible options are:
            * Max: returns the max of all layer predictions.
            * Mean: returns the mean of all layer predictions.
            * Median: returns the median of all layer predictions.
            * Last: returns the predictions of the last layer
        random_state: int = 42
            The random state to reproduce the model initialization and training. By default, `42`.
        verbose: bool = True
            Whether to show loading bars during predictions.
        """
        super().__init__(random_state=random_state)

        self._model_kwargs = dict(
            units=units,
            learning_rate=learning_rate,
            first_order_decay_factor=first_order_decay_factor,
            second_order_decay_factor=second_order_decay_factor,
            number_of_steps_per_layer=number_of_steps_per_layer,
            number_of_edges_per_mini_batch=number_of_edges_per_mini_batch,
            number_of_oversampling_neighbourhoods_per_node=number_of_oversampling_neighbourhoods_per_node,
            sample_only_edges_with_heterogeneous_node_types=sample_only_edges_with_heterogeneous_node_types,
            rate_of_neighbours_to_sample=rate_of_neighbours_to_sample,
            maximum_number_of_neighbours=maximum_number_of_neighbours,
            include_node_type_embedding=include_node_type_embedding,
            include_node_embedding=include_node_embedding,
            avoid_false_negatives=avoid_false_negatives,
            pre_train=pre_train,
            threshold=threshold
        )

        self._prediction_reduce = prediction_reduce
        self._verbose = verbose
        self._model = models.EdgePredictionFFGCN(
            **self._model_kwargs,
            random_state=random_state
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters used for this model."""
        return dict(
            **super().parameters(),
            **self._model_kwargs,
            prediction_reduce= self._prediction_reduce
        )

    def clone(self) -> "FFGCN":
        return FFGCN(**self.parameters())

    @classmethod
    def smoke_test_parameters(cls) -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            units = [10, 10],
            number_of_steps_per_layer=1,
        )

    def normalize_features(
        self,
        graph: Graph,
        node_features: Optional[List[np.ndarray]] = None,
        node_type_features: Optional[List[np.ndarray]] = None
    ) -> np.ndarray:
        if (
            (node_features is None or len(node_features) == 0) and
            (node_type_features is None or len(node_type_features) == 0)
        ):
            raise ValueError(
                "No features were provided. The node features you provided "
                f"are {node_features} and the node type features you provided are {node_type_features}. "
                "Please, provide some features."
            )

        new_node_features = []
        if node_features is not None:
            for node_feature in node_features:
                if not node_feature.data.c_contiguous:
                    node_feature = np.ascontiguousarray(node_feature)
                new_node_features.append(node_feature)

        if node_type_features is not None:
            for node_type_feature in node_type_features:
                node_transformer = NodeTransformer(
                    aligned_mapping=True
                )
                node_transformer.fit(
                    node_type_feature=node_type_feature
                )
                node_feature = node_transformer.transform(
                    node_types=graph
                )
                if not node_feature.data.c_contiguous:
                    node_feature = np.ascontiguousarray(node_feature)
                new_node_features.append(node_feature)

        if len(new_node_features) > 0:
            new_node_features = np.hstack(new_node_features)
        else:
            new_node_features = new_node_features[0]

        if new_node_features.dtype != np.float32:
            new_node_features = new_node_features.astype(np.float32)

        if not new_node_features.data.c_contiguous:
            new_node_features = np.ascontiguousarray(new_node_features)
        
        return new_node_features

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
        self._model.fit(
            graph=graph,
            node_features=self.normalize_features(
                graph=graph,
                node_features=node_features,
                node_type_features=node_type_features
            ),
            verbose=self._verbose,
            support=support
        )

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
        return self._predict_proba(
            graph=graph,
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_features=edge_features
        ) > 0.5

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
        return self._model.predict_proba(
            graph=graph,
            node_features=self.normalize_features(
                graph=graph,
                node_features=node_features,
                node_type_features=node_type_features
            ),
            support=support,
            reduce=self._prediction_reduce,
            verbose=self._verbose
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
    def can_use_edge_types(cls) -> bool:
        """Returns whether the model can optionally use edge types."""
        return False

    @classmethod
    def model_name(cls) -> str:
        return "FFGCN"

    @classmethod
    def library_name(cls) -> str:
        return "Ensmallen"

    @classmethod
    def load(cls, path: str) -> "Self":
        """Load a saved version of the model from the provided path.

        Parameters
        -------------------
        path: str
            Path from where to load the model.
        """
        data = compress_json.load(path)
        model = FFGCN(**data["parameters"])
        model._model = models.EdgePredictionPerceptron.loads(
            json.dumps(data["inner_model"])
        )
        for key, value in data["metadata"].items():
            model.__setattr__(key, value)
        return model

    def dumps(self) -> Dict[str, Any]:
        """Dumps the current model as dictionary."""
        return dict(
            parameters=self.parameters(),
            inner_model=json.loads(self._model.dumps()),
            metadata=dict(
                _fitting_was_executed=self._fitting_was_executed
            )
        )

    def dump(self, path: str):
        """Dump the current model at the provided path.

        Parameters
        -------------------
        path: str
            Path from where to dump the model.
        """
        compress_json.dump(
            self.dumps(),
            path
        )
