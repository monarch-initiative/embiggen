"""Module providing Perceptron for edge prediction."""
from typing import Optional,  Dict, Any, List
from ensmallen import Graph
import numpy as np
from ensmallen import models
import compress_json
import json
from embiggen.node_label_prediction.node_label_prediction_model import AbstractNodeLabelPredictionModel


class PerceptronNodeLabelPrediction(AbstractNodeLabelPredictionModel):
    """Perceptron model for edge prediction."""

    def __init__(
        self,
        number_of_epochs: int = 100,
        learning_rate: float = 0.01,
        first_order_decay_factor: float = 0.9,
        second_order_decay_factor: float = 0.999,
        random_state: int = 42,
        verbose: bool = True
    ):
        """Create new Perceptron object.

        Parameters
        ------------------------
        number_of_epochs: int = 100
            The number of epochs to train the model for. By default, 100.
        learning_rate: float = 0.01
            Learning rate to use while training the model. By default 0.001.
        first_order_decay_factor: float = 0.9
            First order decay factor for the first order momentum.
            By default 0.9.
        second_order_decay_factor: float = 0.999
            Second order decay factor for the second order momentum.
            By default 0.999.
        random_state: int = 42
            The random state to reproduce the model initialization and training. By default, 42.
        verbose: bool = True
            Whether to show epochs loading bar.
        """
        super().__init__(random_state=random_state)

        self._model_kwargs = dict(
            number_of_epochs=number_of_epochs,
            learning_rate=learning_rate,
            first_order_decay_factor=first_order_decay_factor,
            second_order_decay_factor=second_order_decay_factor,
        )
        self._verbose = verbose
        self._model = models.NodeLabelPredictionPerceptron(
            **self._model_kwargs,
            random_state=random_state
        )

    def parameters(self) -> Dict[str, Any]:
        """Returns parameters used for this model."""
        return dict(
            **super().parameters(),
            **self._model_kwargs
        )

    def clone(self) -> "PerceptronNodeLabelPrediction":
        return PerceptronNodeLabelPrediction(**self.parameters())

    @classmethod
    def smoke_test_parameters(cls) -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            number_of_epochs=1,
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
        new_node_features = []
        if node_features is not None:
            for node_feature in node_features:
                if not node_feature.data.c_contiguous:
                    node_feature = np.ascontiguousarray(node_feature)
                new_node_features.append(node_feature)

        self._model.fit(
            graph=graph,
            node_features=new_node_features,
            verbose=self._verbose,
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
        ).argmax(axis=1)

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
        new_node_features = []
        if node_features is not None:
            for node_feature in node_features:
                if not node_feature.data.c_contiguous:
                    node_feature = np.ascontiguousarray(node_feature)
                new_node_features.append(node_feature)

        return self._model.predict(
            graph=graph,
            node_features=new_node_features,
        )

    @classmethod
    def can_use_edge_weights(cls) -> bool:
        """Returns whether the model can optionally use edge weights."""
        return False

    @classmethod
    def can_use_edge_types(cls) -> bool:
        """Returns whether the model can optionally use edge types."""
        return False

    @classmethod
    def model_name(cls) -> str:
        return "Perceptron"

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
        model = PerceptronNodeLabelPrediction(**data["parameters"])
        model._model = models.NodeLabelPredictionPerceptron.loads(
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
