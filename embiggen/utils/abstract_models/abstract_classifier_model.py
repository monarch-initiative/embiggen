"""Module providing abstract classes for classification models."""
import warnings
from typing import Union, Optional, List, Dict, Any, Tuple, Type
from ensmallen import Graph
import numpy as np
import pandas as pd

from embiggen.utils.list_formatting import format_list
from .abstract_model import AbstractModel, abstract_class
import time
from .abstract_embedding_model import AbstractEmbeddingModel, EmbeddingResult
from tqdm.auto import trange
import functools
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score
)


@abstract_class
class AbstractClassifierModel(AbstractModel):
    """Class defining properties of an abstract classifier model."""

    def __init__(self):
        """Create new instance of model."""
        super().__init__()
        self._fitting_was_executed = False

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
        raise NotImplementedError((
            "The `_fit` method must be implemented "
            "in the child classes of abstract model."
        ))

    def _predict(
        self,
        graph: Graph,
        node_features: Optional[List[np.ndarray]] = None,
        node_type_features: Optional[List[np.ndarray]] = None,
        edge_features: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """Run prediction on the provided graph.

        Parameters
        --------------------
        graph: Graph
            The graph to run predictions on.
        node_features: Optional[List[np.ndarray]] = None
            The node features to use.
        node_type_features: Optional[List[np.ndarray]] = None
            The node type features to use.
        edge_features: Optional[List[np.ndarray]] = None
            The edge features to use.
        """
        raise NotImplementedError((
            "The `_predict` method must be implemented "
            "in the child classes of abstract model."
        ))

    def _predict_proba(
        self,
        graph: Graph,
        node_features: Optional[List[np.ndarray]] = None,
        node_type_features: Optional[List[np.ndarray]] = None,
        edge_features: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """Run prediction on the provided graph.

        Parameters
        --------------------
        graph: Graph
            The graph to run predictions on.
        node_features: Optional[List[np.ndarray]] = None
            The node features to use.
        node_type_features: Optional[List[np.ndarray]] = None
            The node type features to use.
        edge_features: Optional[List[np.ndarray]] = None
            The edge features to use.
        """
        raise NotImplementedError((
            "The `_predict_proba` method must be implemented "
            "in the child classes of abstract model."
        ))

    def is_binary_prediction_task(self) -> bool:
        """Returns whether the model was fit on a binary prediction task."""
        raise NotImplementedError((
            "The `is_binary_prediction_task` method should be implemented "
            "in the child classes of abstract model."
        ))

    def is_multilabel_prediction_task(self) -> bool:
        """Returns whether the model was fit on a multilabel prediction task."""
        raise NotImplementedError((
            "The `is_multilabel_prediction_task` method should be implemented "
            "in the child classes of abstract model."
        ))

    def get_available_evaluation_schemas(self) -> List[str]:
        """Returns available evaluation schemas for this task."""
        raise NotImplementedError((
            "The `get_available_evaluation_schemas` method must be implemented "
            "in the child classes of abstract model."
        ))

    def _get_available_evaluation_schemas_lowercase(self) -> List[str]:
        """Returns lowercase available evaluation schemas for this task."""
        return [
            feature_name.lower()
            for feature_name in self.get_available_evaluation_schemas()
        ]

    def normalize_node_feature(
        self,
        graph: Graph,
        node_feature: Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel]],
        allow_automatic_feature: bool = True,
        skip_evaluation_biased_feature: bool = False,
        smoke_test: bool = False
    ) -> List[np.ndarray]:
        """Normalizes the provided node features and validates them.

        Parameters
        ------------------
        graph: Graph
            The graph to check for.
        node_feature: Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel]],
            The node feature to normalize.
        allow_automatic_feature: bool = True
            Whether to allow feature names creation based on the
            provided feature name, using the default settings,
            or based on a provided abstract embedding model that
            will be called on the provided graph.
        skip_evaluation_biased_feature: bool = False
            Whether to skip feature names that are known to be biased
            when running an holdout. These features should be computed
            exclusively on the training graph and not the entire graph.
        smoke_test: bool = False
            Whether this run should be considered a smoke test
            and therefore use the smoke test configurations for
            the provided model names and feature names.
        """
        if isinstance(node_feature, str):
            if not allow_automatic_feature:
                raise ValueError(
                    (
                        "The node feature `{}` was requested, but "
                        "the `allow_automatic_feature` has been set to False. "
                        "This may be because this is an evaluation execution or "
                        "a prediction, and the string feature should be obtained "
                        "from either the training graph, the test graph or the "
                        "complete graph and it is not clear which one was provided. "
                        "Consider calling the `normalize_node_features` method "
                        "yourselves to completely define your intentions."
                    ).format(node_feature)
                )

            node_feature = AbstractEmbeddingModel.get_model_from_library(
                model_name=node_feature
            )()

        # If this object is an implementation of an abstract
        # embedding model, we compute the embedding.
        if issubclass(node_feature.__class__, AbstractEmbeddingModel):
            if (
                skip_evaluation_biased_feature and
                (
                    self.is_using_edge_types() and node_feature.is_using_edge_types() or
                    self.is_using_node_types() and node_feature.is_using_node_types() or
                    self.is_using_edge_weights() and node_feature.is_using_edge_weights() or
                    self.is_topological() and node_feature.is_topological()
                )
            ):
                yield node_feature
                return None

            if smoke_test:
                node_feature = node_feature.__class__(
                    **node_feature.smoke_test_parameters()
                )

            node_feature = node_feature.fit_transform(
                graph=graph,
                return_dataframe=False,
                verbose=False
            )

        if isinstance(node_feature, EmbeddingResult):
            node_feature = node_feature.get_all_node_embedding()

        if not isinstance(node_feature, list):
            node_feature = [node_feature]

        for nf in node_feature:
            if not isinstance(nf, (np.ndarray, pd.DataFrame)):
                raise ValueError(
                    (
                        "The provided node features are of type `{node_features_type}`, "
                        "while we only currently support numpy arrays and pandas DataFrames. "
                        "What behaviour were you expecting with this feature? "
                        "Please do open an issue on Embiggen and let us know!"
                    ).format(
                        node_features_type=type(nf)
                    )
                )

            if graph.get_nodes_number() != nf.shape[0]:
                raise ValueError(
                    (
                        "The provided node features have {rows_number} rows "
                        "but the provided graph{graph_name} has {nodes_number} nodes. "
                        "Maybe these features refer to another "
                        "version of the graph or another graph "
                        "entirely?"
                    ).format(
                        rows_number=nf.shape[0],
                        graph_name="" if graph.get_name().lower(
                        ) == "graph" else " {}".format(graph.get_name()),
                        nodes_number=graph.get_nodes_number()
                    )
                )

            # If it is a dataframe we align it
            if isinstance(nf, pd.DataFrame):
                yield nf.loc[graph.get_node_names()].to_numpy()
            else:
                # And if it is a numpy array we must believe that the user knows what
                # they are doing, as we cannot ensure alignment.
                yield nf

    def normalize_node_features(
        self,
        graph: Graph,
        node_features: Optional[Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel], List[Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel]]]]] = None,
        allow_automatic_feature: bool = True,
        skip_evaluation_biased_feature: bool = False,
        smoke_test: bool = False
    ) -> List[np.ndarray]:
        """Normalizes the provided node features and validates them.

        Parameters
        ------------------
        graph: Graph
            The graph to check for.
        node_features: Optional[Union[str, pd.DataFrame, np.ndarray, List[Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel]]]]] = None
            The node features to normalize.
        allow_automatic_feature: bool = True
            Whether to allow feature names creation based on the
            provided feature name, using the default settings,
            or based on a provided abstract embedding model that
            will be called on the provided graph.
        skip_evaluation_biased_feature: bool = False
            Whether to skip feature names that are known to be biased
            when running an holdout. These features should be computed
            exclusively on the training graph and not the entire graph.
        smoke_test: bool = False
            Whether this run should be considered a smoke test
            and therefore use the smoke test configurations for
            the provided model names and feature names.
        """
        if node_features is None:
            return None

        if not isinstance(node_features, (list, tuple)):
            node_features = [node_features]

        return [
            normalized_node_feature
            for node_feature in node_features
            for normalized_node_feature in self.normalize_node_feature(
                graph=graph,
                node_feature=node_feature,
                allow_automatic_feature=allow_automatic_feature,
                skip_evaluation_biased_feature=skip_evaluation_biased_feature,
                smoke_test=smoke_test
            )
        ]

    def normalize_node_type_feature(
        self,
        graph: Graph,
        node_type_feature: Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel]],
        allow_automatic_feature: bool = True,
        skip_evaluation_biased_feature: bool = False,
        smoke_test: bool = False
    ) -> List[np.ndarray]:
        """Normalizes the provided node type features and validates them.

        Parameters
        ------------------
        graph: Graph
            The graph to check for.
        node_type_feature: Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel]],
            The node type feature to normalize.
        allow_automatic_feature: bool = True
            Whether to allow feature names creation based on the
            provided feature name, using the default settings,
            or based on a provided abstract embedding model that
            will be called on the provided graph.
        skip_evaluation_biased_feature: bool = False
            Whether to skip feature names that are known to be biased
            when running an holdout. These features should be computed
            exclusively on the training graph and not the entire graph.
        smoke_test: bool = False
            Whether this run should be considered a smoke test
            and therefore use the smoke test configurations for
            the provided model names and feature names.
        """
        if isinstance(node_type_feature, str):
            if not allow_automatic_feature:
                raise ValueError(
                    (
                        "The node type feature `{}` was requested, but "
                        "the `allow_automatic_feature` has been set to False. "
                        "This may be because this is an evaluation execution or "
                        "a prediction, and the string feature should be obtained "
                        "from either the training graph, the test graph or the "
                        "complete graph and it is not clear which one was provided. "
                        "Consider calling the `normalize_node_type_features` method "
                        "yourselves to completely define your intentions."
                    ).format(node_type_feature)
                )

            node_type_feature = AbstractEmbeddingModel.get_model_from_library(
                model_name=node_type_feature
            )()

        # If this object is an implementation of an abstract
        # embedding model, we compute the embedding.
        if issubclass(node_type_feature.__class__, AbstractEmbeddingModel):
            if (
                skip_evaluation_biased_feature and
                (
                    self.is_using_edge_types() and node_type_feature.is_using_edge_types() or
                    self.is_using_node_types() and node_type_feature.is_using_node_types() or
                    self.is_using_edge_weights() and node_type_feature.is_using_edge_weights() or
                    self.is_topological() and node_type_feature.is_topological()
                )
            ):
                yield node_type_feature
                return None

            if smoke_test:
                node_type_feature = node_type_feature.__class__(
                    **node_type_feature.smoke_test_parameters()
                )

            node_type_feature = node_type_feature.fit_transform(
                graph=graph,
                return_dataframe=False,
                verbose=False
            )

        if isinstance(node_type_feature, EmbeddingResult):
            node_type_feature = node_type_feature.get_all_node_type_embeddings()

        if not isinstance(node_type_feature, list):
            node_type_feature = [node_type_feature]

        for nf in node_type_feature:
            if not isinstance(nf, (np.ndarray, pd.DataFrame)):
                raise ValueError(
                    (
                        "The provided node type features are of type `{node_type_features_type}`, "
                        "while we only currently support numpy arrays and pandas DataFrames. "
                        "What behaviour were you expecting with this feature? "
                        "Please do open an issue on Embiggen and let us know!"
                    ).format(
                        node_type_features_type=type(nf)
                    )
                )

            if graph.get_node_types_number() != nf.shape[0]:
                raise ValueError(
                    (
                        "The provided node type features have {rows_number} rows "
                        "but the provided graph{graph_name} has {nodes_number} nodes. "
                        "Maybe these features refer to another "
                        "version of the graph or another graph "
                        "entirely?"
                    ).format(
                        rows_number=nf.shape[0],
                        graph_name="" if graph.get_name().lower(
                        ) == "graph" else " {}".format(graph.get_name()),
                        nodes_number=graph.get_node_types_number()
                    )
                )

            # If it is a dataframe we align it
            if isinstance(nf, pd.DataFrame):
                yield nf.loc[graph.get_unique_node_type_names()].to_numpy()
            else:
                # And if it is a numpy array we must believe that the user knows what
                # they are doing, as we cannot ensure alignment.
                yield nf

    def normalize_node_type_features(
        self,
        graph: Graph,
        node_type_features: Optional[Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel], List[Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel]]]]] = None,
        allow_automatic_feature: bool = True,
        skip_evaluation_biased_feature: bool = False,
        smoke_test: bool = False
    ) -> List[np.ndarray]:
        """Normalizes the provided node type features and validates them.

        Parameters
        ------------------
        graph: Graph
            The graph to check for.
        node_type_features: Optional[Union[str, pd.DataFrame, np.ndarray, List[Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel]]]]] = None
            The node features to normalize.
        allow_automatic_feature: bool = True
            Whether to allow feature names creation based on the
            provided feature name, using the default settings,
            or based on a provided abstract embedding model that
            will be called on the provided graph.
        skip_evaluation_biased_feature: bool = False
            Whether to skip feature names that are known to be biased
            when running an holdout. These features should be computed
            exclusively on the training graph and not the entire graph.
        smoke_test: bool = False
            Whether this run should be considered a smoke test
            and therefore use the smoke test configurations for
            the provided model names and feature names.
        """
        if node_type_features is None:
            return None

        if not isinstance(node_type_features, (list, tuple)):
            node_type_features = [node_type_features]

        return [
            normalized_node_type_feature
            for node_type_feature in node_type_features
            for normalized_node_type_feature in self.normalize_node_type_feature(
                graph=graph,
                node_type_feature=node_type_feature,
                allow_automatic_feature=allow_automatic_feature,
                skip_evaluation_biased_feature=skip_evaluation_biased_feature,
                smoke_test=smoke_test
            )
        ]

    def normalize_edge_feature(
        self,
        graph: Graph,
        edge_feature: Optional[Union[str, pd.DataFrame, np.ndarray, EmbeddingResult, Type[AbstractEmbeddingModel]]] = None,
        allow_automatic_feature: bool = True,
        skip_evaluation_biased_feature: bool = False,
        smoke_test: bool = False
    ) -> List[np.ndarray]:
        """Normalizes the provided edge features and validates them.

        Parameters
        ------------------
        graph: Graph
            The graph to check for.
        edge_feature: Optional[Union[str, pd.DataFrame, np.ndarray]] = None
            The edge feature to normalize.
        allow_automatic_feature: bool = True
            Whether to allow feature names creation based on the
            provided feature name, using the default settings,
            or based on a provided abstract embedding model that
            will be called on the provided graph.
        skip_evaluation_biased_feature: bool = False
            Whether to skip feature names that are known to be biased
            when running an holdout. These features should be computed
            exclusively on the training graph and not the entire graph.
        smoke_test: bool = False
            Whether this run should be considered a smoke test
            and therefore use the smoke test configurations for
            the provided model names and feature names.
        """
        if isinstance(edge_feature, str):
            if not allow_automatic_feature:
                raise ValueError(
                    (
                        "The edge feature `{}` was requested, but "
                        "the `allow_automatic_feature` has been set to False. "
                        "This may be because this is an evaluation execution or "
                        "a prediction, and the string feature should be obtained "
                        "from either the training graph, the test graph or the "
                        "complete graph and it is not clear which one was provided. "
                        "Consider calling the `normalize_edge_features` method "
                        "yourselves to completely define your intentions."
                    ).format(edge_feature)
                )

            edge_feature = AbstractEmbeddingModel.get_model_from_library(
                model_name=edge_feature
            )()

        # If this object is an implementation of an abstract
        # embedding model, we compute the embedding.
        if issubclass(edge_feature.__class__, AbstractEmbeddingModel):
            if (
                skip_evaluation_biased_feature and
                (
                    self.is_using_edge_types() and edge_feature.is_using_edge_types() or
                    self.is_using_node_types() and edge_feature.is_using_node_types() or
                    self.is_using_edge_weights() and edge_feature.is_using_edge_weights() or
                    self.is_topological() and edge_feature.is_topological()
                )
            ):
                yield edge_feature
                return None

            if smoke_test:
                edge_feature = edge_feature.__class__(
                    **edge_feature.smoke_test_parameters()
                )

            edge_feature = edge_feature.fit_transform(
                graph=graph,
                return_dataframe=False,
                verbose=False
            )

        if isinstance(edge_feature, EmbeddingResult):
            edge_feature = edge_feature.get_all_edge_embedding()

        if not isinstance(edge_feature, list):
            edge_feature = [edge_feature]

        for ef in edge_feature:
            if not isinstance(ef, (np.ndarray, pd.DataFrame)):
                raise ValueError(
                    (
                        "The provided edge features are of type `{edge_features_type}`, "
                        "while we only currently support numpy arrays and pandas DataFrames. "
                        "What behaviour were you expecting with this feature? "
                        "Please do open an issue on Embiggen and let us know!"
                    ).format(
                        edge_features_type=type(ef)
                    )
                )

            if graph.get_directed_edges_number() != ef.shape[0]:
                raise ValueError(
                    (
                        "The provided edge features have {rows_number} rows "
                        "but the provided graph{graph_name} has {edges_number} edges. "
                        "Maybe these features refer to another "
                        "version of the graph or another graph "
                        "entirely?"
                    ).format(
                        rows_number=ef.shape[0],
                        graph_name="" if graph.get_name().lower(
                        ) == "graph" else " {}".format(graph.get_name()),
                        edges_number=graph.get_edges_number()
                    )
                )

            # If it is a dataframe we align it
            if isinstance(ef, pd.DataFrame):
                yield ef.loc[graph.get_edge_names()].to_numpy()
            else:
                # And if it is a numpy array we must believe that the user knows what
                # they are doing, as we cannot ensure alignment.
                yield ef

    def normalize_edge_features(
        self,
        graph: Graph,
        edge_features: Optional[Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel], List[Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel]]]]] = None,
        allow_automatic_feature: bool = True,
        skip_evaluation_biased_feature: bool = False,
        smoke_test: bool = False
    ) -> List[np.ndarray]:
        """Normalizes the provided edge features and validates them.

        Parameters
        ------------------
        graph: Graph
            The graph to check for.
        edge_features: Optional[Union[str, pd.DataFrame, np.ndarray, List[Union[str, pd.DataFrame, np.ndarray]]]] = None
            The edge features to normalize.
        allow_automatic_feature: bool = True
            Whether to allow feature names creation based on the
            provided feature name, using the default settings,
            or based on a provided abstract embedding model that
            will be called on the provided graph.
        skip_evaluation_biased_feature: bool = False
            Whether to skip feature names that are known to be biased
            when running an holdout. These features should be computed
            exclusively on the training graph and not the entire graph.
        smoke_test: bool = False
            Whether this run should be considered a smoke test
            and therefore use the smoke test configurations for
            the provided model names and feature names.
        """
        if edge_features is None:
            return None

        if not isinstance(edge_features, list):
            edge_features = [edge_features]

        return [
            normalized_edge_feature
            for edge_feature in edge_features
            for normalized_edge_feature in self.normalize_edge_feature(
                graph=graph,
                edge_feature=edge_feature,
                allow_automatic_feature=allow_automatic_feature,
                skip_evaluation_biased_feature=skip_evaluation_biased_feature,
                smoke_test=smoke_test
            )
        ]

    def fit(
        self,
        graph: Graph,
        support: Optional[Graph] = None,
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
    ):
        """Execute predictions on the provided graph.

        Parameters
        --------------------
        graph: Graph
            The graph to run predictions on.
        support: Optional[Graph] = None
            The graph describiding the topological structure that
            includes also the above graph. This parameter
            is mostly useful for topological classifiers
            such as Graph Convolutional Networks.
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node features to use.
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The edge features to use.
        """
        if not graph.has_nodes():
            raise ValueError("The provided graph is empty.")

        if node_type_features is not None and not graph.has_node_types():
            raise ValueError(
                "The node type features have been provided but the current "
                f"instance of graph {graph.get_name()} does not have node types. "
                "It is unclear how to proceed with this data."
            )
        try:
            self._fit(
                graph=graph,
                support=support,
                node_features=self.normalize_node_features(
                    graph=graph,
                    node_features=node_features,
                    allow_automatic_feature=True,
                ),
                node_type_features=self.normalize_node_type_features(
                    graph=graph,
                    node_type_features=node_type_features,
                    allow_automatic_feature=True,
                ),
                edge_features=self.normalize_edge_features(
                    graph=graph,
                    edge_features=edge_features,
                    allow_automatic_feature=True,
                ),
            )
        except Exception as e:
            raise RuntimeError(
                f"An exception was raised while calling the `_fit` method of {self.model_name()} "
                f"implemented using the {self.library_name()} for the {self.task_name()} task. "
                f"Specifically, the class of the model is {self.__class__.__name__}."
            ) from e

        self._fitting_was_executed = True

    def predict(
        self,
        graph: Graph,
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
    ) -> np.ndarray:
        """Execute predictions on the provided graph.

        Parameters
        --------------------
        graph: Graph
            The graph to run predictions on.
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node features to use.
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The edge features to use.
        """
        if not graph.has_nodes():
            raise ValueError("The provided graph is empty.")

        if not self._fitting_was_executed:
            raise ValueError((
                "The prediction cannot be executed without "
                "having first executed the fitting of the model. "
                "Do call the `.fit` method before the `.predict` "
                "method."
            ))
        try:
            return self._predict(
                graph=graph,
                node_features=self.normalize_node_features(
                    graph=graph,
                    node_features=node_features,
                    allow_automatic_feature=False,
                ),
                node_type_features=self.normalize_node_type_features(
                    graph=graph,
                    node_type_features=node_type_features,
                    allow_automatic_feature=True,
                ),
                edge_features=self.normalize_edge_features(
                    graph=graph,
                    edge_features=edge_features,
                    allow_automatic_feature=False,
                ),
            )
        except Exception as e:
            raise RuntimeError(
                f"An exception was raised while calling the `._predict` method of {self.model_name()} "
                f"implemented using the {self.library_name()} for the {self.task_name()} task. "
                f"Specifically, the class of the model is {self.__class__.__name__}."
            ) from e

    def predict_proba(
        self,
        graph: Graph,
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
    ) -> np.ndarray:
        """Execute predictions on the provided graph.

        Parameters
        --------------------
        graph: Graph
            The graph to run predictions on.
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node features to use.
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The edge features to use.
        """
        if not graph.has_nodes():
            raise ValueError("The provided graph is empty.")

        if not self._fitting_was_executed:
            raise ValueError((
                "The prediction cannot be executed without "
                "having first executed the fitting of the model. "
                "Do call the `.fit` method before the `.predict_proba` "
                "method."
            ))

        try:
            return self._predict_proba(
                graph=graph,
                node_features=self.normalize_node_features(
                    graph=graph,
                    node_features=node_features,
                    allow_automatic_feature=False,
                ),
                node_type_features=self.normalize_node_type_features(
                    graph=graph,
                    node_type_features=node_type_features,
                    allow_automatic_feature=True,
                ),
                edge_features=self.normalize_edge_features(
                    graph=graph,
                    edge_features=edge_features,
                    allow_automatic_feature=False,
                ),
            )
        except Exception as e:
            raise RuntimeError(
                f"An exception was raised while calling the `._predict_proba` method of {self.model_name()} "
                f"implemented using the {self.library_name()} for the {self.task_name()} task. "
                f"Specifically, the class of the model is {self.__class__.__name__}."
            ) from e

    def evaluate_predictions(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray
    ) -> Dict[str, Any]:
        """Return evaluations for the provided predictions.

        Parameters
        ----------------
        predictions: np.ndarray
            The predictions to be evaluated.
        ground_truth: np.ndarray
            The ground truth to evaluate the predictions against.
        """

        return {
            **{
                metric.__name__: metric(ground_truth, predictions)
                for metric in (
                    accuracy_score,
                    balanced_accuracy_score,
                )
            },
            **{
                metric.__name__: metric(
                    ground_truth,
                    predictions,
                    average="binary" if self.is_binary_prediction_task() else "macro"
                )
                for metric in (
                    f1_score,
                    precision_score,
                    recall_score,
                )
            }
        }

    def evaluate_prediction_probabilities(
        self,
        prediction_probabilities: np.ndarray,
        ground_truth: np.ndarray
    ) -> Dict[str, Any]:
        """Return evaluations for the provided predictions.

        Parameters
        ----------------
        prediction_probabilities: np.ndarray
            The predictions to be evaluated.
        ground_truth: np.ndarray
            The ground truth to evaluate the predictions against.
        """
        metrics = []
        if self.is_binary_prediction_task():
            # AUPRC in sklearn is only supported for binary labels
            metrics.append(average_precision_score)

        @functools.wraps(roc_auc_score)
        def wrapper_roc_auc_score(*args, **kwargs):
            return roc_auc_score(*args, **kwargs, multi_class="ovr")

        metrics.append(wrapper_roc_auc_score)

        return {
            metric.__name__: metric(
                ground_truth,
                prediction_probabilities,
            )
            for metric in metrics
        }

    def split_graph_following_evaluation_schema(
        self,
        graph: Graph,
        evaluation_schema: str,
        number_of_holdouts: int,
        random_state: int,
        holdouts_kwargs: Dict[str, Any],
        holdout_number: int,
    ) -> Tuple[Graph]:
        """Return train and test graphs tuple following the provided evaluation schema.

        Parameters
        ----------------------
        graph: Graph
            The graph to split.
        evaluation_schema: str
            The evaluation schema to follow.
        number_of_holdouts: int
            The number of holdouts that will be generated throught the evaluation.
        random_state: int
            The random state for the evaluation
        holdouts_kwargs: Dict[str, Any]
            The kwargs to be forwarded to the holdout method.
        holdout_number: int
            The current holdout number.
        """
        raise NotImplementedError(
            "The `split_graph_following_evaluation_schema` method should be implemented "
            "in the child classes of abstract classifier model."
        )

    def _evaluate(
        self,
        graph: Graph,
        train: Graph,
        test: Graph,
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[str, pd.DataFrame, np.ndarray]]]] = None,
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[str, pd.DataFrame, np.ndarray]]]] = None,
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[str, pd.DataFrame, np.ndarray]]]] = None,
        subgraph_of_interest: Optional[Graph] = None,
        random_state: int = 42,
        verbose: bool = True,
        **kwargs: Dict
    ) -> List[Dict[str, Any]]:
        """Return model evaluation on the provided graphs."""
        raise NotImplementedError(
            "The _evaluate method should be implemented in the child "
            "classes of abstract classifier model."
        )

    def evaluate(
        self,
        graph: Graph,
        evaluation_schema: str,
        holdouts_kwargs: Dict[str, Any],
        node_features: Optional[Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel], List[Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel]]]]] = None,
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        edge_features: Optional[Union[str, pd.DataFrame, np.ndarray, List[Union[str, pd.DataFrame, np.ndarray]]]] = None,
        subgraph_of_interest: Optional[Graph] = None,
        number_of_holdouts: int = 10,
        random_state: int = 42,
        verbose: bool = True,
        smoke_test: bool = False,
        **validation_kwargs: Dict
    ) -> pd.DataFrame:
        """Execute evaluation on the provided graph.

        Parameters
        --------------------
        graph: Graph
            The graph to run predictions on.
        evaluation_schema: str
            The schema for the evaluation to follow.
        holdouts_kwargs: Dict[str, Any]
            Parameters to forward to the desired evaluation schema.
        node_features: Optional[Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel], List[Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel]]]]] = None
            The node features to use.
        node_type_features: Optional[Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel], List[Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel]]]]] = None
            The node features to use.
        edge_features: Optional[Union[str, pd.DataFrame, np.ndarray, List[Union[str, pd.DataFrame, np.ndarray]]]] = None
            The edge features to use.
        subgraph_of_interest: Optional[Graph] = None
            Optional subgraph where to focus the task.
        skip_evaluation_biased_feature: bool = False
            Whether to skip feature names that are known to be biased
            when running an holdout. These features should be computed
            exclusively on the training graph and not the entire graph.
        number_of_holdouts: int = 10
            The number of holdouts to execute.
        random_state: int = 42
            The random state to use for the holdouts.
        verbose: bool = True
            Whether to show a loading bar while computing holdouts.
        smoke_test: bool = False
            Whether this run should be considered a smoke test
            and therefore use the smoke test configurations for
            the provided model names and feature names.
        **validation_kwargs: Dict
            kwargs to be forwarded to the model `_evaluate` method.
        """
        if self._fitting_was_executed:
            warnings.warn((
                "Do be advised that this model was already fitted, "
                "and you will be therefore be running a classification "
                "evaluation using a warm start."
            ))

        if not isinstance(number_of_holdouts, int) or number_of_holdouts == 0:
            raise ValueError(
                "The number of holdouts must be a strictly positive integer, "
                f"but {number_of_holdouts} was provided."
            )

        if evaluation_schema.lower() not in self._get_available_evaluation_schemas_lowercase():
            raise ValueError(
                (
                    f"The provided evaluation schema `{evaluation_schema}` is not among the supported "
                    "evaluation schemas for this task, which are:\n- {}"
                ).format(
                    "\n- ".join(self.get_available_evaluation_schemas())
                )
            )

        if subgraph_of_interest is not None:
            if self.task_name() not in ("Edge Prediction", "Edge Label Prediction"):
                raise ValueError(
                    "A subgraph of interest was provided, but this parameter "
                    "is only currently supported for Edge Prediction and "
                    f"Edge Label Prediction tasks and not the {self.task_name()} task."
                )
            
            if not graph.contains(subgraph_of_interest):
                raise ValueError(
                    "The provided subgraph of interest is not "
                    f"contained in the provided graph {graph.get_name()}."
                )

            if not subgraph_of_interest.has_edges():
                raise ValueError(
                    "The provided subgraph of interest does not "
                    "have any edges!"
                )
            
            # We check whether the subgraph of interest shares the same vocabulary
            # of the main graph. If this is true, we can skip the filtering step to
            # drop the nodes from the train and test graph.
            subgraph_of_interest_has_compatible_nodes = graph.has_compatible_node_vocabularies(
                subgraph_of_interest
            )

        # Retrieve the set of provided automatic features parameters
        # so we can put them in the report.
        feature_parameters = {
            parameter_name: value
            for features in (
                node_features
                if isinstance(node_features, (list, tuple))
                else (node_features,),
                node_type_features
                if isinstance(node_type_features, (list, tuple))
                else (node_type_features,),
                edge_features
                if isinstance(edge_features, (list, tuple))
                else (edge_features,),
                (self,)
            )
            for feature in features
            if issubclass(feature.__class__, AbstractModel)
            for parameter_name, value in feature.parameters().items()
        }

        # Retrieve the set of provided automatic features names
        # so we can put them in the report.
        automatic_feature_names = {
            feature.model_name()
            for features in (
                node_features
                if isinstance(node_features, (list, tuple))
                else (node_features,),
                node_type_features
                if isinstance(node_type_features, (list, tuple))
                else (node_type_features,),
                edge_features
                if isinstance(edge_features, (list, tuple))
                else (edge_features,),
            )
            for feature in features
            if issubclass(feature.__class__, AbstractEmbeddingModel)
        }

        # We normalize and/or compute the node features, having
        # the care of skipping the features that induce bias when
        # computed on the entire graph.
        # This way we compute only once the features that do not
        # cause biases for this task, while recomputing those
        # that cause biases at each holdout, avoiding said biases.
        node_features = self.normalize_node_features(
            graph,
            node_features=node_features,
            allow_automatic_feature=True,
            skip_evaluation_biased_feature=True,
            smoke_test=smoke_test
        )

        # We execute the same thing as described above,
        # but now for the node type features instead that for
        # the node features.
        node_type_features = self.normalize_node_type_features(
            graph,
            node_type_features=node_type_features,
            allow_automatic_feature=True,
            skip_evaluation_biased_feature=True,
            smoke_test=smoke_test
        )

        # We execute the same thing as described above,
        # but now for the edge features instead that for
        # the node features.
        edge_features = self.normalize_edge_features(
            graph,
            edge_features=edge_features,
            allow_automatic_feature=True,
            skip_evaluation_biased_feature=True,
            smoke_test=smoke_test
        )

        # We create the list where we store the holdouts performance.
        performance = []

        # We start to iterate on the holdouts.
        for holdout_number in trange(
            number_of_holdouts,
            disable=not verbose,
            leave=False,
            dynamic_ncols=True,
            desc=f"Evaluating {self.model_name()} on {graph.get_name()}"
        ):
            # We create a copy of the current classifier.
            classifier = self.clone()

            # We create the graph split using the provided schema.
            train, test = self.split_graph_following_evaluation_schema(
                graph=graph,
                evaluation_schema=evaluation_schema,
                random_state=random_state,
                holdouts_kwargs=holdouts_kwargs,
                holdout_number=holdout_number,
                number_of_holdouts=number_of_holdouts
            )

            # We enable in the train and test graphs the same
            # speedups enabled in the provided graph.
            train.enable(
                vector_sources=graph.has_sources_tradeoff_enabled(),
                vector_destinations=graph.has_destinations_tradeoff_enabled(),
                vector_cumulative_node_degrees=graph.has_cumulative_node_degrees_tradeoff_enabled(),
                vector_reciprocal_sqrt_degrees=graph.has_reciprocal_sqrt_degrees_tradeoff_enabled()
            )
            test.enable(
                vector_sources=graph.has_sources_tradeoff_enabled(),
                vector_destinations=graph.has_destinations_tradeoff_enabled(),
                vector_cumulative_node_degrees=graph.has_cumulative_node_degrees_tradeoff_enabled(),
                vector_reciprocal_sqrt_degrees=graph.has_reciprocal_sqrt_degrees_tradeoff_enabled()
            )

            # We compute the remaining features
            holdout_node_features = self.normalize_node_features(
                train,
                node_features=node_features,
                allow_automatic_feature=True,
                skip_evaluation_biased_feature=False,
                smoke_test=smoke_test
            )

            # We compute the remaining features
            holdout_node_type_features = self.normalize_node_type_features(
                train,
                node_type_features=node_type_features,
                allow_automatic_feature=True,
                skip_evaluation_biased_feature=False,
                smoke_test=smoke_test
            )

            # We execute the same thing as described above,
            # but now for the edge features instead that for
            # the node features.
            holdout_edge_features = self.normalize_edge_features(
                train,
                edge_features=edge_features,
                allow_automatic_feature=True,
                skip_evaluation_biased_feature=False,
                smoke_test=smoke_test
            )

            if subgraph_of_interest is not None:
                # First we align the train and test graph to have
                # the same node dictionary of the subgraph of interest
                # when the subgraph of interest does not have
                # the same node dictionary as the original graph.
                if not subgraph_of_interest_has_compatible_nodes:
                    train = train.filter_from_names(
                        node_names_to_keep_from_graph=subgraph_of_interest
                    )
                    test = test.filter_from_names(
                        node_names_to_keep_from_graph=subgraph_of_interest
                    )

                    # We enable in the train and test graphs of interest the same
                    # speedups enabled in the provided graph.
                    train.enable(
                        vector_sources=graph.has_sources_tradeoff_enabled(),
                        vector_destinations=graph.has_destinations_tradeoff_enabled(),
                        vector_cumulative_node_degrees=graph.has_cumulative_node_degrees_tradeoff_enabled(),
                        vector_reciprocal_sqrt_degrees=graph.has_reciprocal_sqrt_degrees_tradeoff_enabled()
                    )

                    test.enable(
                        vector_sources=graph.has_sources_tradeoff_enabled(),
                        vector_destinations=graph.has_destinations_tradeoff_enabled(),
                        vector_cumulative_node_degrees=graph.has_cumulative_node_degrees_tradeoff_enabled(),
                        vector_reciprocal_sqrt_degrees=graph.has_reciprocal_sqrt_degrees_tradeoff_enabled()
                    )
                    
                    # We adjust the node features to only include the node features
                    # that the subgraph of interest allows us to use.
                    if holdout_node_features is not None:
                        # We obtain the mapping from the old to the new node IDs
                        node_ids_mapping = train.get_node_ids_mapping_from_graph(
                            graph
                        )

                        holdout_node_features = [
                            holdout_node_feature[node_ids_mapping]
                            for holdout_node_feature in holdout_node_features
                        ]

                train_of_interest = train & subgraph_of_interest
                test_of_interest = test & subgraph_of_interest

                # We validate that the two graphs are still
                # valid for this task.
                for graph_partition, graph_partition_name in (
                    (train_of_interest, "train"),
                    (test_of_interest, "test"),
                ):
                    if not graph_partition.has_nodes():
                        raise ValueError(
                            f"The {graph_partition_name} graph {graph_partition.get_name()} obtained from the evaluation "
                            f"schema {evaluation_schema}, once filtered using the provided "
                            "subgraph of interest, does not have any more nodes."
                        )
                    if (
                        self.task_name() in ("Edge Prediction", "Edge Label Prediction") and
                        not graph_partition.has_edges()
                    ):
                        raise ValueError(
                            f"The {graph_partition_name} graph {graph_partition.get_name()} obtained from the evaluation "
                            f"schema {evaluation_schema}, once filtered using the provided "
                            "subgraph of interest, does not have any more edges which are "
                            f"essential when running a {self.task_name()} task."
                        )
                # We enable in the train and test graphs of interest the same
                # speedups enabled in the provided graph.
                train_of_interest.enable(
                    vector_sources=graph.has_sources_tradeoff_enabled(),
                    vector_destinations=graph.has_destinations_tradeoff_enabled(),
                    vector_cumulative_node_degrees=graph.has_cumulative_node_degrees_tradeoff_enabled(),
                    vector_reciprocal_sqrt_degrees=graph.has_reciprocal_sqrt_degrees_tradeoff_enabled()
                )

                test_of_interest.enable(
                    vector_sources=graph.has_sources_tradeoff_enabled(),
                    vector_destinations=graph.has_destinations_tradeoff_enabled(),
                    vector_cumulative_node_degrees=graph.has_cumulative_node_degrees_tradeoff_enabled(),
                    vector_reciprocal_sqrt_degrees=graph.has_reciprocal_sqrt_degrees_tradeoff_enabled()
                )
            else:
                train_of_interest = train
                test_of_interest = test

            # Fit the model using the training graph
            training_start = time.time()
            classifier.fit(
                graph=train_of_interest,
                support=train,
                node_features=holdout_node_features,
                node_type_features=holdout_node_type_features,
                edge_features=holdout_edge_features
            )
            required_training_time = time.time() - training_start

            try:
                # We add the newly computed performance.
                holdout_performance = classifier._evaluate(
                    graph=graph,
                    train=train_of_interest,
                    test=test_of_interest,
                    node_features=holdout_node_features,
                    node_type_features=holdout_node_type_features,
                    edge_features=holdout_edge_features,
                    subgraph_of_interest=subgraph_of_interest,
                    random_state=random_state * holdout_number,
                    verbose=verbose,
                    **validation_kwargs
                )
            except RuntimeError as e:
                raise e
            except Exception as e:
                raise RuntimeError(
                    f"An exception was raised while calling the `._evaluate` method of {self.model_name()} "
                    f"implemented using the {self.library_name()} for the {self.task_name()} task. "
                    f"Specifically, the class of the model is {self.__class__.__name__}. "
                ) from e

            for hp in holdout_performance:
                hp["required_training_time"] = required_training_time
                performance.append(hp)

        # We save the constant values for this model
        # execution.
        performance = pd.DataFrame(performance)
        performance["task_name"] = self.task_name()
        performance["model_name"] = self.model_name()
        performance["library_name"] = self.library_name()
        performance["graph_name"] = graph.get_name()
        performance["nodes_number"] = graph.get_nodes_number()
        performance["edges_number"] = graph.get_number_of_directed_edges()
        performance["number_of_holdouts"] = number_of_holdouts
        performance["evaluation_schema"] = evaluation_schema
        performance["automatic_feature_names"] = format_list(automatic_feature_names)
        for parameter, value in feature_parameters.items():
            if parameter in performance.columns:
                raise ValueError(
                    "There has been a collision between the parameters used in the "
                    f"{self.model_name()} implemented using the {self.library_name()} for the {self.task_name()} task "
                    "and the parameter used for the validation and reporting of the task itself. "
                    f"The parameter that has caused the collision is {parameter}. "
                    "Please do change the name of the parameter in your model."
                )
            performance[parameter] = str(value)

        return performance
