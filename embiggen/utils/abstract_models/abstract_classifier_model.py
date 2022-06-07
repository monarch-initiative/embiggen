"""Module providing abstract classes for classification models."""
from typing import Union, Optional, List, Dict, Any, Tuple, Type, Iterator
from ensmallen import Graph, express_measures
import numpy as np
import pandas as pd
import time
from tqdm.auto import trange, tqdm
from embiggen.utils.abstract_models.list_formatting import format_list
from cache_decorator import Cache

from embiggen.utils.abstract_models.abstract_model import AbstractModel, abstract_class
from embiggen.utils.abstract_models.abstract_embedding_model import AbstractEmbeddingModel, EmbeddingResult
import functools
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
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
        raise NotImplementedError((
            "The `_predict` method must be implemented "
            "in the child classes of abstract model."
        ))

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

    @staticmethod
    def get_available_evaluation_schemas() -> List[str]:
        """Returns available evaluation schemas for this task."""
        raise NotImplementedError((
            "The `get_available_evaluation_schemas` method must be implemented "
            "in the child classes of abstract model."
        ))

    @classmethod
    def normalize_node_feature(
        cls,
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
                    cls.task_involves_edge_types() and node_feature.is_using_edge_types() or
                    cls.task_involves_node_types() and node_feature.is_using_node_types() or
                    cls.task_involves_edge_weights() and node_feature.is_using_edge_weights() or
                    cls.task_involves_topology() and node_feature.is_topological()
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
                # If this is an Ensmallen model, we can enable the verbosity
                # as it will only show up in the jupyter kernel and it won't bother the
                # other loading bars.
                verbose="Ensmallen" == node_feature.library_name()
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

    @classmethod
    def normalize_node_features(
        cls,
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
            for normalized_node_feature in cls.normalize_node_feature(
                graph=graph,
                node_feature=node_feature,
                allow_automatic_feature=allow_automatic_feature,
                skip_evaluation_biased_feature=skip_evaluation_biased_feature,
                smoke_test=smoke_test
            )
        ]

    @classmethod
    def normalize_node_type_feature(
        cls,
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
                    cls.task_involves_edge_types() and node_type_feature.is_using_edge_types() or
                    cls.task_involves_node_types() and node_type_feature.is_using_node_types() or
                    cls.task_involves_edge_weights() and node_type_feature.is_using_edge_weights() or
                    cls.task_involves_topology() and node_type_feature.is_topological()
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

    @classmethod
    def normalize_node_type_features(
        cls,
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
            for normalized_node_type_feature in cls.normalize_node_type_feature(
                graph=graph,
                node_type_feature=node_type_feature,
                allow_automatic_feature=allow_automatic_feature,
                skip_evaluation_biased_feature=skip_evaluation_biased_feature,
                smoke_test=smoke_test
            )
        ]

    @classmethod
    def normalize_edge_feature(
        cls,
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
                    cls.task_involves_edge_types() and edge_feature.is_using_edge_types() or
                    cls.task_involves_node_types() and edge_feature.is_using_node_types() or
                    cls.task_involves_edge_weights() and edge_feature.is_using_edge_weights() or
                    cls.task_involves_topology() and edge_feature.is_topological()
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

    @classmethod
    def normalize_edge_features(
        cls,
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
            for normalized_edge_feature in cls.normalize_edge_feature(
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

        if (self.requires_node_types() or self.is_using_node_types()) and not graph.has_node_types():
            raise ValueError(
                f"The provided graph {graph.get_name()} does not have node types, but "
                f"the {self.model_name()} requires or is parametrized to use node types."
            )

        if self.requires_edge_types() and not graph.has_edge_types():
            raise ValueError(
                f"The provided graph {graph.get_name()} does not have edge types, but "
                f"the {self.model_name()} requires edge types."
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
        support: Optional[Graph] = None,
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
    ) -> np.ndarray:
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
                support=support,
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
        support: Optional[Graph] = None,
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
    ) -> np.ndarray:
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
                support=support,
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
        ground_truth: np.ndarray,
        predictions: np.ndarray,
    ) -> Dict[str, Any]:
        """Return evaluations for the provided predictions.

        Parameters
        ----------------
        ground_truth: np.ndarray
            The ground truth to evaluate the predictions against.
        predictions: np.ndarray
            The predictions to be evaluated.
        """
        if self.is_binary_prediction_task():
            return express_measures.all_binary_metrics(
                ground_truth.flatten(),
                predictions.flatten()
            )

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
                    average="binary" if self.is_binary_prediction_task() else "macro",
                    zero_division=0
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
        ground_truth: np.ndarray,
        prediction_probabilities: np.ndarray,
    ) -> Dict[str, Any]:
        """Return evaluations for the provided predictions.

        Parameters
        ----------------
        ground_truth: np.ndarray
            The ground truth to evaluate the predictions against.
        prediction_probabilities: np.ndarray
            The predictions to be evaluated.
        """
        metrics = []
        if self.is_binary_prediction_task():
            return {
                "auroc": express_measures.binary_auroc(
                    ground_truth.flatten(),
                    prediction_probabilities.flatten()
                ),
                "auprc": express_measures.binary_auprc(
                    ground_truth.flatten(),
                    prediction_probabilities.flatten()
                )
            }

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

    @staticmethod
    def split_graph_following_evaluation_schema(
        graph: Graph,
        evaluation_schema: str,
        holdout_number: int,
        **holdouts_kwargs: Dict
    ) -> Tuple[Graph]:
        """Return train and test graphs tuple following the provided evaluation schema.

        Parameters
        ----------------------
        graph: Graph
            The graph to split.
        evaluation_schema: str
            The evaluation schema to follow.
        holdout_number: int
            The current holdout number.
        holdouts_kwargs: Dict[str, Any]
            The kwargs to be forwarded to the holdout method.
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
        support: Optional[Graph] = None,
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

    @classmethod
    def _prepare_evaluation(
        cls,
        graph: Graph,
        train: Graph,
        test: Graph,
        support: Optional[Graph] = None,
        subgraph_of_interest: Optional[Graph] = None,
        random_state: int = 42,
        verbose: bool = True,
        **kwargs: Dict
    ) -> Dict[str, Any]:
        """Return additional custom parameters for the current holdout."""
        raise NotImplementedError(
            "The _evaluate method should be implemented in the child "
            "classes of abstract classifier model."
        )

    @classmethod
    def iterate_classifier_models(
        cls,
        models: Union[str, Type["AbstractClassifierModel"], List[Type["AbstractClassifierModel"]]],
        library_names: Optional[Union[str, List[str]]],
        smoke_test: bool
    ) -> Iterator[Type["AbstractClassifierModel"]]:
        """Return iterator over the provided models after validation.

        Parameters
        -------------------
        models: Union[Type[AbstractClassifierModel], List[Type[AbstractClassifierModel]]]
            The models to validate and iterate on.
        expected_parent_class: Type[AbstractClassifierModel]
            The parent class to check the model against.
        library_names: Optional[Union[str, List[str]]] = None
            Library names from where to retrieve the provided model names.
        smoke_test: bool = False
            Whether this run should be considered a smoke test
            and therefore use the smoke test configurations for
            the provided model names and feature names.
        """
        if not isinstance(models, (list, tuple, pd.Series)):
            models = [models]

        number_of_models = len(models)

        if not isinstance(library_names, (list, tuple, pd.Series)):
            library_names = [library_names] * number_of_models

        if number_of_models == 0:
            raise ValueError(
                "An empty list of models was provided."
            )

        number_of_libraries = len(library_names)

        if number_of_libraries != number_of_models:
            raise ValueError(
                f"The number of the provided models {number_of_models} "
                f"is different from the number of provided libraries {number_of_libraries}."
            )

        models = [
            cls.get_model_from_library(
                model,
                task_name=cls.task_name(),
                library_name=library_name
            )()
            if isinstance(model, str)
            else model
            for model, library_name in zip(
                models,
                library_names
            )
        ]

        for model in models:
            if not issubclass(model.__class__, cls):
                raise ValueError(
                    "The provided classifier model is expected to be "
                    f"an implementation of the {cls.__name__} class, but you provided "
                    f"an object of type {type(model)} that does not hereditate from "
                    "the expected class."
                )

        # If this is a smoke test, we replace all of the
        # provided models with their smoke test version.
        if smoke_test:
            models = [
                model.__class__(**model.smoke_test_parameters())
                for model in models
            ]

        for model in tqdm(
            models,
            desc=f"{cls.task_name()} Models",
            total=number_of_models,
            disable=number_of_models == 1,
            dynamic_ncols=True,
            leave=False
        ):
            yield model.clone()

    @Cache(
        cache_path="{cache_dir}/{self.task_name()}/{graph.get_name()}/holdout_{holdout_number}/{self.model_name()}/{self.library_name()}/{_hash}.csv.gz",
        cache_dir="experiments",
        enable_cache_arg_name="enable_cache",
        args_to_ignore=["verbose", "smoke_test"],
        capture_enable_cache_arg_name=True,
        use_approximated_hash=True
    )
    def __train_and_evaluate_model(
        self,
        graph: Graph,
        train_of_interest: Graph,
        test_of_interest: Graph,
        train: Graph,
        subgraph_of_interest: Graph,
        node_features: Optional[List[np.ndarray]],
        node_type_features: Optional[List[np.ndarray]],
        edge_features: Optional[List[np.ndarray]],
        random_state: int,
        holdout_number: int,
        evaluation_schema: str,
        automatic_features_names: List[str],
        automatic_features_parameters: Dict[str, Any],
        **validation_kwargs
    ) -> pd.DataFrame:
        """Run inner training and evaluation."""
        # Fit the model using the training graph
        training_start = time.time()
        self.fit(
            graph=train_of_interest,
            support=train,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_features=edge_features
        )
        time_required_for_training = time.time() - training_start

        start_evaluation = time.time()

        try:
            # We add the newly computed performance.
            model_performance = pd.DataFrame(self._evaluate(
                graph=graph,
                support=train,
                train=train_of_interest,
                test=test_of_interest,
                node_features=node_features,
                node_type_features=node_type_features,
                edge_features=edge_features,
                subgraph_of_interest=subgraph_of_interest,
                random_state=random_state * holdout_number,
                verbose=False,
                **validation_kwargs
            ))
        except RuntimeError as e:
            raise e
        except Exception as e:
            raise RuntimeError(
                f"An exception was raised while calling the `._evaluate` method of {self.model_name()} "
                f"implemented using the {self.library_name()} for the {self.task_name()} task. "
                f"Specifically, the class of the model is {self.__class__.__name__}. "
            ) from e

        time_required_for_evaluation = time.time() - start_evaluation

        model_performance["time_required_for_training"] = time_required_for_training
        model_performance["time_required_for_evaluation"] = time_required_for_evaluation
        model_performance["task_name"] = self.task_name()
        model_performance["model_name"] = self.model_name()
        model_performance["library_name"] = self.library_name()
        model_performance["graph_name"] = graph.get_name()
        model_performance["nodes_number"] = graph.get_nodes_number()
        model_performance["edges_number"] = graph.get_number_of_directed_edges()
        model_performance["evaluation_schema"] = evaluation_schema
        if automatic_features_names:
            model_performance["automatic_features_names"] = format_list(
                automatic_features_names
            )
        for parameter, value in automatic_features_parameters.items():
            if parameter in model_performance.columns:
                raise ValueError(
                    "There has been a collision between the parameters used in "
                    f"one of the classifiers, {self.model_name()},  and the parameter "
                    "used for the validation and reporting of the task itself. "
                    f"The parameter that has caused the collision is {parameter}. "
                    "Please do change the name of the parameter in your model."
                )
            model_performance[parameter] = str(value)
        
        return model_performance

    @classmethod
    @Cache(
        cache_path="{cache_dir}/{cls.task_name()}/{graph.get_name()}/holdout_{holdout_number}/{_hash}.csv.gz",
        cache_dir="experiments",
        enable_cache_arg_name="enable_cache",
        args_to_ignore=["verbose", "smoke_test"],
        capture_enable_cache_arg_name=False,
        use_approximated_hash=True
    )
    def __evaluate_on_single_holdout(
        cls,
        models: Union[Type["AbstractClassifierModel"], List[Type["AbstractClassifierModel"]]],
        library_names: Optional[Union[str, List[str]]],
        graph: Graph,
        subgraph_of_interest: Graph,
        node_features: Optional[List[np.ndarray]],
        node_type_features: Optional[List[np.ndarray]],
        edge_features: Optional[List[np.ndarray]],
        random_state: int,
        holdout_number: int,
        evaluation_schema: str,
        enable_cache: bool,
        smoke_test: bool,
        holdouts_kwargs: Dict[str, Any],
        subgraph_of_interest_has_compatible_nodes: Optional[bool],
        automatic_features_names: List[str],
        automatic_features_parameters: Dict[str, Any],
        verbose: bool,
        **validation_kwargs
    ) -> pd.DataFrame:
        starting_setting_up_holdout = time.time()

        # We create the graph split using the provided schema.
        train, test = cls.split_graph_following_evaluation_schema(
            graph=graph,
            evaluation_schema=evaluation_schema,
            random_state=random_state,
            holdout_number=holdout_number,
            **holdouts_kwargs
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
        holdout_node_features = cls.normalize_node_features(
            train,
            node_features=node_features,
            allow_automatic_feature=True,
            skip_evaluation_biased_feature=False,
            smoke_test=smoke_test
        )

        # We compute the remaining features
        holdout_node_type_features = cls.normalize_node_type_features(
            train,
            node_type_features=node_type_features,
            allow_automatic_feature=True,
            skip_evaluation_biased_feature=False,
            smoke_test=smoke_test
        )

        # We execute the same thing as described above,
        # but now for the edge features instead that for
        # the node features.
        holdout_edge_features = cls.normalize_edge_features(
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
                    cls.task_name() in ("Edge Prediction", "Edge Label Prediction") and
                    not graph_partition.has_edges()
                ):
                    raise ValueError(
                        f"The {graph_partition_name} graph {graph_partition.get_name()} obtained from the evaluation "
                        f"schema {evaluation_schema}, once filtered using the provided "
                        "subgraph of interest, does not have any more edges which are "
                        f"essential when running a {cls.task_name()} task."
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

        additional_validation_kwargs = cls._prepare_evaluation(
            graph=graph,
            support=train,
            train=train_of_interest,
            test=test_of_interest,
            subgraph_of_interest=subgraph_of_interest,
            random_state=random_state * holdout_number,
            verbose=verbose,
            **validation_kwargs
        )

        time_required_for_setting_up_holdout = time.time() - starting_setting_up_holdout

        holdout_performance = pd.concat([
            classifier.__train_and_evaluate_model(
                graph=graph,
                train_of_interest=train_of_interest,
                test_of_interest=test_of_interest,
                train=train,
                subgraph_of_interest=subgraph_of_interest,
                node_features=holdout_node_features,
                node_type_features=holdout_node_type_features,
                edge_features=holdout_edge_features,
                random_state=random_state,
                holdout_number=holdout_number,
                evaluation_schema=evaluation_schema,
                enable_cache=enable_cache,
                automatic_features_names=automatic_features_names,
                automatic_features_parameters=automatic_features_parameters,
                **additional_validation_kwargs,
                **validation_kwargs,
            )
            for classifier in cls.iterate_classifier_models(
                models=models,
                library_names=library_names,
                smoke_test=smoke_test
            )
        ])

        holdout_performance["time_required_for_setting_up_holdout"] = time_required_for_setting_up_holdout

        return holdout_performance

    @classmethod
    @Cache(
        cache_path="{cache_dir}/{cls.task_name()}/{graph.get_name()}/{_hash}.csv.gz",
        cache_dir="experiments",
        enable_cache_arg_name="enable_cache",
        args_to_ignore=["verbose", "smoke_test"],
        capture_enable_cache_arg_name=False,
        use_approximated_hash=True
    )
    def evaluate(
        cls,
        models: Union[Type["AbstractClassifierModel"], List[Type["AbstractClassifierModel"]]],
        graph: Graph,
        evaluation_schema: str,
        holdouts_kwargs: Dict[str, Any],
        library_names: Optional[Union[str, List[str]]] = None,
        node_features: Optional[Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel], List[Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel]]]]] = None,
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        edge_features: Optional[Union[str, pd.DataFrame, np.ndarray, List[Union[str, pd.DataFrame, np.ndarray]]]] = None,
        subgraph_of_interest: Optional[Graph] = None,
        number_of_holdouts: int = 10,
        random_state: int = 42,
        verbose: bool = True,
        enable_cache: bool = False,
        smoke_test: bool = False,
        **validation_kwargs: Dict
    ) -> pd.DataFrame:
        """Execute evaluation on the provided graph.

        Parameters
        --------------------
        models: Union[Type["AbstractClassifierModel"], List[Type["AbstractClassifierModel"]]]
            The model(s) to be evaluated.
        graph: Graph
            The graph to run predictions on.
        evaluation_schema: str
            The schema for the evaluation to follow.
        holdouts_kwargs: Dict[str, Any]
            Parameters to forward to the desired evaluation schema.
        library_names: Optional[Union[str, List[str]]] = None
            The library names of the models, needed when a desired model is implemented in multiple
            libraries and it is unclear which one to use.
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
        enable_cache: bool = False
            Whether to enable the cache.
        smoke_test: bool = False
            Whether this run should be considered a smoke test
            and therefore use the smoke test configurations for
            the provided model names and feature names.
        **validation_kwargs: Dict
            kwargs to be forwarded to the model `_evaluate` method.
        """
        if not isinstance(number_of_holdouts, int) or number_of_holdouts <= 0:
            raise ValueError(
                "The number of holdouts must be a strictly positive integer, "
                f"but {number_of_holdouts} was provided."
            )

        if subgraph_of_interest is not None:
            if cls.task_name() not in ("Edge Prediction", "Edge Label Prediction"):
                raise ValueError(
                    "A subgraph of interest was provided, but this parameter "
                    "is only currently supported for Edge Prediction and "
                    f"Edge Label Prediction tasks and not the {cls.task_name()} task."
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
        else:
            subgraph_of_interest_has_compatible_nodes = None

        # Retrieve the set of provided automatic features parameters
        # so we can put them in the report.
        automatic_features_parameters = {
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
            )
            for feature in features
            if issubclass(feature.__class__, AbstractModel)
            for parameter_name, value in feature.parameters().items()
        }

        # Retrieve the set of provided automatic features names
        # so we can put them in the report.
        automatic_features_names = {
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
        node_features = cls.normalize_node_features(
            graph,
            node_features=node_features,
            allow_automatic_feature=True,
            skip_evaluation_biased_feature=True,
            smoke_test=smoke_test
        )

        # We execute the same thing as described above,
        # but now for the node type features instead that for
        # the node features.
        node_type_features = cls.normalize_node_type_features(
            graph,
            node_type_features=node_type_features,
            allow_automatic_feature=True,
            skip_evaluation_biased_feature=True,
            smoke_test=smoke_test
        )

        # We execute the same thing as described above,
        # but now for the edge features instead that for
        # the node features.
        edge_features = cls.normalize_edge_features(
            graph,
            edge_features=edge_features,
            allow_automatic_feature=True,
            skip_evaluation_biased_feature=True,
            smoke_test=smoke_test
        )

        # We start to iterate on the holdouts.
        performance = pd.concat([
            cls.__evaluate_on_single_holdout(
                models=models,
                library_names=library_names,
                graph=graph,
                subgraph_of_interest=subgraph_of_interest,
                node_features=node_features,
                node_type_features=node_type_features,
                edge_features=edge_features,
                random_state=random_state,
                holdout_number=holdout_number,
                evaluation_schema=evaluation_schema,
                enable_cache=enable_cache,
                smoke_test=smoke_test,
                holdouts_kwargs=holdouts_kwargs,
                subgraph_of_interest_has_compatible_nodes=subgraph_of_interest_has_compatible_nodes,
                verbose=verbose,
                automatic_features_names=automatic_features_names,
                automatic_features_parameters=automatic_features_parameters,
                **validation_kwargs
            )
            for holdout_number in trange(
                number_of_holdouts,
                disable=not verbose,
                leave=False,
                dynamic_ncols=True,
                desc=f"Evaluating on {graph.get_name()}"
            )
        ])

        # Adding to the report the informations relative to
        # the whole validation run, which are NOT necessary
        # to make unique the cache hash of the single holdouts
        # or the single models.
        # Be extremely weary of adding informations at this
        # high level, as often the best place should be
        # in the core loop, where the actual model is trained,
        # as often such information changes how the model
        # is trained.
        performance["number_of_holdouts"] = number_of_holdouts

        # We save the constant values for this model
        # execution.
        return performance
