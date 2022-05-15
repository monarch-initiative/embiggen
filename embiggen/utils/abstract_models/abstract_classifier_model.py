"""Module providing abstract classes for classification models."""
import warnings
from typing import Union, Optional, List, Dict, Any, Tuple, Type
from ensmallen import Graph
import numpy as np
import pandas as pd
from .abstract_model import AbstractModel, abstract_class
import time
from .abstract_embedding_model import AbstractEmbeddingModel
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
        node_features: Optional[List[np.ndarray]] = None,
        edge_features: Optional[List[np.ndarray]] = None,
    ):
        """Run fitting on the provided graph.

        Parameters
        --------------------
        graph: Graph
            The graph to run predictions on.
        node_features: Optional[List[np.ndarray]] = None
            The node features to use.
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
        edge_features: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """Run prediction on the provided graph.

        Parameters
        --------------------
        graph: Graph
            The graph to run predictions on.
        node_features: Optional[List[np.ndarray]] = None
            The node features to use.
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
        edge_features: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """Run prediction on the provided graph.

        Parameters
        --------------------
        graph: Graph
            The graph to run predictions on.
        node_features: Optional[List[np.ndarray]] = None
            The node features to use.
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

    def get_evaluation_biased_feature_names(self) -> List[str]:
        """Returns names of features that are biased in an evaluation setting."""
        raise NotImplementedError((
            "The `get_evaluation_biased_feature_names` method must be implemented "
            "in the child classes of abstract model."
        ))

    def get_available_evaluation_schemas(self) -> List[str]:
        """Returns available evaluation schemas for this task."""
        raise NotImplementedError((
            "The `get_available_evaluation_schemas` method must be implemented "
            "in the child classes of abstract model."
        ))

    def _get_evaluation_biased_feature_names_lowercase(self) -> List[str]:
        """Returns lowercase names of features that are biased in an evaluation setting."""
        return [
            feature_name.lower()
            for feature_name in self.get_evaluation_biased_feature_names()
        ]

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
            if (
                skip_evaluation_biased_feature and
                node_feature.lower() in self._get_evaluation_biased_feature_names_lowercase()
            ):
                return node_feature

            node_feature = AbstractEmbeddingModel.get_model_from_library(
                model_name=node_feature
            )()

        # If this object is an implementation of an abstract
        # embedding model, we compute the embedding.
        if issubclass(node_feature.__class__, AbstractEmbeddingModel):
            if (
                skip_evaluation_biased_feature and
                node_feature.model_name().lower() in self._get_evaluation_biased_feature_names_lowercase()
            ):
                return node_feature

            if smoke_test:
                node_feature = node_feature.__class__(
                    **node_feature.smoke_test_parameters()
                )

            node_feature = node_feature.fit_transform(
                graph=graph,
                return_dataframe=True,
                verbose=False
            )

        if not isinstance(node_feature, (np.ndarray, pd.DataFrame)):
            raise ValueError(
                (
                    "The provided node features are of type `{node_features_type}`, "
                    "while we only currently support numpy arrays and pandas DataFrames. "
                    "What behaviour were you expecting with this feature? "
                    "Please do open an issue on Embiggen and let us know!"
                ).format(
                    node_features_type=type(node_feature)
                )
            )

        if graph.get_nodes_number() != node_feature.shape[0]:
            raise ValueError(
                (
                    "The provided node features have {rows_number} rows "
                    "but the provided graph{graph_name} has {nodes_number} nodes. "
                    "Maybe these features refer to another "
                    "version of the graph or another graph "
                    "entirely?"
                ).format(
                    rows_number=node_feature.shape[0],
                    graph_name="" if graph.get_name().lower(
                    ) == "graph" else " {}".format(graph.get_name()),
                    nodes_number=graph.get_nodes_number()
                )
            )

        # If it is a dataframe we align it
        if isinstance(node_feature, pd.DataFrame):
            return node_feature.loc[graph.get_node_names()].to_numpy()

        # And if it is a numpy array we must believe that the user knows what
        # they are doing, as we cannot ensure alignment.
        return node_feature

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

        if not isinstance(node_features, list):
            node_features = [node_features]

        return [
            self.normalize_node_feature(
                graph=graph,
                node_feature=node_feature,
                allow_automatic_feature=allow_automatic_feature,
                skip_evaluation_biased_feature=skip_evaluation_biased_feature,
                smoke_test=smoke_test
            )
            for node_feature in node_features
        ]

    def normalize_edge_feature(
        self,
        graph: Graph,
        edge_feature: Optional[Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel]]] = None,
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
        if edge_feature is None:
            return None

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
            if (
                skip_evaluation_biased_feature and
                edge_feature.lower() in self._get_evaluation_biased_feature_names_lowercase()
            ):
                return edge_feature

            edge_feature = AbstractEmbeddingModel.get_model_from_library(
                model_name=edge_feature
            )()

        # If this object is an implementation of an abstract
        # embedding model, we compute the embedding.
        if issubclass(edge_feature.__class__, AbstractEmbeddingModel):
            if (
                skip_evaluation_biased_feature and
                edge_feature.model_name().lower() in self._get_evaluation_biased_feature_names_lowercase()
            ):
                return edge_feature

            if smoke_test:
                edge_feature = edge_feature.__class__(
                    **edge_feature.smoke_test_parameters()
                )

            edge_feature = edge_feature.fit_transform(
                graph=graph,
                return_dataframe=True,
                verbose=False
            )

        if not isinstance(edge_feature, (np.ndarray, pd.DataFrame)):
            raise ValueError(
                (
                    "The provided edge features are of type `{edge_features_type}`, "
                    "while we only currently support numpy arrays and pandas DataFrames. "
                    "What behaviour were you expecting with this feature? "
                    "Please do open an issue on Embiggen and let us know!"
                ).format(
                    edge_features_type=type(edge_feature)
                )
            )

        if graph.get_directed_edges_number() != edge_feature.shape[0]:
            raise ValueError(
                (
                    "The provided edge features have {rows_number} rows "
                    "but the provided graph{graph_name} has {edges_number} edges. "
                    "Maybe these features refer to another "
                    "version of the graph or another graph "
                    "entirely?"
                ).format(
                    rows_number=edge_feature.shape[0],
                    graph_name="" if graph.get_name().lower(
                    ) == "graph" else " {}".format(graph.get_name()),
                    edges_number=graph.get_edges_number()
                )
            )

        # If it is a dataframe we align it
        if isinstance(edge_feature, pd.DataFrame):
            return edge_feature.loc[graph.get_edge_names()].to_numpy()

        # And if it is a numpy array we must believe that the user knows what
        # they are doing, as we cannot ensure alignment.
        return edge_feature

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
            self.normalize_edge_feature(
                graph=graph,
                edge_feature=edge_feature,
                allow_automatic_feature=allow_automatic_feature,
                skip_evaluation_biased_feature=skip_evaluation_biased_feature
            )
            for edge_feature in edge_features
        ]

    def fit(
        self,
        graph: Graph,
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        skip_evaluation_biased_feature: bool = False
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
        skip_evaluation_biased_feature: bool = False
            Whether to skip feature names that are known to be biased
            when running an holdout. These features should be computed
            exclusively on the training graph and not the entire graph.
        """
        if not graph.has_nodes():
            raise ValueError("The provided graph is empty.")

        self._fit(
            graph=graph,
            node_features=self.normalize_node_features(
                graph=graph,
                node_features=node_features,
                allow_automatic_feature=True,
                skip_evaluation_biased_feature=skip_evaluation_biased_feature
            ),
            edge_features=self.normalize_edge_features(
                graph=graph,
                edge_features=edge_features,
                allow_automatic_feature=True,
                skip_evaluation_biased_feature=skip_evaluation_biased_feature
            ),
        )

        self._fitting_was_executed = True

    def predict(
        self,
        graph: Graph,
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
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

        return self._predict(
            graph=graph,
            node_features=self.normalize_node_features(
                graph=graph,
                node_features=node_features,
                allow_automatic_feature=False,
            ),
            edge_features=self.normalize_edge_features(
                graph=graph,
                edge_features=edge_features,
                allow_automatic_feature=False,
            ),
        )

    def predict_proba(
        self,
        graph: Graph,
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
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

        return self._predict_proba(
            graph=graph,
            node_features=self.normalize_node_features(
                graph=graph,
                node_features=node_features,
                allow_automatic_feature=False,
            ),
            edge_features=self.normalize_edge_features(
                graph=graph,
                edge_features=edge_features,
                allow_automatic_feature=False,
            ),
        )

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
        if self.is_binary_prediction_task():
            average_methods = ("binary",)
        else:
            average_methods = ("macro", "weighted")

        return {
            **{
                metric.__name__: metric(ground_truth, predictions)
                for metric in (
                    accuracy_score,
                    balanced_accuracy_score,
                )
            },
            **{
                metric.__name__
                if average_method == "binary"
                else metric.__name__ + " " + average_method: metric(ground_truth, predictions, average=average_method)
                for metric in (
                    f1_score,
                    precision_score,
                    recall_score,
                )
                for average_method in average_methods
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
            average_methods_probs = ("micro", "macro", "weighted")
            # AUPRC in sklearn is only supported for binary labels
            metrics.append(average_precision_score)
        else:
            average_methods_probs = ("macro", "weighted")

        @functools.wraps(roc_auc_score)
        def wrapper_roc_auc_score(*args, **kwargs):
            return roc_auc_score(*args, **kwargs, multi_class="ovr")

        metrics.append(wrapper_roc_auc_score)

        return {
            metric.__name__ + " " + average_method: metric(
                ground_truth,
                prediction_probabilities,
                average=average_method
            )
            for metric in metrics
            for average_method in average_methods_probs
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
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[str, pd.DataFrame, np.ndarray]]]] = None,
        random_state: int = 42,
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
        edge_features: Optional[Union[str, pd.DataFrame, np.ndarray, List[Union[str, pd.DataFrame, np.ndarray]]]] = None,
        subgraph_of_interest: Optional[Graph] = None,
        number_of_holdouts: int = 10,
        random_state: int = 42,
        verbose: bool = True,
        smoke_test: bool = False,
        **kwargs: Dict
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
        **kwargs: Dict
            kwargs to be forwarded to the model `_evaluate` method.
        """
        if self._fitting_was_executed:
            warnings.warn((
                "Do be advised that this model was already fitted, "
                "and you will be therefore be running a classification "
                "evaluation using a warm start. "
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

        # We normalize and/or compute the node features, having
        # the care of skipping the features that induce bias when
        # computed on the entire graph.
        # This way we compute only once the features that do not
        # cause biases for this task, while recomputing those
        # that cause biases at each holdout, avoiding said biases.
        node_features = self.normalize_node_feature(
            graph,
            node_feature=node_features,
            allow_automatic_feature=True,
            skip_evaluation_biased_feature=True,
            smoke_test=smoke_test
        )

        # We execute the same thing as described above,
        # but now for the edge features instead that for
        # the node features.
        edge_features = self.normalize_edge_feature(
            graph,
            edge_feature=edge_features,
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

            # We compute the remaining features
            holdout_node_features = self.normalize_node_feature(
                train,
                node_feature=node_features,
                allow_automatic_feature=True,
                skip_evaluation_biased_feature=False,
                smoke_test=smoke_test
            )

            # We execute the same thing as described above,
            # but now for the edge features instead that for
            # the node features.
            holdout_edge_features = self.normalize_edge_feature(
                train,
                edge_feature=edge_features,
                allow_automatic_feature=True,
                skip_evaluation_biased_feature=False,
                smoke_test=smoke_test
            )

            if subgraph_of_interest is not None:
                train = train.filter_from_names(
                    node_names_to_keep_from_graph=subgraph_of_interest
                )
                test = test.filter_from_names(
                    node_names_to_keep_from_graph=subgraph_of_interest
                )

            # Fit the model using the training graph
            training_start = time.time()
            classifier.fit(
                graph=train,
                node_features=holdout_node_features,
                edge_features=holdout_edge_features
            )
            required_training_time = time.time() - training_start

            # We add the newly computed performance.
            holdout_performance = classifier._evaluate(
                graph=graph if subgraph_of_interest is None else subgraph_of_interest,
                train=train,
                test=test,
                node_features=holdout_node_features,
                edge_features=holdout_edge_features,
                random_state=random_state,
                **kwargs
            )
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
        performance["number_of_holdouts"] = number_of_holdouts
        performance["evaluation_schema"] = evaluation_schema
        for parameter, value in self.parameters().items():
            performance[parameter] = str(value)

        return performance
