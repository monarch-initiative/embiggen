"""Module providing abstract edge prediction model."""
from typing import Optional, Union, List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import math
from ensmallen import Graph
from ..utils import AbstractClassifierModel, AbstractEmbeddingModel, abstract_class


@abstract_class
class AbstractEdgePredictionModel(AbstractClassifierModel):
    """Class defining an abstract edge prediction model."""

    @staticmethod
    def task_name() -> str:
        """Returns name of the task this model is used for."""
        return "Edge Prediction"

    def is_binary_prediction_task(self) -> bool:
        """Returns whether the model was fit on a binary prediction task."""
        # Edge prediction is always a binary prediction task.
        return True

    @staticmethod
    def is_topological() -> bool:
        return True

    def get_available_evaluation_schemas(self) -> List[str]:
        """Returns available evaluation schemas for this task."""
        return [
            "Connected Monte Carlo",
            "Monte Carlo",
            "Kfold"
        ]

    def split_graph_following_evaluation_schema(
        self,
        graph: Graph,
        evaluation_schema: str,
        number_of_holdouts: int,
        random_state: int,
        holdouts_kwargs: Dict[str, Any],
        holdout_number: int
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
        if evaluation_schema == "Connected Monte Carlo":
            return graph.connected_holdout(
                **holdouts_kwargs,
                random_state=random_state+holdout_number,
                verbose=False
            )
        if evaluation_schema == "Monte Carlo":
            return graph.random_holdout(
                **holdouts_kwargs,
                random_state=random_state+holdout_number,
                verbose=False
            )
        if evaluation_schema == "Kfold":
            return graph.get_edge_prediction_kfold(
                **holdouts_kwargs,
                k=number_of_holdouts,
                k_index=holdout_number,
                random_state=random_state,
                verbose=False
            )
        raise ValueError(
            f"The requested evaluation schema `{evaluation_schema}` "
            "is not available."
        )

    def _evaluate(
        self,
        graph: Graph,
        train: Graph,
        test: Graph,
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[str, pd.DataFrame, np.ndarray]]]] = None,
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[str, pd.DataFrame, np.ndarray]]]] = None,
        random_state: int = 42,
        sample_only_edges_with_heterogeneous_node_types: bool = False,
        unbalance_rates: Tuple[float] = (1.0, )
    ) -> List[Dict[str, Any]]:
        """Return model evaluation on the provided graphs."""
        edges_number = graph.get_edges_number()
        train_size = train.get_edges_number() / edges_number
        performance = []
        existent_train_predictions = self.predict(
            train,
            node_features=node_features,
            edge_features=edge_features
        )
        existent_test_predictions = self.predict(
            test,
            node_features=node_features,
            edge_features=edge_features
        )
        existent_train_prediction_probabilities = self.predict_proba(
            train,
            node_features=node_features,
            edge_features=edge_features
        )
        if existent_train_prediction_probabilities.shape[1] > 1:
            existent_train_prediction_probabilities = existent_train_prediction_probabilities[:, 1]
        existent_test_prediction_probabilities = self.predict_proba(
            test,
            node_features=node_features,
            edge_features=edge_features
        )
        if existent_test_prediction_probabilities.shape[1] > 1:
            existent_test_prediction_probabilities = existent_test_prediction_probabilities[:, 1]

        for unbalance_rate in unbalance_rates:
            negative_graph = train.sample_negative_graph(
                number_of_negative_samples=int(
                    math.ceil(edges_number*unbalance_rate)),
                random_state=random_state,
                sample_only_edges_with_heterogeneous_node_types=sample_only_edges_with_heterogeneous_node_types,
                graph_to_avoid=graph
            )

            assert(negative_graph.has_edges())

            non_existent_train, non_existent_test = negative_graph.random_holdout(
                train_size=train_size,
                random_state=random_state,
                verbose=False,
            )

            assert(non_existent_train.has_edges())
            assert(non_existent_test.has_edges())

            for evaluation_mode, (existent_predictions, existent_prediction_probabilitiess, non_existent_graph) in (
                (
                    "train",
                    (
                        existent_train_predictions,
                        existent_train_prediction_probabilities,
                        non_existent_train
                    )
                ),
                (
                    "test",
                    (
                        existent_test_predictions,
                        existent_test_prediction_probabilities,
                        non_existent_test
                    )
                ),
            ):
                non_existent_predictions = self.predict(
                    non_existent_graph,
                    node_features=node_features,
                    edge_features=edge_features
                )
                non_existent_prediction_probabilities = self.predict_proba(
                    non_existent_graph,
                    node_features=node_features,
                    edge_features=edge_features
                )

                if non_existent_prediction_probabilities.shape[1] > 1:
                    non_existent_prediction_probabilities = non_existent_prediction_probabilities[:, 1]
                
                predictions = np.concatenate((
                    existent_predictions,
                    non_existent_predictions
                ))

                prediction_probabilities = np.concatenate((
                    existent_prediction_probabilitiess,
                    non_existent_prediction_probabilities
                ))

                labels = np.concatenate((
                    np.ones_like(existent_predictions),
                    np.zeros_like(non_existent_predictions),
                ))

                performance.append({
                    "evaluation_mode": evaluation_mode,
                    "unbalance_rate": unbalance_rate,
                    "sample_only_edges_with_heterogeneous_node_types": sample_only_edges_with_heterogeneous_node_types,
                    "train_size": train_size,
                    **self.evaluate_predictions(
                        predictions,
                        labels
                    ),
                    **self.evaluate_prediction_probabilities(
                        prediction_probabilities,
                        labels
                    ),
                })

        return performance

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
        if edge_features is not None:
            raise NotImplementedError(
                "Currently edge features are not supported in edge prediction models."
            )

        return super().predict(graph, node_features)

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
        if edge_features is not None:
            raise NotImplementedError(
                "Currently edge features are not supported in edge prediction models."
            )

        return super().predict_proba(graph, node_features)

    def fit(
        self,
        graph: Graph,
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
    ):
        """Execute fitting on the provided graph.

        Parameters
        --------------------
        graph: Graph
            The graph to run predictions on.
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node features to use.
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The edge features to use.
        """
        if edge_features is not None:
            raise NotImplementedError(
                "Currently edge features are not supported in edge prediction models."
            )

        super().fit(
            graph=graph,
            node_features=node_features,
            edge_features=None,
        )

    def evaluate(
        self,
        graph: Graph,
        evaluation_schema: str,
        holdouts_kwargs: Dict[str, Any],
        node_features: Optional[Union[str, pd.DataFrame, np.ndarray, AbstractEmbeddingModel, List[Union[str, pd.DataFrame, np.ndarray, AbstractEmbeddingModel]]]] = None,
        edge_features: Optional[Union[str, pd.DataFrame, np.ndarray, List[Union[str, pd.DataFrame, np.ndarray]]]] = None,
        subgraph_of_interest: Optional[Graph] = None,
        number_of_holdouts: int = 10,
        random_state: int = 42,
        smoke_test: bool = False,
        verbose: bool = True,
        sample_only_edges_with_heterogeneous_node_types: bool = False,
        unbalance_rates: Tuple[float] = (1.0, )
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
        node_features: Optional[Union[str, pd.DataFrame, np.ndarray, AbstractEmbeddingModel, List[Union[str, pd.DataFrame, np.ndarray, AbstractEmbeddingModel]]]] = None
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
        smoke_test: bool = False
            Whether this run should be considered a smoke test
            and therefore use the smoke test configurations for
            the provided model names and feature names.
        verbose: bool = True
            Whether to show a loading bar while computing holdouts.
        sample_only_edges_with_heterogeneous_node_types: bool = False
            Whether to sample negative edges exclusively between nodes with different node types.
            This can be useful when executing a bipartite edge prediction task.
        unbalance_rates: Tuple[float] = (1.0, )
            Unbalance rate for the non-existent graphs generation.
        """
        if edge_features is not None:
            raise NotImplementedError(
                "Currently edge features are not supported in edge prediction models."
            )

        return super().evaluate(
            graph=graph,
            evaluation_schema=evaluation_schema,
            holdouts_kwargs=holdouts_kwargs,
            node_features=node_features,
            edge_features=None,
            subgraph_of_interest=subgraph_of_interest,
            number_of_holdouts=number_of_holdouts,
            random_state=random_state,
            verbose=verbose,
            smoke_test=smoke_test,
            sample_only_edges_with_heterogeneous_node_types=sample_only_edges_with_heterogeneous_node_types,
            unbalance_rates=unbalance_rates,
        )
