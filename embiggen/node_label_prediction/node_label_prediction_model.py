"""Module providing abstract node label prediction model."""
from typing import Optional, Union, List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import math
from ensmallen import Graph
from ..utils import AbstractClassifierModel, AbstractEmbeddingModel, abstract_class


@abstract_class
class AbstractNodeLabelPredictionModel(AbstractClassifierModel):
    """Class defining an abstract node label prediction model."""

    def __init__(self):
        self._is_binary_prediction_task = None
        super().__init__()

    @staticmethod
    def requires_node_types() -> bool:
        """Returns whether this method requires node types."""
        return True

    @staticmethod
    def task_name() -> str:
        """Returns name of the task this model is used for."""
        return "Node Label Prediction"

    @staticmethod
    def library_name() -> str:
        """Return name of the model."""
        return "scikit-learn"

    def get_evaluation_biased_feature_names(self) -> List[str]:
        """Returns names of features that are biased in an evaluation setting."""
        # TODO: Extend this list.
        return [
            "TransE"
        ]

    def get_available_evaluation_schemas(self) -> List[str]:
        """Returns available evaluation schemas for this task."""
        return [
            "Stratified Monte Carlo",
            "Monte Carlo",
            "Stratified Kfold"
            "Kfold"
        ]

    def is_binary_prediction_task(self) -> bool:
        """Returns whether the model was fit on a binary prediction task."""
        return self._is_binary_prediction_task

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
        if evaluation_schema in ("Stratified Monte Carlo", "Monte Carlo"):
            return graph.get_node_label_holdout_graphs(
                **holdouts_kwargs,
                use_stratification="Stratified" in evaluation_schema,
                random_state=random_state+holdout_number,
            )
        if evaluation_schema in ("Kfold", "Stratified Kfold"):
            return graph.get_node_label_kfold(
                k=number_of_holdouts,
                k_index=holdout_number,
                use_stratification="Stratified" in evaluation_schema,
                random_state=random_state,
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
    ) -> List[Dict[str, Any]]:
        """Return model evaluation on the provided graphs."""
        nodes_number = graph.get_nodes_number()
        train_size = train.get_known_node_types() / nodes_number
        performance = []
        for evaluation_mode, evaluation_graph in (
            ("train", train),
            ("test", test),
        ):
            predictions = self.predict(
                evaluation_graph,
                node_features=node_features,
                edge_features=edge_features
            )
            prediction_probabilities = self.predict_proba(
                evaluation_graph,
                node_features=node_features,
                edge_features=edge_features
            )
            if evaluation_graph.has_multilabel_node_types():
                labels = evaluation_graph.get_one_hot_encoded_node_types()
            else:
                labels = np.fromiter(
                    (
                        0 if node_type_id is None else node_type_id[0]
                        for node_type_id in (
                            evaluation_graph.get_node_type_ids_from_node_id(node_id)
                            for node_id in range(evaluation_graph.get_nodes_number())
                        )
                    ),
                    dtype=np.uint32
                )
            
            if graph.has_unknown_node_types():
                mask = graph.get_known_node_types_mask()
                prediction_probabilities = prediction_probabilities[mask]
                predictions = predictions[mask]
                labels = labels[mask]

            performance.append({
                "evaluation_mode": evaluation_mode,
                "train_size": train_size,
                "edges_number": edges_number,
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
        if edge_features is not None:
            raise NotImplementedError(
                "Currently edge features are not supported in edge prediction models."
            )

        self._is_binary_prediction_task = graph.get_node_types_number() == 2

        super().fit(
            graph=graph,
            node_features=node_features,
            edge_features=None,
            skip_evaluation_biased_feature=skip_evaluation_biased_feature,
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
        """
        if edge_features is not None:
            raise NotImplementedError(
                "Currently edge features are not supported in node-label prediction models."
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
            smoke_test=smoke_test,
            verbose=verbose,
        )
