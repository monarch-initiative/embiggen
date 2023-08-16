"""Module providing abstract edge prediction model."""
import math
import gc
import warnings
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
from ensmallen import Graph
from tqdm.auto import tqdm

from embiggen.utils import AbstractEdgeFeature
from embiggen.utils.abstract_models import (AbstractClassifierModel,
                                            abstract_class)


@abstract_class
class AbstractEdgePredictionModel(AbstractClassifierModel):
    """Class defining an abstract edge prediction model."""

    @classmethod
    def task_name(cls) -> str:
        """Returns name of the task this model is used for."""
        return "Edge Prediction"

    def is_binary_prediction_task(self) -> bool:
        """Returns whether the model was fit on a binary prediction task."""
        # Edge prediction is always a binary prediction task.
        return True

    @classmethod
    def is_topological(cls) -> bool:
        return True

    @classmethod
    def get_available_evaluation_schemas(cls) -> List[str]:
        """Returns available evaluation schemas for this task."""
        return ["Connected Monte Carlo", "Monte Carlo", "Kfold"]

    @classmethod
    def edge_features_check(
        cls,
        edge_features: Optional[
            Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]
        ],
    ) -> Optional[Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]]:
        """Check the provided edge features."""
        if edge_features is None:
            edge_features = []
        if not isinstance(edge_features, list):
            edge_features = [edge_features]
        for edge_feature in edge_features:
            if not issubclass(type(edge_feature), AbstractEdgeFeature):
                raise NotImplementedError(
                    "Currently, we solely support edge features that are subclasses of AbstractEdgeFeature. "
                    "This is because most commonly, it is not possible to precompute edge features for all "
                    "possible edges of a complete graph and thus, we need to compute them on the fly. "
                    "To do so, we need a common interface that allows us to query the edge features on "
                    "demand, lazily, hence avoiding unsustainable memory peaks."
                    f"You have provided an egde feature of type {type(edge_feature)}, which is not a subclass of AbstractEdgeFeature."
                )
        return edge_features

    @classmethod
    def split_graph_following_evaluation_schema(
        cls,
        graph: Graph,
        evaluation_schema: str,
        random_state: int,
        holdout_number: int,
        number_of_holdouts: int,
        **holdouts_kwargs: Dict[str, Any],
    ) -> Tuple[Graph]:
        """Return train and test graphs tuple following the provided evaluation schema.

        Parameters
        ----------------------
        graph: Graph
            The graph to split.
        evaluation_schema: str
            The evaluation schema to follow.
        random_state: int
            The random state for the evaluation
        holdout_number: int
            The current holdout number.
        number_of_holdouts: int
            The total number of holdouts.
        holdouts_kwargs: Dict[str, Any]
            The kwargs to be forwarded to the holdout method.
        """
        if evaluation_schema == "Connected Monte Carlo":
            return graph.connected_holdout(
                **holdouts_kwargs,
                random_state=random_state + holdout_number,
                verbose=False,
            )
        if evaluation_schema == "Monte Carlo":
            return graph.random_holdout(
                **holdouts_kwargs,
                random_state=random_state + holdout_number,
                verbose=False,
            )
        if evaluation_schema == "Kfold":
            return graph.get_edge_prediction_kfold(
                k=number_of_holdouts,
                k_index=holdout_number,
                random_state=random_state,
                verbose=False,
            )
        super().split_graph_following_evaluation_schema(
            graph=graph,
            evaluation_schema=evaluation_schema,
            random_state=random_state,
            holdout_number=holdout_number,
            number_of_holdouts=number_of_holdouts,
            **holdouts_kwargs,
        )

    @staticmethod
    def __iterate_negative_graphs(
        graph: Graph,
        train: Graph,
        test: Graph,
        support: Graph,
        subgraph_of_interest: Optional[Graph],
        random_state: int,
        verbose: bool,
        source_node_types_names: Optional[List[str]],
        destination_node_types_names: Optional[List[str]],
        source_edge_types_names: Optional[List[str]],
        destination_edge_types_names: Optional[List[str]],
        source_nodes_prefixes: Optional[List[str]],
        destination_nodes_prefixes: Optional[List[str]],
        validation_unbalance_rates: Tuple[float],
        use_scale_free_distribution: bool,
    ) -> Iterator[Tuple[Graph]]:
        """Return iterator over the negative graphs for evaluation."""
        if subgraph_of_interest is None:
            sampler_graph = graph
        else:
            sampler_graph = subgraph_of_interest

        if not use_scale_free_distribution:
            warnings.warn(
                "Please do be advised that you have DISABLED the use of scale free sampling "
                "for the negative edges for the EVALUATION (not the training) "
                "of a model. This is a POOR CHOICE as it will introduce a positive bias "
                "as edges sampled uniformely have a significantly different node degree "
                "distribution than the positive edges in the graph, and are therefore much easier "
                "to predict. The only case where it makes sense to use this parameter is when "
                "evaluating how strongly this bias would have affected your task. "
                "DO NOT USE THIS CONFIGURATION IN ANY OTHER USE CASE."
            )

        train_size = train.get_number_of_edges() / (
            train.get_number_of_edges() + test.get_number_of_edges()
        )

        return (
            sampler_graph.sample_negative_graph(
                number_of_negative_samples=int(
                    math.ceil(sampler_graph.get_number_of_edges() * unbalance_rate)
                ),
                random_state=random_state * (i + 1),
                use_scale_free_distribution=use_scale_free_distribution,
                source_node_types_names=source_node_types_names,
                destination_node_types_names=destination_node_types_names,
                source_edge_types_names=source_edge_types_names,
                destination_edge_types_names=destination_edge_types_names,
                source_nodes_prefixes=source_nodes_prefixes,
                destination_nodes_prefixes=destination_nodes_prefixes,
                support=support,
                graph_to_avoid=graph,
            ).random_holdout(
                train_size=train_size,
                random_state=random_state,
                verbose=False,
            )
            for i, unbalance_rate in tqdm(
                enumerate(validation_unbalance_rates),
                disable=not verbose or len(validation_unbalance_rates) == 1,
                total=len(validation_unbalance_rates),
                leave=False,
                dynamic_ncols=True,
                desc="Building negative graphs for evaluation",
            )
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
        source_node_types_names: Optional[List[str]] = None,
        destination_node_types_names: Optional[List[str]] = None,
        source_edge_types_names: Optional[List[str]] = None,
        destination_edge_types_names: Optional[List[str]] = None,
        source_nodes_prefixes: Optional[List[str]] = None,
        destination_nodes_prefixes: Optional[List[str]] = None,
        validation_unbalance_rates: Tuple[float] = (1.0,),
        use_scale_free_distribution: bool = True,
    ) -> Dict[str, Any]:
        """Return additional custom parameters for the current holdout."""
        return dict(
            negative_graphs=list(
                cls.__iterate_negative_graphs(
                    graph=graph,
                    train=train,
                    test=test,
                    support=support,
                    subgraph_of_interest=subgraph_of_interest,
                    random_state=random_state,
                    verbose=verbose,
                    source_node_types_names=source_node_types_names,
                    destination_node_types_names=destination_node_types_names,
                    source_edge_types_names=source_edge_types_names,
                    destination_edge_types_names=destination_edge_types_names,
                    source_nodes_prefixes=source_nodes_prefixes,
                    destination_nodes_prefixes=destination_nodes_prefixes,
                    validation_unbalance_rates=validation_unbalance_rates,
                    use_scale_free_distribution=use_scale_free_distribution,
                )
            )
        )

    def _evaluate(
        self,
        graph: Graph,
        train: Graph,
        test: Graph,
        support: Graph,
        node_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[str, pd.DataFrame, np.ndarray]]]
        ] = None,
        node_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[str, pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[str, pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[str, pd.DataFrame, np.ndarray]]]
        ] = None,
        subgraph_of_interest: Optional[Graph] = None,
        random_state: int = 42,
        verbose: bool = True,
        negative_graphs: Optional[List[Tuple[Graph]]] = None,
        source_node_types_names: Optional[List[str]] = None,
        destination_node_types_names: Optional[List[str]] = None,
        source_edge_types_names: Optional[List[str]] = None,
        destination_edge_types_names: Optional[List[str]] = None,
        source_nodes_prefixes: Optional[List[str]] = None,
        destination_nodes_prefixes: Optional[List[str]] = None,
        validation_unbalance_rates: Tuple[float] = (1.0,),
        use_scale_free_distribution: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return model evaluation on the provided graphs."""
        performance = []

        train_size = (
            train.get_number_of_directed_edges() / graph.get_number_of_directed_edges()
        )

        train_predict_proba = self.predict_proba(
            train,
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_type_features=edge_type_features,
            edge_features=edge_features,
        )

        test_predict_proba = self.predict_proba(
            test,
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_type_features=edge_type_features,
            edge_features=edge_features,
        )

        negative_graph_iterator = (
            self.__iterate_negative_graphs(
                graph=graph,
                train=train,
                test=test,
                support=support,
                subgraph_of_interest=subgraph_of_interest,
                random_state=random_state,
                verbose=verbose,
                source_node_types_names=source_node_types_names,
                destination_node_types_names=destination_node_types_names,
                source_edge_types_names=source_edge_types_names,
                destination_edge_types_names=destination_edge_types_names,
                source_nodes_prefixes=source_nodes_prefixes,
                destination_nodes_prefixes=destination_nodes_prefixes,
                validation_unbalance_rates=validation_unbalance_rates,
                use_scale_free_distribution=use_scale_free_distribution,
            )
            if negative_graphs is None
            else negative_graphs
        )

        for unbalance_rate, (negative_train, negative_test) in tqdm(
            zip(validation_unbalance_rates, negative_graph_iterator),
            disable=not verbose or len(validation_unbalance_rates) == 1,
            total=len(validation_unbalance_rates),
            leave=False,
            dynamic_ncols=True,
            desc="Evaluating on unbalances",
        ):
            for evaluation_mode, (existent_predict_proba, non_existent_graph) in (
                ("train", (train_predict_proba, negative_train)),
                ("test", (test_predict_proba, negative_test)),
            ):
                non_existent_predict_proba = self.predict_proba(
                    non_existent_graph,
                    support=support,
                    node_features=node_features,
                    node_type_features=node_type_features,
                    edge_type_features=edge_type_features,
                    edge_features=edge_features,
                )

                if (
                    len(non_existent_predict_proba.shape) > 1
                    and non_existent_predict_proba.shape[1] > 1
                ):
                    non_existent_predict_proba = non_existent_predict_proba[:, 1]

                predict_proba = np.concatenate(
                    (existent_predict_proba, non_existent_predict_proba)
                )

                labels = np.concatenate(
                    (
                        np.ones_like(existent_predict_proba, dtype=bool),
                        np.zeros_like(non_existent_predict_proba, dtype=bool),
                    )
                )

                performance.append(
                    {
                        "evaluation_mode": evaluation_mode,
                        "train_size": train_size,
                        "validation_unbalance_rate": unbalance_rate,
                        "use_scale_free_distribution": use_scale_free_distribution,
                        **self.evaluate_predictions(
                            labels,
                            predict_proba,
                        ),
                        **self.evaluate_prediction_probabilities(
                            labels,
                            predict_proba,
                        ),
                    }
                )

        return performance

    def predict(
        self,
        graph: Graph,
        support: Optional[Graph] = None,
        node_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        node_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_features: Optional[
            Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]
        ] = None,
        return_predictions_dataframe: bool = False,
        return_edge_type_names: bool = True,
        return_node_names: bool = True,
    ) -> Union[np.ndarray, pd.DataFrame, Iterator[np.ndarray]]:
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
        edge_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        edge_features: Optional[Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]] = None
            The edge features to use.
        return_predictions_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the predictions is returned as it weights much less.
        return_edge_type_names: bool = True
            Whether to return edge type names when returning the prediction DataFrame.
            This value is ignored when either the values to be returned to the user
            are not requested to be a DataFrame or the graph does not have edge types.
        return_node_names: bool = True
            Whether to return node names when returning the prediction DataFrame.
            This value is ignored when the values to be returned the user has not
            requested for a prediction dataframe to be returned.
        """
        if graph.has_edges():
            predictions = (
                super()
                .predict(
                    graph,
                    support=support,
                    node_features=node_features,
                    node_type_features=node_type_features,
                    edge_type_features=edge_type_features,
                    edge_features=self.edge_features_check(edge_features),
                )
                .flatten()
            )
        else:
            predictions = np.array([])

        if return_predictions_dataframe:
            predictions = pd.DataFrame(
                {
                    "predictions": predictions,
                    "sources": graph.get_source_names(directed=True)
                    if return_node_names
                    else graph.get_directed_source_node_ids(),
                    "destinations": graph.get_destination_names(directed=True)
                    if return_node_names
                    else graph.get_directed_destination_node_ids(),
                    **(
                        {
                            "edge_types": graph.get_directed_edge_type_names()
                            if return_edge_type_names
                            else graph.get_directed_edge_type_ids()
                        }
                        if graph.has_edge_types()
                        else {}
                    ),
                },
            )

        return predictions

    def predict_bipartite_graph_from_edge_node_ids(
        self,
        graph: Graph,
        source_node_ids: List[int],
        destination_node_ids: List[int],
        support: Optional[Graph] = None,
        node_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        node_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_features: Optional[
            Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]
        ] = None,
        return_predictions_dataframe: bool = False,
        return_edge_type_names: bool = True,
        return_node_names: bool = True,
    ) -> Union[np.ndarray, pd.DataFrame, Iterator[np.ndarray]]:
        """Execute predictions on the provided graph bipartite portion.

        Parameters
        --------------------
        graph: Graph
            The graph from which to extract the edges.
        source_node_ids: List[int]
            The source nodes of the bipartite graph.
        destination_node_ids: List[int]
            The destination nodes of the bipartite graph.
        support: Optional[Graph] = None
            The graph describiding the topological structure that
            includes also the above graph. This parameter
            is mostly useful for topological classifiers
            such as Graph Convolutional Networks.
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node features to use.
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        edge_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        edge_features: Optional[Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]] = None
            The edge features to use.
        return_predictions_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the predictions is returned as it weights much less.
        return_edge_type_names: bool = True
            Whether to return edge type names when returning the prediction DataFrame.
            This value is ignored when either the values to be returned to the user
            are not requested to be a DataFrame or the graph does not have edge types.
        return_node_names: bool = True
            Whether to return node names when returning the prediction DataFrame.
            This value is ignored when the values to be returned the user has not
            requested for a prediction dataframe to be returned.
        """
        return self.predict(
            graph.build_bipartite_graph_from_edge_node_ids(
                source_node_ids=source_node_ids,
                destination_node_ids=destination_node_ids,
                directed=True,
            ),
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_type_features=edge_type_features,
            edge_features=edge_features,
            return_predictions_dataframe=return_predictions_dataframe,
            return_edge_type_names=return_edge_type_names,
            return_node_names=return_node_names,
        )

    def predict_bipartite_graph_from_edge_node_names(
        self,
        graph: Graph,
        source_node_names: List[str],
        destination_node_names: List[str],
        support: Optional[Graph] = None,
        node_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        node_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_features: Optional[
            Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]
        ] = None,
        return_predictions_dataframe: bool = False,
        return_edge_type_names: bool = True,
        return_node_names: bool = True,
    ) -> Union[np.ndarray, pd.DataFrame, Iterator[np.ndarray]]:
        """Execute predictions on the provided graph bipartite portion.

        Parameters
        --------------------
        graph: Graph
            The graph from which to extract the edges.
        source_node_names: List[str]
            The source nodes of the bipartite graph.
        destination_node_names: List[str]
            The destination nodes of the bipartite graph.
        support: Optional[Graph] = None
            The graph describiding the topological structure that
            includes also the above graph. This parameter
            is mostly useful for topological classifiers
            such as Graph Convolutional Networks.
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node features to use.
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        edge_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        edge_features: Optional[Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]] = None
            The edge features to use.
        return_predictions_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the predictions is returned as it weights much less.
        return_edge_type_names: bool = True
            Whether to return edge type names when returning the prediction DataFrame.
            This value is ignored when either the values to be returned to the user
            are not requested to be a DataFrame or the graph does not have edge types.
        return_node_names: bool = True
            Whether to return node names when returning the prediction DataFrame.
            This value is ignored when the values to be returned the user has not
            requested for a prediction dataframe to be returned.
        """
        return self.predict(
            graph.build_bipartite_graph_from_edge_node_names(
                source_node_names=source_node_names,
                destination_node_names=destination_node_names,
                directed=True,
            ),
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_type_features=edge_type_features,
            edge_features=edge_features,
            return_predictions_dataframe=return_predictions_dataframe,
            return_edge_type_names=return_edge_type_names,
            return_node_names=return_node_names,
        )

    def predict_bipartite_graph_from_edge_node_prefixes(
        self,
        graph: Graph,
        source_node_prefixes: List[str],
        destination_node_prefixes: List[str],
        support: Optional[Graph] = None,
        node_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        node_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_features: Optional[
            Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]
        ] = None,
        return_predictions_dataframe: bool = False,
        return_edge_type_names: bool = True,
        return_node_names: bool = True,
    ) -> Union[np.ndarray, pd.DataFrame, Iterator[np.ndarray]]:
        """Execute predictions on the provided graph bipartite portion.

        Parameters
        --------------------
        graph: Graph
            The graph from which to extract the edges.
        source_node_prefixes: List[str]
            The source node prefixes of the bipartite graph.
        destination_node_prefixes: List[str]
            The destination node prefixes of the bipartite graph.
        support: Optional[Graph] = None
            The graph describiding the topological structure that
            includes also the above graph. This parameter
            is mostly useful for topological classifiers
            such as Graph Convolutional Networks.
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node features to use.
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        edge_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        edge_features: Optional[Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]] = None
            The edge features to use.
        return_predictions_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the predictions is returned as it weights much less.
        return_edge_type_names: bool = True
            Whether to return edge type names when returning the prediction DataFrame.
            This value is ignored when either the values to be returned to the user
            are not requested to be a DataFrame or the graph does not have edge types.
        return_node_names: bool = True
            Whether to return node names when returning the prediction DataFrame.
            This value is ignored when the values to be returned the user has not
            requested for a prediction dataframe to be returned.
        """
        return self.predict(
            graph.build_bipartite_graph_from_edge_node_prefixes(
                source_node_prefixes=source_node_prefixes,
                destination_node_prefixes=destination_node_prefixes,
                directed=True,
            ),
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_type_features=edge_type_features,
            edge_features=edge_features,
            return_predictions_dataframe=return_predictions_dataframe,
            return_edge_type_names=return_edge_type_names,
            return_node_names=return_node_names,
        )

    def predict_bipartite_graph_from_edge_node_types(
        self,
        graph: Graph,
        source_node_types: List[str],
        destination_node_types: List[str],
        support: Optional[Graph] = None,
        node_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        node_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_features: Optional[
            Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]
        ] = None,
        return_predictions_dataframe: bool = False,
        return_edge_type_names: bool = True,
        return_node_names: bool = True,
    ) -> Union[np.ndarray, pd.DataFrame, Iterator[np.ndarray]]:
        """Execute predictions on the provided graph bipartite portion.

        Parameters
        --------------------
        graph: Graph
            The graph from which to extract the edges.
        source_node_types: List[str]
            The source node prefixes of the bipartite graph.
        destination_node_types: List[str]
            The destination node prefixes of the bipartite graph.
        support: Optional[Graph] = None
            The graph describiding the topological structure that
            includes also the above graph. This parameter
            is mostly useful for topological classifiers
            such as Graph Convolutional Networks.
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node features to use.
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        edge_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        edge_features: Optional[Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]] = None
            The edge features to use.
        return_predictions_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the predictions is returned as it weights much less.
        return_edge_type_names: bool = True
            Whether to return edge type names when returning the prediction DataFrame.
            This value is ignored when either the values to be returned to the user
            are not requested to be a DataFrame or the graph does not have edge types.
        return_node_names: bool = True
            Whether to return node names when returning the prediction DataFrame.
            This value is ignored when the values to be returned the user has not
            requested for a prediction dataframe to be returned.
        """
        return self.predict(
            graph.build_bipartite_graph_from_edge_node_types(
                source_node_types=source_node_types,
                destination_node_types=destination_node_types,
                directed=True,
            ),
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_type_features=edge_type_features,
            edge_features=edge_features,
            return_predictions_dataframe=return_predictions_dataframe,
            return_edge_type_names=return_edge_type_names,
            return_node_names=return_node_names,
        )

    def predict_clique_graph_from_node_ids(
        self,
        graph: Graph,
        node_ids: List[int],
        support: Optional[Graph] = None,
        node_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        node_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_features: Optional[
            Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]
        ] = None,
        return_predictions_dataframe: bool = False,
        return_edge_type_names: bool = True,
        return_node_names: bool = True,
    ) -> Union[np.ndarray, pd.DataFrame, Iterator[np.ndarray]]:
        """Execute predictions on the provided graph bipartite portion.

        Parameters
        --------------------
        graph: Graph
            The graph from which to extract the edges.
        node_ids: List[int]
            The nodes of the bipartite graph.
        support: Optional[Graph] = None
            The graph describiding the topological structure that
            includes also the above graph. This parameter
            is mostly useful for topological classifiers
            such as Graph Convolutional Networks.
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node features to use.
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        edge_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        edge_features: Optional[Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]] = None
            The edge features to use.
        return_predictions_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the predictions is returned as it weights much less.
        return_edge_type_names: bool = True
            Whether to return edge type names when returning the prediction DataFrame.
            This value is ignored when either the values to be returned to the user
            are not requested to be a DataFrame or the graph does not have edge types.
        return_node_names: bool = True
            Whether to return node names when returning the prediction DataFrame.
            This value is ignored when the values to be returned the user has not
            requested for a prediction dataframe to be returned.
        """
        return self.predict(
            graph.build_clique_graph_from_node_ids(node_ids=node_ids, directed=True),
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_type_features=edge_type_features,
            edge_features=edge_features,
            return_predictions_dataframe=return_predictions_dataframe,
            return_edge_type_names=return_edge_type_names,
            return_node_names=return_node_names,
        )

    def predict_clique_graph_from_node_names(
        self,
        graph: Graph,
        node_names: List[str],
        support: Optional[Graph] = None,
        node_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        node_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_features: Optional[
            Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]
        ] = None,
        return_predictions_dataframe: bool = False,
        return_edge_type_names: bool = True,
        return_node_names: bool = True,
    ) -> Union[np.ndarray, pd.DataFrame, Iterator[np.ndarray]]:
        """Execute predictions on the provided graph bipartite portion.

        Parameters
        --------------------
        graph: Graph
            The graph from which to extract the edges.
        node_names: List[str]
            The nodes of the bipartite graph.
        support: Optional[Graph] = None
            The graph describiding the topological structure that
            includes also the above graph. This parameter
            is mostly useful for topological classifiers
            such as Graph Convolutional Networks.
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node features to use.
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        edge_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        edge_features: Optional[Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]] = None
            The edge features to use.
        return_predictions_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the predictions is returned as it weights much less.
        return_edge_type_names: bool = True
            Whether to return edge type names when returning the prediction DataFrame.
            This value is ignored when either the values to be returned to the user
            are not requested to be a DataFrame or the graph does not have edge types.
        return_node_names: bool = True
            Whether to return node names when returning the prediction DataFrame.
            This value is ignored when the values to be returned the user has not
            requested for a prediction dataframe to be returned.
        """
        return self.predict(
            graph.build_clique_graph_from_node_names(
                node_names=node_names, directed=True
            ),
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_type_features=edge_type_features,
            edge_features=edge_features,
            return_predictions_dataframe=return_predictions_dataframe,
            return_edge_type_names=return_edge_type_names,
            return_node_names=return_node_names,
        )

    def predict_clique_graph_from_node_prefixes(
        self,
        graph: Graph,
        node_prefixes: List[str],
        support: Optional[Graph] = None,
        node_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        node_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_features: Optional[
            Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]
        ] = None,
        return_predictions_dataframe: bool = False,
        return_edge_type_names: bool = True,
        return_node_names: bool = True,
    ) -> Union[np.ndarray, pd.DataFrame, Iterator[np.ndarray]]:
        """Execute predictions on the provided graph bipartite portion.

        Parameters
        --------------------
        graph: Graph
            The graph from which to extract the edges.
        node_prefixes: List[str]
            The node prefixes of the bipartite graph.
        support: Optional[Graph] = None
            The graph describiding the topological structure that
            includes also the above graph. This parameter
            is mostly useful for topological classifiers
            such as Graph Convolutional Networks.
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node features to use.
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        edge_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        edge_features: Optional[Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]] = None
            The edge features to use.
        return_predictions_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the predictions is returned as it weights much less.
        return_edge_type_names: bool = True
            Whether to return edge type names when returning the prediction DataFrame.
            This value is ignored when either the values to be returned to the user
            are not requested to be a DataFrame or the graph does not have edge types.
        return_node_names: bool = True
            Whether to return node names when returning the prediction DataFrame.
            This value is ignored when the values to be returned the user has not
            requested for a prediction dataframe to be returned.
        """
        return self.predict(
            graph.build_clique_graph_from_node_prefixes(
                node_prefixes=node_prefixes, directed=True
            ),
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_type_features=edge_type_features,
            edge_features=edge_features,
            return_predictions_dataframe=return_predictions_dataframe,
            return_edge_type_names=return_edge_type_names,
            return_node_names=return_node_names,
        )

    def predict_clique_graph_from_node_type_names(
        self,
        graph: Graph,
        node_type_names: List[str],
        support: Optional[Graph] = None,
        node_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        node_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_features: Optional[
            Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]
        ] = None,
        return_predictions_dataframe: bool = False,
        return_edge_type_names: bool = True,
        return_node_names: bool = True,
    ) -> Union[np.ndarray, pd.DataFrame, Iterator[np.ndarray]]:
        """Execute predictions on the provided graph bipartite portion.

        Parameters
        --------------------
        graph: Graph
            The graph from which to extract the edges.
        node_type_names: List[str]
            The node prefixes of the bipartite graph.
        support: Optional[Graph] = None
            The graph describiding the topological structure that
            includes also the above graph. This parameter
            is mostly useful for topological classifiers
            such as Graph Convolutional Networks.
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node features to use.
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        edge_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        edge_features: Optional[Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]] = None
            The edge features to use.
        return_predictions_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the predictions is returned as it weights much less.
        return_edge_type_names: bool = True
            Whether to return edge type names when returning the prediction DataFrame.
            This value is ignored when either the values to be returned to the user
            are not requested to be a DataFrame or the graph does not have edge types.
        return_node_names: bool = True
            Whether to return node names when returning the prediction DataFrame.
            This value is ignored when the values to be returned the user has not
            requested for a prediction dataframe to be returned.
        """
        return self.predict(
            graph.build_clique_graph_from_node_type_names(
                node_type_names=node_type_names, directed=True
            ),
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_type_features=edge_type_features,
            edge_features=edge_features,
            return_predictions_dataframe=return_predictions_dataframe,
            return_edge_type_names=return_edge_type_names,
            return_node_names=return_node_names,
        )

    def predict_proba(
        self,
        graph: Graph,
        support: Optional[Graph] = None,
        node_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        node_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_features: Optional[
            Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]
        ] = None,
        path: Optional[str] = None,
        consume_predictions: bool = False,
        return_predictions_dataframe: bool = False,
        return_edge_type_names: bool = True,
        return_node_names: bool = True,
    ) -> Union[np.ndarray, pd.DataFrame, Iterator[np.ndarray]]:
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
        edge_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        edge_features: Optional[Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]] = None
            The edge features to use.
        path: Optional[str] = None
            The path to the file where to save the predictions.
        return_predictions_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the predictions is returned as it weights much less.
        return_edge_type_names: bool = True
            Whether to return edge type names when returning the prediction DataFrame.
            This value is ignored when either the values to be returned to the user
            are not requested to be a DataFrame or the graph does not have edge types.
        return_node_names: bool = True
            Whether to return node names when returning the prediction DataFrame.
            This value is ignored when the values to be returned the user has not
            requested for a prediction dataframe to be returned.
        """
        predictions: Union[Iterator[np.ndarray], np.ndarray] = super().predict_proba(
            graph,
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_type_features=edge_type_features,
            edge_features=self.edge_features_check(edge_features),
        )

        if isinstance(predictions, np.ndarray) and path is not None:
            predictions = [predictions]

        if consume_predictions and return_predictions_dataframe:
            raise ValueError(
                "Cannot consume predictions and return a DataFrame at the same time."
            )

        if not consume_predictions:
            prediction_mini_batches = []
        
        if path is not None:
            edge_id = 0
            extension = path.split(".")[-1]
            if extension == "csv":
                separator = ","
            elif extension == "tsv":
                separator = "\t"
            elif extension == "txt":
                separator = " "
            else:
                raise ValueError(
                    f"Unsupported file extension {extension}. "
                    "Please use either csv, tsv or txt."
                )
            with open(path, "w", encoding="utf8") as file:
                file.writeline(
                    separator.join(
                        [
                            "source",
                            "destination",
                            *(("edge_type",) if return_edge_type_names else ()),
                            "prediction",
                        ]
                    )
                )
                for prediction_mini_batch in predictions:
                    if not consume_predictions:
                        prediction_mini_batches.append(prediction_mini_batch)
                    for prediction_score in prediction_mini_batch:
                        if return_node_names:
                            src, dst = graph.get_edge_node_names_from_edge_id(edge_id)

                            # We need to normalize the node names in the unfortunate
                            # case that they contain the selected separator.
                            if separator in src:
                                src = src.replace("\"", "\\\"")
                                src = f"\"{src}\""
                            if separator in dst:
                                dst = dst.replace("\"", "\\\"")
                                dst = f"\"{dst}\""
                        else:
                            src, dst = graph.get_edge_node_ids_from_edge_id(edge_id)
                        file.writeline(
                            separator.join(
                                [
                                    src,
                                    dst,
                                    *(graph.get_edge_type_name_from_edge_id(edge_id) if return_edge_type_names else ()),
                                    prediction_score,
                                ]
                            )
                        )
                        edge_id += 1
                    if consume_predictions:
                        # We make sure that the predictions are consumed
                        # and we don't cause a memory peak by various odd
                        # behaviour of the GC.
                        del prediction_mini_batch
                        gc.collect()

            if consume_predictions:
                predictions = None
            else:
                predictions = prediction_mini_batches

        if not consume_predictions and not isinstance(predictions, np.ndarray):
            if not isinstance(predictions, list):
                predictions = list(predictions)
            predictions = np.concatenate(predictions)

        if return_predictions_dataframe:
            predictions = pd.DataFrame(
                {
                    "predictions": predictions,
                    "sources": graph.get_source_names(directed=True)
                    if return_node_names
                    else graph.get_directed_source_node_ids(),
                    "destinations": graph.get_destination_names(directed=True)
                    if return_node_names
                    else graph.get_directed_destination_node_ids(),
                    **(
                        {
                            "edge_types": graph.get_directed_edge_type_names()
                            if return_edge_type_names
                            else graph.get_directed_edge_type_ids()
                        }
                        if graph.has_edge_types()
                        else {}
                    ),
                },
            )
        
        return predictions

    def predict_proba_bipartite_graph_from_edge_node_ids(
        self,
        graph: Graph,
        source_node_ids: List[int],
        destination_node_ids: List[int],
        support: Optional[Graph] = None,
        node_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        node_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_features: Optional[
            Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]
        ] = None,
        path: Optional[str] = None,
        consume_predictions: bool = False,
        return_predictions_dataframe: bool = False,
        return_edge_type_names: bool = True,
        return_node_names: bool = True,
    ) -> Union[np.ndarray, pd.DataFrame, Iterator[np.ndarray]]:
        """Execute predictions probabilities on the provided graph bipartite portion.

        Parameters
        --------------------
        graph: Graph
            The graph from which to extract the edges.
        source_node_ids: List[int]
            The source nodes of the bipartite graph.
        destination_node_ids: List[int]
            The destination nodes of the bipartite graph.
        support: Optional[Graph] = None
            The graph describiding the topological structure that
            includes also the above graph. This parameter
            is mostly useful for topological classifiers
            such as Graph Convolutional Networks.
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node features to use.
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        edge_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        edge_features: Optional[Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]] = None
            The edge features to use.
        path: Optional[str] = None
            The path to the file where to save the predictions.    
        consume_predictions: bool = False
            Whether to consume the predictions iterator as it is being used
            instead of collecting it into a list. This is useful when the
            predictions are just being stored to disk and the graph size
            is large enough to be problematic to store in main memory.
        return_predictions_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the predictions is returned as it weights much less.
        return_edge_type_names: bool = True
            Whether to return edge type names when returning the prediction DataFrame.
            This value is ignored when either the values to be returned to the user
            are not requested to be a DataFrame or the graph does not have edge types.
        return_node_names: bool = True
            Whether to return node names when returning the prediction DataFrame.
            This value is ignored when the values to be returned the user has not
            requested for a prediction dataframe to be returned.
        """
        return self.predict_proba(
            graph.build_bipartite_graph_from_edge_node_ids(
                source_node_ids=source_node_ids,
                destination_node_ids=destination_node_ids,
                directed=True,
            ),
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_type_features=edge_type_features,
            edge_features=edge_features,
            path=path,
            consume_predictions=consume_predictions,
            return_predictions_dataframe=return_predictions_dataframe,
            return_edge_type_names=return_edge_type_names,
            return_node_names=return_node_names,
        )

    def predict_proba_bipartite_graph_from_edge_node_names(
        self,
        graph: Graph,
        source_node_names: List[str],
        destination_node_names: List[str],
        support: Optional[Graph] = None,
        node_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        node_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_features: Optional[
            Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]
        ] = None,
        path: Optional[str] = None,
        consume_predictions: bool = False,
        return_predictions_dataframe: bool = False,
        return_edge_type_names: bool = True,
        return_node_names: bool = True,
    ) -> Union[np.ndarray, pd.DataFrame, Iterator[np.ndarray]]:
        """Execute predictions probabilities on the provided graph bipartite portion.

        Parameters
        --------------------
        graph: Graph
            The graph from which to extract the edges.
        source_node_names: List[str]
            The source nodes of the bipartite graph.
        destination_node_names: List[str]
            The destination nodes of the bipartite graph.
        support: Optional[Graph] = None
            The graph describiding the topological structure that
            includes also the above graph. This parameter
            is mostly useful for topological classifiers
            such as Graph Convolutional Networks.
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node features to use.
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        edge_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        edge_features: Optional[Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]] = None
            The edge features to use.
        path: Optional[str] = None
            The path to the file where to save the predictions.    
        consume_predictions: bool = False
            Whether to consume the predictions iterator as it is being used
            instead of collecting it into a list. This is useful when the
            predictions are just being stored to disk and the graph size
            is large enough to be problematic to store in main memory.
        return_predictions_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the predictions is returned as it weights much less.
        return_edge_type_names: bool = True
            Whether to return edge type names when returning the prediction DataFrame.
            This value is ignored when either the values to be returned to the user
            are not requested to be a DataFrame or the graph does not have edge types.
        return_node_names: bool = True
            Whether to return node names when returning the prediction DataFrame.
            This value is ignored when the values to be returned the user has not
            requested for a prediction dataframe to be returned.
        """
        return self.predict_proba(
            graph.build_bipartite_graph_from_edge_node_names(
                source_node_names=source_node_names,
                destination_node_names=destination_node_names,
                directed=True,
            ),
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_type_features=edge_type_features,
            edge_features=edge_features,
            path=path,
            consume_predictions=consume_predictions,
            return_predictions_dataframe=return_predictions_dataframe,
            return_edge_type_names=return_edge_type_names,
            return_node_names=return_node_names,
        )

    def predict_proba_bipartite_graph_from_edge_node_prefixes(
        self,
        graph: Graph,
        source_node_prefixes: List[str],
        destination_node_prefixes: List[str],
        support: Optional[Graph] = None,
        node_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        node_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_features: Optional[
            Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]
        ] = None,
        path: Optional[str] = None,
        consume_predictions: bool = False,
        return_predictions_dataframe: bool = False,
        return_edge_type_names: bool = True,
        return_node_names: bool = True,
    ) -> Union[np.ndarray, pd.DataFrame, Iterator[np.ndarray]]:
        """Execute predictions probabilities on the provided graph bipartite portion.

        Parameters
        --------------------
        graph: Graph
            The graph from which to extract the edges.
        source_node_prefixes: List[str]
            The source node prefixes of the bipartite graph.
        destination_node_prefixes: List[str]
            The destination node prefixes of the bipartite graph.
        support: Optional[Graph] = None
            The graph describiding the topological structure that
            includes also the above graph. This parameter
            is mostly useful for topological classifiers
            such as Graph Convolutional Networks.
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node features to use.
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        edge_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        edge_features: Optional[Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]] = None
            The edge features to use.
        path: Optional[str] = None
            The path to the file where to save the predictions.    
        consume_predictions: bool = False
            Whether to consume the predictions iterator as it is being used
            instead of collecting it into a list. This is useful when the
            predictions are just being stored to disk and the graph size
            is large enough to be problematic to store in main memory.
        return_predictions_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the predictions is returned as it weights much less.
        return_edge_type_names: bool = True
            Whether to return edge type names when returning the prediction DataFrame.
            This value is ignored when either the values to be returned to the user
            are not requested to be a DataFrame or the graph does not have edge types.
        return_node_names: bool = True
            Whether to return node names when returning the prediction DataFrame.
            This value is ignored when the values to be returned the user has not
            requested for a prediction dataframe to be returned.
        """
        return self.predict_proba(
            graph.build_bipartite_graph_from_edge_node_prefixes(
                source_node_prefixes=source_node_prefixes,
                destination_node_prefixes=destination_node_prefixes,
                directed=True,
            ),
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_type_features=edge_type_features,
            edge_features=edge_features,
            path=path,
            consume_predictions=consume_predictions,
            return_predictions_dataframe=return_predictions_dataframe,
            return_edge_type_names=return_edge_type_names,
            return_node_names=return_node_names,
        )

    def predict_proba_bipartite_graph_from_edge_node_types(
        self,
        graph: Graph,
        source_node_types: List[str],
        destination_node_types: List[str],
        support: Optional[Graph] = None,
        node_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        node_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_features: Optional[
            Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]
        ] = None,
        path: Optional[str] = None,
        consume_predictions: bool = False,
        return_predictions_dataframe: bool = False,
        return_edge_type_names: bool = True,
        return_node_names: bool = True,
    ) -> Union[np.ndarray, pd.DataFrame, Iterator[np.ndarray]]:
        """Execute predictions probabilities on the provided graph bipartite portion.

        Parameters
        --------------------
        graph: Graph
            The graph from which to extract the edges.
        source_node_types: List[str]
            The source node prefixes of the bipartite graph.
        destination_node_types: List[str]
            The destination node prefixes of the bipartite graph.
        support: Optional[Graph] = None
            The graph describiding the topological structure that
            includes also the above graph. This parameter
            is mostly useful for topological classifiers
            such as Graph Convolutional Networks.
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node features to use.
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        edge_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        edge_features: Optional[Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]] = None
            The edge features to use.
        path: Optional[str] = None
            The path to the file where to save the predictions.    
        consume_predictions: bool = False
            Whether to consume the predictions iterator as it is being used
            instead of collecting it into a list. This is useful when the
            predictions are just being stored to disk and the graph size
            is large enough to be problematic to store in main memory.
        return_predictions_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the predictions is returned as it weights much less.
        return_edge_type_names: bool = True
            Whether to return edge type names when returning the prediction DataFrame.
            This value is ignored when either the values to be returned to the user
            are not requested to be a DataFrame or the graph does not have edge types.
        return_node_names: bool = True
            Whether to return node names when returning the prediction DataFrame.
            This value is ignored when the values to be returned the user has not
            requested for a prediction dataframe to be returned.
        """
        return self.predict_proba(
            graph.build_bipartite_graph_from_edge_node_types(
                source_node_types=source_node_types,
                destination_node_types=destination_node_types,
                directed=True,
            ),
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_type_features=edge_type_features,
            edge_features=edge_features,
            path=path,
            consume_predictions=consume_predictions,
            return_predictions_dataframe=return_predictions_dataframe,
            return_edge_type_names=return_edge_type_names,
            return_node_names=return_node_names,
        )

    def predict_proba_clique_graph_from_node_ids(
        self,
        graph: Graph,
        node_ids: List[int],
        support: Optional[Graph] = None,
        node_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        node_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_features: Optional[
            Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]
        ] = None,
        path: Optional[str] = None,
        consume_predictions: bool = False,
        return_predictions_dataframe: bool = False,
        return_edge_type_names: bool = True,
        return_node_names: bool = True,
    ) -> Union[np.ndarray, pd.DataFrame, Iterator[np.ndarray]]:
        """Execute predictions probabilities on the provided graph bipartite portion.

        Parameters
        --------------------
        graph: Graph
            The graph from which to extract the edges.
        node_ids: List[int]
            The nodes of the bipartite graph.
        support: Optional[Graph] = None
            The graph describiding the topological structure that
            includes also the above graph. This parameter
            is mostly useful for topological classifiers
            such as Graph Convolutional Networks.
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node features to use.
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        edge_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        edge_features: Optional[Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]] = None
            The edge features to use.
        path: Optional[str] = None
            The path to the file where to save the predictions.    
        consume_predictions: bool = False
            Whether to consume the predictions iterator as it is being used
            instead of collecting it into a list. This is useful when the
            predictions are just being stored to disk and the graph size
            is large enough to be problematic to store in main memory.
        return_predictions_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the predictions is returned as it weights much less.
        return_edge_type_names: bool = True
            Whether to return edge type names when returning the prediction DataFrame.
            This value is ignored when either the values to be returned to the user
            are not requested to be a DataFrame or the graph does not have edge types.
        return_node_names: bool = True
            Whether to return node names when returning the prediction DataFrame.
            This value is ignored when the values to be returned the user has not
            requested for a prediction dataframe to be returned.
        """
        return self.predict_proba(
            graph.build_clique_graph_from_node_ids(node_ids=node_ids, directed=True),
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_type_features=edge_type_features,
            edge_features=edge_features,
            path=path,
            consume_predictions=consume_predictions,
            return_predictions_dataframe=return_predictions_dataframe,
            return_edge_type_names=return_edge_type_names,
            return_node_names=return_node_names,
        )

    def predict_proba_clique_graph_from_node_names(
        self,
        graph: Graph,
        node_names: List[str],
        support: Optional[Graph] = None,
        node_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        node_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_features: Optional[
            Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]
        ] = None,
        path: Optional[str] = None,
        consume_predictions: bool = False,
        return_predictions_dataframe: bool = False,
        return_edge_type_names: bool = True,
        return_node_names: bool = True,
    ) -> Union[np.ndarray, pd.DataFrame, Iterator[np.ndarray]]:
        """Execute predictions probabilities on the provided graph bipartite portion.

        Parameters
        --------------------
        graph: Graph
            The graph from which to extract the edges.
        node_names: List[str]
            The nodes of the bipartite graph.
        support: Optional[Graph] = None
            The graph describiding the topological structure that
            includes also the above graph. This parameter
            is mostly useful for topological classifiers
            such as Graph Convolutional Networks.
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node features to use.
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        edge_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        edge_features: Optional[Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]] = None
            The edge features to use.
        path: Optional[str] = None
            The path to the file where to save the predictions.    
        consume_predictions: bool = False
            Whether to consume the predictions iterator as it is being used
            instead of collecting it into a list. This is useful when the
            predictions are just being stored to disk and the graph size
            is large enough to be problematic to store in main memory.
        return_predictions_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the predictions is returned as it weights much less.
        return_edge_type_names: bool = True
            Whether to return edge type names when returning the prediction DataFrame.
            This value is ignored when either the values to be returned to the user
            are not requested to be a DataFrame or the graph does not have edge types.
        return_node_names: bool = True
            Whether to return node names when returning the prediction DataFrame.
            This value is ignored when the values to be returned the user has not
            requested for a prediction dataframe to be returned.
        """
        return self.predict_proba(
            graph.build_clique_graph_from_node_names(
                node_names=node_names, directed=True
            ),
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_type_features=edge_type_features,
            edge_features=edge_features,
            path=path,
            consume_predictions=consume_predictions,
            return_predictions_dataframe=return_predictions_dataframe,
            return_edge_type_names=return_edge_type_names,
            return_node_names=return_node_names,
        )

    def predict_proba_clique_graph_from_node_prefixes(
        self,
        graph: Graph,
        node_prefixes: List[str],
        support: Optional[Graph] = None,
        node_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        node_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_features: Optional[
            Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]
        ] = None,
        path: Optional[str] = None,
        consume_predictions: bool = False,
        return_predictions_dataframe: bool = False,
        return_edge_type_names: bool = True,
        return_node_names: bool = True,
    ) -> Union[np.ndarray, pd.DataFrame, Iterator[np.ndarray]]:
        """Execute predictions probabilities on the provided graph bipartite portion.

        Parameters
        --------------------
        graph: Graph
            The graph from which to extract the edges.
        node_prefixes: List[str]
            The node prefixes of the bipartite graph.
        support: Optional[Graph] = None
            The graph describiding the topological structure that
            includes also the above graph. This parameter
            is mostly useful for topological classifiers
            such as Graph Convolutional Networks.
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node features to use.
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        edge_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        edge_features: Optional[Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]] = None
            The edge features to use.
        path: Optional[str] = None
            The path to the file where to save the predictions.    
        consume_predictions: bool = False
            Whether to consume the predictions iterator as it is being used
            instead of collecting it into a list. This is useful when the
            predictions are just being stored to disk and the graph size
            is large enough to be problematic to store in main memory.
        return_predictions_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the predictions is returned as it weights much less.
        return_edge_type_names: bool = True
            Whether to return edge type names when returning the prediction DataFrame.
            This value is ignored when either the values to be returned to the user
            are not requested to be a DataFrame or the graph does not have edge types.
        return_node_names: bool = True
            Whether to return node names when returning the prediction DataFrame.
            This value is ignored when the values to be returned the user has not
            requested for a prediction dataframe to be returned.
        """
        return self.predict_proba(
            graph.build_clique_graph_from_node_prefixes(
                node_prefixes=node_prefixes, directed=True
            ),
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_type_features=edge_type_features,
            edge_features=edge_features,
            path=path,
            consume_predictions=consume_predictions,
            return_predictions_dataframe=return_predictions_dataframe,
            return_edge_type_names=return_edge_type_names,
            return_node_names=return_node_names,
        )

    def predict_proba_clique_graph_from_node_type_names(
        self,
        graph: Graph,
        node_type_names: List[str],
        support: Optional[Graph] = None,
        node_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        node_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_features: Optional[
            Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]
        ] = None,
        path: Optional[str] = None,
        consume_predictions: bool = False,
        return_predictions_dataframe: bool = False,
        return_edge_type_names: bool = True,
        return_node_names: bool = True,
    ) -> Union[np.ndarray, pd.DataFrame, Iterator[np.ndarray]]:
        """Execute predictions probabilities on the provided graph bipartite portion.

        Parameters
        --------------------
        graph: Graph
            The graph from which to extract the edges.
        node_type_names: List[str]
            The node prefixes of the bipartite graph.
        support: Optional[Graph] = None
            The graph describiding the topological structure that
            includes also the above graph. This parameter
            is mostly useful for topological classifiers
            such as Graph Convolutional Networks.
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node features to use.
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        edge_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        edge_features: Optional[Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]] = None
            The edge features to use.
        path: Optional[str] = None
            The path to the file where to save the predictions.    
        consume_predictions: bool = False
            Whether to consume the predictions iterator as it is being used
            instead of collecting it into a list. This is useful when the
            predictions are just being stored to disk and the graph size
            is large enough to be problematic to store in main memory.
        return_predictions_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the predictions is returned as it weights much less.
        return_edge_type_names: bool = True
            Whether to return edge type names when returning the prediction DataFrame.
            This value is ignored when either the values to be returned to the user
            are not requested to be a DataFrame or the graph does not have edge types.
        return_node_names: bool = True
            Whether to return node names when returning the prediction DataFrame.
            This value is ignored when the values to be returned the user has not
            requested for a prediction dataframe to be returned.
        """
        return self.predict_proba(
            graph.build_clique_graph_from_node_type_names(
                node_type_names=node_type_names, directed=True
            ),
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_type_features=edge_type_features,
            edge_features=edge_features,
            path=path,
            consume_predictions=consume_predictions,
            return_predictions_dataframe=return_predictions_dataframe,
            return_edge_type_names=return_edge_type_names,
            return_node_names=return_node_names,
        )

    def fit(
        self,
        graph: Graph,
        support: Optional[Graph] = None,
        node_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        node_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_type_features: Optional[
            Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]
        ] = None,
        edge_features: Optional[
            Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]
        ] = None,
    ):
        """Execute fitting on the provided graph.

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
        edge_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        edge_features: Optional[Union[Type[AbstractEdgeFeature], List[Type[AbstractEdgeFeature]]]] = None
            The edge features to use.
        """
        super().fit(
            graph=graph,
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_type_features=edge_type_features,
            edge_features=self.edge_features_check(edge_features),
        )

    def is_using_node_types(self) -> bool:
        """Whether the current model is using node types."""
        return self._is_using_node_type_features or self.requires_node_types()

    def is_using_edge_types(self) -> bool:
        """Whether the current model is using edge types."""
        return self._is_using_edge_type_features or self.requires_edge_types()

    @classmethod
    def task_involves_edge_weights(cls) -> bool:
        """Returns whether the model task involves edge weights."""
        return False

    @classmethod
    def task_involves_edge_types(cls) -> bool:
        """Returns whether the model task involves edge types."""
        return False

    @classmethod
    def task_involves_node_types(cls) -> bool:
        """Returns whether the model task involves node types."""
        return False

    @classmethod
    def task_involves_topology(cls) -> bool:
        """Returns whether the model task involves topology."""
        return True

    @classmethod
    def supports_multilabel_prediction(cls) -> bool:
        """Returns whether the model supports multilabel prediction.

        Implementation details
        ----------------------
        The edge prediction task is, by definition, a binary prediction
        task. Therefore, this method always returns False.
        """
        return False

    def is_multilabel_prediction_task(self) -> bool:
        """Returns whether the model is a multilabel prediction task.

        Implementation details
        ----------------------
        The edge prediction task is, by definition, a binary prediction
        task. Therefore, this method always returns False.
        """
        return False

    @classmethod
    def can_use_node_type_features(cls) -> bool:
        """Returns whether the model can use node type features."""
        return True