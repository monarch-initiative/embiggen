"""Module providing abstract edge prediction model."""
import warnings
from typing import Optional, Union, List, Dict, Any, Tuple, Iterator
import pandas as pd
import numpy as np
import math
from ensmallen import Graph
from tqdm.auto import tqdm
from embiggen.utils.abstract_models import AbstractClassifierModel, abstract_class, format_list


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
        return [
            "Connected Monte Carlo",
            "Monte Carlo",
            "Kfold"
        ]

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
                k=number_of_holdouts,
                k_index=holdout_number,
                random_state=random_state,
                verbose=False
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
        support: Optional[Graph],
        subgraph_of_interest: Optional[Graph],
        random_state: int,
        verbose: bool,
        validation_sample_only_edges_with_heterogeneous_node_types: bool,
        validation_unbalance_rates: Tuple[float],
        use_scale_free_distribution: bool
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

        train_size = (
            train.get_number_of_edges() / (train.get_number_of_edges() + test.get_number_of_edges())
        )

        return (
            sampler_graph.sample_negative_graph(
                number_of_negative_samples=int(
                    math.ceil(sampler_graph.get_number_of_edges()*unbalance_rate)
                ),
                random_state=random_state*(i+1),
                sample_only_edges_with_heterogeneous_node_types=validation_sample_only_edges_with_heterogeneous_node_types,
                use_scale_free_distribution=use_scale_free_distribution,
                support=support,
                graph_to_avoid=graph
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
                desc="Building negative graphs for evaluation"
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
        validation_sample_only_edges_with_heterogeneous_node_types: bool = False,
        validation_unbalance_rates: Tuple[float] = (1.0, ),
        use_scale_free_distribution: bool = True
    ) -> Dict[str, Any]:
        """Return additional custom parameters for the current holdout."""
        return dict(
            negative_graphs=list(cls.__iterate_negative_graphs(
                graph=graph,
                train=train,
                test=test,
                support=support,
                subgraph_of_interest=subgraph_of_interest,
                random_state=random_state,
                verbose=verbose,
                validation_sample_only_edges_with_heterogeneous_node_types=validation_sample_only_edges_with_heterogeneous_node_types,
                validation_unbalance_rates=validation_unbalance_rates,
                use_scale_free_distribution=use_scale_free_distribution
            ))
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
        negative_graphs: Optional[List[Tuple[Graph]]] = None,
        validation_sample_only_edges_with_heterogeneous_node_types: bool = False,
        validation_unbalance_rates: Tuple[float] = (1.0, ),
        use_scale_free_distribution: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return model evaluation on the provided graphs."""
        performance = []

        train_predic_proba = self.predict_proba(
            train,
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_features=edge_features
        )

        test_predict_proba = self.predict_proba(
            test,
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_features=edge_features
        )

        negative_graph_iterator = self.__iterate_negative_graphs(
            graph=graph,
            train=train,
            test=test,
            support=support,
            subgraph_of_interest=subgraph_of_interest,
            random_state=random_state,
            verbose=verbose,
            validation_sample_only_edges_with_heterogeneous_node_types=validation_sample_only_edges_with_heterogeneous_node_types,
            validation_unbalance_rates=validation_unbalance_rates,
            use_scale_free_distribution=use_scale_free_distribution
        ) if negative_graphs is None else negative_graphs

        for unbalance_rate, (negative_train, negative_test) in tqdm(
            zip(validation_unbalance_rates, negative_graph_iterator),
            disable=not verbose or len(validation_unbalance_rates) == 1,
            total=len(validation_unbalance_rates),
            leave=False,
            dynamic_ncols=True,
            desc=f"Evaluating on unbalances"
        ):
            for evaluation_mode, (existent_predict_proba, non_existent_graph) in (
                ("train", (train_predic_proba, negative_train)),
                ("test", (test_predict_proba, negative_test)),
            ):
                non_existent_predict_proba = self.predict_proba(
                    non_existent_graph,
                    support=support,
                    node_features=node_features,
                    node_type_features=node_type_features,
                    edge_features=edge_features
                )

                if len(non_existent_predict_proba.shape) > 1 and non_existent_predict_proba.shape[1] > 1:
                    non_existent_predict_proba = non_existent_predict_proba[:, 1]

                predict_proba = np.concatenate((
                    existent_predict_proba,
                    non_existent_predict_proba
                ))

                labels = np.concatenate((
                    np.ones_like(existent_predict_proba, dtype=bool),
                    np.zeros_like(non_existent_predict_proba, dtype=bool),
                ))

                performance.append({
                    "evaluation_mode": evaluation_mode,
                    "validation_unbalance_rate": unbalance_rate,
                    "use_scale_free_distribution": use_scale_free_distribution,
                    "validation_sample_only_edges_with_heterogeneous_node_types": validation_sample_only_edges_with_heterogeneous_node_types,
                    **self.evaluate_predictions(
                        labels,
                        predict_proba,
                    ),
                    **self.evaluate_prediction_probabilities(
                        labels,
                        predict_proba,
                    ),
                })

        return performance

    def predict(
        self,
        graph: Graph,
        support: Optional[Graph] = None,
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        return_predictions_dataframe: bool = False
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
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The edge features to use.
        return_predictions_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the predictions is returned as it weights much less.
        """
        if edge_features is not None:
            raise NotImplementedError(
                "Currently edge features are not supported in edge prediction models."
            )

        predictions = super().predict(
            graph,
            support=support,
            node_features=node_features,
            node_type_features=node_type_features
        ).flatten()

        if return_predictions_dataframe:
            predictions = pd.DataFrame(
                {
                    "predictions": predictions,
                    "sources": graph.get_directed_source_node_ids(),
                    "destinations": graph.get_directed_destination_node_ids(),
                },
            )

        return predictions

    def predict_bipartite_graph_from_edge_node_ids(
        self,
        graph: Graph,
        source_node_ids: List[int],
        destination_node_ids: List[int],
        support: Optional[Graph] = None,
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        return_predictions_dataframe: bool = False
    ) -> np.ndarray:
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
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The edge features to use.
        return_predictions_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the predictions is returned as it weights much less.
        """
        return self.predict(
            graph.build_bipartite_graph_from_edge_node_ids(
                source_node_ids=source_node_ids,
                destination_node_ids=destination_node_ids,
                directed=True
            ),
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_features=edge_features,
            return_predictions_dataframe=return_predictions_dataframe
        )

    def predict_bipartite_graph_from_edge_node_names(
        self,
        graph: Graph,
        source_node_names: List[str],
        destination_node_names: List[str],
        support: Optional[Graph] = None,
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        return_predictions_dataframe: bool = False
    ) -> np.ndarray:
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
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The edge features to use.
        return_predictions_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the predictions is returned as it weights much less.
        """
        return self.predict(
            graph.build_bipartite_graph_from_edge_node_names(
                source_node_names=source_node_names,
                destination_node_names=destination_node_names,
                directed=True
            ),
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_features=edge_features,
            return_predictions_dataframe=return_predictions_dataframe
        )

    def predict_bipartite_graph_from_edge_node_prefixes(
        self,
        graph: Graph,
        source_node_prefixes: List[str],
        destination_node_prefixes: List[str],
        support: Optional[Graph] = None,
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        return_predictions_dataframe: bool = False
    ) -> np.ndarray:
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
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The edge features to use.
        return_predictions_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the predictions is returned as it weights much less.
        """
        return self.predict(
            graph.build_bipartite_graph_from_edge_node_prefixes(
                source_node_prefixes=source_node_prefixes,
                destination_node_prefixes=destination_node_prefixes,
                directed=True
            ),
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_features=edge_features,
            return_predictions_dataframe=return_predictions_dataframe
        )

    def predict_bipartite_graph_from_edge_node_types(
        self,
        graph: Graph,
        source_node_types: List[str],
        destination_node_types: List[str],
        support: Optional[Graph] = None,
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        return_predictions_dataframe: bool = False
    ) -> np.ndarray:
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
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The edge features to use.
        return_predictions_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the predictions is returned as it weights much less.
        """
        return self.predict(
            graph.build_bipartite_graph_from_edge_node_types(
                source_node_types=source_node_types,
                destination_node_types=destination_node_types,
                directed=True
            ),
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_features=edge_features,
            return_predictions_dataframe=return_predictions_dataframe
        )

    def predict_clique_graph_from_node_ids(
        self,
        graph: Graph,
        node_ids: List[int],
        support: Optional[Graph] = None,
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        return_predictions_dataframe: bool = False
    ) -> np.ndarray:
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
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The edge features to use.
        return_predictions_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the predictions is returned as it weights much less.
        """
        return self.predict(
            graph.build_clique_graph_from_node_ids(
                node_ids=node_ids,
                directed=True
            ),
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_features=edge_features,
            return_predictions_dataframe=return_predictions_dataframe
        )

    def predict_clique_graph_from_node_names(
        self,
        graph: Graph,
        node_names: List[str],
        support: Optional[Graph] = None,
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        return_predictions_dataframe: bool = False
    ) -> np.ndarray:
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
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The edge features to use.
        return_predictions_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the predictions is returned as it weights much less.
        """
        return self.predict(
            graph.build_clique_graph_from_node_names(
                node_names=node_names,
                directed=True
            ),
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_features=edge_features,
            return_predictions_dataframe=return_predictions_dataframe
        )

    def predict_clique_graph_from_node_prefixes(
        self,
        graph: Graph,
        node_prefixes: List[str],
        support: Optional[Graph] = None,
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        return_predictions_dataframe: bool = False
    ) -> np.ndarray:
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
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The edge features to use.
        return_predictions_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the predictions is returned as it weights much less.
        """
        return self.predict(
            graph.build_clique_graph_from_node_prefixes(
                node_prefixes=node_prefixes,
                directed=True
            ),
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_features=edge_features,
            return_predictions_dataframe=return_predictions_dataframe
        )

    def predict_clique_graph_from_node_types(
        self,
        graph: Graph,
        node_types: List[str],
        support: Optional[Graph] = None,
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        return_predictions_dataframe: bool = False
    ) -> np.ndarray:
        """Execute predictions on the provided graph bipartite portion.

        Parameters
        --------------------
        graph: Graph
            The graph from which to extract the edges.
        node_types: List[str]
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
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The edge features to use.
        return_predictions_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the predictions is returned as it weights much less.
        """
        return self.predict(
            graph.build_clique_graph_from_node_types(
                node_types=node_types,
                directed=True
            ),
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_features=edge_features,
            return_predictions_dataframe=return_predictions_dataframe
        )

    def predict_proba(
        self,
        graph: Graph,
        support: Optional[Graph] = None,
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        return_predictions_dataframe: bool = False
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
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The node type features to use.
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The edge features to use.
        return_predictions_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the predictions is returned as it weights much less.
        """
        if edge_features is not None:
            raise NotImplementedError(
                "Currently edge features are not supported in edge prediction models."
            )

        predictions = super().predict_proba(
            graph,
            support=support,
            node_features=node_features,
            node_type_features=node_type_features
        ).flatten()

        if np.isnan(predictions).any():
            raise ValueError(
                "There are NaN values in the predicted probabilities!"
            )

        if return_predictions_dataframe:
            predictions = pd.DataFrame(
                {
                    "predictions": predictions,
                    "sources": graph.get_directed_source_node_ids(),
                    "destinations": graph.get_directed_destination_node_ids(),
                },
            )

        return predictions

    def predict_proba_bipartite_graph_from_edge_node_ids(
        self,
        graph: Graph,
        source_node_ids: List[int],
        destination_node_ids: List[int],
        support: Optional[Graph] = None,
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        return_predictions_dataframe: bool = False
    ) -> np.ndarray:
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
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The edge features to use.
        return_predictions_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the predictions is returned as it weights much less.
        """
        return self.predict_proba(
            graph.build_bipartite_graph_from_edge_node_ids(
                source_node_ids=source_node_ids,
                destination_node_ids=destination_node_ids,
                directed=True
            ),
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_features=edge_features,
            return_predictions_dataframe=return_predictions_dataframe
        )

    def predict_proba_bipartite_graph_from_edge_node_names(
        self,
        graph: Graph,
        source_node_names: List[str],
        destination_node_names: List[str],
        support: Optional[Graph] = None,
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        return_predictions_dataframe: bool = False
    ) -> np.ndarray:
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
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The edge features to use.
        return_predictions_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the predictions is returned as it weights much less.
        """
        return self.predict_proba(
            graph.build_bipartite_graph_from_edge_node_names(
                source_node_names=source_node_names,
                destination_node_names=destination_node_names,
                directed=True
            ),
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_features=edge_features,
            return_predictions_dataframe=return_predictions_dataframe
        )

    def predict_proba_bipartite_graph_from_edge_node_prefixes(
        self,
        graph: Graph,
        source_node_prefixes: List[str],
        destination_node_prefixes: List[str],
        support: Optional[Graph] = None,
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        return_predictions_dataframe: bool = False
    ) -> np.ndarray:
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
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The edge features to use.
        return_predictions_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the predictions is returned as it weights much less.
        """
        return self.predict_proba(
            graph.build_bipartite_graph_from_edge_node_prefixes(
                source_node_prefixes=source_node_prefixes,
                destination_node_prefixes=destination_node_prefixes,
                directed=True
            ),
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_features=edge_features,
            return_predictions_dataframe=return_predictions_dataframe
        )

    def predict_proba_bipartite_graph_from_edge_node_types(
        self,
        graph: Graph,
        source_node_types: List[str],
        destination_node_types: List[str],
        support: Optional[Graph] = None,
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        return_predictions_dataframe: bool = False
    ) -> np.ndarray:
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
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The edge features to use.
        return_predictions_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the predictions is returned as it weights much less.
        """
        return self.predict_proba(
            graph.build_bipartite_graph_from_edge_node_types(
                source_node_types=source_node_types,
                destination_node_types=destination_node_types,
                directed=True
            ),
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_features=edge_features,
            return_predictions_dataframe=return_predictions_dataframe
        )

    def predict_proba_clique_graph_from_node_ids(
        self,
        graph: Graph,
        node_ids: List[int],
        support: Optional[Graph] = None,
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        return_predictions_dataframe: bool = False
    ) -> np.ndarray:
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
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The edge features to use.
        return_predictions_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the predictions is returned as it weights much less.
        """
        return self.predict_proba(
            graph.build_clique_graph_from_node_ids(
                node_ids=node_ids,
                directed=True
            ),
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_features=edge_features,
            return_predictions_dataframe=return_predictions_dataframe
        )

    def predict_proba_clique_graph_from_node_names(
        self,
        graph: Graph,
        node_names: List[str],
        support: Optional[Graph] = None,
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        return_predictions_dataframe: bool = False
    ) -> np.ndarray:
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
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The edge features to use.
        return_predictions_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the predictions is returned as it weights much less.
        """
        return self.predict_proba(
            graph.build_clique_graph_from_node_names(
                node_names=node_names,
                directed=True
            ),
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_features=edge_features,
            return_predictions_dataframe=return_predictions_dataframe
        )

    def predict_proba_clique_graph_from_node_prefixes(
        self,
        graph: Graph,
        node_prefixes: List[str],
        support: Optional[Graph] = None,
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        return_predictions_dataframe: bool = False
    ) -> np.ndarray:
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
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The edge features to use.
        return_predictions_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the predictions is returned as it weights much less.
        """
        return self.predict_proba(
            graph.build_clique_graph_from_node_prefixes(
                node_prefixes=node_prefixes,
                directed=True
            ),
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_features=edge_features,
            return_predictions_dataframe=return_predictions_dataframe
        )

    def predict_proba_clique_graph_from_node_types(
        self,
        graph: Graph,
        node_types: List[str],
        support: Optional[Graph] = None,
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        return_predictions_dataframe: bool = False
    ) -> np.ndarray:
        """Execute predictions probabilities on the provided graph bipartite portion.

        Parameters
        --------------------
        graph: Graph
            The graph from which to extract the edges.
        node_types: List[str]
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
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The edge features to use.
        return_predictions_dataframe: bool = False
            Whether to return a pandas DataFrame, which as indices has the node IDs.
            By default, a numpy array with the predictions is returned as it weights much less.
        """
        return self.predict_proba(
            graph.build_clique_graph_from_node_types(
                node_types=node_types,
                directed=True
            ),
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_features=edge_features,
            return_predictions_dataframe=return_predictions_dataframe
        )

    def fit(
        self,
        graph: Graph,
        support: Optional[Graph] = None,
        node_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        node_type_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None,
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
        edge_features: Optional[Union[pd.DataFrame, np.ndarray, List[Union[pd.DataFrame, np.ndarray]]]] = None
            The edge features to use.
        """
        if edge_features is not None:
            raise NotImplementedError(
                "Currently edge features are not supported in edge prediction models."
            )

        super().fit(
            graph=graph,
            support=support,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_features=None,
        )

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
