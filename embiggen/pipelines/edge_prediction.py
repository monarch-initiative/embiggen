"""Submodule providing pipelines for edge prediction."""
import itertools
from typing import Union, Callable, Tuple, List, Optional, Dict
import pandas as pd
from ensmallen import Graph
from time import time
from tqdm.auto import trange, tqdm
import math
from ensmallen.datasets import get_dataset
from sklearn.base import ClassifierMixin
import copy
import warnings
from ..edge_prediction import SklearnModelEdgePredictionAdapter, is_tensorflow_edge_prediction_method, get_tensorflow_model
from ..utils import is_sklearn_classifier_model, get_sklearn_default_classifier, is_default_sklearn_classifier
from .compute_node_embedding import compute_node_embedding


def get_negative_graphs(
    graph: Graph,
    random_state: int,
    sample_only_edges_with_heterogeneous_node_types: bool,
    unbalance_rate: float,
    train_size: float,
) -> Tuple[Graph, Graph]:
    """Return tuple with training and test negative graphs.

    Implementative details
    ----------------------------
    We generate first one larger graph of negative edges,
    and then we split it into a training and test subgraphs
    using the `random hodout` schema, that is we do not preserve
    the number of connected components as we do not care about them
    since we are not using the topology. Do note that this may
    not be the same case when using a GCN for edge prediction.

    Parameters
    ----------------------------
    graph: Graph
        The graph from which to sample the negative edges.
    random_state: int
        Random state to sample the negatives and execute the random split.
    sample_only_edges_with_heterogeneous_node_types: bool
        Whether to only sample edges between heterogeneous node types.
        This may be useful when training a model to predict between
        two portions in a bipartite graph.
    unbalance_rate: float
        Quantity over over (or under) sampling for the negatives respectively
        to the number of edges in the provided graph.
    train_size: float
        Split size of the training graph.
    """
    # For both the training and the test set we sample the same
    # number of negative edges are there are existing edges in the training and test graphs, respectively.
    # This is done in order to avoid an excessive bias in the evaluation of the edge prediction.
    # Across multiple holdouts, statistically it is unlikely to sample
    # consistently unknown positive edges, and therefore we should be able
    # to remove this negative bias from the evaluation.
    # Of course, this only apply to graphs where we can assume that there is
    # not a massive amount of unknown positive edges.
    negative_graph = graph.sample_negatives(
        number_of_negative_samples=int(
            math.ceil(graph.get_edges_number()*unbalance_rate)),
        random_state=random_state,
        sample_only_edges_with_heterogeneous_node_types=sample_only_edges_with_heterogeneous_node_types,
        verbose=False
    )
    return negative_graph.random_holdout(
        train_size=train_size,
        random_state=random_state,
        verbose=False,
    )


def evaluate_embedding_for_edge_prediction(
    embedding_method: Union[str, Callable[[Graph, int], pd.DataFrame]],
    graphs: Union[Graph, str, List[str], List[Graph]],
    model: Union[str, Callable, ClassifierMixin],
    repositories: Optional[Union[str, List[str]]] = None,
    versions: Optional[Union[str, List[str]]] = None,
    number_of_holdouts: int = 10,
    train_size: float = 0.8,
    unbalance_rates: Tuple[Union[float, str]] = (10.0, 100.0, "auto"),
    holdouts_random_state: int = 42,
    edge_embedding_method: str = "Concatenate",
    edge_types: Optional[List[str]] = None,
    minimum_node_degree: Optional[int] = None,
    maximum_node_degree: Optional[int] = None,
    use_only_cpu: Union[bool, str] = "auto",
    only_execute_embeddings: bool = False,
    embedding_kwargs: Optional[Dict] = None,
    embedding_fit_kwargs: Optional[Dict] = None,
    classifier_kwargs: Optional[Dict] = None,
    classifier_fit_kwargs: Optional[Dict] = None,
    subgraph_of_interest_for_edge_prediction: Optional[Graph] = None,
    sample_only_edges_with_heterogeneous_node_types: bool = False,
    graph_normalization_callback: Callable[[Graph], Graph] = None,
) -> pd.DataFrame:
    """Return the evaluation of an embedding for edge prediction on the given model.

    Parameters
    ----------------------
    embedding_method: Union[str, Callable[[Graph, int], pd.DataFrame]]
        Either the name of the embedding method or a method to be called.
    graphs: Union[Graph, str, List[str], List[Graph]]
        The graph to run the embedding and edge prediction on.
        If a string was provided, we will retrieve the graphs from Ensmallen's repositories.
        If a list was provided, we will iterate on all graphs.
    model: Union[str, Callable, ClassifierMixin]
        Either the name of the model or a method returning a model.
    repositories: Optional[Union[str, List[str]]] = None
        Repositorie(s) of the graphs to be retrieved.
        This only applies when the provided graph(s) are all strings,
        otherwise an exception will be raised.
    versions: Optional[Union[str, List[str]]] = None
        Version(s) of the graphs to be retrieved.
        This only applies when the provided graph(s) are all strings,
        otherwise an exception will be raised.
    number_of_holdouts: int = 10
        The number of the holdouts to run.
    train_size: float = 0.8
        Split size of the training graph.
    unbalance_rates: Tuple[Union[float, str]] = (10.0, 100.0, "auto")
        List of unbalance to evaluate.
        Do note that an unbalance of one is always included.
        With "auto", we use the true unbalance of the graph.
    holdouts_random_state: int = 42
        The seed to be used to reproduce the holdout.
    edge_types: Optional[List[str]] = None
        Edge types to focus the edge prediction on, if any.
    minimum_node_degree: Optional[int] = None
        The minimum node degree of either the source or destination node to be sampled. By default 0.
    maximum_node_degree: Optional[int] = None
        The maximum node degree of either the source or destination node to be sampled. By default, the number of nodes.
    use_only_cpu: Union[bool, str] = "auto",
        Whether to only use CPU.
        Do note that for CBOW and SkipGram models,
        this will switch the implementation from the
        TensorFlow implementation and will use our Rust Ensmallen one.
    only_execute_embeddings: bool = False
        Whether to only execute the computation of the embedding or also the edge prediction.
        This flag can be useful when the two operations should be executed on different machines
        or at different times.
    embedding_fit_kwargs: Optional[Dict] = None
        The kwargs to be forwarded to the embedding fit method
    embedding_kwargs: Optional[Dict] = None
        The kwargs to be forwarded to the embedding method
    subgraph_of_interest_for_edge_prediction: Optional[Graph] = None
        The subgraph to use for the edge prediction training and evaluation, if any.
        If none are provided, we sample the negative edges from the entire graph.
    sample_only_edges_with_heterogeneous_node_types: bool = False
        Whether to only sample edges between heterogeneous node types.
        This may be useful when training a model to predict between
        two portions in a bipartite graph.
    graph_normalization_callback: Callable[[Graph], Graph] = None
        Graph normalization procedure to call on graphs that have been loaded from
        the Ensmallen automatic retrieval.
    """

    if embedding_kwargs is None:
        embedding_kwargs = {}

    if classifier_kwargs is None:
        classifier_kwargs = {}

    if classifier_fit_kwargs is None:
        classifier_fit_kwargs = {}

    if edge_types is not None and isinstance(edge_types, str):
        edge_types = [edge_types]

    if isinstance(model, str):
        model_name = model
    elif is_sklearn_classifier_model(model):
        model_name = model.__class__.__name__
    elif callable(model):
        model_name = model.__name__

    if subgraph_of_interest_for_edge_prediction is not None and not subgraph_of_interest_for_edge_prediction.has_edges():
        raise ValueError(
            "The provided subgraph of interest does not have any edge."
        )

    if isinstance(graphs, (Graph, str)):
        graphs = [graphs]

    number_of_graphs = len(graphs)

    if versions is not None:
        if any(not isinstance(graph, str) for graph in graphs):
            raise ValueError(
                "Graph versions were provided, but the graphs are not ",
                "graph names from Ensmallen's automatic retrieval."
            )
        if isinstance(versions, str):
            versions = [versions] * number_of_graphs
    else:
        versions = [None] *number_of_graphs

    if repositories is not None:
        if any(not isinstance(graph, str) for graph in graphs):
            raise ValueError(
                "Graph repositories were provided, but the graphs are not ",
                "graph names from Ensmallen's automatic retrieval."
            )
        if isinstance(repositories, str):
            repositories = [repositories]*len(graphs)
    else:
        repositories = [None] *number_of_graphs

    holdouts = []
    for graph, repository, version in tqdm(
        zip(graphs, repositories, versions),
        desc="Executing graph",
        disable=number_of_graphs==1,
        dynamic_ncols=True,
        leave=False
    ):
        if isinstance(graph, str):
            graph = get_dataset(
                graph,
                repository=repository,
                version=version
            )()
            if graph_normalization_callback is not None:
                graph = graph_normalization_callback(graph)

        graph_name = graph.get_name()
        graph_unbalance_rate =  0.9 * (graph.get_nodes_number() * (graph.get_nodes_number() - 1)) / graph.get_number_of_directed_edges()

        unbalance_rates_for_graph = [
            graph_unbalance_rate
            if unbalance_rate == "auto"
            else unbalance_rate
            for unbalance_rate in unbalance_rates
            if unbalance_rate != "auto" and unbalance_rate < graph_unbalance_rate
        ]

        if len(unbalance_rates_for_graph) < len(unbalance_rates):
            warnings.warn(
                (
                    "Be advised that the provided unbalance rates included "
                    "rates that were higher than the maximum possible unbalance rate "
                    "possible in this graph {:4f}."
                ).format(
                    graph_unbalance_rate
                )
            )

        for holdout_number in trange(
            number_of_holdouts,
            desc=f"Executing holdouts on {graph_name}",
            dynamic_ncols=True,
            leave=False
        ):
            connected_holdout_random_state = holdouts_random_state*holdout_number
            connected_holdouts_parameters = dict(
                random_state=connected_holdout_random_state,
                edge_types=edge_types,
                minimum_node_degree=minimum_node_degree,
                maximum_node_degree=maximum_node_degree,
            )
            negative_graph_parameters = dict(
                sample_only_edges_with_heterogeneous_node_types=sample_only_edges_with_heterogeneous_node_types,
            )
            train_graph, test_graph = graph.connected_holdout(
                train_size,
                **connected_holdouts_parameters,
                verbose=False
            )

            start_embedding = time()
            if isinstance(embedding_method, str):
                embedding, _ = compute_node_embedding(
                    graph=train_graph,
                    node_embedding_method_name=embedding_method,
                    use_only_cpu=use_only_cpu,
                    fit_kwargs=embedding_fit_kwargs,
                    verbose=False,
                    **embedding_kwargs
                )
            else:
                embedding = embedding_method(
                    train_graph,
                    holdout_number,
                    **embedding_kwargs
                )
            seconds_necessary_for_embedding = time() - start_embedding

            if only_execute_embeddings:
                continue

            # If requested, we focus the training and test graphs
            # into the area of interest.
            if subgraph_of_interest_for_edge_prediction is not None:
                # We remove all the nodes from the training and test graphs
                # which are not part of the subgraph of interest.
                subgraph_node_names = subgraph_of_interest_for_edge_prediction.get_node_names()
                train_graph = train_graph.filter_from_names(
                    node_names_to_keep=subgraph_node_names
                )
                test_graph = test_graph.filter_from_names(
                    node_names_to_keep=subgraph_node_names
                )

                # We remove all the edges from the training and test graphs
                # which are not part of the subgraph of interest.
                train_graph = train_graph & subgraph_of_interest_for_edge_prediction
                test_graph = test_graph & subgraph_of_interest_for_edge_prediction

                assert train_graph.has_edges()
                assert test_graph.has_edges()

                # We realign the considered training and test graph with
                # the provided subgraph of interest node names dictionary.
                train_graph = train_graph.remap_from_graph(
                    subgraph_of_interest_for_edge_prediction
                )
                test_graph = test_graph.remap_from_graph(
                    subgraph_of_interest_for_edge_prediction
                )

                assert train_graph.has_edges()
                assert test_graph.has_edges()

            graph_to_use_to_sample_negatives = (
                # We sample the negative edges from the entire graph
                graph
                # when no subgraph of interest has been provided
                if subgraph_of_interest_for_edge_prediction is None
                # else, if the subgraph was provided, we only extract the negative
                # edges from this portion of the graph, making the evaluation of the edge prediction task
                # more significant for the actual desired task: often, when doing an edge predition
                # in a graph, we actually intend to predict some type of edges of interest.
                # When this is the case, we do not care to train or evaluate the model on edges that
                # are not in the portion of the graph.
                # Consider a graph representing a social network: if we are interested in learning the
                # connections between users living in small towns, if we also take into account users
                # living in large metropolis we would both train the model on unrelevant data and evaluate
                # the model of a different task, which may be more or less difficult.
                else subgraph_of_interest_for_edge_prediction
            )

            # Force alignment
            embedding = embedding.loc[train_graph.get_node_names()]

            train_graph.enable(
                vector_sources=True,
                vector_destinations=True,
                vector_cumulative_node_degrees=True
            )

            train_negative_graph, test_negative_graph = get_negative_graphs(
                graph_to_use_to_sample_negatives,
                # Inside ensmallen we pass this number in a splitmix function
                # therefore even a sum should have quite enough additional entropy.
                random_state=holdouts_random_state + holdout_number,
                **negative_graph_parameters,
                unbalance_rate=1.0,
                train_size=train_size,
            )

            if isinstance(model, str):
                if is_tensorflow_edge_prediction_method(model):
                    model_instance = get_tensorflow_model(
                        model,
                        graph=graph,
                        embedding=embedding,
                        edge_embedding_method=edge_embedding_method,
                        **classifier_kwargs
                    )
                elif is_default_sklearn_classifier(model):
                    model_instance = SklearnModelEdgePredictionAdapter(
                        get_sklearn_default_classifier(
                            model, **classifier_kwargs)
                    )
                else:
                    raise ValueError(
                        "The provided model name {} is not available.".format(
                            model)
                    )
            elif callable(model):
                if is_sklearn_classifier_model(model):
                    model_instance = SklearnModelEdgePredictionAdapter(
                        model(**classifier_kwargs))
                else:
                    model_instance = model(graph, embedding)
            elif is_sklearn_classifier_model(model):
                # If the provide model is already an sklearn model,
                # we proceed to wrap it and we avoid to training the original
                # models multiple times by making a deep copy of it.
                model_instance = SklearnModelEdgePredictionAdapter(
                    copy.deepcopy(model)
                )
            else:
                raise ValueError(
                    "It is not clear what to do with the provided model object of type {}.".format(
                        type(model)
                    )
                )

            start_training = time()
            if isinstance(model_instance, SklearnModelEdgePredictionAdapter):
                evaluation_kwargs = dict(
                    node_features=embedding.values,
                    edge_embedding_method=edge_embedding_method,
                    aligned_node_mapping=True,
                )
                model_instance.fit(
                    positive_graph=train_graph,
                    negative_graph=train_negative_graph,
                    # TODO: In the future consider adding these.
                    # edge_features: Optional[np.ndarray] = None,
                    **evaluation_kwargs
                )
            elif is_tensorflow_edge_prediction_method(model):
                evaluation_kwargs = dict()
                model_instance.fit(
                    train_graph=train_graph,
                    valid_graph=test_graph,
                    negative_valid_graph=test_negative_graph,
                    **classifier_fit_kwargs,
                    sample_only_edges_with_heterogeneous_node_types=sample_only_edges_with_heterogeneous_node_types,
                    verbose=False
                )
            else:
                raise NotImplementedError(
                    "Unclear what to do with this model."
                )
            seconds_necessary_to_train_model = time() - start_training

            start_train_evaluation = time()
            train_performance = model_instance.evaluate(
                positive_graph=train_graph,
                negative_graph=train_negative_graph,
                **evaluation_kwargs
            )
            train_evaluation_time = time() - start_train_evaluation

            train_performance["evaluation_type"] = "train"
            train_performance["unbalance"] = 1.0
            train_performance["size"] = train_size
            train_performance["graph_name"] = graph_name
            train_performance["required_training_time"] = seconds_necessary_to_train_model
            train_performance["seconds_necessary_for_embedding"] = seconds_necessary_for_embedding
            train_performance["train_evaluation_time"] = train_evaluation_time

            start_test_evaluation = time()
            test_performance = model_instance.evaluate(
                positive_graph=test_graph,
                negative_graph=test_negative_graph,
                **evaluation_kwargs
            )
            test_evaluation_time = time() - start_test_evaluation

            test_performance["evaluation_type"] = "test"
            test_performance["unbalance"] = 1.0
            test_performance["size"] = 1.0 - train_size
            test_performance["graph_name"] = graph_name
            test_performance["seconds_necessary_for_embedding"] = seconds_necessary_for_embedding
            train_performance["test_evaluation_time"] = test_evaluation_time

            for parameter_name, parameter in itertools.chain(
                connected_holdouts_parameters.items(),
                negative_graph_parameters.items(),
            ):
                if parameter is not None:
                    train_performance[parameter_name] = parameter
                    test_performance[parameter_name] = parameter

            holdouts.append(train_performance)
            holdouts.append(test_performance)

            for i, unbalance_rate in enumerate(tqdm(
                unbalance_rates_for_graph,
                desc="Evaluating on datasets with different unbalance",
                dynamic_ncols=True,
                leave=False
            ), start=1):
                train_negative_graph, test_negative_graph = get_negative_graphs(
                    graph_to_use_to_sample_negatives,
                    # Inside ensmallen we pass this number in a splitmix function
                    # therefore even a sum should have quite enough additional entropy.
                    random_state=holdouts_random_state + holdout_number + i,
                    sample_only_edges_with_heterogeneous_node_types=sample_only_edges_with_heterogeneous_node_types,
                    unbalance_rate=unbalance_rate,
                    train_size=train_size,
                )

                start_test_evaluation = time()
                train_performance = model_instance.evaluate(
                    positive_graph=train_graph,
                    negative_graph=train_negative_graph,
                    **evaluation_kwargs
                )
                train_evaluation_time = time() - start_train_evaluation

                train_performance["evaluation_type"] = "train"
                train_performance["unbalance"] = unbalance_rate
                train_performance["size"] = train_size
                train_performance["graph_name"] = graph_name
                train_performance["required_training_time"] = seconds_necessary_to_train_model
                train_performance["seconds_necessary_for_embedding"] = seconds_necessary_for_embedding
                train_performance["train_evaluation_time"] = train_evaluation_time

                start_test_evaluation = time()
                test_performance = model_instance.evaluate(
                    positive_graph=test_graph,
                    negative_graph=test_negative_graph,
                    **evaluation_kwargs
                )
                test_evaluation_time = time() - start_test_evaluation

                test_performance["evaluation_type"] = "test"
                test_performance["unbalance"] = unbalance_rate
                test_performance["size"] = 1.0 - train_size
                test_performance["graph_name"] = graph_name
                test_performance["seconds_necessary_for_embedding"] = seconds_necessary_for_embedding
                test_performance["test_evaluation_time"] = test_evaluation_time

                for parameter_name, parameter in itertools.chain(
                    connected_holdouts_parameters.items(),
                    negative_graph_parameters.items(),
                ):
                    if parameter is not None:
                        train_performance[parameter_name] = parameter
                        test_performance[parameter_name] = parameter

                holdouts.append(train_performance)
                holdouts.append(test_performance)

    holdouts = pd.DataFrame(holdouts)
    if isinstance(embedding_method, str):
        holdouts["embedding_method"] = embedding_method
    holdouts["model_name"] = model_name

    return holdouts
