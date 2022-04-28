"""Submodule providing pipelines for edge prediction."""
from typing import Union, Callable, Tuple, List, Optional, Dict
import pandas as pd
import tensorflow as tf
from ensmallen import Graph
from tqdm.auto import trange, tqdm
import math
from ensmallen.datasets import get_dataset
from yaml import warnings
from ..edge_prediction import Perceptron, MultiLayerPerceptron, EdgePredictionModel
from ..utils import execute_gpu_checks, get_available_gpus_number
from .compute_node_embedding import compute_node_embedding

edge_prediction_models = {
    "Perceptron": Perceptron,
    "MultiLayerPerceptron": MultiLayerPerceptron
}


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
    graph.enable()
    negative_graph = graph.sample_negatives(
        number_of_negative_samples=int(math.ceil(graph.get_edges_number()*unbalance_rate)),
        random_state=random_state,
        sample_only_edges_with_heterogeneous_node_types=sample_only_edges_with_heterogeneous_node_types,
        verbose=False
    )
    negative_graph.enable()
    return negative_graph.random_holdout(
        train_size=train_size,
        random_state=random_state,
        verbose=False,
    )


def evaluate_embedding_for_edge_prediction(
    embedding_method: Union[str, Callable[[Graph, int], pd.DataFrame]],
    graphs: Union[Graph, str, List[str], List[Graph]],
    model_name: Union[str, Callable[[Graph, pd.DataFrame], EdgePredictionModel]],
    epochs: int = 1000,
    number_of_holdouts: int = 10,
    train_size: float = 0.8,
    unbalance_rates: Tuple[Union[float, str]] = (10.0, 100.0, "auto"),
    random_state: int = 42,
    batch_size: int = 2**10,
    edge_embedding_method: str = "Concatenate",
    trainable_embedding: bool = False,
    use_dropout: bool = True,
    dropout_rate: float = 0.1,
    use_edge_metrics: bool = False,
    edge_types: Optional[List[str]] = None,
    use_mirrored_strategy: Union[bool, str] = "auto",
    use_only_cpu: Union[bool, str] = "auto",
    only_execute_embeddings: bool = False,
    embedding_method_fit_kwargs: Optional[Dict] = None,
    embedding_method_kwargs: Optional[Dict] = None,
    subgraph_of_interest_for_edge_prediction: Optional[Graph] = None,
    sample_only_edges_with_heterogeneous_node_types: bool = False,
    devices: Union[List[str], str] = None,
    graph_normalization_callback: Callable[[Graph], Graph] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    """Return the evaluation of an embedding for edge prediction on the given model.

    Parameters
    ----------------------
    embedding_method: Union[str, Callable[[Graph, int], pd.DataFrame]]
        Either the name of the embedding method or a method to be called.
    graphs: Union[Graph, str, List[str], List[Graph]]
        The graph to run the embedding and edge prediction on.
        If a string was provided, we will retrieve the graphs from Ensmallen's repositories.
        If a list was provided, we will iterate on all graphs.
    model_name: Union[str, Callable[[Graph, pd.DataFrame], EdgePredictionModel]]
        Either the name of the model or a method returning a model.
    epochs: int = 1000
        Number of epochs to train the perceptron model for.
    number_of_holdouts: int = 10
        The number of the holdouts to run.
    train_size: float = 0.8
        Split size of the training graph.
    unbalance_rates: Tuple[Union[float, str]] = (10.0, 100.0, "auto")
        List of unbalance to evaluate.
        Do note that an unbalance of one is always included.
        With "auto", we use the true unbalance of the graph.
    random_state: int = 42
        The seed to be used to reproduce the holdout.
    batch_size: int = 2**10
        Size of the batch to be considered.
    edge_types: Optional[List[str]] = None
        Edge types to focus the edge prediction on, if any.
    use_mirrored_strategy: Union[bool, str] = "auto"
        Whether to use mirror strategy to distribute the
        computation across multiple devices.
        This is automatically enabled if more than one
        GPU is detected and the flag `use_only_gpu` was
        not provided, or if the list of devices to use
        was provided and it includes at least a GPU.
    use_only_cpu: Union[bool, str] = "auto",
        Whether to only use CPU.
        Do note that for CBOW and SkipGram models,
        this will switch the implementation from the
        TensorFlow implementation and will use our Rust Ensmallen one.
    only_execute_embeddings: bool = False
        Whether to only execute the computation of the embedding or also the edge prediction.
        This flag can be useful when the two operations should be executed on different machines
        or at different times.
    embedding_method_fit_kwargs: Optional[Dict] = None
        The kwargs to be forwarded to the embedding fit method
    embedding_method_kwargs: Optional[Dict] = None
        The kwargs to be forwarded to the embedding method
    subgraph_of_interest_for_edge_prediction: Optional[Graph] = None
        The subgraph to use for the edge prediction training and evaluation, if any.
        If none are provided, we sample the negative edges from the entire graph.
    sample_only_edges_with_heterogeneous_node_types: bool = False
        Whether to only sample edges between heterogeneous node types.
        This may be useful when training a model to predict between
        two portions in a bipartite graph.
    devices: Union[List[str], str] = None
        The list of devices to use when training the embedding and edge prediction models
        in a MirroredStrategy, that is across multiple GPUs. Thise feature is mainly useful
        when there are multiple GPUs available AND the graph is large enough to actually
        use the GPUs (for instance when it has at least a few million nodes).
    graph_normalization_callback: Callable[[Graph], Graph] = None
        Graph normalization procedure to call on graphs that have been loaded from
        the Ensmallen automatic retrieval.
    verbose: bool = True
        Whether to show the loading bars.
    """
    if isinstance(model_name, str) and model_name not in edge_prediction_models:
        raise ValueError(
            f"The given edge prediction model `{model_name}` is not supported. "
            f"The supported node embedding methods are `{edge_prediction_models}`."
        )

    if embedding_method_kwargs is None:
        embedding_method_kwargs = {}

    if subgraph_of_interest_for_edge_prediction is not None and not subgraph_of_interest_for_edge_prediction.has_edges():
        raise ValueError(
            "The provided subgraph of interest does not have any edge."
        )

    # If devices are given as a single device we adapt this into a list.
    if isinstance(devices, str):
        devices = [devices]

    # If in the list of provided devices there is a GPU specified,
    # and there are more than one GPU, we need to use the MirroredStrategy
    # to distribute its computation.
    if devices and any(
        "GPU" in device
        for device in devices
    ) and get_available_gpus_number() > 1:
        use_mirrored_strategy = True

    # If the embedding method is a string, we execute this check also within
    # the compute node embedding pipeline.
    if not isinstance(embedding_method, str):
        execute_gpu_checks(use_mirrored_strategy)

    if isinstance(graphs, (Graph, str)):
        graphs = [graphs]


    holdouts = []
    histories = []
    for graph in tqdm(
        graphs,
        desc="Executing graph",
        disable=len(graphs) <= 1,
        dynamic_ncols=True,
        leave=False
    ):
        if isinstance(graph, str):
            graph = get_dataset(graph)()
            if graph_normalization_callback is not None:
                graph = graph_normalization_callback(graph)
        
        graph_name = graph.get_name()
        graph_unbalance_rate = graph.get_nodes_number() * (graph.get_nodes_number() - 1) / graph.get_number_of_directed_edges()

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
            train_graph, test_graph = graph.connected_holdout(
                train_size,
                random_state=random_state*holdout_number,
                edge_types=edge_types,
                verbose=False
            )

            if isinstance(embedding_method, str):
                embedding, _ = compute_node_embedding(
                    graph=train_graph,
                    node_embedding_method_name=embedding_method,
                    use_mirrored_strategy=use_mirrored_strategy,
                    use_only_cpu=use_only_cpu,
                    fit_kwargs=embedding_method_fit_kwargs,
                    devices=devices,
                    verbose=False,
                    **embedding_method_kwargs
                )
            else:
                embedding = embedding_method(
                    train_graph,
                    holdout_number,
                    use_mirrored_strategy=use_mirrored_strategy,
                    **embedding_method_kwargs
                )

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

            # Simplify the train graph if needed.
            if train_graph.has_edge_weights():
                train_graph.remove_inplace_edge_weights()

            if train_graph.has_edge_types():
                train_graph.remove_inplace_edge_types()

            # Simplify the test graph if needed.
            if test_graph.has_edge_weights():
                test_graph.remove_inplace_edge_weights()

            if test_graph.has_edge_types():
                test_graph.remove_inplace_edge_types()

            train_graph.enable(
                vector_sources=True,
                vector_destinations=True,
                vector_cumulative_node_degrees=True
            )

            train_negative_graph, test_negative_graph = get_negative_graphs(
                graph_to_use_to_sample_negatives,
                # Inside ensmallen we pass this number in a splitmix function
                # therefore even a sum should have quite enough additional entropy.
                random_state=random_state + holdout_number,
                sample_only_edges_with_heterogeneous_node_types=sample_only_edges_with_heterogeneous_node_types,
                unbalance_rate=1.0,
                train_size=train_size,
            )

            if tf.config.list_physical_devices('GPU') and use_mirrored_strategy:
                strategy = tf.distribute.MirroredStrategy(devices)
            else:
                # Use the Default Strategy
                strategy = tf.distribute.get_strategy()

            with strategy.scope():
                if isinstance(model_name, str):
                    model = edge_prediction_models.get(model_name)(
                        graph=graph,
                        embedding=embedding,
                        edge_embedding_method=edge_embedding_method,
                        trainable_embedding=trainable_embedding,
                        use_dropout=use_dropout,
                        dropout_rate=dropout_rate,
                        use_edge_metrics=use_edge_metrics,
                    )
                else:
                    model = model_name(graph, embedding)

            histories.append(model.fit(
                train_graph=train_graph,
                valid_graph=test_graph,
                negative_valid_graph=test_negative_graph,
                batch_size=batch_size,
                epochs=epochs,
                sample_only_edges_with_heterogeneous_node_types=sample_only_edges_with_heterogeneous_node_types,
                verbose=False
            ))

            train_performance = model.evaluate(
                graph=train_graph,
                negative_graph=train_negative_graph,
                batch_size=batch_size,
                verbose=False
            )
            train_performance["evaluation_type"] = "train"
            train_performance["unbalance"] = 1.0
            train_performance["graph_name"] = graph_name

            test_performance = model.evaluate(
                graph=test_graph,
                negative_graph=test_negative_graph,
                batch_size=batch_size,
                verbose=False
            )
            test_performance["evaluation_type"] = "test"
            test_performance["unbalance"] = 1.0
            test_performance["graph_name"] = graph_name

            if isinstance(embedding_method, str):
                train_performance["embedding_method"] = embedding_method
                test_performance["embedding_method"] = embedding_method

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
                    random_state=random_state + holdout_number + i,
                    sample_only_edges_with_heterogeneous_node_types=sample_only_edges_with_heterogeneous_node_types,
                    unbalance_rate=unbalance_rate,
                    train_size=train_size,
                )
                train_performance = model.evaluate(
                    graph=train_graph,
                    negative_graph=train_negative_graph,
                    batch_size=batch_size,
                    verbose=False
                )
                train_performance["evaluation_type"] = "train"
                train_performance["unbalance"] = unbalance_rate
                train_performance["graph_name"] = graph_name
                
                test_performance = model.evaluate(
                    graph=test_graph,
                    negative_graph=test_negative_graph,
                    batch_size=batch_size,
                    verbose=False
                )
                test_performance["evaluation_type"] = "test"
                test_performance["unbalance"] = unbalance_rate
                test_performance["graph_name"] = graph_name

                holdouts.append(train_performance)
                holdouts.append(test_performance)

                if isinstance(embedding_method, str):
                    train_performance["embedding_method"] = embedding_method
                    test_performance["embedding_method"] = embedding_method


    return pd.DataFrame(holdouts), histories
