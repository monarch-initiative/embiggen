"""Submodule providing pipelines for edge prediction."""
from typing import Union, Callable, Tuple, List, Optional, Dict
import pandas as pd
import tensorflow as tf
from ensmallen import Graph
from tqdm.auto import trange
from ..edge_prediction import Perceptron, MultiLayerPerceptron, EdgePredictionModel
from ..utils import execute_gpu_checks
from .compute_node_embedding import compute_node_embedding

edge_prediction_models = {
    "Perceptron": Perceptron,
    "MultiLayerPerceptron": MultiLayerPerceptron
}


def _evaluate_embedding_for_edge_prediction(
    model_name: str,
    graph: Graph,
    embedding: pd.DataFrame,
    train_graph: Graph,
    test_graph: Graph,
    train_negative_graph: Graph,
    test_negative_graph: Graph,
    batch_size: int,
    use_mirrored_strategy: bool
) -> Tuple[pd.DataFrame, Dict, Dict]:
    if isinstance(model_name, str):
        model = edge_prediction_models.get(model_name)(
            graph=graph,
            embedding=embedding
        )
    else:
        model = model_name(graph, embedding)

    history = model.fit(
        train_graph=train_graph,
        valid_graph=test_graph,
        negative_valid_graph=test_negative_graph,
        batch_size=batch_size,
        support_mirrored_strategy=use_mirrored_strategy
    )
    train_performance = model.evaluate(
        graph=train_graph,
        negative_graph=train_negative_graph,
        batch_size=batch_size,
        support_mirrored_strategy=use_mirrored_strategy
    )
    train_performance["evaluation_type"] = "train"
    test_performance = model.evaluate(
        graph=test_graph,
        negative_graph=test_negative_graph,
        batch_size=batch_size,
        support_mirrored_strategy=use_mirrored_strategy
    )
    test_performance["evaluation_type"] = "test"

    return history, train_performance, test_performance


def evaluate_embedding_for_edge_prediction(
    embedding_method: Union[str, Callable[[Graph, int], pd.DataFrame]],
    graph: Graph,
    model_name: Union[str, Callable[[Graph, pd.DataFrame], EdgePredictionModel]],
    number_of_holdouts: int = 10,
    training_size: float = 0.8,
    random_seed: int = 42,
    batch_size: int = 2**10,
    edge_types: Optional[List[str]] = None,
    use_mirrored_strategy: bool = False,
    only_execute_embeddings: bool = False,
    embedding_method_fit_kwargs: Optional[Dict] = None,
    embedding_method_kwargs: Optional[Dict] = None,
    subgraph_of_interest_for_edge_prediction: Optional[Graph] = None,
    devices: Union[List[str], str] = None,
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    """Return the evaluation of an embedding for edge prediction on the given model.

    Parameters
    ----------------------
    embedding_method: Union[str, Callable[[Graph, int], pd.DataFrame]]
        Either the name of the embedding method or a method to be called.
    graph: Graph
        The graph to run the embedding and edge prediction on.
    model_name: Union[str, Callable[[Graph, pd.DataFrame], EdgePredictionModel]]
        Either the name of the model or a method returning a model.
    number_of_holdouts: int = 10
        The number of the holdouts to run.
    training_size: float = 0.8
        Split size of the training graph.
    random_seed: int = 42
        The seed to be used to reproduce the holdout.
    batch_size: int = 2**10
        Size of the batch to be considered.
    edge_types: Optional[List[str]] = None
        Edge types to focus the edge prediction on, if any.
    use_mirrored_strategy: bool = False
        Whether to use mirrored strategy for the embedding and edge prediction models.
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
    devices: Union[List[str], str] = None
        The list of devices to use when training the embedding and edge prediction models
        in a MirroredStrategy, that is across multiple GPUs. Thise feature is mainly useful
        when there are multiple GPUs available AND the graph is large enough to actually
        use the GPUs (for instance when it has at least a few million nodes).
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

    # If the embedding method is a string, we execute this check also within
    # the compute node embedding pipeline.
    if isinstance(embedding_method, str):
        execute_gpu_checks(use_mirrored_strategy)

    holdouts = []
    histories = []
    for holdout_number in trange(
        number_of_holdouts,
        desc="Executing holdouts"
    ):
        train_graph, test_graph = graph.connected_holdout(
            training_size,
            random_state=random_seed*holdout_number,
            edge_types=edge_types,
            verbose=True
        )

        if isinstance(embedding_method, str):
            embedding, _ = compute_node_embedding(
                graph=train_graph,
                node_embedding_method_name=embedding_method,
                use_mirrored_strategy=use_mirrored_strategy,
                fit_kwargs=embedding_method_fit_kwargs,
                devices=devices,
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

        # For both the training and the test set we sample the same
        # number of negative edges are there are existing edges in the training and test graphs, respectively.
        # This is done in order to avoid an excessive bias in the evaluation of the edge prediction.
        # Across multiple holdouts, statistically it is unlikely to sample
        # consistently unknown positive edges, and therefore we should be able
        # to remove this negative bias from the evaluation.
        # Of course, this only apply to graphs where we can assume that there is
        # not a massive amount of unknown positive edges.
        train_negative_graph = graph_to_use_to_sample_negatives.sample_negatives(
            negatives_number=train_graph.get_edges_number(),
            random_state=random_seed*holdout_number,
            verbose=True
        )

        test_negative_graph = graph_to_use_to_sample_negatives.sample_negatives(
            negatives_number=test_graph.get_edges_number(),
            # We add an arbitrary constant to the random state to make
            # the initial sampling of the training graph different from
            # the initial sampling of the test graph.
            random_state=(random_seed + 23456787)*holdout_number,
            verbose=True
        )

        # Consinstency check on graph size
        assert train_negative_graph.get_edges_number() == train_graph.get_edges_number()
        assert train_negative_graph.get_nodes_number() == train_graph.get_nodes_number()
        assert test_negative_graph.get_edges_number() == test_graph.get_edges_number()
        assert test_negative_graph.get_nodes_number() == test_graph.get_nodes_number()

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

        if use_mirrored_strategy:
            # The following Try-Except statement is needed because of a
            # weird IndexError exception raised in recent (> 2.6) versions
            # of TensorFlow. According to the TensorFlow documentation,
            # the usage of MirroredStrategy within this code snipped should
            # be correct, nonetheless it raises the exception.
            # Since the execution of the model is correct, we patch it
            # this way to avoid loosing model training.
            try:
                strategy = tf.distribute.MirroredStrategy(devices=devices)
                with strategy.scope():
                    # This is a candidate patch to a MirroredStrategy
                    history, train_performance, test_performance = _evaluate_embedding_for_edge_prediction(
                        model_name=model_name,
                        graph=graph,
                        embedding=embedding,
                        train_graph=train_graph,
                        test_graph=test_graph,
                        train_negative_graph=train_negative_graph,
                        test_negative_graph=test_negative_graph,
                        batch_size=batch_size,
                        use_mirrored_strategy=use_mirrored_strategy
                    )
                    holdouts.append(train_performance)
                    holdouts.append(test_performance)
                    histories.append(history)
            except IndexError:
                pass
        else:
            history, train_performance, test_performance = _evaluate_embedding_for_edge_prediction(
                model_name=model_name,
                graph=graph,
                embedding=embedding,
                train_graph=train_graph,
                test_graph=test_graph,
                train_negative_graph=train_negative_graph,
                test_negative_graph=test_negative_graph,
                batch_size=batch_size,
                use_mirrored_strategy=use_mirrored_strategy
            )
            holdouts.append(train_performance)
            holdouts.append(test_performance)
            histories.append(history)

    return pd.DataFrame(holdouts), histories
