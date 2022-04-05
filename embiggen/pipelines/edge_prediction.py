"""Submodule providing pipelines for edge prediction."""
from typing import Union, Callable, Tuple, List, Optional, Dict
import pandas as pd
from ensmallen import Graph
from tqdm.auto import trange
from ..edge_prediction import Perceptron, MultiLayerPerceptron, EdgePredictionModel
from ..utils import execute_gpu_checks
from .compute_node_embedding import compute_node_embedding

edge_prediction_models = {
    "Perceptron": Perceptron,
    "MultiLayerPerceptron": MultiLayerPerceptron
}


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
    subgraph_of_interest_for_edge_prediction: Optional[Graph] = None
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    """Return the evaluation of an embedding for edge prediction on the given model.
    """
    if isinstance(model_name, str) and model_name not in edge_prediction_models:
        raise ValueError(
            (
                "The given edge prediction model `{}` is not supported. "
                "The supported node embedding methods are `{}`."
            ).format(
                model_name,
                edge_prediction_models
            )
        )

    if embedding_method_kwargs is None:
        embedding_method_kwargs = {}

    if subgraph_of_interest_for_edge_prediction is not None and not graph.contains(subgraph_of_interest_for_edge_prediction):
        raise ValueError(
            "The provided subgraph of interest for the edge prediction task is NOT a subgraph "
            "of the provided graph, and therefore cannot be used as a subgraph of interest."
        )

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
            train_graph = train_graph & subgraph_of_interest_for_edge_prediction
            test_graph = test_graph & subgraph_of_interest_for_edge_prediction

        negative_graph = (
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
        ).sample_negatives(
            negatives_number=test_graph.get_number_of_directed_edges(),
            random_state=random_seed*holdout_number,
            verbose=True
        )

        # Consinstency check on graph size
        assert negative_graph.get_edges_number() == negative_graph.get_edges_number()

        # Force alignment
        embedding = embedding.loc[train_graph.get_node_names()]

        # Simplify the test graph if needed.
        if test_graph.has_edge_weights():
            test_graph.remove_inplace_edge_weights()

        if test_graph.has_edge_types():
            test_graph.remove_inplace_edge_types()

        if isinstance(model_name, str):
            model = edge_prediction_models.get(model_name)(
                graph=graph,
                embedding=embedding
            )
        else:
            model = model_name(graph, embedding)

        histories.append(model.fit(
            train_graph=train_graph,
            valid_graph=test_graph,
            negative_valid_graph=negative_graph,
            batch_size=batch_size,
            support_mirrored_strategy=use_mirrored_strategy
        ))
        holdouts.append(model.evaluate(
            graph=test_graph,
            negative_graph=negative_graph,
            batch_size=batch_size,
            support_mirrored_strategy=use_mirrored_strategy
        ))

    return pd.DataFrame(holdouts), histories
