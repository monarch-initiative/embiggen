"""Submodule providing classification evaluation pipeline."""
from typing import Union, List, Optional, Iterator, Type, Dict, Any
from ensmallen import Graph
from ensmallen.datasets import get_dataset
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from cache_decorator import Cache
from .abstract_models import AbstractClassifierModel, AbstractEmbeddingModel


def iterate_graphs(
    graphs: Union[Graph, List[Graph]],
    repositories: Optional[Union[str, List[str]]] = None,
    versions: Optional[Union[str, List[str]]] = None,
) -> Iterator[Graph]:
    """Returns iterator over provided graphs.

    Parameters
    ------------------
    graphs: Union[Graph, List[Graph]]
        The graph or graphs to iterate on.
    repositories: Optional[Union[str, List[str]]] = None
        The repositories from where to retrieve these graphs.
        This only applies for the graph names that are available
        from the ensmallen automatic retrieval.
    versions: Optional[Union[str, List[str]]] = None
        The versions of the graphs to be retrieved.
        When this is left to none, the retrieved version will be
        the one that has been indicated to be the most recent one.
        This only applies for the graph names that are available
        from the ensmallen automatic retrieval.
    """
    if not isinstance(graphs, (list, tuple)):
        graphs = [graphs]

    for graph in graphs:
        if not isinstance(graph, (str, Graph)):
            raise ValueError(
                "The graph objects should either be strings when "
                "they are graphs to be automatically retrieved or "
                "alternatively graph object instances, but you "
                f"provided an object of type {type(graph)}."
            )

    number_of_graphs = len(graphs)

    if number_of_graphs == 0:
        raise ValueError(
            "An empty list of graphs was provided."
        )

    if not isinstance(repositories, (list, tuple)):
        repositories = [repositories] * number_of_graphs

    number_of_repositories = len(repositories)

    if number_of_graphs != number_of_repositories:
        raise ValueError(
            f"The number of provided graphs `{number_of_graphs}` does not match "
            f"the number of provided repositories `{number_of_repositories}`."
        )

    if not isinstance(versions, (list, tuple)):
        versions = [versions] * number_of_graphs

    number_of_versions = len(versions)

    if number_of_graphs != number_of_versions:
        raise ValueError(
            f"The number of provided graphs `{number_of_graphs}` does not match "
            f"the number of provided versions `{number_of_versions}`."
        )

    for graph, version, repository in tqdm(
        zip(graphs, versions, repositories),
        desc="Graphs",
        total=number_of_graphs,
        disable=number_of_graphs == 1,
        dynamic_ncols=True,
        leave=False
    ):
        if isinstance(graph, str):
            yield get_dataset(
                name=graph,
                repository=repository,
                version=version
            )
        yield graph


def iterate_classifier_models(
    models: Union[Type[AbstractClassifierModel], List[Type[AbstractClassifierModel]]],
    expected_parent_class: Type[AbstractClassifierModel]
) -> Iterator[Type[AbstractClassifierModel]]:
    """Return iterator over the provided models after validation.

    Parameters
    -------------------
    models: Union[Type[AbstractClassifierModel], List[Type[AbstractClassifierModel]]]
        The models to validate and iterate on.
    expected_parent_class: Type[AbstractClassifierModel]
        The parent class to check the model against.
    """
    if not isinstance(models, (list, tuple)):
        models = [models]

    number_of_models = len(models)

    if number_of_models == 0:
        raise ValueError(
            "An empty list of models was provided."
        )

    for model in models:
        if not issubclass(model, expected_parent_class):
            raise ValueError(
                "The provided classifier model is expected to be "
                f"an implementation of the {expected_parent_class.__name__} class, but you provided "
                f"an object of type {type(model)} that does not hereditate from "
                "the expected class."
            )

    for model in tqdm(
        models,
        desc="Models",
        total=number_of_models,
        disable=number_of_models == 1,
        dynamic_ncols=True,
        leave=False
    ):
        yield model


@Cache(
    cache_path="{cache_dir}/{task_name}/{model_name}/{graph_name}/{_hash}.csv.gz",
    cache_dir="experiments"
)
def evaluate_classifier(
    classifier: Type[AbstractClassifierModel],
    task_name: str,
    model_name: str,
    graph_name: str,
    graph: Graph,
    evaluation_schema: str,
    holdouts_kwargs: Dict[str, Any],
    node_features: Optional[Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel], List[Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel]]]]] = None,
    edge_features: Optional[Union[str, pd.DataFrame, np.ndarray, List[Union[str, pd.DataFrame, np.ndarray]]]] = None,
    subgraph_of_interest: Optional[Graph] = None,
    number_of_holdouts: int = 10,
    random_state: int = 42,
    **evaluation_kwargs: Dict
) -> pd.DataFrame:
    """Executes cache evaluation for model."""
    return classifier.evaluate(
        graph,
        evaluation_schema=evaluation_schema,
        holdouts_kwargs=holdouts_kwargs,
        node_features=node_features,
        edge_features=edge_features,
        subgraph_of_interest=subgraph_of_interest,
        number_of_holdouts=number_of_holdouts,
        random_state=random_state,
        **evaluation_kwargs
    )


def classification_evaluation_pipeline(
    evaluation_schema: str,
    holdouts_kwargs: Dict[str, Any],
    graphs: Union[str, Graph, List[Graph], List[str]],
    models: Union[Type[AbstractClassifierModel], List[Type[AbstractClassifierModel]]],
    expected_parent_class: Type[AbstractClassifierModel],
    node_features: Optional[Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel], List[Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel]]]]] = None,
    edge_features: Optional[Union[str, pd.DataFrame, np.ndarray, List[Union[str, pd.DataFrame, np.ndarray]]]] = None,
    subgraph_of_interest: Optional[Graph] = None,
    number_of_holdouts: int = 10,
    random_state: int = 42,
    repositories: Optional[Union[str, List[str]]] = None,
    versions: Optional[Union[str, List[str]]] = None,
    **evaluation_kwargs
) -> pd.DataFrame:
    """Execute classification pipeline for all provided models and graphs.
    
    Parameters
    ---------------------
    evaluation_schema: str
        The evaluation schema to follow.
    holdouts_kwargs: Dict[str, Any]
        The parameters for the selected holdouts method.
    graphs: Union[str, Graph, List[Graph], List[str]]
        The graphs or graph names to run this evaluation on.
    models: Union[Type[AbstractClassifierModel], List[Type[AbstractClassifierModel]]]
        The models to evaluate.
    expected_parent_class: Type[AbstractClassifierModel]
        The expected parent class of the models, necessary to validate that the models are what we expect.
    node_features: Optional[Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel], List[Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel]]]]] = None
        The node features to use.
    edge_features: Optional[Union[str, pd.DataFrame, np.ndarray, List[Union[str, pd.DataFrame, np.ndarray]]]] = None
        The edge features to use.
    subgraph_of_interest: Optional[Graph] = None
        The subgraph of interest to focus the task on.
    number_of_holdouts: int = 10
        The number of holdouts to execute.
    random_state: int = 42
        Random state to reproduce this evaluation.
    repositories: Optional[Union[str, List[str]]] = None
        Repositories from where to retrieve the provided graph names
        from the Ensmallen automatic retrieval.
    versions: Optional[Union[str, List[str]]] = None
        Graph versions to retrieve.
    **evaluation_kwargs: Dict
        Keyword arguments to forward to evaluation.
    """
    return pd.concat([
        evaluate_classifier(
            classifier=classifier,
            task_name=classifier.task_name(),
            model_name=classifier.name(),
            graph_name=graph.get_name(),
            graph=graph,
            evaluation_schema=evaluation_schema,
            holdouts_kwargs=holdouts_kwargs,
            node_features=node_features,
            edge_features=edge_features,
            subgraph_of_interest=subgraph_of_interest,
            number_of_holdouts=number_of_holdouts,
            random_state=random_state,
            **evaluation_kwargs
        )
        for graph in iterate_graphs(graphs, repositories, versions)
        for classifier in iterate_classifier_models(models, expected_parent_class)
    ])
