"""Submodule providing classification evaluation pipeline."""
from typing import Union, List, Optional, Iterator, Type, Dict, Any
from ensmallen import Graph
from ensmallen.datasets import get_dataset
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from embiggen.utils.abstract_models import AbstractClassifierModel, AbstractEmbeddingModel


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

    for graph in graphs:
        if not isinstance(graph, (str, Graph)):
            raise ValueError(
                "The provided classifier graph is expected to be "
                "either an Ensmallen graph object or a string with the graph name "
                f"but an object of type {type(graph)} was provided."
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
            )()
        else:
            yield graph


def classification_evaluation_pipeline(
    evaluation_schema: str,
    holdouts_kwargs: Dict[str, Any],
    graphs: Union[str, Graph, List[Graph], List[str]],
    models: Union[Type[AbstractClassifierModel], List[Type[AbstractClassifierModel]]],
    expected_parent_class: Type[AbstractClassifierModel],
    node_features: Optional[Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel], List[Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel]]]]] = None,
    node_type_features: Optional[Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel], List[Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel]]]]] = None,
    edge_features: Optional[Union[str, pd.DataFrame, np.ndarray, List[Union[str, pd.DataFrame, np.ndarray]]]] = None,
    library_names: Optional[Union[str, List[str]]] = None,
    subgraph_of_interest: Optional[Graph] = None,
    number_of_holdouts: int = 10,
    random_state: int = 42,
    repositories: Optional[Union[str, List[str]]] = None,
    versions: Optional[Union[str, List[str]]] = None,
    enable_cache: bool = False,
    smoke_test: bool = False,
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
    library_names: Optional[Union[str, List[str]]] = None
        Library names from where to retrieve the provided model names.
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
    enable_cache: bool = False
        Whether to enable the cache.
    smoke_test: bool = False
        Whether this run should be considered a smoke test
        and therefore use the smoke test configurations for
        the provided model names and feature names.
        This parameter will also turn off the cache.
    **evaluation_kwargs: Dict
        Keyword arguments to forward to evaluation.
    """
    return pd.concat([
        expected_parent_class.evaluate(
            models=models,
            graph=graph,
            evaluation_schema=evaluation_schema,
            holdouts_kwargs=holdouts_kwargs,
            node_features=node_features,
            node_type_features=node_type_features,
            edge_features=edge_features,
            subgraph_of_interest=subgraph_of_interest,
            number_of_holdouts=number_of_holdouts,
            random_state=random_state,
            enable_cache=enable_cache and not smoke_test,
            smoke_test=smoke_test,
            **evaluation_kwargs
        )
        for graph in iterate_graphs(graphs, repositories, versions)
    ])

    