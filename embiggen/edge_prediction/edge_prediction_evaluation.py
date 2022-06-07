"""Submodule providing edge prediction evaluation pipeline."""
from typing import Dict, Any, Union, List, Type, Optional, Tuple
from ensmallen import Graph
import pandas as pd
import numpy as np
from embiggen.utils import classification_evaluation_pipeline, AbstractEmbeddingModel
from embiggen.edge_prediction.edge_prediction_model import AbstractEdgePredictionModel


def edge_prediction_evaluation(
    holdouts_kwargs: Dict[str, Any],
    graphs: Union[str, Graph, List[Graph], List[str]],
    models: Union[Type[AbstractEdgePredictionModel], List[Type[AbstractEdgePredictionModel]]],
    evaluation_schema: str = "Connected Monte Carlo",
    node_features: Optional[Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel], List[Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel]]]]] = None,
    node_type_features: Optional[Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel], List[Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel]]]]] = None,
    library_names: Optional[Union[str, List[str]]] = None,
    subgraph_of_interest: Optional[Graph] = None,
    number_of_holdouts: int = 10,
    random_state: int = 42,
    repositories: Optional[Union[str, List[str]]] = None,
    versions: Optional[Union[str, List[str]]] = None,
    validation_sample_only_edges_with_heterogeneous_node_types: bool = False,
    validation_unbalance_rates: Tuple[float] = (1.0, ),
    enable_cache: bool = False,
    smoke_test: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """Execute edge prediction evaluation pipeline for all provided models and graphs.

    Parameters
    ---------------------
    holdouts_kwargs: Dict[str, Any]
        The parameters for the selected holdouts method.
    graphs: Union[str, Graph, List[Graph], List[str]]
        The graphs or graph names to run this evaluation on.
    models: Union[Type[AbstractEdgePredictionModel], List[Type[AbstractEdgePredictionModel]]]
        The models to evaluate.
    evaluation_schema: str = "Connected Monte Carlo"
        The evaluation schema to follow.
    node_features: Optional[Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel], List[Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel]]]]] = None
        The node features to use.
    node_type_features: Optional[Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel], List[Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel]]]]] = None
        The node type features to use.
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
    validation_sample_only_edges_with_heterogeneous_node_types: bool = False
        Whether to sample negative edges exclusively between nodes with different node types.
        This can be useful when executing a bipartite edge prediction task.
    validation_unbalance_rates: Tuple[float] = (1.0, )
        Unbalance rate for the non-existent graphs generation.
    enable_cache: bool = False
        Whether to enable the cache.
    smoke_test: bool = False
        Whether this run should be considered a smoke test
        and therefore use the smoke test configurations for
        the provided model names and feature names.
        This parameter will also turn off the cache.
    verbose: bool = True
        Whether to show loading bars
    """
    return classification_evaluation_pipeline(
        evaluation_schema=evaluation_schema,
        holdouts_kwargs=holdouts_kwargs,
        graphs=graphs,
        models=models,
        expected_parent_class=AbstractEdgePredictionModel,
        node_features=node_features,
        node_type_features=node_type_features,
        library_names=library_names,
        subgraph_of_interest=subgraph_of_interest,
        number_of_holdouts=number_of_holdouts,
        random_state=random_state,
        repositories=repositories,
        versions=versions,
        enable_cache=enable_cache,
        smoke_test=smoke_test,
        verbose=verbose,
        validation_sample_only_edges_with_heterogeneous_node_types=validation_sample_only_edges_with_heterogeneous_node_types,
        validation_unbalance_rates=validation_unbalance_rates
    )
