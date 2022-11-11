"""Submodule providing edge prediction evaluation pipeline."""
from typing import Dict, Any, Union, List, Type, Optional, Tuple, Callable
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
    graph_callback: Optional[Callable[[Graph], Graph]] = None,
    subgraph_of_interest: Optional[Graph] = None,
    use_subgraph_as_support: bool = False,
    number_of_holdouts: int = 10,
    random_state: int = 42,
    repositories: Optional[Union[str, List[str]]] = None,
    versions: Optional[Union[str, List[str]]] = None,
    validation_sample_only_edges_with_heterogeneous_node_types: bool = False,
    source_node_types_names: Optional[List[str]] = None,
    destination_node_types_names: Optional[List[str]] = None,
    source_edge_types_names: Optional[List[str]] = None,
    destination_edge_types_names: Optional[List[str]] = None,
    source_nodes_prefixes: Optional[List[str]] = None,
    destination_nodes_prefixes: Optional[List[str]] = None,
    validation_unbalance_rates: Tuple[float] = (1.0, ),
    use_scale_free_distribution: bool = True,
    enable_cache: bool = False,
    precompute_constant_stocastic_features: bool = False,
    smoke_test: bool = False,
    number_of_slurm_nodes: Optional[int] = None,
    slurm_node_id_variable: str = "SLURM_GRAPE_ID",
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
        There are a number of supported evaluation schemas, specifically:
        - Connected Monte Carlo
            A random holdout with repeated sampling of the edges across the different
            repetitions, that assures that there will be exactly the same connected components
            in the training set. This is the ideal evaluation schema when making a closed
            world assumption, that is when you do not want to evaluate the edge prediction performance
            of a model for edges between distinct connected components.
            This is generally used expecially when you do not have additional node features that may
            help the model learn that two different components are connected.
        - Monte Carlo
            A random holdout with repeated sampling of the edges across the different 
            repetitions which DOES NOT HAVE any assurance about creating new connected components.
            This is a correct evaluation schema when you want to evaluate the edge prediction
            performance of a model across different connected components, which you may want to
            do when you have additional node features that may
            help the model learn that two different components are connected.
        - Kfold
            A k-fold holdout which will split the set of edges into k different `folds`, where
            k is the total number of holdouts that will be executed.
            Do note that this procedure DOES NOT HAVE any assurance about creating new connected components.
            This is a correct evaluation schema when you want to evaluate the edge prediction
            performance of a model across different connected components, which you may want to
            do when you have additional node features that may
            help the model learn that two different components are connected.
    node_features: Optional[Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel], List[Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel]]]]] = None
        The node features to use.
    node_type_features: Optional[Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel], List[Union[str, pd.DataFrame, np.ndarray, Type[AbstractEmbeddingModel]]]]] = None
        The node type features to use.
    library_names: Optional[Union[str, List[str]]] = None
        Library names from where to retrieve the provided model names.
    graph_callback: Optional[Callable[[Graph], Graph]] = None
        Callback to use for graph normalization and sanitization, must be
        a function that receives and returns a graph object.
        For instance this may be used for filtering the uncertain edges
        in graphs such as STRING PPIs.
    subgraph_of_interest: Optional[Graph] = None
        Optional subgraph where to focus the task.
        This is applied to the train and test graph
        after the desired holdout schema is applied.
    use_subgraph_as_support: bool = False
        Whether to use the provided subgraph as support or
        to use the train graph (not filtered by the subgraph).
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
    source_node_types_names: Optional[List[str]]
        Node type names of the nodes to be samples as sources.
        If a node has any of the provided node types, it can be sampled as a source node.
    destination_node_types_names: Optional[List[str]]
        Node type names of the nodes to be samples as destinations.
        If a node has any of the provided node types, it can be sampled as a destination node.
    source_edge_types_names: Optional[List[str]]
        Edge type names of the nodes to be samples as sources.
        If a node has any of the provided edge types, it can be sampled as a source node.
    destination_edge_types_names: Optional[List[str]]
        Edge type names of the nodes to be samples as destinations.
        If a node has any of the provided edge types, it can be sampled as a destination node.
    source_nodes_prefixes: Optional[List[str]]
        Prefixes of the nodes names to be samples as sources.
        If a node starts with any of the provided prefixes, it can be sampled as a source node.
    destination_nodes_prefixes: Optional[List[str]]
        Prefixes of the nodes names to be samples as destinations.
        If a node starts with any of the provided prefixes, it can be sampled as a destinations node.
    validation_unbalance_rates: Tuple[float] = (1.0, )
        Unbalance rate for the non-existent graphs generation.
    use_scale_free_distribution: bool = True
        Whether to use the scale free sampling of the NEGATIVE edges for the EVALUATION
        of the edge prediction performance of the provided models.
        Please DO BE ADVISED that not using a scale free sampling for the negative
        edges is a poor choice and will cause a significant positive bias
        in the model performance.
    enable_cache: bool = False
        Whether to enable the cache.
    precompute_constant_stocastic_features: bool = False
        Whether to precompute once the constant automatic stocastic
        features before starting the embedding loop. This means that,
        when left set to false, while the features will be computed
        using the same input data, the random state between runs will
        be different and therefore the experiment performance will
        capture more of the variance derived from the stocastic aspect
        of the considered method. When set to true, they are only computed
        once and therefore the experiment will be overall faster.
    smoke_test: bool = False
        Whether this run should be considered a smoke test
        and therefore use the smoke test configurations for
        the provided model names and feature names.
        This parameter will also turn off the cache.
    number_of_slurm_nodes: Optional[int] = None
        Number of SLURM nodes to consider as available.
        This variable is used to parallelize the holdouts accordingly.
    slurm_node_id_variable: str = "SLURM_GRAPE_ID"
        Name of the system variable to use as SLURM node id.
        It must be set in the slurm bash script.
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
        graph_callback=graph_callback,
        subgraph_of_interest=subgraph_of_interest,
        use_subgraph_as_support=use_subgraph_as_support,
        number_of_holdouts=number_of_holdouts,
        random_state=random_state,
        repositories=repositories,
        versions=versions,
        enable_cache=enable_cache,
        precompute_constant_stocastic_features=precompute_constant_stocastic_features,
        smoke_test=smoke_test,
        number_of_slurm_nodes=number_of_slurm_nodes,
        slurm_node_id_variable=slurm_node_id_variable,
        verbose=verbose,
        validation_sample_only_edges_with_heterogeneous_node_types=validation_sample_only_edges_with_heterogeneous_node_types,
        source_node_types_names=source_node_types_names,
        destination_node_types_names=destination_node_types_names,
        source_edge_types_names=source_edge_types_names,
        destination_edge_types_names=destination_edge_types_names,
        source_nodes_prefixes=source_nodes_prefixes,
        destination_nodes_prefixes=destination_nodes_prefixes,
        validation_unbalance_rates=validation_unbalance_rates,
        use_scale_free_distribution=use_scale_free_distribution
    )
