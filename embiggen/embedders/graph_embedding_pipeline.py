"""Module offering pipeline to embed graph nodes, node types and edge types."""
from typing import Union, Dict, Optional, Type
from ensmallen import Graph

from embiggen.utils.pipeline import iterate_graphs
from embiggen.utils.abstract_models import AbstractEmbeddingModel, EmbeddingResult



def embed_graph(
    graph: Union[Graph, str],
    embedding_model: Union[str, Type[AbstractEmbeddingModel]],
    repository: Optional[str] = None,
    version: Optional[str] = None,
    library_name: Optional[str] = None,
    smoke_test: bool = False,
    return_dataframe: bool = True,
    verbose: bool = True,
    **kwargs: Dict
) -> EmbeddingResult:
    """Return embedding of the provided graph.

    Parameters
    ---------------------
    graph: Union[Graph, str]
        The graph or graph name to embed.
        If a graph name is provided, it will be retrieved from Ensmallen's automatic retrieval.
    embedding_model: Union[str, Type[AbstractEmbeddingModel]]
        Model or model name to use.
        If a model name is provided, it will be retrieved from the models' library.
    repository: Optional[str] = None
        Repository from where to retrieve the provided graph names
        from the Ensmallen automatic retrieval.
    version: Optional[str] = None
        Graph version to retrieve.
    library_name: Optional[str] = None
        The library from where to retrieve the embedding model.
    enable_cache: bool = False
        Whether to enable the cache.
    smoke_test: bool = False
        Whether this run should be considered a smoke test
        and therefore use the smoke test configurations for
        the provided model names and feature names.
    cache_directory: str = "embedding"
        Path where to store the cache if it is enabled.
    return_dataframe: bool = True
        Whether to return a pandas DataFrame with the embedding.
    verbose: bool
        Whether to show loading bars.
    **kwargs: Dict
        Kwargs to forward to the embedding model creation.
        If a model name was NOT provided, an exception will
        be raised as it is unclear how to behave.
    """

    graph = next(iterate_graphs(
        graphs=graph,
        repositories=repository,
        versions=version
    ))

    if isinstance(embedding_model, str):
        embedding_model: Type[AbstractEmbeddingModel] = AbstractEmbeddingModel.get_model_from_library(
            model_name=embedding_model,
            library_name=library_name
        )(**kwargs)
    elif kwargs is not None:
        raise ValueError(
            "Please be advised that even though you have provided yourself "
            "the embedding model, you have also provided the kwargs which "
            "would normally be forwarded to the creation of the embedding "
            "model. It is unclear what to do with these arguments."
        )

    if not issubclass(embedding_model.__class__, AbstractEmbeddingModel):
        raise ValueError(
            "The provided object is not an embedding model, that is, "
            "it does not extend the class `AbstractEmbeddingModel`."
        )

    if smoke_test:
        try:
            embedding_model = embedding_model.__class__(
                **embedding_model.smoke_test_parameters()
            )
        except Exception as e:
            raise ValueError(
                "An exception was raised while trying to create "
                f"a smoke test version of the model called {embedding_model.model_name()} "
                f"from the library {library_name}, specifically "
                f"implemented in the class {embedding_model.__class__.__name__}. "
                "Most likely there is an error in this class smoke test parameters. "
                f"The body of the exception was: {str(e)}. "
            ) from e

    if embedding_model.requires_nodes_sorted_by_decreasing_node_degree():
        graph = graph.sort_by_decreasing_outbound_node_degree()

    try:
        return embedding_model.fit_transform(
            graph,
            return_dataframe=return_dataframe,
            verbose=verbose
        )
    except Exception as e:
        raise ValueError(
            "An exception was raised while trying to compute "
            f"a node embedding on the graph {graph.get_name()} "
            f"using the model called {embedding_model.model_name()} "
            f"from the library {library_name}, specifically "
            f"implemented in the class {embedding_model.__class__.__name__}. "
            f"The body of the exception was: {str(e)}."
        ) from e