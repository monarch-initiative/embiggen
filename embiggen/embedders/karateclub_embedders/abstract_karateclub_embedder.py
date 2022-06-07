"""Submodule providing wrapper for the Karate Club models."""
from typing import Type, Dict, Any
from ensmallen import Graph
import numpy as np
import pandas as pd
from karateclub.estimator import Estimator
from embiggen.utils.abstract_models import AbstractEmbeddingModel, EmbeddingResult, abstract_class
from embiggen.utils.networkx_utils import convert_ensmallen_graph_to_networkx_graph

@abstract_class
class AbstractKarateClubEmbedder(AbstractEmbeddingModel):

    def __init__(
        self,
        embedding_size: int,
        enable_cache: bool = False
    ):
        """Create new embedding model.

        Parameters
        ---------------------
        embedding_size: int
            The dimensionality of the embedding.
        enable_cache: bool = False
            Whether to enable the cache, that is to
            store the computed embedding.
        """
        super().__init__(
            embedding_size=embedding_size,
            enable_cache=enable_cache
        )

    @staticmethod
    def smoke_test_parameters() -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            embedding_size=2,
        )

    @staticmethod
    def library_name() -> str:
        return "Karate Club"

    @staticmethod
    def task_name() -> str:
        return "Node Embedding"

    def _build_model(self) -> Type[Estimator]:
        """Returnd the built estimator."""
        raise NotImplementedError(
            f"In the child class {self.__class__.__name__} of {super().__name__.__name__} "
            f"implementing the model {self.model_name()} we could not find the method "
            "called `_build_model`. Please do implement it."
        )

    def _fit_transform(
        self,
        graph: Graph,
        return_dataframe: bool = True,
        verbose: bool = True
    ) -> EmbeddingResult:
        """Return node embedding.

        Parameters
        ---------------
        graph: Graph
            The graph to embed.
        return_dataframe: bool = True
            Whether to return a DataFrame.
        verbose: bool = True
            Whether to show a loading bar.
        """
        model: Type[Estimator] = self._build_model()

        if not issubclass(model.__class__, Estimator):
            raise NotImplementedError(
                "The model created with the `_build_model` in the child "
                f"class {self.__class__.__name__} for the model {self.model_name()} "
                f"in the library {self.library_name()} did not return a "
                f"Estimator but an object of type {type(model)}. "
                "It is not clear what to do with this object."
            )
        
        model.fit(convert_ensmallen_graph_to_networkx_graph(graph))

        node_embeddings: np.ndarray = model.get_embedding()

        if not issubclass(node_embeddings.__class__, np.ndarray):
            raise NotImplementedError(
                "The model created with the `get_embedding` in the child "
                f"class {self.__class__.__name__} for the model {self.model_name()} "
                f"in the library {self.library_name()} did not return a "
                f"Numpy Array but an object of type {type(model)}. "
                "It is not clear what to do with this object."
            )

        if return_dataframe:
            node_embeddings: pd.DataFrame = pd.DataFrame(
                node_embeddings,
                index=graph.get_node_names()
            )

        return EmbeddingResult(
            embedding_method_name=self.model_name(),
            node_embeddings=node_embeddings
        )
