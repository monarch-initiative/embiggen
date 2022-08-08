"""Submodule providing wrapper for PyKEEN's TransE model."""
from typing import Union, Type, List
from ensmallen import Graph
from pykeen.models import ERModel
from pykeen.nn.representation import Representation
from embiggen.embedders.pykeen_embedders.pykeen_embedder import PyKEENEmbedder
import pandas as pd
from embiggen.utils.abstract_models import abstract_class, EmbeddingResult

try:
    from pykeen.models import EntityRelationEmbeddingModel
except ImportError:
    # The following is just to patch the removal of the
    # class EntityRelationEmbeddingModel in PyKEEN.
    class EntityRelationEmbeddingModel:
        pass

@abstract_class
class EntityRelationEmbeddingModelPyKEEN(PyKEENEmbedder):

    def _extract_embeddings(
        self,
        graph: Graph,
        model: Union[EntityRelationEmbeddingModel, ERModel],
        return_dataframe: bool
    ) -> EmbeddingResult:
        """Returns embedding from the model.

        Parameters
        ------------------
        graph: Graph
            The graph that was embedded.
        model: Type[Model]
            The Keras model used to embed the graph.
        return_dataframe: bool
            Whether to return a dataframe of a numpy array.
        """
        if isinstance(model, EntityRelationEmbeddingModel):
            node_embeddings: List[Type[Representation]] = [model.entity_embeddings]
            edge_type_embeddings: List[Type[Representation]] = [model.relation_embeddings]
        elif isinstance(model, ERModel):
            node_embeddings: List[Type[Representation]] = model.entity_representations
            edge_type_embeddings: List[Type[Representation]] = model.relation_representations
        else:
            raise NotImplementedError(
                f"The provided model has type {type(model)}, which "
                "is not currently supported. The supported types "
                "are `EntityRelationEmbeddingModel` and `ERModel`."
            )

        node_embeddings = [
            array
            for array in (
                node_embedding().cpu().detach().numpy().reshape((
                    graph.get_number_of_nodes(),
                    -1
                ))
                for node_embedding in node_embeddings
            )
            if array.size > 0
        ]

        edge_type_embeddings = [
            array
            for array in (
                edge_type_embedding().cpu().detach().numpy().reshape((
                    graph.get_number_of_edge_types(),
                    -1
                ))
                for edge_type_embedding in edge_type_embeddings
            )
            if array.size > 0
        ]

        if return_dataframe:
            node_embeddings = [
                pd.DataFrame(
                    node_embedding,
                    index=graph.get_node_names()
                )
                for node_embedding in node_embeddings
            ]

            edge_type_embeddings = [
                pd.DataFrame(
                    edge_type_embedding,
                    index=graph.get_unique_edge_type_names()
                )
                for edge_type_embedding in edge_type_embeddings
            ]

        return EmbeddingResult(
            embedding_method_name=self.model_name(),
            node_embeddings=node_embeddings,
            edge_type_embeddings=edge_type_embeddings
        )