"""Submodule providing wrapper for PyKeen's TransE model."""
from typing import Union
from ensmallen import Graph
from pykeen.models import EntityRelationEmbeddingModel, ERModel
from embiggen.embedders.pykeen_embedders.pykeen_embedder import PyKeenEmbedder
import pandas as pd
from embiggen.utils.abstract_models import abstract_class, EmbeddingResult


@abstract_class
class EntityRelationEmbeddingModelPyKeen(PyKeenEmbedder):

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
            node_embeddings = [model.entity_embeddings]
            edge_type_embeddings = [model.relation_embeddings]
        elif isinstance(model, ERModel):
            node_embeddings = model.entity_representations
            edge_type_embeddings = model.relation_representations
        else:
            raise NotImplementedError(
                f"The provided model has type {type(model)}, which "
                "is not currently supported. The supported types "
                "are `EntityRelationEmbeddingModel` and `ERModel`."
            )

        node_embeddings = [
            node_embedding._embeddings.weight.cpu().detach().numpy()
            for node_embedding in node_embeddings
        ]

        edge_type_embeddings = [
            edge_type_embedding._embeddings.weight.cpu().detach().numpy()
            for edge_type_embedding in edge_type_embeddings
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