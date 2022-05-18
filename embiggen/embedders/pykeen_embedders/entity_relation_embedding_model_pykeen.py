"""Submodule providing wrapper for PyKeen's TransE model."""
from typing import Union, Dict, Any
from ensmallen import Graph
from pykeen.models import EntityRelationEmbeddingModel
from .pykeen_embedder import PyKeenEmbedder
import numpy as np
import pandas as pd
from ...utils import abstract_class


@abstract_class
class EntityRelationEmbeddingModelPyKeen(PyKeenEmbedder):

    @staticmethod
    def smoke_test_parameters() -> Dict[str, Any]:
        """Returns parameters for smoke test."""
        return dict(
            embedding_size=5,
            epochs=1,
            scoring_fct_norm=1
        )

    def parameters(self) -> Dict[str, Any]:
        return {
            **super().parameters(),
            **dict(
                scoring_fct_norm=self._scoring_fct_norm
            )
        }

    def _extract_embeddings(
        self,
        graph: Graph,
        model: EntityRelationEmbeddingModel,
        return_dataframe: bool
    ) -> Union[Dict[str, np.ndarray], Dict[str, pd.DataFrame]]:
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
        if return_dataframe:
            return {
                "node_embedding": pd.DataFrame(
                    model.entity_embeddings._embeddings.weight.cpu().detach().numpy(),
                    index=graph.get_node_names()
                ),
                "edge_type_embedding": pd.DataFrame(
                    model.relation_embeddings._embeddings.weight.cpu().detach().numpy(),
                    index=graph.get_unique_edge_type_names()
                )
            }

        return {
            "node_embedding": model.entity_embeddings._embeddings.weight.cpu().detach().numpy(),
            "edge_type_embedding": model.relation_embeddings._embeddings.weight.cpu().detach().numpy(),
        }
