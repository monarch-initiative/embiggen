"""Submodule providing wrapper for PyKeen's TransE model."""
from typing import Union, Type, Dict, Any
from ensmallen import Graph
from pykeen.models import TransE
from .entity_relation_embedding_model_pykeen import EntityRelationEmbeddingModelPyKeen
from pykeen.training import TrainingLoop
from torch.optim import Optimizer


class TransEPyKeen(EntityRelationEmbeddingModelPyKeen):

    @staticmethod
    def model_name() -> str:
        """Return name of the model."""
        return "TransE"

    def _build_model(self, graph: Graph) -> TransE:
        """Build new TransE model for embedding.

        Parameters
        ------------------
        graph: Graph
            The graph to build the model for.
        """
        return TransE(
            embedding_dim=self._embedding_size,
            scoring_fct_norm=self._scoring_fct_norm
        )