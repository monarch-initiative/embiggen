"""Submodule providing wrapper for PyKeen's ProjE model."""
from typing import Union, Type, Dict, Any, Optional
from pykeen.training import TrainingLoop
from pykeen.models import ProjE
from embiggen.embedders.pykeen_embedders.entity_relation_embedding_model_pykeen import EntityRelationEmbeddingModelPyKeen
from pykeen.triples import CoreTriplesFactory


class ProjEPyKeen(EntityRelationEmbeddingModelPyKeen):

    @classmethod
    def model_name(cls) -> str:
        """Return name of the model."""
        return "ProjE"

    def _build_model(
        self,
        triples_factory: CoreTriplesFactory
    ) -> ProjE:
        """Build new ProjE model for embedding.

        Parameters
        ------------------
        graph: Graph
            The graph to build the model for.
        """
        return ProjE(
            triples_factory=triples_factory,
            embedding_dim=self._embedding_size,
            random_seed=self._random_state
        )
