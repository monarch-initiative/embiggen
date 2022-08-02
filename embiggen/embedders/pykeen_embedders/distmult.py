"""Submodule providing wrapper for PyKeen's DistMult model."""
from typing import Union, Type, Dict, Any, Optional
from pykeen.training import TrainingLoop
from pykeen.models import DistMult
from embiggen.embedders.pykeen_embedders.entity_relation_embedding_model_pykeen import EntityRelationEmbeddingModelPyKeen
from pykeen.triples import CoreTriplesFactory


class DistMultPyKeen(EntityRelationEmbeddingModelPyKeen):

    @classmethod
    def model_name(cls) -> str:
        """Return name of the model."""
        return "DistMult"

    def _build_model(
        self,
        triples_factory: CoreTriplesFactory
    ) -> DistMult:
        """Build new DistMult model for embedding.

        Parameters
        ------------------
        graph: Graph
            The graph to build the model for.
        """
        return DistMult(
            triples_factory=triples_factory,
            embedding_dim=self._embedding_size,
            random_seed=self._random_state
        )
