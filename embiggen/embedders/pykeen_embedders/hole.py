"""Submodule providing wrapper for PyKeen's HolE model."""
from typing import Union, Type, Dict, Any
from pykeen.training import TrainingLoop
from pykeen.models import HolE
from embiggen.embedders.pykeen_embedders.entity_relation_embedding_model_pykeen import EntityRelationEmbeddingModelPyKeen
from pykeen.triples import CoreTriplesFactory


class HolEPyKeen(EntityRelationEmbeddingModelPyKeen):

    @classmethod
    def model_name(cls) -> str:
        """Return name of the model."""
        return "HolE"

    def _build_model(
        self,
        triples_factory: CoreTriplesFactory
    ) -> HolE:
        """Build new HolE model for embedding.

        Parameters
        ------------------
        graph: Graph
            The graph to build the model for.
        """
        return HolE(
            triples_factory=triples_factory,
            embedding_dim=self._embedding_size,
            random_seed=self._random_state
        )
