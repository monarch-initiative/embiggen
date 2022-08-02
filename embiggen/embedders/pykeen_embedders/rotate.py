"""Submodule providing wrapper for PyKeen's RotatE model."""
from typing import Union, Type, Dict, Any
from pykeen.training import TrainingLoop
from pykeen.models import RotatE
from embiggen.embedders.pykeen_embedders.entity_relation_embedding_model_pykeen import EntityRelationEmbeddingModelPyKeen
from pykeen.triples import CoreTriplesFactory


class RotatEPyKeen(EntityRelationEmbeddingModelPyKeen):

    @classmethod
    def model_name(cls) -> str:
        """Return name of the model."""
        return "RotatE"

    def _build_model(
        self,
        triples_factory: CoreTriplesFactory
    ) -> RotatE:
        """Build new RotatE model for embedding.

        Parameters
        ------------------
        graph: Graph
            The graph to build the model for.
        """
        return RotatE(
            triples_factory=triples_factory,
            embedding_dim=self._embedding_size,
            random_seed=self._random_state
        )
