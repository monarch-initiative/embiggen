"""Submodule providing wrapper for PyKeen's ComplEx model."""
from typing import Union, Type, Dict, Any, Optional
from pykeen.training import TrainingLoop
from pykeen.models import ComplEx
from embiggen.embedders.pykeen_embedders.entity_relation_embedding_model_pykeen import EntityRelationEmbeddingModelPyKeen
from pykeen.triples import CoreTriplesFactory


class ComplExPyKeen(EntityRelationEmbeddingModelPyKeen):

    @classmethod
    def model_name(cls) -> str:
        """Return name of the model."""
        return "ComplEx"

    def _build_model(
        self,
        triples_factory: CoreTriplesFactory
    ) -> ComplEx:
        """Build new ComplEx model for embedding.

        Parameters
        ------------------
        graph: Graph
            The graph to build the model for.
        """
        return ComplEx(
            triples_factory=triples_factory,
            embedding_dim=self._embedding_size,
            random_seed=self._random_state
        )
